# ===== ESPN · schnelle Ergebnisse (bis football-data.co.uk nachzieht) =====
# Kostenlose, inoffizielle ESPN-Scoreboard-Schnittstelle (kein Schluessel noetig).
# Fuellt NUR die Anzeige: Endstand, Halbzeitstand und (falls vorhanden) Ecken fuer vergangene Spiele,
# die bei football-data noch kein Ergebnis haben. Die Modelle trainieren weiter nur mit football-data.
# Faellt ESPN aus oder aendert sich das Format, passiert nichts - die Seite zeigt dann "Ergebnis folgt".
from datetime import datetime, timedelta

import requests

import odds_api   # Namens-Abgleich (_aehnlich) wiederverwenden

B = "https://site.api.espn.com/apis/site/v2/sports/soccer/{}/scoreboard"
TAGE_ZURUECK = 4
LIGEN = {"D1": "ger.1", "D2": "ger.2", "E0": "eng.1", "E1": "eng.2", "E2": "eng.3", "E3": "eng.4", "EC": "eng.5",
         "SP1": "esp.1", "SP2": "esp.2", "I1": "ita.1", "I2": "ita.2", "F1": "fra.1", "F2": "fra.2", "N1": "ned.1",
         "P1": "por.1", "B1": "bel.1", "G1": "gre.1", "SC0": "sco.1", "SC1": "sco.2", "SC2": "sco.3", "SC3": "sco.4",
         "DNK": "den.1", "NOR": "nor.1", "SWE": "swe.1", "FIN": "fin.1", "AUT": "aut.1", "SWZ": "sui.1",
         "POL": "pol.1", "ROU": "rou.1", "IRL": "irl.1", "RUS": "rus.1", "USA": "usa.1", "MEX": "mex.1",
         "BRA": "bra.1", "ARG": "arg.1", "JPN": "jpn.1", "CHN": "chn.1"}


# Namen, die sich zwischen football-data und ESPN stark unterscheiden
ALIAS = {"aarhus": "agf", "fc copenhagen": "fc kobenhavn", "la coruna": "deportivo", "atl. san luis": "atletico san luis",
         "unam pumas": "pumas unam", "tijuana": "club tijuana", "los angeles fc": "lafc", "brondby": "brondby if",
         "nordsjaelland": "fc nordsjaelland", "midtjylland": "fc midtjylland", "cfr cluj": "cfr 1907 cluj",
         "fc rapid bucuresti": "rapid bucuresti", "csikszereda m. ciuc": "csikszereda", "wisla plock": "wisla plock",
         "vps": "vps vaasa", "hjk": "hjk helsinki", "ac oulu": "ac oulu", "st. gallen": "st gallen"}
ALLE = "all"          # ESPN-Sammel-Scoreboard (alle Ligen eines Tages) als Ersatz, wenn eine Liga leer bleibt


def _n(x):
    k = str(x).strip().lower()
    return ALIAS.get(k, k)


def _passt(e, s):
    wh, wa = odds_api._aehnlich(_n(e["heim"]), _n(s["heim"])), odds_api._aehnlich(_n(e["gast"]), _n(s["gast"]))
    # beide aehnlich - oder ein Team eindeutig (jedes Team spielt hoechstens einmal am Tag)
    ok = min(wh, wa) >= 0.6 or (max(wh, wa) >= 0.85 and not s.get("alle"))   # Sammel-Scoreboard: beide Teams muessen passen
    return (wh + wa) / 2 if ok else 0.0


def _zahl(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _stat(c, namen):
    for s in c.get("statistics") or []:
        if s.get("name") in namen:
            return _zahl(s.get("displayValue", s.get("value")))
    return None


def _spiele(slug, tag):
    r = requests.get(B.format(slug), params={"dates": tag.strftime("%Y%m%d")}, timeout=20,
                     headers={"User-Agent": "Mozilla/5.0 (score-code-agent)"})
    if r.status_code != 200:
        return None
    out = []
    for ev in r.json().get("events", []):
        try:
            comp = ev["competitions"][0]
            if not comp.get("status", ev.get("status", {})).get("type", {}).get("completed"):
                continue
            t = {c["homeAway"]: c for c in comp["competitors"]}
            h, a = t["home"], t["away"]
            sh, sa = _zahl(h.get("score")), _zahl(a.get("score"))
            if sh is None or sa is None:
                continue
            ht = None
            lh, la = h.get("linescores") or [], a.get("linescores") or []
            if lh and la:
                x, y = _zahl(lh[0].get("value")), _zahl(la[0].get("value"))
                if x is not None and y is not None:
                    ht = f"{x}:{y}"
            eh, ea = _stat(h, ("wonCorners", "cornerKicks", "corners")), _stat(a, ("wonCorners", "cornerKicks", "corners"))
            out.append({"heim": h["team"].get("displayName") or h["team"].get("name"),
                        "gast": a["team"].get("displayName") or a["team"].get("name"),
                        "datum": ev.get("date", "")[:10], "ergebnis": f"{sh}:{sa}", "ht": ht,
                        "ecken": [eh, ea] if eh is not None and ea is not None else None})
        except Exception:
            continue
    return out


def ergaenzen(liste, heute, log=print):
    """Setzt Ergebnisse fuer vergangene Spiele ohne Ergebnis. Gibt die Anzahl zurueck."""
    offen = [e for e in liste if not e.get("ergebnis") and e["datum"] < heute.strftime("%Y-%m-%d")
             and e["datum"] >= (heute - timedelta(days=TAGE_ZURUECK)).strftime("%Y-%m-%d") and e["liga"] in LIGEN]
    if not offen:
        log("ESPN: keine offenen Ergebnisse")
        return 0
    n_ges, stat, alle_cache = 0, {}, {}
    for liga in sorted({e["liga"] for e in offen}):
        kand = [e for e in offen if e["liga"] == liga]
        tage = sorted({e["datum"] for e in kand})
        espn = []
        for d in {(datetime.strptime(t, "%Y-%m-%d") + timedelta(days=k)) for t in tage for k in (0, 1)}:
            try:
                x = _spiele(LIGEN[liga], d)
            except Exception as ex:
                x = None
                log(f"ESPN {liga}: Fehler {ex}")
            if x:
                espn += x
        if not espn:      # Liga bei ESPN leer -> Sammel-Scoreboard aller Ligen probieren
            for d in {datetime.strptime(t, "%Y-%m-%d") + timedelta(days=k) for t in tage for k in (0, 1)}:
                if d not in alle_cache:
                    try:
                        alle_cache[d] = [dict(x, alle=True) for x in (_spiele(ALLE, d) or [])]
                    except Exception:
                        alle_cache[d] = []
                espn += alle_cache[d]
        n = 0
        for e in kand:
            best, wert = None, 0.0
            for s in espn:
                if abs((datetime.strptime(s["datum"], "%Y-%m-%d") - datetime.strptime(e["datum"], "%Y-%m-%d")).days) > 1:
                    continue
                w = _passt(e, s)
                if w > wert:
                    best, wert = s, w
            if best is not None and wert >= 0.45:
                e["ergebnis"] = best["ergebnis"]
                if best["ht"] and not e.get("ergebnis_ht"):
                    e["ergebnis_ht"] = best["ht"]
                if best["ecken"] and not e.get("ecken"):
                    e["ecken"] = best["ecken"]
                e["equelle"] = "ESPN"
                n += 1
        stat[liga] = f"{n}/{len(kand)}"
        if n < len(kand):     # Diagnose: was liefert ESPN, was fehlt?
            fehlt = [f'{e["datum"]} {e["heim"]}-{e["gast"]}' for e in kand if not e.get("equelle")][:3]
            bsp = [f'{s["datum"]} {s["heim"]}-{s["gast"]}' for s in espn][:6]
            log(f"   ESPN {liga} ({LIGEN[liga]}): {len(espn)} fertige Spiele bei ESPN | fehlt z. B.: {fehlt} | ESPN z. B.: {bsp}")
        n_ges += n
    log(f"ESPN: {n_ges} von {len(offen)} offenen Ergebnissen ergänzt | " + ", ".join(f"{k} {v}" for k, v in stat.items()))
    return n_ges
