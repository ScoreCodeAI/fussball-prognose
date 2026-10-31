# ===== THE ODDS API · aktuelle Quoten fuer die kommenden Spiele =====
# - Nur Ligen, die in den naechsten STUNDEN Spiele haben, werden abgerufen (je Liga 2 Credits: 1X2 + OVER/UNDER).
#   Der Spielplan-Abruf (/events) ist kostenlos.
# - Je Spiel: hoechste Quote je Ausgang ueber alle Buchmacher, Pinnacle (oder Durchschnitt) fuer den Spielkopf.
# - OVER/UNDER nur von Buchmachern mit genau der Linie 2,5 (viele liefern 3,0 / 3,25 als Hauptlinie).
# - Alles wird in docs/odds_archiv.json gespeichert -> vergangene Spiele kosten keine Credits.
# - Teamnamen werden ueber Aehnlichkeit + Datum zugeordnet. Nicht zugeordnet -> football-data bleibt.
import difflib
import json
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone

import requests

B = "https://api.the-odds-api.com/v4/"
STUNDEN = 30          # Spiele bis so viele Stunden nach dem Update (Update laeuft frueh morgens)
MIN_CREDITS = 6       # Reserve: darunter wird nichts mehr abgerufen
JETZT = None          # nur fuer Tests: feste Uhrzeit
LIGEN = {"D1": "soccer_germany_bundesliga", "D2": "soccer_germany_bundesliga2", "E0": "soccer_epl",
         "E1": "soccer_efl_champ", "E2": "soccer_england_league1", "E3": "soccer_england_league2",
         "SP1": "soccer_spain_la_liga", "SP2": "soccer_spain_segunda_division", "I1": "soccer_italy_serie_a",
         "I2": "soccer_italy_serie_b", "F1": "soccer_france_ligue_one", "F2": "soccer_france_ligue_two",
         "N1": "soccer_netherlands_eredivisie", "P1": "soccer_portugal_primeira_liga", "B1": "soccer_belgium_first_div",
         "G1": "soccer_greece_super_league", "SC0": "soccer_spl", "DNK": "soccer_denmark_superliga",
         "NOR": "soccer_norway_eliteserien", "SWE": "soccer_sweden_allsvenskan", "FIN": "soccer_finland_veikkausliiga",
         "AUT": "soccer_austria_bundesliga", "SWZ": "soccer_switzerland_superleague", "POL": "soccer_poland_ekstraklasa",
         "IRL": "soccer_league_of_ireland", "RUS": "soccer_russia_premier_league", "USA": "soccer_usa_mls",
         "MEX": "soccer_mexico_ligamx", "BRA": "soccer_brazil_campeonato", "ARG": "soccer_argentina_primera_division",
         "JPN": "soccer_japan_j_league", "CHN": "soccer_china_superleague"}
WEG = {"fc", "cf", "sc", "ac", "afc", "ssc", "as", "us", "sv", "vfb", "vfl", "tsg", "fk", "if", "bk", "sk", "cd", "ud",
       "rc", "rcd", "club", "de", "the", "1", "04", "05", "1899", "1900", "1909", "1846", "calcio", "bc", "sd"}


ALIAS = {"m'gladbach": "monchengladbach", "wolves": "wolverhampton", "paris sg": "paris saint germain",
         "nott'm forest": "nottingham forest", "ein frankfurt": "eintracht frankfurt", "ath madrid": "atletico madrid",
         "ath bilbao": "athletic bilbao", "sheffield weds": "sheffield wednesday", "qpr": "queens park rangers",
         "west brom": "west bromwich", "psv eindhoven": "psv", "sp lisbon": "sporting", "fortuna dusseldorf": "dusseldorf"}


def _norm(n):
    n = str(n).strip().lower()
    n = ALIAS.get(n, n)
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    n = n.replace("utd", "united").replace("man ", "manchester ").replace("st ", "saint ").replace("&", "and")
    w = [x for x in re.split(r"[^a-z0-9]+", n) if x and x not in WEG]
    return " ".join(w)


def _aehnlich(a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.95
    return difflib.SequenceMatcher(None, a, b).ratio()


def _auswerten(ev):
    """Ein Spiel der Odds API -> {x: Hoechstquoten, ou: Hoechstquoten 2,5, kopf: Pinnacle/Durchschnitt, n}"""
    h, a = ev["home_team"], ev["away_team"]
    x, ou, pin, alle = [[], [], []], [[], []], None, []
    for bm in ev.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m["key"] == "h2h":
                d = {o["name"]: o["price"] for o in m["outcomes"]}
                if h in d and a in d and "Draw" in d:
                    q = [d[h], d["Draw"], d[a]]
                    for i in range(3):
                        x[i].append(q[i])
                    alle.append(q)
                    if bm["key"] == "pinnacle":
                        pin = q
            elif m["key"] == "totals":
                d = {o["name"]: o["price"] for o in m["outcomes"] if abs(float(o.get("point", 0)) - 2.5) < 1e-9}
                if "Over" in d and "Under" in d:
                    ou[0].append(d["Over"]); ou[1].append(d["Under"])
    if not alle:
        return None
    kopf = pin or [round(sum(q[i] for q in alle) / len(alle), 2) for i in range(3)]
    return {"x": [max(v) for v in x], "ou": [max(v) for v in ou] if ou[0] else None, "kopf": kopf,
            "n": len(alle), "n_ou": len(ou[0]), "zeit": ev["commence_time"]}


def _key(liga, datum, heim, gast):
    return f"{liga}|{datum}|{heim}|{gast}"


def holen(fx_spiele, archiv_pfad, log=print):
    """fx_spiele: Liste kommender Spiele (liga, datum 'YYYY-MM-DD', heim, gast).
    Gibt (archiv, status) zurueck. archiv: {key: auswertung}."""
    try:
        archiv = json.load(open(archiv_pfad, encoding="utf-8"))
    except Exception:
        archiv = {}
    status = {"aktiv": False, "rest": None, "verbraucht": None, "abgerufen": [], "zeit": None}
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        log("Odds API: kein Schluessel (ODDS_API_KEY) - nur football-data-Quoten")
        return archiv, status
    status["aktiv"] = True
    jetzt = JETZT or datetime.now(timezone.utc)
    bis = jetzt + timedelta(hours=STUNDEN)
    ligen = sorted({s["liga"] for s in fx_spiele if s["liga"] in LIGEN})
    for liga in ligen:
        sport = LIGEN[liga]
        try:
            r = requests.get(B + f"sports/{sport}/events", params={"apiKey": key,
                             "commenceTimeFrom": jetzt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "commenceTimeTo": bis.strftime("%Y-%m-%dT%H:%M:%SZ")}, timeout=30)
            evs = r.json() if r.status_code == 200 else []
        except Exception:
            evs = []
        if not evs:
            continue
        heute_txt = jetzt.strftime("%Y-%m-%d")
        if any(k.startswith(liga + "|") and v.get("abruf", "").startswith(heute_txt) for k, v in archiv.items()):
            log(f"Odds API {liga}: heute schon abgerufen - keine Credits verbraucht")
            continue
        rest = status["rest"]
        if rest is not None and rest < MIN_CREDITS:
            log(f"Odds API: Credits fast aufgebraucht ({rest}) - {liga} uebersprungen")
            continue
        try:
            r = requests.get(B + f"sports/{sport}/odds", params={"apiKey": key, "regions": "eu",
                             "markets": "h2h,totals", "oddsFormat": "decimal",
                             "commenceTimeFrom": jetzt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "commenceTimeTo": bis.strftime("%Y-%m-%dT%H:%M:%SZ")}, timeout=30)
        except Exception as ex:
            log(f"Odds API {liga}: Fehler {ex}")
            continue
        status["rest"] = _int(r.headers.get("x-requests-remaining"))
        status["verbraucht"] = _int(r.headers.get("x-requests-used"))
        if r.status_code != 200:
            log(f"Odds API {liga}: Status {r.status_code}")
            continue
        n = 0
        kandidaten = [s for s in fx_spiele if s["liga"] == liga]
        for ev in r.json():
            ausw = _auswerten(ev)
            if not ausw:
                continue
            tag = ev["commence_time"][:10]
            best, wert = None, 0.0
            for s in kandidaten:
                if abs((datetime.fromisoformat(s["datum"]) - datetime.fromisoformat(tag)).days) > 1:
                    continue
                wh, wa = _aehnlich(s["heim"], ev["home_team"]), _aehnlich(s["gast"], ev["away_team"])
                # beide aehnlich - oder ein Team eindeutig und das andere zumindest grob (jedes Team spielt nur 1x am Tag)
                ok = min(wh, wa) >= 0.6 or (max(wh, wa) >= 0.9 and min(wh, wa) >= 0.35)
                w = (wh + wa) / 2 if ok else 0.0
                if w > wert:
                    best, wert = s, w
            if best is not None and wert >= 0.6:
                ausw["abruf"] = jetzt.strftime("%Y-%m-%d %H:%M")
                archiv[_key(liga, best["datum"], best["heim"], best["gast"])] = ausw
                n += 1
        status["abgerufen"].append(f"{liga}:{n}")
        log(f"Odds API {liga}: {n} Spiele zugeordnet | Credits uebrig {status['rest']}")
    status["zeit"] = jetzt.strftime("%d.%m. %H:%M")
    # Archiv klein halten: nur letzte 60 Tage
    grenze = (jetzt - timedelta(days=60)).strftime("%Y-%m-%d")
    archiv = {k: v for k, v in archiv.items() if k.split("|")[1] >= grenze}
    os.makedirs(os.path.dirname(archiv_pfad), exist_ok=True)
    json.dump(archiv, open(archiv_pfad, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    if status["rest"] is None:       # keine Liga abgerufen -> Kontostand kostenlos abfragen
        try:
            r = requests.get(B + "sports", params={"apiKey": key}, timeout=30)
            status["rest"] = _int(r.headers.get("x-requests-remaining"))
            status["verbraucht"] = _int(r.headers.get("x-requests-used"))
        except Exception:
            pass
    return archiv, status


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def anwenden(liste, archiv):
    """Setzt e['hq'], e['quoten'] und e['qquelle'] fuer Spiele mit Odds-API-Quoten."""
    n = 0
    for e in liste:
        a = archiv.get(_key(e["liga"], e["datum"], e["heim"], e["gast"]))
        e.setdefault("qquelle", {"x": "football-data", "ou": "football-data"})
        if not a:
            continue
        hq = dict(e.get("hq") or {})
        hq["x"] = [round(v, 2) for v in a["x"]]
        e["qquelle"]["x"] = "ODDS API"
        if a.get("ou"):
            hq["ou"] = [round(v, 2) for v in a["ou"]]
            e["qquelle"]["ou"] = "ODDS API"
        e["hq"] = hq
        q = dict(e.get("quoten") or {})
        q["1"], q["X"], q["2"] = a["kopf"]
        e["quoten"] = q
        e["qinfo"] = {"buchmacher": a["n"], "abruf": a.get("abruf")}
        n += 1
    return n
