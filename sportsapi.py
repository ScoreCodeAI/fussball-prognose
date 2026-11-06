# ===== SPORTSAPI PRO · Hauptquelle fuer aktuelle Daten (Gratis-Plan, 100 Abrufe/Tag) =====
# Liefert fuer unsere Ligen: Ergebnis, Halbzeit, ALLE Statistiken je Halbzeit (xG, xGOT, Grosschancen,
# Schuesse, Ecken, Karten, ...) und die Quoten (1X2, Doppelte Chance, 1. HZ, Beide treffen, Tore 0,5-7,5,
# Karten, Ecken, Erstes Tor).
# Abruf-Reihenfolge (Bremse bei BUDGET Abrufen):
#   1) Spielplan der letzten 3 und naechsten 2 Tage          (~6 Abrufe)
#   2) Quoten der Spiele in den naechsten 48 Stunden          (max. QUOTEN_MAX, frueheste zuerst)
#   3) Statistiken aller beendeten Spiele ohne Statistik       (neueste zuerst - KEIN Spiel wird ausgelassen,
#                                                               was heute nicht passt, kommt morgen dran)
#   4) Rest: Vergangenheit Tag fuer Tag rueckwaerts nachladen  (Spielplan + Statistiken)
# Archiv: archiv/sportsapi.json.gz (wird vom Workflow gespeichert).
import gzip
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

import odds_api

B = "https://v2.football.sportsapipro.com"
BUDGET = 90
QUOTEN_MAX = 0          # SportsAPI-Quoten aus (alle Abrufe fuer Statistik/xG). Wieder an: 20
QUOTEN_LIGEN = {"D1", "E0", "SP1", "I1", "F1", "N1", "E1", "D2"}   # Quoten nur hier (Statistik/xG hat Vorrang)
RUECK_BIS = "2025-07-01"          # so weit zurueck wird nachgeladen (letzte Saison komplett)
DE = ZoneInfo("Europe/Berlin")
LIGEN = {("Germany", "Bundesliga"): "D1", ("Germany", "2. Bundesliga"): "D2", ("England", "Premier League"): "E0",
         ("England", "Championship"): "E1", ("England", "League One"): "E2", ("England", "League Two"): "E3",
         ("Spain", "LaLiga"): "SP1", ("Spain", "LaLiga 2"): "SP2", ("Italy", "Serie A"): "I1", ("Italy", "Serie B"): "I2",
         ("France", "Ligue 1"): "F1", ("France", "Ligue 2"): "F2", ("Netherlands", "VriendenLoterij Eredivisie"): "N1",
         ("Netherlands", "Eredivisie"): "N1", ("Belgium", "Pro League"): "B1", ("Greece", "Stoiximan Super League"): "G1",
         ("Greece", "Super League"): "G1", ("Scotland", "Scottish Premiership"): "SC0", ("Scotland", "Premiership"): "SC0",
         ("USA", "MLS"): "USA"}
PFAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archiv", "sportsapi.json.gz")


# ---------- Archiv ----------
def laden():
    try:
        with gzip.open(PFAD, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"spiele": {}, "rueck": None}


def speichern(a):
    os.makedirs(os.path.dirname(PFAD), exist_ok=True)
    with gzip.open(PFAD, "wt", encoding="utf-8") as f:
        json.dump(a, f, ensure_ascii=False, separators=(",", ":"))


# ---------- Hilfen ----------
class Abruf:
    def __init__(self, key, log):
        self.key, self.log, self.n, self.rest = key, log, 0, None

    def __call__(self, pfad, params=None):
        if self.n >= BUDGET or (self.rest is not None and self.rest <= 100 - BUDGET):
            return None
        self.n += 1
        try:
            r = requests.get(B + pfad, headers={"x-api-key": self.key}, params=params, timeout=30)
            v = r.headers.get("x-ratelimit-remaining")
            self.rest = int(v) if v and v.isdigit() else self.rest
            if r.status_code == 429:
                self.rest = 0
                return None
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None


def _ev(j):
    if isinstance(j, dict):
        for k in ("events", "matches"):
            if isinstance(j.get(k), list):
                return j[k]
        d = j.get("data")
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            return _ev(d)
    return j if isinstance(j, list) else []


def _code(e):
    t = e.get("tournament") or {}
    land = (t.get("category") or {}).get("name", "") if isinstance(t, dict) else ""
    return LIGEN.get((land, t.get("name", "") if isinstance(t, dict) else ""))


def _spiel(e, liga):
    hs, as_ = e.get("homeScore") or {}, e.get("awayScore") or {}
    st = (e.get("status") or {}).get("type", "")
    ts = e.get("startTimestamp")
    zeit = datetime.fromtimestamp(ts, timezone.utc).astimezone(DE) if ts else None
    fertig = st == "finished" and hs.get("current") is not None
    ht = None
    if hs.get("period1") is not None and as_.get("period1") is not None:
        ht = [int(hs["period1"]), int(as_["period1"])]
    return {"liga": liga, "datum": zeit.strftime("%Y-%m-%d") if zeit else None, "zeit": zeit.strftime("%H:%M") if zeit else "",
            "heim": (e.get("homeTeam") or {}).get("name"), "gast": (e.get("awayTeam") or {}).get("name"),
            "fertig": fertig, "ft": [int(hs.get("normaltime", hs["current"])), int(as_.get("normaltime", as_["current"]))] if fertig else None,
            "ht": ht, "ts": ts}


def _statistik(j):
    """Alle Statistik-Werte je Halbzeit: {"ALL": {key: [heim, gast]}, "1ST": {...}, "2ND": {...}}"""
    out = {}

    def zahl(v):
        try:
            return float(str(v).replace("%", "").split("/")[0].split(" ")[0])
        except Exception:
            return None

    def lauf(o, periode):
        if isinstance(o, dict):
            periode = o.get("period", periode)
            k = o.get("key")
            if k and ("homeValue" in o or "home" in o):
                h = o.get("homeValue", zahl(o.get("home")))
                a = o.get("awayValue", zahl(o.get("away")))
                if h is not None and a is not None:
                    out.setdefault(periode or "ALL", {})[k] = [round(float(h), 3), round(float(a), 3)]
            for v in o.values():
                lauf(v, periode)
        elif isinstance(o, list):
            for v in o:
                lauf(v, periode)
    lauf(j, None)
    return out


def _bruch(s):
    try:
        a, b = str(s).split("/")
        return round(1 + float(a) / float(b), 3)
    except Exception:
        try:
            return round(float(s), 3)
        except Exception:
            return None


def _quoten(j):
    """{markt: {auswahl: quote}} - Markt-Name inkl. Linie, z. B. 'Match goals 2.5', '1st half', 'Cards in match 5.5'"""
    d = j.get("data", j) if isinstance(j, dict) else {}
    out = {}
    for m in (d.get("markets") or []) if isinstance(d, dict) else []:
        name = m.get("marketName", "")
        if name in ("Asian handicap", "Draw no bet"):          # nicht gewuenscht
            continue
        if m.get("choiceGroup"):
            name = f"{name} {m['choiceGroup']}"
        q = {c.get("name"): _bruch(c.get("fractionalValue")) for c in m.get("choices", [])}
        out[name] = {k: v for k, v in q.items() if v}
    return out


# ---------- Hauptabruf ----------
def holen(log=print, jetzt=None):
    a = laden()
    key = os.environ.get("SPORTSAPIPRO_KEY", "")
    status = {"aktiv": bool(key), "abrufe": 0, "rest": None, "neu_stat": 0, "neu_quoten": 0, "spiele": len(a["spiele"])}
    if not key:
        log("SportsAPI: kein Schluessel (SPORTSAPIPRO_KEY) - nur Archiv")
        return a, status
    hole = Abruf(key, log)
    jetzt = jetzt or datetime.now(timezone.utc)
    S = a["spiele"]

    def plan(tag):
        j = hole(f"/api/schedule/{tag}")
        n = 0
        for e in _ev(j):
            c = _code(e)
            if not c or not e.get("id"):
                continue
            s = S.setdefault(str(e["id"]), {})
            s.update(_spiel(e, c))
            n += 1
        return j is not None, n

    # 1) Spielplan -3 .. +2 Tage (API rechnet in New-York-Zeit -> Fenster deckt das ab)
    for k in range(-3, 3):
        plan((jetzt + timedelta(days=k)).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d"))
    # 2) Quoten der naechsten 48 h (einmal je Tag und Spiel)
    heute = jetzt.astimezone(DE).strftime("%Y-%m-%d")
    bald = sorted([(s["ts"], i) for i, s in S.items() if not s.get("fertig") and s.get("ts")
                   and 0 < s["ts"] - jetzt.timestamp() < 48 * 3600 and s.get("q_tag") != heute and s.get("liga") in QUOTEN_LIGEN])
    for _, i in bald[:QUOTEN_MAX]:
        j = hole(f"/api/match/{i}/odds", {"scope": "all"})
        if j is None:
            break
        S[i]["quoten"] = _quoten(j)
        S[i]["q_tag"] = heute
        status["neu_quoten"] += 1
    # 3) + 4) Statistiken (neueste zuerst), dazwischen Vergangenheit nachladen
    rueck = a.get("rueck") or (jetzt - timedelta(days=4)).strftime("%Y-%m-%d")
    while hole.n < BUDGET:
        offen = sorted([(s.get("ts") or 0, i) for i, s in S.items() if s.get("fertig") and "stat" not in s], reverse=True)
        if offen:
            i = offen[0][1]
            j = hole(f"/api/match/{i}/statistics")
            if j is None:
                break
            S[i]["stat"] = _statistik(j)
            status["neu_stat"] += 1
            continue
        if rueck <= RUECK_BIS:
            break
        ok, _ = plan(rueck)
        if not ok:
            break
        rueck = (datetime.strptime(rueck, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    a["rueck"] = rueck
    speichern(a)
    status.update(abrufe=hole.n, rest=hole.rest, spiele=len(S),
                  mit_stat=sum(1 for s in S.values() if s.get("stat")), zurueck_bis=rueck)
    log(f"SportsAPI: {hole.n} Abrufe (uebrig {hole.rest}) | {len(S)} Spiele im Archiv, {status['mit_stat']} mit Statistik | "
        f"neu: {status['neu_stat']} Statistiken, {status['neu_quoten']} Quoten | nachgeladen bis {rueck}")
    return a, status


# ---------- Zuordnung zu football-data ----------
def _finde(index, liga, datum, heim, gast):
    best, wert = None, 0.0
    for d in (datum, *[(datetime.strptime(datum, "%Y-%m-%d") + timedelta(days=k)).strftime("%Y-%m-%d") for k in (-1, 1)]):
        for s in index.get((liga, d), []):
            wh, wa = odds_api._aehnlich(heim, s["heim"]), odds_api._aehnlich(gast, s["gast"])
            ok = min(wh, wa) >= 0.6 or (max(wh, wa) >= 0.85 and min(wh, wa) >= 0.3)
            w = (wh + wa) / 2 if ok else 0
            if w > wert:
                best, wert = s, w
    return best if wert >= 0.5 else None


def _index(a):
    idx = {}
    for s in a["spiele"].values():
        if s.get("datum"):
            idx.setdefault((s["liga"], s["datum"]), []).append(s)
    return idx


def _w(st, k, p="ALL"):
    v = (st or {}).get(p, {}).get(k)
    return v if v else [None, None]


def ergaenze_df(df, a, log=print):
    """Spiele-Tabelle (football-data) um SportsAPI ergaenzen:
       - neue Spalten: xG, xGOT, Grosschancen, Schuesse im Strafraum, Karten (alle Spiele mit Statistik)
       - fehlende Werte (Halbzeit, Schuesse, Ecken, Karten) auffuellen
       - Spiele, die football-data noch nicht hat (letzte Tage), als neue Zeilen anhaengen"""
    import numpy as np
    import pandas as pd
    neu_spalten = {"HXG": ("expectedGoals", 0), "AXG": ("expectedGoals", 1), "HXGOT": ("expectedGoalsOnTarget", 0),
                   "AXGOT": ("expectedGoalsOnTarget", 1), "HBC": ("bigChanceCreated", 0), "ABC": ("bigChanceCreated", 1),
                   "HSIB": ("totalShotsInsideBox", 0), "ASIB": ("totalShotsInsideBox", 1),
                   "HXG2": ("expectedGoals", 0, "2ND"), "AXG2": ("expectedGoals", 1, "2ND")}
    fuell = {"HS": ("totalShotsOnGoal", 0), "AS": ("totalShotsOnGoal", 1), "HST": ("shotsOnGoal", 0), "AST": ("shotsOnGoal", 1),
             "HC": ("cornerKicks", 0), "AC": ("cornerKicks", 1), "HY": ("yellowCards", 0), "AY": ("yellowCards", 1),
             "HR": ("redCards", 0), "AR": ("redCards", 1)}
    for c in list(neu_spalten) + list(fuell):
        if c not in df:
            df[c] = np.nan
    idx = _index(a)
    df = df.reset_index(drop=True)
    benutzt, n_stat = set(), 0
    for i, r in df[df["Date"] >= pd.Timestamp(RUECK_BIS) - pd.Timedelta(days=2)].iterrows():
        s = _finde(idx, r["Div"], r["Date"].strftime("%Y-%m-%d"), r["Home"], r["Away"])
        if not s:
            continue
        benutzt.add(id(s))
        st = s.get("stat")
        if s.get("ht") and pd.isna(r.get("HTHG")):
            df.at[i, "HTHG"], df.at[i, "HTAG"] = s["ht"]
        if not st:
            continue
        n_stat += 1
        for c, spec in {**neu_spalten, **fuell}.items():
            v = _w(st, spec[0], spec[2] if len(spec) > 2 else "ALL")[spec[1]]
            if v is not None and (c in neu_spalten or pd.isna(r.get(c))):
                df.at[i, c] = v
    # Spiele, die football-data noch nicht hat
    zeilen = []
    for s in a["spiele"].values():
        if id(s) in benutzt or not s.get("fertig") or not s.get("datum") or s["datum"] < RUECK_BIS:
            continue
        z = {"Div": s["liga"], "Date": pd.Timestamp(s["datum"]), "Home": s["heim"], "Away": s["gast"],
             "HG": s["ft"][0], "AG": s["ft"][1], "HTHG": (s.get("ht") or [np.nan, np.nan])[0], "HTAG": (s.get("ht") or [np.nan, np.nan])[1]}
        for c, spec in {**neu_spalten, **fuell}.items():
            z[c] = _w(s.get("stat"), spec[0], spec[2] if len(spec) > 2 else "ALL")[spec[1]]
        zeilen.append(z)
    if zeilen:
        neu = pd.DataFrame(zeilen)
        # nur anhaengen, wenn die Teams der Liga bekannt sind (Namen von football-data) -> sonst neue Namen = neue Teams
        bek = {(d, t) for d, t in zip(df["Div"], df["Home"])} | {(d, t) for d, t in zip(df["Div"], df["Away"])}
        neu["Home"] = [_name(df, r.Div, r.Home) for r in neu.itertuples()]
        neu["Away"] = [_name(df, r.Div, r.Away) for r in neu.itertuples()]
        neu = neu[[(d, h) in bek and (d, g) in bek for d, h, g in zip(neu["Div"], neu["Home"], neu["Away"])]]
        df = pd.concat([df, neu], ignore_index=True).drop_duplicates(subset=["Div", "Date", "Home", "Away"], keep="first")
    log(f"SportsAPI-Daten im Modell: {n_stat} Spiele mit Statistik (xG ...), {len(zeilen)} Spiele neuer als football-data")
    return df.sort_values("Date").reset_index(drop=True)


_NAMEN = {}


def _name(df, liga, n):
    k = (liga, n)
    if k not in _NAMEN:
        teams = set(df.loc[df["Div"] == liga, "Home"].tail(800))
        best = max(teams, key=lambda t: odds_api._aehnlich(t, n), default=n)
        _NAMEN[k] = best if teams and odds_api._aehnlich(best, n) >= 0.6 else n
    return _NAMEN[k]


def anwenden(liste, a, log=print):
    """Webseite: Ergebnisse/Halbzeit/Ecken/Karten/xG und SportsAPI-Quoten an die Spiele haengen."""
    idx = _index(a)
    n_e = n_q = 0
    for e in liste:
        s = _finde(idx, e["liga"], e["datum"], e["heim"], e["gast"])
        if not s:
            continue
        st = s.get("stat")
        if s.get("fertig"):
            if not e.get("ergebnis"):
                e["ergebnis"] = f'{s["ft"][0]}:{s["ft"][1]}'
                e["equelle"] = "SportsAPI"
                n_e += 1
            if s.get("ht") and not e.get("ergebnis_ht"):
                e["ergebnis_ht"] = f'{s["ht"][0]}:{s["ht"][1]}'
            ec = _w(st, "cornerKicks")
            if ec[0] is not None and not e.get("ecken"):
                e["ecken"] = [int(ec[0]), int(ec[1])]
            ka = _w(st, "yellowCards")
            if ka[0] is not None:
                rk = _w(st, "redCards")
                e["karten"] = [int(ka[0] + (rk[0] or 0)), int(ka[1] + (rk[1] or 0))]
        if st:
            e["stat"] = {p: {k: v for k, v in d.items()} for p, d in st.items()}
        if s.get("quoten"):
            e["squoten"] = s["quoten"]
            n_q += 1
    log(f"SportsAPI auf der Seite: {n_e} Ergebnisse ergänzt, {n_q} Spiele mit SportsAPI-Quoten")


def xg_form(df, liste, log=print, n=10):
    """Je Team die letzten n Spiele MIT xG: Tore vs xG und Gegentore vs xG gegen.
    Positiv = mehr Tore als xG (Glueck/Effizienz), negativ = Pech/Ineffizienz -> spielt besser als die Ergebnisse."""
    import pandas as pd
    if "HXG" not in df:
        return
    d = df.dropna(subset=["HXG", "AXG"]).sort_values("Date")
    if d.empty:
        return
    rows = []
    for r in d.itertuples(index=False):
        rows.append((r.Div, r.Home, r.Date, r.HG, r.AG, r.HXG, r.AXG))
        rows.append((r.Div, r.Away, r.Date, r.AG, r.HG, r.AXG, r.HXG))
    t = pd.DataFrame(rows, columns=["liga", "team", "date", "tore", "geg", "xg", "xga"])
    n_ok = 0
    for e in liste:
        out = {}
        for seite, team in (("h", e["heim"]), ("a", e["gast"])):
            x = t[(t["liga"] == e["liga"]) & (t["team"] == team) & (t["date"] < pd.Timestamp(e["datum"]))].tail(n)
            if len(x) >= 3:
                out[seite] = {"n": int(len(x)), "tore": int(x["tore"].sum()), "xg": round(float(x["xg"].sum()), 2),
                              "geg": int(x["geg"].sum()), "xga": round(float(x["xga"].sum()), 2)}
        if out:
            e["xgform"] = out
            n_ok += 1
    log(f"xG-Form: {n_ok} Spiele mit Tore-gegen-xG-Vergleich")
