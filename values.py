# ===== VALUES - Wettquoten-Vergleich fuer kommende Spiele =====
# Eigenstaendiger Zusatz: veraendert KEIN Modell und KEINEN Backtest.
# Wird am Ende von engine.py aufgerufen und schreibt die Liste "values" in daten.json.
#
# Zwei Arten von Value:
#   1) Markt-Value:  ein Buchmacher zahlt mehr als die faire Quote (Pinnacle ohne Marge).
#   2) Modell-Value: die Modelle sehen eine hoehere Chance als der Markt - gegen die BESTE Quote.
#
# Alle Buchmacher-Spalten aus fixtures.csv werden automatisch erkannt.
import math
import statistics

import numpy as np
import pandas as pd

# ---------- Einstellungen ----------
MIN_EV_MARKT = 0.03       # Markt-Value ab +3 %
MIN_MODELLE = 2           # so viele Modelle muessen nach IHRER kalibrierten Regel Value sehen
QUOTE_MIN, QUOTE_MAX = 1.30, 8.0
MAX_EV = 0.40             # unrealistisch hohe Werte (Datenfehler) ausblenden
MIN_BUCHMACHER = 3        # fuer Konsens, falls Pinnacle fehlt

BUCHMACHER = {            # Spalten-Praefix -> Anzeigename
    "B365": "bet365", "BW": "Bwin", "PS": "Pinnacle", "WH": "William Hill", "1XB": "1xBet",
    "BFE": "Betfair Exch.", "BF": "Betfair", "IW": "Interwetten", "VC": "BetVictor", "LB": "Ladbrokes",
    "BV": "BetVictor", "GB": "Gamebookers", "SJ": "Stan James", "BS": "Blue Square", "SB": "Sportingbet",
    "CL": "Coral", "Max": "Höchstquote", "P": "Pinnacle",
}
NICHT_VERGLEICHEN = {"Avg"}          # Durchschnitt ist kein buchbarer Buchmacher
SCHARF_1X2, SCHARF_OU = "PS", "P"    # Pinnacle-Spalten


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) and v > 1.0 else None


def marge_entfernen(quoten):
    """Power-Methode: (1/q)^k mit Summe 1 - beruecksichtigt den Favoriten-Aussenseiter-Effekt."""
    imp = [1 / q for q in quoten]
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if sum(p ** k for p in imp) > 1 else (lo, k)
    p = [x ** ((lo + hi) / 2) for x in imp]
    s = sum(p)
    return [x / s for x in p]


def _praefixe(spalten, suffixe):
    """Alle Praefixe, fuer die jede Ausgangs-Spalte existiert (z. B. B365H/B365D/B365A)."""
    out = []
    for c in spalten:
        s0 = suffixe[0]
        if c.endswith(s0):
            pre = c[: -len(s0)]
            if pre and all(pre + s in spalten for s in suffixe):
                if pre.endswith("C") and pre[:-1] + s0 in spalten:     # Schlussquoten (z. B. PSCH) auslassen
                    continue
                out.append(pre)
    return out


def _markt(row, praefixe, suffixe, scharf):
    """-> (faire Wahrscheinlichkeiten, Referenz, {Praefix: [Quoten]})"""
    quoten = {}
    for pre in praefixe:
        q = [_f(row.get(pre + s)) for s in suffixe]
        if all(q):
            quoten[pre] = q
    if scharf in quoten:
        return marge_entfernen(quoten[scharf]), "Pinnacle", quoten
    einzeln = [q for p, q in quoten.items() if p not in NICHT_VERGLEICHEN and p != "Max"]
    if len(einzeln) >= MIN_BUCHMACHER:
        pr = [marge_entfernen(q) for q in einzeln]
        med = [statistics.median(x[i] for x in pr) for i in range(len(suffixe))]
        s = sum(med)
        return [m / s for m in med], f"Konsens {len(einzeln)} Buchm.", quoten
    return None, None, quoten


def _beste(quoten, i):
    """Beste buchbare Quote fuer Ausgang i -> (Quote, Name, Anzahl Buchmacher)"""
    best, name, n = None, None, 0
    for pre, q in quoten.items():
        if pre in NICHT_VERGLEICHEN:
            continue
        n += pre != "Max"
        if best is None or q[i] > best + 1e-9 or (abs(q[i] - best) < 1e-9 and name == "Höchstquote"):
            best, name = q[i], BUCHMACHER.get(pre, pre)
    return best, name, n


MAERKTE = [  # (Gruppe, Spalten-Suffixe, Pinnacle-Praefix, Ausgaenge, Prognose-Feld)
    ("1X2", ["H", "D", "A"], SCHARF_1X2, ["1", "X", "2"], ["p1", "px", "p2"]),
    ("Tore", [">2.5", "<2.5"], SCHARF_OU, ["Über 2,5", "Unter 2,5"], ["ue25", "-ue25"]),
]
TIPP_NAME = {"1": "Heimsieg", "X": "Remis", "2": "Auswärtssieg"}


def _modell_p(prog, feld):
    v = prog.get(feld.lstrip("-"))
    if v is None:
        return None
    v = float(v) / 100
    return 1 - v if feld.startswith("-") else v


def berechne(liste, fx, modelle):
    """liste: Spiele aus engine.py, fx: fixtures-DataFrame, modelle: Modell-Infos."""
    spalten = set(fx.columns)
    zeilen = {}
    for _, row in fx.iterrows():
        if pd.isna(row.get("Date")):
            continue
        zeilen[(row["Div"], row["Date"].strftime("%Y-%m-%d"), row.get("HomeTeam"), row.get("AwayTeam"))] = row

    ids = [m["id"] for m in modelle]
    kurz = {m["id"]: m["kurz"] for m in modelle}
    regel = {m["id"]: m.get("einstellung_live") or {"mischung": 0.5, "min_value": 0.10,
                                                    "quote_min": 1.5, "quote_max": 5.0} for m in modelle}
    out = []
    for sp in liste:
        if sp.get("ergebnis"):
            continue
        row = zeilen.get((sp["liga"], sp["datum"], sp["heim"], sp["gast"]))
        if row is None:
            continue
        for gruppe, suffixe, scharf, ausgaenge, felder in MAERKTE:
            praef = _praefixe(spalten, suffixe)
            fair, ref, quoten = _markt(row, praef, suffixe, scharf)
            if fair is None:
                continue
            for i, (aus, feld) in enumerate(zip(ausgaenge, felder)):
                q, buch, n_buch = _beste(quoten, i)
                if q is None or not (QUOTE_MIN <= q <= QUOTE_MAX):
                    continue
                ev_markt = q * fair[i] - 1
                mp = {mid: _modell_p(sp["prog"][mid], feld) for mid in ids if mid in sp["prog"]}
                mp = {k: v for k, v in mp.items() if v is not None}
                p_mod = float(np.mean(list(mp.values()))) if mp else None
                # Jedes Modell nach seiner eigenen, im Backtest kalibrierten Regel (wie in engine.py),
                # aber gegen die BESTE verfuegbare Quote und die faire Pinnacle-Wahrscheinlichkeit
                ja, edges = [], []
                for mid, pm in mp.items():
                    r = regel[mid]
                    e = (r["mischung"] * pm + (1 - r["mischung"]) * fair[i]) * q - 1
                    if r["quote_min"] <= q <= r["quote_max"] and r["min_value"] < e < MAX_EV:
                        ja.append(kurz[mid])
                        edges.append(e)
                ev_mod = float(np.mean(edges)) if edges else None
                markt_ok = MIN_EV_MARKT <= ev_markt <= MAX_EV
                modell_ok = len(ja) >= MIN_MODELLE
                if not (markt_ok or modell_ok):
                    continue
                out.append({
                    "id": sp["id"], "liga": sp["liga"], "datum": sp["datum"], "zeit": sp["zeit"],
                    "heim": sp["heim"], "gast": sp["gast"], "gruppe": gruppe,
                    "tipp": TIPP_NAME.get(aus, aus), "quote": round(q, 2), "buchmacher": buch,
                    "n_buch": n_buch, "fair": round(1 / fair[i], 2), "p_markt": round(fair[i] * 100, 1),
                    "ev_markt": round(ev_markt * 100, 1), "referenz": ref,
                    "p_modelle": None if p_mod is None else round(p_mod * 100, 1),
                    "ev_modelle": None if ev_mod is None else round(ev_mod * 100, 1),
                    "modelle_ja": ja, "modelle_n": len(mp),
                    "art": "beide" if markt_ok and modell_ok else "markt" if markt_ok else "modell",
                })
    out += _ohne_quoten(liste, kurz)
    rang = {"beide": 0, "markt": 1, "modell": 2}
    out.sort(key=lambda v: (v["datum"], rang[v["art"]], -max(v["ev_markt"], v["ev_modelle"] or -99)))
    return out


# ---------- Maerkte ohne Buchmacher-Quoten (BTTS, Halbzeit, HZ/ES, Correct Score) ----------
# Vergleich Modell gegen MARKT-Erwartung (aus Pinnacle 1X2 + Ueber/Unter berechnet, siehe markt.py).
# Ein Tipp kommt in die Liste, wenn mind. MIN_MODELLE Modelle denselben Ausgang als Vorteil sehen.
MAERKTE_OHNE = {"bts": "BTTS", "ht": "Halbzeit", "htft": "HZ/ES", "cs": "Correct Score"}
MIN_MODELLE_OHNE = 3      # strenger, weil hier keine echte Buchmacher-Quote vorliegt


def _ohne_quoten(liste, kurz):
    out = []
    for sp in liste:
        if sp.get("ergebnis") or not sp.get("markt"):
            continue
        for mk, name in MAERKTE_OHNE.items():
            stimmen = {}
            for mid, pr in sp["prog"].items():
                v = (pr.get("vt") or {}).get(mk)
                if v:
                    stimmen.setdefault(v[0], []).append((mid, v))
            for lab, liste_m in stimmen.items():
                if len(liste_m) < MIN_MODELLE_OHNE:
                    continue
                pm = float(np.mean([v[1] for _, v in liste_m]))
                pk = liste_m[0][1][2]
                out.append({
                    "id": sp["id"], "liga": sp["liga"], "datum": sp["datum"], "zeit": sp["zeit"],
                    "heim": sp["heim"], "gast": sp["gast"], "gruppe": name, "tipp": lab,
                    "quote": None, "buchmacher": None, "n_buch": 0,
                    "fair": round(100 / pk, 2), "p_markt": pk, "ev_markt": 0.0, "referenz": "Markt berechnet",
                    "p_modelle": round(pm, 1), "ev_modelle": round((pm / pk - 1) * 100, 1),
                    "fair_modell": round(100 / pm, 2),
                    "modelle_ja": [kurz[m] for m, _ in liste_m], "modelle_n": len(sp["prog"]),
                    "art": "modell"})
    return out
