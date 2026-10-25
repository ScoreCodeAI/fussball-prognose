# ===== VALUES - Betfair-Exchange-Quoten gegen faire Pinnacle-Wahrscheinlichkeit =====
# Eigenstaendiger Zusatz: veraendert KEIN Modell und KEINEN Backtest.
#
#   Markt-Value:  Betfair zahlt mehr als die faire Quote (Pinnacle ohne Marge).
#   Modell-Value: mind. MIN_MODELLE Modelle sehen nach ihrer kalibrierten Regel Value
#                 gegen die Betfair-Quote.
# Maerkte: 1X2 und OVER/UNDER 2,5. Sortiert: groesster Vorteil zuerst.
#
# Zusaetzlich: SARIMAX-Values fuer OVER/UNDER 2,5 gegen drei Quellen (Betfair, Pinnacle, Hoechstquote).
import math

import numpy as np
import pandas as pd

import markt

# ---------- Einstellungen ----------
MIN_EV_MARKT = 0.02       # Markt-Value ab +2 %
MIN_MODELLE = 2           # so viele Modelle muessen nach IHRER kalibrierten Regel Value sehen
QUOTE_MIN, QUOTE_MAX = 1.30, 8.0
MAX_EV = 0.40             # unrealistisch hohe Werte (Datenfehler) ausblenden
BETFAIR = "BFE"           # Betfair Exchange (Back-Quote, ohne Kommission)

MAERKTE = [  # (Gruppe, Spalten-Suffixe, Ausgaenge, Prognose-Feld, Pinnacle-Quellen)
    ("1X2", ["H", "D", "A"], ["Heimsieg", "Remis", "Auswärtssieg"], ["p1", "px", "p2"], markt.QUELLEN_1X2),
    ("OU", [">2.5", "<2.5"], ["OVER 2,5", "UNDER 2,5"], ["ue25", "-ue25"], markt.QUELLEN_OU),
]
OU_QUELLEN = [("BFE", "Betfair"), ("P", "Pinnacle"), ("Max", "Höchstquote")]


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) and v > 1.0 else None


def _modell_p(prog, feld):
    v = prog.get(feld.lstrip("-"))
    if v is None:
        return None
    v = float(v) / 100
    return 1 - v if feld.startswith("-") else v


def _zeilen(fx):
    z = {}
    for _, row in fx.iterrows():
        if not pd.isna(row.get("Date")):
            z[(row["Div"], row["Date"].strftime("%Y-%m-%d"), row.get("HomeTeam"), row.get("AwayTeam"))] = row
    return z


def _regeln(modelle):
    return {m["id"]: m.get("einstellung_live") or {"mischung": 0.5, "min_value": 0.10,
                                                  "quote_min": 1.5, "quote_max": 5.0} for m in modelle}


def berechne(liste, fx, modelle):
    """liste: Spiele aus engine.py, fx: fixtures-DataFrame, modelle: echte Modelle (mit Backtest)."""
    zeilen = _zeilen(fx)
    kurz = {m["id"]: m["kurz"] for m in modelle}
    regel = _regeln(modelle)
    out = []
    for sp in liste:
        if sp.get("ergebnis"):
            continue
        row = zeilen.get((sp["liga"], sp["datum"], sp["heim"], sp["gast"]))
        if row is None:
            continue
        for gruppe, suffixe, ausgaenge, felder, quellen in MAERKTE:
            fair, ref = markt._hole(row, quellen, suffixe)
            bf = [_f(row.get(BETFAIR + s)) for s in suffixe]
            if fair is None or not all(bf):
                continue
            for i, (aus, feld) in enumerate(zip(ausgaenge, felder)):
                q = bf[i]
                if not (QUOTE_MIN <= q <= QUOTE_MAX):
                    continue
                ev_markt = q * fair[i] - 1
                ja, edges, pm_all = [], [], []
                for mid in kurz:
                    pm = _modell_p(sp["prog"].get(mid) or {}, feld)
                    if pm is None:
                        continue
                    pm_all.append(pm)
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
                    "heim": sp["heim"], "gast": sp["gast"], "gruppe": gruppe, "tipp": aus,
                    "quote": round(q, 2), "buchmacher": "Betfair", "fair": round(1 / fair[i], 2),
                    "p_markt": round(fair[i] * 100, 1), "ev_markt": round(ev_markt * 100, 1), "referenz": ref,
                    "p_modelle": round(float(np.mean(pm_all)) * 100, 1) if pm_all else None,
                    "ev_modelle": None if ev_mod is None else round(ev_mod * 100, 1),
                    "modelle_ja": ja, "modelle_n": len(pm_all),
                    "art": "beide" if markt_ok and modell_ok else "markt" if markt_ok else "modell"})
    out.sort(key=lambda v: -max(v["ev_markt"], v["ev_modelle"] if v["ev_modelle"] is not None else -99))
    return out


def sarimax_ou(liste, fx, modelle, mid="modell_sarimax"):
    """SARIMAX OVER/UNDER 2,5: Vorteil gegen Betfair, Pinnacle und Hoechstquote.
    Rechnung wie die Value-Box des Modells (kalibrierte Mischung mit der fairen Pinnacle-Chance)."""
    m = next((x for x in modelle if x["id"] == mid), None)
    if m is None:
        return 0
    r = _regeln([m])[mid]
    zeilen = _zeilen(fx)
    n = 0
    for sp in liste:
        pr = sp["prog"].get(mid)
        if sp.get("ergebnis") or not pr:
            continue
        row = zeilen.get((sp["liga"], sp["datum"], sp["heim"], sp["gast"]))
        if row is None:
            continue
        fair, _ = markt._hole(row, markt.QUELLEN_OU, [">2.5", "<2.5"])
        if fair is None:
            continue
        pm = [pr["ue25"] / 100, 1 - pr["ue25"] / 100]
        zeilen_out = []
        for i, aus in enumerate(["OVER 2,5", "UNDER 2,5"]):
            pf = r["mischung"] * pm[i] + (1 - r["mischung"]) * fair[i]
            for pre, name in OU_QUELLEN:
                q = _f(row.get(pre + (">2.5" if i == 0 else "<2.5")))
                if not q:
                    continue
                e = pf * q - 1
                zeilen_out.append({"tipp": aus, "quelle": name, "quote": round(q, 2), "ev": round(e * 100, 1),
                                   "value": bool(r["min_value"] < e < MAX_EV and QUOTE_MIN <= q <= QUOTE_MAX)})
        if any(z["value"] for z in zeilen_out):
            zeilen_out.sort(key=lambda z: -z["ev"])
            pr["ouv"] = {"p": round(pm[0] * 100, 1), "fair": round(fair[0] * 100, 1), "zeilen": zeilen_out}
            n += 1
    return n
