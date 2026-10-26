# ===== MODELL SV · SPALTEN-VALUE =====
# Georgis Idee: jeder Ausgang wird EINZELN als Spalte betrachtet (1 | X | 2, OVER | UNDER).
#   1) Faire Wahrscheinlichkeit je Spalte = Konsens ALLER Buchmacher
#      (jede Quotenzeile ohne Marge, dann Median ueber die Buchmacher)
#   2) In jeder Spalte wird die hoechste Quote mit der fairen verglichen:
#      EV = hoechste Quote x faire Wahrscheinlichkeit - 1  -> ab MIN_EV ist die Spalte "Value"
# Nur Maerkte mit echten Quoten (1X2, OVER/UNDER 2,5) - nichts wird simuliert.
import math
import statistics

import markt
import quoten_modell

MODELL_INFO = {"id": "sv", "typ": "sv", "name": "SV · Spalten-Value", "kurz": "Spalten", "icon": "SV",
               "farbe1": "#ff4d7d", "farbe2": "#ff2bd6",
               "beschreibung": "Jeder Ausgang einzeln: höchste Quote gegen den fairen Konsens aller Buchmacher"}

MIN_EV, MAX_EV, MAX_QUOTE = 0.03, 0.40, 10.0
MIN_BUCHMACHER = 3
BUCHMACHER = {"x": ["B365", "BW", "PS", "WH", "1XB", "BFE", "BF", "IW", "VC", "LB", "BV", "CL", "GB", "SB"],
              "ou": ["B365", "P", "BFE", "BF", "1XB", "WH", "BW", "BV", "VC", "LB", "CL"]}
SUFFIXE = {"x": ["H", "D", "A"], "ou": [">2.5", "<2.5"]}
LABELS = {"x": ["1", "X", "2"], "ou": ["O", "U"]}
HOECHST = {"x": "Max", "ou": "Max"}


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) and v > 1.0 else None


def analyse(row, m):
    zeilen = []
    for b in BUCHMACHER[m]:
        q = [_f(row.get(b + s)) for s in SUFFIXE[m]]
        if all(q):
            zeilen.append(q)
    if len(zeilen) < MIN_BUCHMACHER:
        return None
    fair = [statistics.median(p[i] for p in (markt.marge_entfernen(z) for z in zeilen)) for i in range(len(SUFFIXE[m]))]
    s = sum(fair)
    fair = [p / s for p in fair]
    hoch = [_f(row.get(HOECHST[m] + suf)) for suf in SUFFIXE[m]]
    out = {"n": len(zeilen), "fair": [], "ev": [], "q": [], "value": []}
    for i in range(len(fair)):
        spalte = [z[i] for z in zeilen] + ([hoch[i]] if hoch[i] else [])
        q = max(spalte)
        ev = q * fair[i] - 1
        out["fair"].append(round(fair[i] * 100, 1))
        out["q"].append(round(q, 2))
        out["ev"].append(round(ev * 100, 1))
        out["value"].append(bool(MIN_EV <= ev <= MAX_EV and q <= MAX_QUOTE))
    # Spalten-Check: hoechste Quote x faire Wahrscheinlichkeit (ueber 100 % = Value)
    out["check"] = [round(q * f * 100, 2) for q, f in zip(out["q"], fair)]
    # Quoten-Analyse (frueher Modell Q) auf den hoechsten Quoten der Spalten
    odds = dict(zip(LABELS[m], out["q"]))
    matches, payout, marge, ev = quoten_modell.run_analysis(odds)
    treffer = [k for k, v in matches.items() if round(v, 2) == round(payout, 2)]
    tq = treffer[0] if treffer else min(matches, key=lambda k: abs(matches[k] - payout))
    ev_ok = [k for k, v in ev.items() if round(abs(v), 2) == round(marge, 2)] or \
        [min(ev, key=lambda k: abs(abs(ev[k]) - marge))]
    out["qa"] = {"tipp": tq, "ev": ev_ok, "evw": {k: ev[k] for k in ev_ok}, "payout": payout, "marge": marge}
    best = [i for i, v in enumerate(out["value"]) if v]
    out["tipp"] = LABELS[m][max(best, key=lambda i: out["ev"][i])] if best else None
    out["labels"] = LABELS[m]
    return out


def eintrag(row):
    if row is None:
        return None
    x, ou = analyse(row, "x"), analyse(row, "ou")
    if not x and not ou:
        return None
    return {"typ": "sv", "x": x, "ou": ou}
