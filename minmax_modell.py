# ===== MODELL MM · MIN-MAX-ANALYSE (Georgis Prozess) =====
# Je Ausgang: niedrigste und hoechste Quote aller Buchmacher -> Durchschnitt (Min+Max)/2
# Daraus: Payout, Marge, Uebereinstimmung, Uebereinstimmung + Marge, x faire Wahrscheinlichkeit, EV
#   TIPP     = Ausgang, dessen "Uebereinstimmung + Marge" genau Payout + Marge trifft
#   EV-TIPP  = Ausgang, dessen EV genau der Marge entspricht (EV = -Marge)
# Fuer 1X2 und OVER/UNDER 2,5 (fuer BTTS und Halbzeit gibt es keine Quoten).
import math

INFO = {"id": "mm", "typ": "mm", "name": "Min-Max · Quoten-Analyse", "kurz": "Min-Max", "icon": "⚖",
        "farbe1": "#ffd84d", "farbe2": "#2bf5a0",
        "beschreibung": "Durchschnitt aus niedrigster und hoechster Quote, Payout, Marge, EV"}

SPALTEN = {"x": (["H", "D", "A"], ["1", "X", "2"]), "ou": ([">2.5", "<2.5"], ["O", "U"])}
BUCHMACHER = {"x": ["PS", "B365", "BW", "WH", "1XB", "BFE", "BF", "IW", "VC", "LB", "BV", "CL", "Max"],
              "ou": ["P", "B365", "BFE", "BF", "1XB", "WH", "BW", "BV", "VC", "LB", "CL", "Max"]}
MIN_BUCHMACHER = 2


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) and v > 1.0 else None


def analyse(row, m):
    sufs, labs = SPALTEN[m]
    mm = {}
    for suf, lab in zip(sufs, labs):
        qs = [x for x in (_f(row.get(b + suf)) for b in BUCHMACHER[m]) if x]
        if len(qs) < MIN_BUCHMACHER:
            return None
        mm[lab] = (min(qs), max(qs))
    odds = {k: round((a + b) / 2, 2) for k, (a, b) in mm.items()}
    payout = 100 / sum(1 / o for o in odds.values())
    fp = {k: round(100 / o, 2) for k, o in odds.items()}
    rp = {k: round(1 / o * payout, 2) for k, o in odds.items()}
    pay = round(payout, 2)
    marge = round(100 - payout, 2)
    ue = {k: round(100 - abs(fp[k] - rp[k]) / fp[k] * 100, 2) for k in fp}
    uem = {k: round(ue[k] + marge, 2) for k in ue}
    mul = {k: round(odds[k] * fp[k], 2) for k in fp}
    ev = {k: round((odds[k] * rp[k] / 100 - 1) * 100, 2) for k in fp}
    ziel = round(pay + marge, 2)
    tipp = [k for k in labs if uem[k] == ziel]
    evt = [k for k in labs if round(-ev[k], 2) == marge]
    return {"labels": labs, "pay": pay, "marge": marge, "ue": [ue[k] for k in labs], "uem": [uem[k] for k in labs],
            "mul": [mul[k] for k in labs], "ev": [ev[k] for k in labs], "tipp": tipp, "evt": evt}


def eintrag(row):
    if row is None:
        return None
    x, ou = analyse(row, "x"), analyse(row, "ou")
    if not x and not ou:
        return None
    return {"typ": "mm", "x": x, "ou": ou}
