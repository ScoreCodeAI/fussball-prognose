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
SCHLUSS = {"x": ["PSCH", "PSCD", "PSCA"], "ou": ["PC>2.5", "PC<2.5"]}      # Pinnacle-Schlussquoten (fuer CLV)
NAMEN = {"B365": "bet365", "BW": "bwin", "PS": "Pinnacle", "P": "Pinnacle", "WH": "William Hill", "1XB": "1xBet",
         "BFE": "Betfair Exchange", "BF": "Betfair", "IW": "Interwetten", "VC": "BetVictor", "LB": "Ladbrokes",
         "BV": "BetVictor", "CL": "Coral", "GB": "Gamebookers", "SB": "Sportingbet", "Max": "Höchstquote"}


def shin(quoten):
    """Marge nach Shin entfernen (Aussenseiter tragen mehr Marge). Bei negativer Marge: gleichmaessig."""
    from scipy.optimize import brentq
    pi = [1 / q for q in quoten]
    S = sum(pi)
    if S <= 1:
        return [p / S for p in pi]
    f = lambda z: sum(math.sqrt(z * z + 4 * (1 - z) * p * p / S) - z for p in pi) / (2 * (1 - z)) - 1
    try:
        z = brentq(f, 0, 0.4)
        p = [(math.sqrt(z * z + 4 * (1 - z) * x * x / S) - z) / (2 * (1 - z)) for x in pi]
    except Exception:
        p = pi
    t = sum(p)
    return [x / t for x in p]


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) and v > 1.0 else None


def analyse(row, m):
    zeilen, wer = [], []
    namen = row.get("_namen") or {}
    oa = sorted({k[3:-len(sfx)] for k in row.keys() if str(k).startswith("OA:") for sfx in SUFFIXE[m][:1] if str(k).endswith(sfx)})
    for b in (["OA:" + k for k in oa] if oa else BUCHMACHER[m]):
        q = [_f(row.get(b + s)) for s in SUFFIXE[m]]
        if all(q):
            zeilen.append(q)
            wer.append(b)
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
    # Durchschnitt je Spalte OHNE Marge: mittlere Wahrscheinlichkeit, auf 100 % normiert -> Quote -> x faire Chance
    avg_p = [statistics.mean(1 / z[i] for z in zeilen) for i in range(len(fair))]
    t = sum(avg_p)
    avg_p = [x / t for x in avg_p]
    out["avg_p"] = [round(x * 100, 1) for x in avg_p]
    out["avg_q"] = [round(1 / x, 2) for x in avg_p]
    out["avg_check"] = [round(f / x * 100, 2) for f, x in zip(fair, avg_p)]
    # (2) Wer zahlt die hoechste Quote je Spalte, und wie weit ueber dem Konsens (Median der Spalte)?
    out["wer"], out["abw"] = [], []
    for i in range(len(fair)):
        spalte = [z[i] for z in zeilen]
        j = max(range(len(spalte)), key=lambda k: spalte[k])
        hq = hoch[i] if hoch[i] and hoch[i] > spalte[j] else None
        nm = namen.get(wer[j][3:], wer[j][3:]) if wer[j].startswith("OA:") else NAMEN.get(wer[j], wer[j])
        out["wer"].append(NAMEN.get("Max", "Max") if hq else nm)
        out["abw"].append(round(((hq or spalte[j]) / statistics.median(spalte) - 1) * 100, 1))
    # (3) Faire Chance mit Shin aus der Durchschnittsquote -> hoechste Quote x Shin-Chance
    sh = shin([statistics.mean(z[i] for z in zeilen) for i in range(len(fair))])
    out["shin"] = [round(x * 100, 2) for x in sh]
    out["shin_check"] = [round(q * x * 100, 2) for q, x in zip(out["q"], sh)]
    # (1) CLV: hoechste Quote gegen Pinnacle-Schlussquote ohne Marge (nur fuer beendete Spiele vorhanden)
    sq = [_f(row.get(c)) for c in SCHLUSS[m]]
    if all(sq):
        fs = shin(sq)
        out["clv"] = [round((q * f - 1) * 100, 1) for q, f in zip(out["q"], fs)]
        out["schluss"] = [round(x, 2) for x in sq]
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
