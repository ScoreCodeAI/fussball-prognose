# ===== MODELL Q · QUOTEN-ANALYSE =====
# Georgis Berechnung 1:1 (Payout, Uebereinstimmung, Marge, Erwartungswert) -
# angewendet auf die HOECHSTEN Quoten (Max-Spalten von football-data.co.uk).
#
#   Tipp     = Ausgang, dessen "Uebereinstimmung" genau dem Payout entspricht (✅)
#   EV-Tipp  = alle Ausgaenge, deren |EV| genau der Marge entspricht (✅)
#              1X2: zwei Haken -> Doppelte Chance (1X, X2, 12)
#
# Kein Score-Modell: wird NACH den anderen Modellen in daten.json eingetragen
# und veraendert weder Backtest noch die anderen Modelle.
import math

MODELL_INFO = {"id": "quoten", "typ": "quoten", "name": "Quoten · Analyse", "kurz": "Quoten", "icon": "Q",
               "farbe1": "#f5b82b", "farbe2": "#ff7a45",
               "beschreibung": "Payout, Übereinstimmung, Marge und EV aus den Höchstquoten"}

SPALTEN = {"x": [("1", "MaxH"), ("X", "MaxD"), ("2", "MaxA")],
           "ou": [("O", "Max>2.5"), ("U", "Max<2.5")]}
ERSATZ = {"MaxH": "B365H", "MaxD": "B365D", "MaxA": "B365A", "Max>2.5": "B365>2.5", "Max<2.5": "B365<2.5"}


# ---------------------------
# Funktionen (unveraendert aus Georgis Code)
# ---------------------------
def calculate_real_probabilities(odds, payout=None):
    if payout is None:
        payout = 100 / sum(1 / odd for odd in odds.values())
    rp = {outcome: round((1 / odd) * payout, 2) for outcome, odd in odds.items()}
    return rp, round(payout, 2)


def calculate_bookmaker_margin(odds, payout=None):
    if payout is None:
        payout = 100 / sum(1 / odd for odd in odds.values())
    return round(100 - payout, 2)


def calculate_fair_probabilities(odds):
    return {outcome: round((1 / odd) * 100, 2) for outcome, odd in odds.items()}


def calculate_expected_value(odds, real_probs):
    return {outcome: round((odd * (real_probs[outcome] / 100) - 1) * 100, 2) for outcome, odd in odds.items()}


def run_analysis(odds):
    fair_probabilities = calculate_fair_probabilities(odds)
    real_probabilities, auto_payout = calculate_real_probabilities(odds)
    matches = {k: round(100 - (abs(fair_probabilities[k] - real_probabilities[k]) / fair_probabilities[k] * 100), 2)
               for k in fair_probabilities}
    margin = calculate_bookmaker_margin(odds)
    expected_values = calculate_expected_value(odds, real_probabilities)
    return matches, auto_payout, margin, expected_values


# ---------------------------
# Auswertung -> Tipp und EV-Tipp
# ---------------------------
def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) and v > 1.0 else None


def _quoten(row, markt):
    out = {}
    for name, spalte in SPALTEN[markt]:
        v = _f(row.get(spalte)) or _f(row.get(ERSATZ[spalte]))
        if v is None:
            return None
        out[name] = v
    return out


def analyse(row, markt):
    if row is None:
        return None
    odds = _quoten(row, markt)
    if not odds:
        return None
    matches, payout, margin, ev = run_analysis(odds)
    treffer = [k for k, v in matches.items() if round(v, 2) == round(payout, 2)]
    # wie im Code: ✅ bei exakter Gleichheit; falls keiner exakt trifft -> naechster am Payout
    tipp = treffer[0] if treffer else min(matches, key=lambda k: abs(matches[k] - payout))
    ev_ok = [k for k, v in ev.items() if round(abs(v), 2) == round(margin, 2)]
    if not ev_ok:
        ev_ok = [min(ev, key=lambda k: abs(abs(ev[k]) - margin))]
    return {"q": odds, "tipp": tipp, "ev": ev_ok, "evw": {k: ev[k] for k in ev_ok},
            "payout": payout, "marge": margin}


def eintrag(row):
    x, ou = analyse(row, "x"), analyse(row, "ou")
    if not x and not ou:
        return None
    return {"typ": "quoten", "x": x, "ou": ou}
