# ===== DATEN-CHECK (fuer eigenstaendige Modelle, z. B. Ensemble) =====
# Frage je Markt: Wenn das Modell frueher einen aehnlichen Tipp mit aehnlicher Sicherheit gegeben hat
# (gleicher Tipp, Wahrscheinlichkeit im selben 5-%-Band), wie sind diese Spiele WIRKLICH ausgegangen?
# Ehrlich: fuer jedes Spiel zaehlen nur Spiele VOR diesem Tag. Zu wenige aehnliche Spiele -> kein Check.
import numpy as np
import pandas as pd

BAND = 5          # Prozentpunkte
MIN_N = 30
GLAETTUNG = 1.0


def _sg(a, b):
    return "1" if a > b else "X" if a == b else "2"


def _ht(sp):
    t = sp.get("HT")
    if not t:
        return None
    a, b = t.split(":")
    return int(a), int(b)


# Markt -> (Labels, Wahrscheinlichkeiten aus p, echter Ausgang aus Spiel)
def _maerkte():
    ft = lambda sp: (sp["HG"], sp["AG"])
    return {
        "x": (["1", "X", "2"], lambda p: [p["1"], p["X"], p["2"]], lambda sp: _sg(*ft(sp))),
        "ht": (["1", "X", "2"], lambda p: p.get("HT"), lambda sp: _sg(*_ht(sp)) if _ht(sp) else None),
        "ht2": (["1", "X", "2"], lambda p: p.get("HT2"),
                lambda sp: _sg(sp["HG"] - _ht(sp)[0], sp["AG"] - _ht(sp)[1]) if _ht(sp) else None),
        "ou25": (["OVER", "UNDER"], lambda p: [p["Ü2.5"], p["U2.5"]], lambda sp: "OVER" if sum(ft(sp)) > 2 else "UNDER"),
        "ou15": (["OVER", "UNDER"], lambda p: [p["OU15"], 1 - p["OU15"]] if "OU15" in p else None,
                 lambda sp: "OVER" if sum(ft(sp)) > 1 else "UNDER"),
        "ou35": (["OVER", "UNDER"], lambda p: [p["OU35"], 1 - p["OU35"]] if "OU35" in p else None,
                 lambda sp: "OVER" if sum(ft(sp)) > 3 else "UNDER"),
        "bts": (["Ja", "Nein"], lambda p: [p["BTS"], 1 - p["BTS"]], lambda sp: "Ja" if min(ft(sp)) > 0 else "Nein"),
        "htou05": (["OVER", "UNDER"], lambda p: [p["HTOU05"], 1 - p["HTOU05"]] if "HTOU05" in p else None,
                   lambda sp: ("OVER" if sum(_ht(sp)) > 0 else "UNDER") if _ht(sp) else None),
        "htou15": (["OVER", "UNDER"], lambda p: [p["HTOU15"], 1 - p["HTOU15"]] if "HTOU15" in p else None,
                   lambda sp: ("OVER" if sum(_ht(sp)) > 1 else "UNDER") if _ht(sp) else None),
        "htft": ([a + "/" + b for a in "1X2" for b in "1X2"],
                 lambda p: [p["HTFT"][a + "/" + b] for a in "1X2" for b in "1X2"] if "HTFT" in p else None,
                 lambda sp: (_sg(*_ht(sp)) + "/" + _sg(*ft(sp))) if _ht(sp) else None),
    }


def _schluessel(labs, ps):
    i = int(np.argmax(ps))
    return labs[i], int(ps[i] * 100 // BAND)


def _check(zaehler, m, labs, ps):
    if ps is None:
        return None
    tipp, band = _schluessel(labs, ps)
    c, n = np.zeros(len(labs)), 0
    for b in (band, band - 1, band + 1):          # bei zu wenig Faellen Nachbar-Baender dazu
        z = zaehler.get((m, tipp, b))
        if z is not None:
            c += z
            n = int(c.sum())
        if n >= MIN_N:
            break
    if n < MIN_N:
        return {"n": n}
    q = (c + GLAETTUNG) / (c.sum() + GLAETTUNG * len(labs))
    return {"p": [round(float(x) * 100, 1) for x in q], "n": n, "tipp": labs[int(np.argmax(q))]}


def berechne(spiele, live):
    """spiele: Backtest-Spiele (p, HG, AG, HT, Date); live: kommende (p). Setzt sp['dc'] / lv['dc']."""
    M = _maerkte()
    zaehler = {}
    ordn = sorted(spiele, key=lambda s: s["Date"])
    i = 0
    while i < len(ordn):
        tag = pd.Timestamp(ordn[i]["Date"]).normalize()
        j = i
        while j < len(ordn) and pd.Timestamp(ordn[j]["Date"]).normalize() == tag:
            j += 1
        gruppe = ordn[i:j]
        for sp in gruppe:              # erst pruefen (nur fruehere Tage) ...
            sp["dc"] = {m: _check(zaehler, m, labs, f(sp["p"])) for m, (labs, f, _) in M.items()}
        for sp in gruppe:              # ... dann diesen Tag dazuzaehlen
            for m, (labs, f, echt) in M.items():
                ps = f(sp["p"])
                try:
                    e = echt(sp)
                except Exception:
                    e = None
                if ps is None or e is None:
                    continue
                k = (m,) + _schluessel(labs, ps)
                z = zaehler.setdefault(k, np.zeros(len(labs)))
                z[labs.index(e)] += 1
        i = j
    for lv in live:
        lv["dc"] = {m: _check(zaehler, m, labs, f(lv["p"])) for m, (labs, f, _) in M.items()}
