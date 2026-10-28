# ===== KALIBRIERUNG (Platt-Skalierung) =====
# Jedes Modell lernt aus seinen EIGENEN alten Prognosen und den echten Ergebnissen,
# wie seine Prozente tatsaechlich eintreffen, und rechnet sie gerade.
#   1X2:      p_neu ~ exp(a * log p + b_Ausgang)   (3 Parameter)
#   OVER/UNDER, BTTS: logit(p_neu) = a * logit(p) + b
# Ehrlich: kalibriert wird nur mit Spielen VOR dem jeweiligen Spiel (letzte FENSTER Spiele, jede Woche neu).
# Test 10 Ligen ab Juli 2024: Platt ueberall besser (Tor-Modell O/U 0.6894 -> 0.6841, BTTS 0.6908 -> 0.6857),
# Isotonic dagegen schlechter -> nicht verwendet.
import numpy as np
import pandas as pd
from scipy.optimize import minimize

FENSTER = 3000
MIN_SPIELE = 800
EPS = 1e-6


def _ergebnisse(sp):
    hg, ag = sp["HG"], sp["AG"]
    return (0 if hg > ag else 1 if hg == ag else 2), int(hg + ag > 2), int(hg > 0 and ag > 0)


def _p3(P, w):
    Z = w[0] * np.log(np.clip(P, EPS, 1)) + np.array([0, w[1], w[2]])[None, :]
    Q = np.exp(Z - Z.max(1, keepdims=True))
    return Q / Q.sum(1, keepdims=True)


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _p2(p, w):
    return 1 / (1 + np.exp(-(w[0] * _logit(p) + w[1])))


def fit(spiele):
    """spiele: Liste mit 'p' (kern.maerkte) und HG/AG -> Parameter oder None"""
    spiele = spiele[-FENSTER:]
    if len(spiele) < MIN_SPIELE:
        return None
    P = np.array([[s["p"]["1"], s["p"]["X"], s["p"]["2"]] for s in spiele])
    po = np.array([s["p"]["Ü2.5"] for s in spiele])
    pb = np.array([s["p"]["BTS"] for s in spiele])
    y = np.array([_ergebnisse(s) for s in spiele])
    n = np.arange(len(spiele))
    w1 = minimize(lambda w: -np.log(np.clip(_p3(P, w)[n, y[:, 0]], EPS, 1)).mean(), [1, 0, 0], method="Nelder-Mead").x

    def ll2(p, t):
        p = np.clip(p, EPS, 1 - EPS)
        return -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()
    wo = minimize(lambda w: ll2(_p2(po, w), y[:, 1]), [1, 0], method="Nelder-Mead").x
    wb = minimize(lambda w: ll2(_p2(pb, w), y[:, 2]), [1, 0], method="Nelder-Mead").x
    # Sicherheitsgrenzen je Markt: extreme Werte werden nicht verwendet
    ok = lambda w: 0.3 < w[0] < 2.0
    par = {"x": w1, "ou": wo, "bts": wb}
    return {k: [float(v) for v in w] for k, w in par.items() if ok(w)} or None


def anwenden(p, par):
    """p: dict aus kern.maerkte -> neues dict (Halbzeit, Tipp bleiben)"""
    if not par:
        return p
    q = dict(p)
    if "x" in par:
        a = _p3(np.array([[p["1"], p["X"], p["2"]]]), par["x"])[0]
        q["1"], q["X"], q["2"] = float(a[0]), float(a[1]), float(a[2])
    if "ou" in par:
        o = float(_p2(np.array([p["Ü2.5"]]), par["ou"])[0])
        q["Ü2.5"], q["U2.5"] = o, 1 - o
    if "bts" in par:
        q["BTS"] = float(_p2(np.array([p["BTS"]]), par["bts"])[0])
    return q


def _verlust(p, sp):
    y1, yo, yb = _ergebnisse(sp)
    c = lambda v: min(max(v, EPS), 1 - EPS)
    return {"x": -np.log(c([p["1"], p["X"], p["2"]][y1])),
            "ou": -np.log(c(p["Ü2.5"] if yo else 1 - p["Ü2.5"])),
            "bts": -np.log(c(p["BTS"] if yb else 1 - p["BTS"]))}


def walk_forward(spiele, tage=7):
    """Kalibriert Backtest-Spiele ehrlich: jede Woche nur mit frueheren Spielen. Aendert 'p' direkt,
    die Roh-Prognose bleibt in 'p_roh'. Gibt Parameter fuer kommende Spiele zurueck (alle Spiele)."""
    ordn = sorted(range(len(spiele)), key=lambda i: spiele[i]["Date"])
    roh = [dict(spiele[i], p=spiele[i]["p"]) for i in ordn]
    if not roh:
        return None, {}
    start = pd.Timestamp(roh[0]["Date"])
    daten = [pd.Timestamp(s["Date"]) for s in roh]
    par, grenze, bilanz = None, start, {}
    for pos, i in enumerate(ordn):
        d = daten[pos]
        if d >= grenze:
            while grenze <= d:
                grenze += pd.Timedelta(days=tage)
            frueher = sum(1 for x in daten[:pos] if x < d)      # nur Spiele VOR diesem Tag
            par = fit(roh[:frueher])
        spiele[i]["p_roh"] = spiele[i]["p"]
        spiele[i]["p"] = anwenden(spiele[i]["p"], par)
        if par:
            v0, v1 = _verlust(spiele[i]["p_roh"], spiele[i]), _verlust(spiele[i]["p"], spiele[i])
            for k in v0:
                if k in par:
                    bilanz.setdefault(k, [0.0, 0.0, 0])
                    bilanz[k][0] += v0[k]; bilanz[k][1] += v1[k]; bilanz[k][2] += 1
    # Waechter: ein Markt wird nur kalibriert, wenn es im ehrlichen Rueckblick besser war
    final = fit(roh) or {}
    final = {k: w for k, w in final.items() if k in bilanz and bilanz[k][2] >= 200 and bilanz[k][1] < bilanz[k][0]}
    info = {k: (round(b[0] / b[2], 4), round(b[1] / b[2], 4), b[2]) for k, b in bilanz.items()}
    return final or None, info
