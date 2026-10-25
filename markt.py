# ===== MARKT: was erwarten die Buchmacher? =====
# Aus den Quoten (Pinnacle, sonst Durchschnitt/bet365) fuer 1X2 und OVER/UNDER 2,5 werden die
# erwarteten Tore des MARKTES zurueckgerechnet. Daraus folgen Markt-Wahrscheinlichkeiten fuer
# alle Maerkte - auch fuer BTTS, 1. und 2. Halbzeit, fuer die es keine Quoten gibt.
#
# "Vorteil" = Modell-Wahrscheinlichkeit / Markt-Wahrscheinlichkeit - 1
#   -> zeigt, wo ein Modell ANDERS denkt als der Markt, statt einfach den Favoriten zu tippen.
# Wird auch fuer das Modell "P · Pinnacle" verwendet. Veraendert KEIN anderes Modell.
import math

import numpy as np
from scipy.optimize import minimize

import kern

QUELLEN_1X2 = [("PS", "Pinnacle"), ("Avg", "Durchschnitt"), ("B365", "bet365")]
QUELLEN_OU = [("P", "Pinnacle"), ("Avg", "Durchschnitt"), ("B365", "bet365")]

# Markt -> (Mindest-Wahrscheinlichkeit Modell in %, Mindest-Vorteil, Hoechst-Vorteil)
REGELN = {"1x2": (20, 0.15, 0.60), "ou": (30, 0.12, 0.50), "bts": (30, 0.12, 0.50),
          "ht": (20, 0.15, 0.60), "ht2": (20, 0.15, 0.60)}


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) and v > 1.0 else None


def marge_entfernen(quoten):
    """Power-Methode: beruecksichtigt den Favoriten-Aussenseiter-Effekt."""
    imp = [1 / q for q in quoten]
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if sum(p ** k for p in imp) > 1 else (lo, k)
    p = [x ** ((lo + hi) / 2) for x in imp]
    s = sum(p)
    return [x / s for x in p]


def _hole(row, quellen, suffixe):
    for pre, name in quellen:
        q = [_f(row.get(pre + s)) for s in suffixe]
        if all(q):
            return marge_entfernen(q), name
    return None, None


def _fit(p1, px, p2, pue):
    def fehler(x):
        lh, la = np.exp(x)
        M = kern.matrix_dixon_coles(lh, la, -0.05, g=10)
        a, d, b = kern._1x2(M)
        e = (a - p1) ** 2 + (d - px) ** 2 + (b - p2) ** 2
        if pue is not None:
            idx = np.add.outer(np.arange(11), np.arange(11))
            e += (1 - M[idx <= 2].sum() - pue) ** 2
        return e
    res = minimize(fehler, np.log([1.45, 1.15]), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 200})
    return [float(v) for v in np.exp(res.x)]


def block(row):
    """Markt-Wahrscheinlichkeiten (in %) fuer alle Maerkte oder None."""
    if row is None:
        return None
    p, q1 = _hole(row, QUELLEN_1X2, ["H", "D", "A"])
    if p is None:
        return None
    ou, _ = _hole(row, QUELLEN_OU, [">2.5", "<2.5"])
    lh, la = _fit(p[0], p[1], p[2], ou[0] if ou else None)
    ft = kern.matrix_dixon_coles(lh, la, -0.05)
    ht = kern.matrix_unabhaengig(lh * kern.HZ_ANTEIL, la * kern.HZ_ANTEIL)
    m = kern.maerkte(ft, ht)
    r = lambda x: round(float(x) * 100, 1)
    return {"quelle": q1, "xg": [round(lh, 2), round(la, 2)], "tipp": m["tipp"],
            "1x2": [r(x) for x in p],                                   # direkt aus den Quoten
            "ou": [r(ou[0]), r(ou[1])] if ou else [r(m["Ü2.5"]), r(m["U2.5"])],
            "bts": [r(m["BTS"]), r(1 - m["BTS"])],
            "ht": [r(x) for x in m["HT"]],
            "ht2": [r(x) for x in m["HT2"]]}


def als_prognose(mk):
    """Markt-Block -> Prognose-Eintrag im Format der Modelle (fuer Modell P · Pinnacle)."""
    g = lambda x: int(round(x))      # ganze Prozent wie bei den anderen Modellen
    return {"p1": g(mk["1x2"][0]), "px": g(mk["1x2"][1]), "p2": g(mk["1x2"][2]), "ue25": g(mk["ou"][0]),
            "bts": g(mk["bts"][0]), "ht": [g(x) for x in mk["ht"]], "ht2": [g(x) for x in mk["ht2"]],
            "tipp": mk["tipp"], "xg": mk["xg"],
            "value": [], "vt": {}, "quelle_markt": mk["quelle"]}


def _optionen(prog, mk):
    """(Markt, Label, Modell-%, Markt-%) fuer alle Ausgaenge"""
    out = [("1x2", "1", prog["p1"], mk["1x2"][0]), ("1x2", "X", prog["px"], mk["1x2"][1]),
           ("1x2", "2", prog["p2"], mk["1x2"][2]),
           ("ou", "OVER 2,5", prog["ue25"], mk["ou"][0]), ("ou", "UNDER 2,5", 100 - prog["ue25"], mk["ou"][1]),
           ("bts", "Ja", prog["bts"], mk["bts"][0]), ("bts", "Nein", 100 - prog["bts"], mk["bts"][1])]
    if prog.get("ht"):
        out += [("ht", "HZ1 " + k, prog["ht"][i], mk["ht"][i]) for i, k in enumerate(["1", "X", "2"])]
    if prog.get("ht2"):
        out += [("ht2", "HZ2 " + k, prog["ht2"][i], mk["ht2"][i]) for i, k in enumerate(["1", "X", "2"])]
    return out


def vorteile(prog, mk):
    """Bester Vorteil je Markt -> {markt: [Label, Modell-%, Markt-%, Vorteil-%]}"""
    if not mk:
        return {}
    best = {}
    for markt, lab, pm, pk in _optionen(prog, mk):
        if pm is None or pk is None or pk <= 0.3:
            continue
        min_p, min_v, max_v = REGELN[markt]
        v = pm / pk - 1
        if pm >= min_p and min_v <= v <= max_v and (markt not in best or v > best[markt][3] / 100):
            best[markt] = [lab, round(pm, 1), round(pk, 1), round(v * 100, 1)]
    return best
