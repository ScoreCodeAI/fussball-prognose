# ===== MODELL ST · STACKING (Meta-Modell ueber alle Tor-Modelle) =====
# Idee: Kein Modell ist immer das beste. ST lernt aus den alten Prognosen ALLER Tor-Modelle,
# wie stark es jedem Modell trauen soll, und mischt deren Wahrscheinlichkeiten mit diesen Gewichten.
#  - Gewichte w (>= 0, Summe 1) minimieren den Log-Loss von 1X2 + OVER/UNDER 2,5 (+ 1. Halbzeit)
#  - Ehrlich (walk-forward): jede Woche neu, nur mit Spielen VOR dieser Woche (letzte FENSTER Spiele)
#  - Gemischt wird die komplette Prognose (1X2, O/U, Halbzeiten, Tore je Team, alle Kombis).
#    Weil alle Maerkte aus Tor-Matrizen stammen, ist das gleichbedeutend mit einer gemischten Tor-Matrix
#    -> alle Maerkte bleiben zueinander stimmig.
#  - Danach wie jedes Modell: Platt-Kalibrierung mit Waechter (in engine.py).
# Lineares Mischen statt logistischer Regression: im Test blaehte die Regression bei wenig Daten das Remis auf.
import numpy as np
import pandas as pd

import kern
from scipy.optimize import minimize

INFO = {"id": "st", "name": "Stacking · Meta-Modell", "kurz": "Stacking", "icon": "Σ",
        "farbe1": "#ff7ad9", "farbe2": "#7ff0ff",
        "beschreibung": "Lernt jede Woche, wie stark es jedem Tor-Modell trauen soll, und mischt deren Prognosen"}
FENSTER = 6000
MIN_SPIELE = 800
TAGE = 7
EPS = 1e-6


def _key(sp):
    return (sp["Div"], pd.Timestamp(sp["Date"]).strftime("%Y-%m-%d"), sp["Home"], sp["Away"])


def mischen(ps, w):
    """Gewichtete Mischung beliebig verschachtelter Prognose-Dicts (Zahlen, Listen, Dicts)."""
    a = ps[0]
    if isinstance(a, dict):
        out = {}
        for k in a:
            if k == "tipp":
                out[k] = ps[int(np.argmax(w))][k]
                continue
            v = [p.get(k) for p in ps]
            if any(x is None for x in v):
                out[k] = None
                continue
            try:
                out[k] = mischen(v, w)
            except Exception:
                out[k] = v[int(np.argmax(w))]
        return out
    if isinstance(a, (list, tuple)):
        return [mischen([p[i] for p in ps], w) for i in range(len(a))]
    if isinstance(a, (int, float, np.floating)) and not isinstance(a, bool):
        return float(sum(wi * float(x) for wi, x in zip(w, ps)))
    return ps[int(np.argmax(w))]


MARKT = True       # Markt-Baustein (Tor-Matrix aus Buchmacher-Quoten) mit ins Stacking nehmen
_MK = {}


def shin(quoten):
    """Marge nach Shin entfernen: Aussenseiter tragen mehr Marge als Favoriten (Test 60.535 Spiele: besser als gleichmaessig)."""
    from scipy.optimize import brentq
    pi = 1 / np.array(quoten, dtype=float)
    S = pi.sum()
    f = lambda z: (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / S) - z).sum() / (2 * (1 - z)) - 1
    try:
        z = brentq(f, 0, 0.4)
        p = (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / S) - z) / (2 * (1 - z))
    except Exception:
        p = pi
    return p / p.sum()


def markt_p(q, xg=(1.4, 1.1)):
    """Faire Buchmacher-Wahrscheinlichkeiten (1X2 + O/U 2,5) -> komplette Tor-Matrix -> alle Maerkte."""
    try:
        if not all(q.get(k) for k in ("1", "X", "2", "Ü2.5", "U2.5")):
            return None
        key = tuple(round(float(q[k]), 3) for k in ("1", "X", "2", "Ü2.5", "U2.5"))
        if key in _MK:
            return _MK[key]
        i3 = shin([q["1"], q["X"], q["2"]])
        o = shin([q["Ü2.5"], q["U2.5"]])[0]

        def f(t):
            M = kern.matrix_dixon_coles(np.exp(t[0]), np.exp(t[1]), kern.REMIS_RHO)
            p1, px, p2 = kern._1x2(M)
            n = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
            po = 1 - M[n <= 2].sum()
            return (p1 - i3[0]) ** 2 + (px - i3[1]) ** 2 + (p2 - i3[2]) ** 2 + (po - o) ** 2
        t = minimize(f, np.log(np.clip(xg, 0.2, 4)), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-9}).x
        lh, la = float(np.exp(t[0])), float(np.exp(t[1]))
        ft = kern.matrix_dixon_coles(lh, la, kern.REMIS_RHO)
        p = kern.maerkte(ft, kern.matrix_unabhaengig(lh * 0.45, la * 0.45))
        p["_xg"] = [lh, la]
        _MK[key] = p
        return p
    except Exception:
        return None


def _ziele(sp):
    hg, ag = sp["HG"], sp["AG"]
    ht = None
    if sp.get("HT"):
        h1, a1 = map(int, sp["HT"].split(":"))
        ht = 0 if h1 > a1 else 1 if h1 == a1 else 2
    return (0 if hg > ag else 1 if hg == ag else 2), int(hg + ag > 2), ht


def _fit(zeilen):
    """zeilen: [(liste der p je Modell, ziele)] -> Gewichte"""
    n_mod = len(zeilen[0][0])
    X = np.array([[[p["1"], p["X"], p["2"]][z[0]] for p in ps] for ps, z in zeilen])
    O = np.array([[p["Ü2.5"] if z[1] else 1 - p["Ü2.5"] for p in ps] for ps, z in zeilen])
    hz = [(ps, z) for ps, z in zeilen if z[2] is not None and all(p.get("HT") for p in ps)]
    H = np.array([[p["HT"][z[2]] for p in ps] for ps, z in hz]) if hz else None

    def verlust(t):
        w = np.exp(t - t.max()); w /= w.sum()
        v = -np.log(np.clip(X @ w, EPS, 1)).mean() - np.log(np.clip(O @ w, EPS, 1)).mean()
        if H is not None and len(H):
            v += 0.5 * -np.log(np.clip(H @ w, EPS, 1)).mean()
        return v + 1e-3 * float(t @ t)          # leichte Regularisierung -> keine Extremgewichte
    t = minimize(verlust, np.zeros(n_mod), method="L-BFGS-B").x
    w = np.exp(t - t.max())
    return w / w.sum()


def berechne(bt, live_p, namen, log=print):
    """bt: {modell: [Backtest-Spiele mit p]}, live_p: {modell: {key: (p, xg)}}
    -> (backtest_spiele, live {key: (p, xg)}, gewichte_aktuell)"""
    ids = [m for m in namen if m in bt and bt[m]]
    gross = max((len(bt[m]) for m in ids), default=0)
    weg = [m for m in ids if len(bt[m]) < 0.5 * gross]     # zu kurzer Rueckblick (z. B. SARIMAX: nur 5 Wochen)
    ids = [m for m in ids if m not in weg]
    if weg:
        log("   Stacking: ohne " + ", ".join(weg) + " (zu kurzer Backtest, Gewichte waeren nicht ehrlich lernbar)")
    maps = {m: {_key(sp): sp for sp in bt[m]} for m in ids}
    gemeinsam = set.intersection(*[set(v) for v in maps.values()]) if ids else set()
    spiele = sorted((maps[ids[0]][k] for k in gemeinsam), key=lambda s: s["Date"])
    namen_w = list(ids) + (["markt"] if MARKT else [])
    n_mod = len(namen_w)
    gleich = np.ones(n_mod) / n_mod

    def mit_markt(ps, q, xg):
        if not MARKT:
            return ps
        mp = markt_p(q or {}, xg)
        if mp is None:                       # keine Quoten -> neutral (Durchschnitt der Modelle)
            mp = mischen(ps, np.ones(len(ps)) / len(ps))
        return ps + [{k: v for k, v in mp.items() if k != "_xg"}]
    out, zeilen, w, grenze = [], [], gleich, None
    for sp in spiele:
        d = pd.Timestamp(sp["Date"])
        if grenze is None or d >= grenze:
            grenze = (grenze or d) + pd.Timedelta(days=TAGE)
            while grenze <= d:
                grenze += pd.Timedelta(days=TAGE)
            frueher = [z for z in zeilen if z[2] < d.normalize()][-FENSTER:]
            w = _fit([(ps, zz) for ps, zz, _ in frueher]) if len(frueher) >= MIN_SPIELE else gleich
        k = _key(sp)
        ps = mit_markt([maps[m][k]["p"] for m in ids], maps[ids[0]][k].get("q"), maps[ids[0]][k]["xg"])
        neu = {x: sp[x] for x in ("Div", "Date", "Home", "Away", "HG", "AG", "HT", "Time") if x in sp}
        neu["p"] = mischen(ps, w)
        neu["xg"] = [float(sum(wi * maps[m][k]["xg"][j] for wi, m in zip(w, ids)) / max(sum(w[:len(ids)]), 1e-9)) for j in (0, 1)]
        neu["q"] = maps[ids[0]][k].get("q") or {}
        out.append(neu)
        zeilen.append((ps, _ziele(sp), d.normalize()))
    # ehrlicher Vergleich (nur Spiele, bei denen schon gelernte Gewichte galten)
    rps = lambda p, o: ((p["1"] - (o == 0)) ** 2 + (p["1"] + p["X"] - (o <= 1)) ** 2) / 2
    ab = MIN_SPIELE
    if len(out) > ab + 100:
        z = {m: np.mean([rps(maps[m][_key(sp)]["p"], _ziele(sp)[0]) for sp in out[ab:]]) for m in ids}
        if MARKT:
            z["MARKT"] = np.mean([rps(markt_p(sp["q"]) or sp["p"], _ziele(sp)[0]) for sp in out[ab:]])
        z["STACKING"] = np.mean([rps(sp["p"], _ziele(sp)[0]) for sp in out[ab:]])
        log(f"   Stacking ehrlicher Test ({len(out) - ab} Spiele) RPS: " + ", ".join(f"{k} {v:.4f}" for k, v in z.items()))
    w_live = _fit([(ps, zz) for ps, zz, _ in zeilen[-FENSTER:]]) if len(zeilen) >= MIN_SPIELE else gleich
    live = {}
    for k in set.intersection(*[set(live_p[m]) for m in ids]) if ids else set():
        lv = live_p[ids[0]][k][2] if len(live_p[ids[0]][k]) > 2 else {}
        ps = mit_markt([live_p[m][k][0] for m in ids], (lv or {}).get("q"), live_p[ids[0]][k][1])
        xg = [float(sum(wi * live_p[m][k][1][j] for wi, m in zip(w_live, ids)) / max(sum(w_live[:len(ids)]), 1e-9)) for j in (0, 1)]
        live[k] = (mischen(ps, w_live), xg)
    log(f"   Stacking: {len(out)} Backtest-Spiele, {len(live)} kommend | Gewichte: " +
        ", ".join(f"{m}={x:.2f}" for m, x in zip(namen_w, w_live)))
    return out, live, {m: round(float(x), 3) for m, x in zip(namen_w, w_live)}
