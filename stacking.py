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
    n_mod = len(ids)
    gleich = np.ones(n_mod) / n_mod
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
        ps = [maps[m][k]["p"] for m in ids]
        neu = {x: sp[x] for x in ("Div", "Date", "Home", "Away", "HG", "AG", "HT", "Time") if x in sp}
        neu["p"] = mischen(ps, w)
        neu["xg"] = [float(sum(wi * maps[m][k]["xg"][j] for wi, m in zip(w, ids))) for j in (0, 1)]
        neu["q"] = maps[ids[0]][k].get("q") or {}
        out.append(neu)
        zeilen.append((ps, _ziele(sp), d.normalize()))
    # ehrlicher Vergleich (nur Spiele, bei denen schon gelernte Gewichte galten)
    rps = lambda p, o: ((p["1"] - (o == 0)) ** 2 + (p["1"] + p["X"] - (o <= 1)) ** 2) / 2
    ab = MIN_SPIELE
    if len(out) > ab + 100:
        z = {m: np.mean([rps(maps[m][_key(sp)]["p"], _ziele(sp)[0]) for sp in out[ab:]]) for m in ids}
        z["STACKING"] = np.mean([rps(sp["p"], _ziele(sp)[0]) for sp in out[ab:]])
        log(f"   Stacking ehrlicher Test ({len(out) - ab} Spiele) RPS: " + ", ".join(f"{k} {v:.4f}" for k, v in z.items()))
    w_live = _fit([(ps, zz) for ps, zz, _ in zeilen[-FENSTER:]]) if len(zeilen) >= MIN_SPIELE else gleich
    live = {}
    for k in set.intersection(*[set(live_p[m]) for m in ids]) if ids else set():
        ps = [live_p[m][k][0] for m in ids]
        xg = [float(sum(wi * live_p[m][k][1][j] for wi, m in zip(w_live, ids))) for j in (0, 1)]
        live[k] = (mischen(ps, w_live), xg)
    log(f"   Stacking: {len(out)} Backtest-Spiele, {len(live)} kommend | Gewichte: " +
        ", ".join(f"{m}={x:.2f}" for m, x in zip(ids, w_live)))
    return out, live, {m: round(float(x), 3) for m, x in zip(ids, w_live)}
