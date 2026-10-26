# ===== BASIS: SARIMAX / Auto-ARIMA (dein Ansatz) =====
# Nach deinem Originalcode:
#  - Heimteam: Zeitreihe seiner Tore in HEIMspielen,  Gastteam: seiner Tore in AUSWAERTSspielen
#  - Halbzeit: eigene Zeitreihen der Halbzeittore, unabhaengig von der Vollzeit
#  - Auto-ARIMA sucht p, d, q selbst (p, q von 0 bis 2, d per KPSS-Test, Auswahl per AIC),
#    so wie pmdarima.auto_arima - hier mit statsmodels nachgebaut, weil pmdarima mit neuen
#    Python-Versionen oft nicht installierbar ist.
#  - Prognose der naechsten Torzahl = Lambda -> Poisson-Ergebnis-Matrix -> alle Maerkte
# Variante "Plus": zusaetzlich die Gegnerstaerke als aeusserer Einfluss (das "X" in SARIMAX):
#  Heimtore haengen von der Auswaerts-Gegentor-Neigung des Gegners ab, Auswaertstore von der
#  Heim-Gegentor-Neigung des Gegners. Diese Neigung wird nur aus Spielen VOR dem jeweiligen
#  Spiel berechnet (kein Blick in die Zukunft).
# Technische Absicherung: Prognosen werden auf sinnvolle Werte begrenzt (keine negativen Tore).
import warnings

import numpy as np
import pandas as pd

import kern

MAX_LAENGE = 250       # Rechenzeit-Grenze je Zeitreihe (ca. 15 Saisons Heim- bzw. Auswaertsspiele)
MIN_LAENGE = 15        # darunter: Durchschnitt statt ARIMA
EWMA_ALPHA = 0.15      # Plus: wie schnell sich die Gegentor-Neigung anpasst (Annahme)
GRENZEN = {"ft": (0.05, 6.0), "ht": (0.02, 4.0)}
GITTER_BETA = [0.0, 0.25, 0.5, 0.75, 1.0]   # Gewicht der Gegnerstaerke (automatisch je Land gewaehlt)
VALID_TAGE = 120


def auto_arima(y, X=None, xf=None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import kpss
    y = np.asarray(y, float)
    d = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = 1 if kpss(y, regression="c", nlags="auto")[1] < 0.05 else 0
    except Exception:
        pass
    cache = {}

    def fit(p, q):
        if (p, q) not in cache:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(y, exog=X, order=(p, d, q), trend="c" if d == 0 else "n").fit(disp=False, maxiter=50)
                cache[(p, q)] = (res.aic if np.isfinite(res.aic) else np.inf, res)
            except Exception:
                cache[(p, q)] = (np.inf, None)
        return cache[(p, q)]

    best = min([(0, 0), (1, 0), (0, 1), (1, 1)], key=lambda k: fit(*k)[0])
    while True:
        nb = [(best[0] + a, best[1] + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= best[0] + a <= 2 and 0 <= best[1] + b <= 2]
        cand = min(nb, key=lambda k: fit(*k)[0])
        if fit(*cand)[0] < fit(*best)[0]:
            best = cand
        else:
            break
    res = fit(*best)[1]
    if res is None:
        return float(np.mean(y))
    fc = res.forecast(1, exog=None if X is None else np.array([[xf]]))
    return float(np.asarray(fc)[0])


def trainiere(df, heute, plus):
    d = df[df["Date"] < heute].sort_values("Date").reset_index(drop=True)
    if len(d) < 50:
        raise ValueError("zu wenig Daten")
    ht_ok = d[["HTHG", "HTAG"]].notna().all(axis=1).sum() >= 200
    letzte = d[d["Date"] >= heute - pd.DateOffset(years=2)]
    m = {"d": d, "plus": plus, "ht_ok": ht_ok, "cache": {},
         "mittel": {"HG": letzte["HG"].mean(), "AG": letzte["AG"].mean(),
                    "HTHG": letzte["HTHG"].mean(), "HTAG": letzte["HTAG"].mean()}}
    _gegner_neigung(m)          # Gegentor-Neigung je Team (auch fuer das Original, als Gegnerstaerke)
    m["beta"] = 0.0 if plus else _waehle_beta(m, heute)
    m["abh"] = _waehle_abh(letzte, m["mittel"])
    return m


def _waehle_beta(m, heute):
    """Wie stark zaehlt die Abwehr des Gegners? Auswahl ueber die letzten VALID_TAGE (schneller Ersatz-Schaetzer)."""
    d = m["d"]
    va = d[d["Date"] >= heute - pd.Timedelta(days=VALID_TAGE)]
    if len(va) < 60:
        return 0.0
    mit = m["mittel"]
    best, best_ll = 0.5, -np.inf
    form_h = d.groupby("Home")["HG"].apply(lambda x: x.shift().rolling(10, min_periods=3).mean())
    form_a = d.groupby("Away")["AG"].apply(lambda x: x.shift().rolling(10, min_periods=3).mean())
    fh = pd.Series(form_h.reset_index(level=0, drop=True)).reindex(va.index).fillna(mit["HG"]).to_numpy()
    fa = pd.Series(form_a.reset_index(level=0, drop=True)).reindex(va.index).fillna(mit["AG"]).to_numpy()
    oh = (va["ex_h_ft"] / mit["HG"]).to_numpy()
    oa = (va["ex_a_ft"] / mit["AG"]).to_numpy()
    yh, ya = va["HG"].to_numpy(float), va["AG"].to_numpy(float)
    lls = {}
    for b in GITTER_BETA:
        lh = np.clip(fh * oh ** b, 0.05, 6)
        la = np.clip(fa * oa ** b, 0.05, 6)
        lls[b] = np.sum(yh * np.log(lh) - lh + ya * np.log(la) - la)
    best = max(lls, key=lls.get)
    # nur uebernehmen, wenn die Verbesserung deutlich ist (sonst Original-Verhalten ohne Gegnerstaerke)
    return best if lls[best] - lls[0.0] > 0.01 * len(va) else 0.0


def _waehle_abh(d, mit):
    """Abhaengigkeit Heim/Gast (unabhaengig, Dixon-Coles oder Frank-Copula) automatisch per Likelihood."""
    try:
        d = d.dropna(subset=["HG", "AG"])
        if len(d) < 200:
            return ("unabhaengig", 0.0)
        n = len(d)
        return kern.waehle_abhaengigkeit(d["HG"].to_numpy(int), d["AG"].to_numpy(int),
                                         np.full(n, mit["HG"]), np.full(n, mit["AG"]), np.ones(n))
    except Exception:
        return ("unabhaengig", 0.0)


def _gegner_neigung(m):
    """Leckagefreie Gegentor-Neigung (EWMA) je Team, getrennt Heim/Auswaerts und Voll-/Halbzeit."""
    d = m["d"]
    mit = m["mittel"]
    zust = {}   # (team, ort, ziel) -> aktueller Wert
    ex = {k: np.empty(len(d)) for k in ("h_ft", "a_ft", "h_ht", "a_ht")}
    H, A = d["Home"].to_numpy(), d["Away"].to_numpy()
    HG, AG = d["HG"].to_numpy(float), d["AG"].to_numpy(float)
    HH, HA = d["HTHG"].to_numpy(float), d["HTAG"].to_numpy(float)
    for r in range(len(d)):
        h, a = H[r], A[r]
        # vor dem Spiel bekannt:
        ex["h_ft"][r] = zust.get((a, "aus", "ft"), mit["HG"])   # Gegner (Gast) kassiert auswaerts
        ex["a_ft"][r] = zust.get((h, "heim", "ft"), mit["AG"])  # Gegner (Heim) kassiert zu Hause
        ex["h_ht"][r] = zust.get((a, "aus", "ht"), mit["HTHG"])
        ex["a_ht"][r] = zust.get((h, "heim", "ht"), mit["HTAG"])
        # nach dem Spiel aktualisieren:
        zust[(a, "aus", "ft")] = (1 - EWMA_ALPHA) * ex["h_ft"][r] + EWMA_ALPHA * HG[r]
        zust[(h, "heim", "ft")] = (1 - EWMA_ALPHA) * ex["a_ft"][r] + EWMA_ALPHA * AG[r]
        if not np.isnan(HH[r]) and not np.isnan(HA[r]):
            zust[(a, "aus", "ht")] = (1 - EWMA_ALPHA) * ex["h_ht"][r] + EWMA_ALPHA * HH[r]
            zust[(h, "heim", "ht")] = (1 - EWMA_ALPHA) * ex["a_ht"][r] + EWMA_ALPHA * HA[r]
    for k, v in ex.items():
        d["ex_" + k] = v
    m["zust"] = zust


def _lambda(m, team, gegner, ort, ziel):
    key = (team, gegner if m["plus"] else None, ort, ziel)
    if key in m["cache"]:
        return m["cache"][key]
    d, mit = m["d"], m["mittel"]
    spalte = {("heim", "ft"): "HG", ("aus", "ft"): "AG", ("heim", "ht"): "HTHG", ("aus", "ht"): "HTAG"}[(ort, ziel)]
    s = d[d["Home"] == team] if ort == "heim" else d[d["Away"] == team]
    s = s.dropna(subset=[spalte]).tail(MAX_LAENGE)
    y = s[spalte].to_numpy(float)
    if len(y) < MIN_LAENGE:
        wert = float(np.mean(y)) if len(y) >= 3 else float(mit[spalte])
    else:
        X = xf = None
        if m["plus"]:
            X = s["ex_" + ("h_" if ort == "heim" else "a_") + ziel].to_numpy(float).reshape(-1, 1)
            # aktuelle Neigung des Gegners: Gast kassiert auswaerts / Heimteam kassiert zu Hause
            xf = m["zust"].get((gegner, "aus" if ort == "heim" else "heim", ziel),
                               mit["HG" if ort == "heim" else "AG"] if ziel == "ft" else mit["HTHG" if ort == "heim" else "HTAG"])
        try:
            wert = auto_arima(y, X, xf)
        except Exception:
            wert = float(np.mean(y))
    lo, hi = GRENZEN[ziel]
    wert = float(np.clip(wert if np.isfinite(wert) else mit[spalte], lo, hi))
    m["cache"][key] = wert
    return wert


def vorhersage(m, heim, gast):
    lh = _lambda(m, heim, gast, "heim", "ft")
    la = _lambda(m, gast, heim, "aus", "ft")
    b = m.get("beta", 0.0)
    if b:   # Gegnerstaerke: kassiert der Gegner viel, steigen die erwarteten Tore
        mit = m["mittel"]
        lh *= (m["zust"].get((gast, "aus", "ft"), mit["HG"]) / mit["HG"]) ** b
        la *= (m["zust"].get((heim, "heim", "ft"), mit["AG"]) / mit["AG"]) ** b
        lh, la = float(np.clip(lh, *GRENZEN["ft"])), float(np.clip(la, *GRENZEN["ft"]))
    out = {"ft": kern.matrix(lh, la, m.get("abh", ("unabhaengig", 0.0))), "ht": None, "xg": [lh, la]}
    if m["ht_ok"]:
        hh = _lambda(m, heim, gast, "heim", "ht")
        ha = _lambda(m, gast, heim, "aus", "ht")
        out["ht"] = kern.matrix_unabhaengig(hh, ha)
    return out
