# ===== MODELL RF · RANDOM FOREST (reine Statistik, ohne Quoten) =====
# Viele unabhaengige Entscheidungsbaeume, jeder auf einer Zufalls-Stichprobe der Spiele und Merkmale;
# die Prognose ist der Durchschnitt aller Baeume (robust, kaum Ueberanpassung).
# Merkmale (gleiche Aufbereitung wie XGBoost, aber OHNE Buchmacher-Quoten):
#   Schuss-Staerken, Tor-Staerken (Angriff/Abwehr), Elo, Form (letzte 5), Ruhetage, Liga.
# Dadurch Prognose fuer ALLE Spiele, auch ohne Quoten, und unabhaengig vom Markt.
# Ehrlich: Training nur mit Spielen VOR der angezeigten Spielwoche.
import numpy as np
import pandas as pd

import kern
import xgb_modell as X

INFO = {"id": "rf", "name": "Random Forest · Statistik", "kurz": "RandomForest", "icon": "🌲",
        "farbe1": "#2bf5a0", "farbe2": "#7ff0ff",
        "beschreibung": "Random Forest aus Schuss- und Tor-Stärken, Elo, Form und Ruhetagen – ohne Buchmacher-Quoten"}
MARKT = {"m1", "mx", "m2", "mo"}
FEAT = [f for f in X.FEAT if f not in MARKT]
BAEUME, BLATT = 400, 60


def _modell():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=BAEUME, min_samples_leaf=BLATT, max_features="sqrt",
                                  n_jobs=-1, random_state=7)


def _ziele(r):
    hg, ag = int(r.HG), int(r.AG)
    z = dict(y1=0 if hg > ag else 1 if hg == ag else 2, yo=int(hg + ag > 2), yb=int(hg > 0 and ag > 0))
    if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)):
        h1, a1 = int(r.HTHG), int(r.HTAG)
        z["yh"] = 0 if h1 > a1 else 1 if h1 == a1 else 2
        z["yh2"] = 0 if hg - h1 > ag - a1 else 1 if hg - h1 == ag - a1 else 2
    return z


def berechne(spiele, liste, log=print):
    vor, ende = X._features(spiele)
    ligen = {d: i for i, d in enumerate(sorted(spiele["Div"].unique()))}
    stichtag = min(pd.Timestamp(e["datum"]) for e in liste) if liste else pd.Timestamp.today()
    rows = []
    for r in spiele.dropna(subset=["HG", "AG"]).itertuples(index=False):
        if r.Date >= stichtag:
            continue
        f = vor.get(X._key(r.Div, r.Date, r.Home, r.Away))
        if f:
            rows.append(dict(f, liga=ligen.get(r.Div, -1), **_ziele(r)))
    T = pd.DataFrame(rows)
    if len(T) < 1000:
        log(f"RandomForest: zu wenig Trainingsspiele ({len(T)})")
        return {}
    modelle = {}
    for ziel in ("y1", "yo", "yb", "yh", "yh2"):
        t = T.dropna(subset=[ziel]) if ziel in T else None
        if t is None or len(t) < 1000 or t[ziel].nunique() < 2:
            continue
        m = _modell(); m.fit(t[FEAT], t[ziel].astype(int)); modelle[ziel] = m
    out = {}
    for i, e in enumerate(liste):
        k = (e["liga"], e["datum"], e["heim"], e["gast"])
        f = vor.get(k) if e.get("ergebnis") else None
        if f is None:
            if e["liga"] not in ende:
                continue
            f = X._zukunft(ende[e["liga"]], e["liga"], e["datum"], e["heim"], e["gast"])
        Xe = pd.DataFrame([dict(f, liga=ligen.get(e["liga"], -1))])[FEAT]
        pr = lambda z: modelle[z].predict_proba(Xe)[0] if z in modelle else None
        p1, po, pb, ph, ph2 = pr("y1"), pr("yo"), pr("yb"), pr("yh"), pr("yh2")
        if p1 is None or po is None:
            continue
        M = kern.matrix_unabhaengig(f["lh"], f["la"])
        a, b = np.unravel_index(M.argmax(), M.shape)
        r0 = lambda x: round(float(x) * 100, 2)
        out[i] = {"p1": r0(p1[0]), "px": r0(p1[1]), "p2": r0(p1[2]), "ue25": r0(po[1]),
                  "bts": r0(pb[1]) if pb is not None else r0(f["sb"]),
                  "ht": [r0(x) for x in ph] if ph is not None else None,
                  "ht2": [r0(x) for x in ph2] if ph2 is not None else None,
                  "tipp": f"{a}:{b}", "xg": [round(f["lh"], 2), round(f["la"], 2)], "value": [],
                  # Halbzeit OVER 0,5 aus den erwarteten Toren (Anteil 1. HZ 45 %, 2. HZ 55 %)
                  "h1o05": r0(1 - np.exp(-0.45 * (f["lh"] + f["la"]))), "h2o05": r0(1 - np.exp(-0.55 * (f["lh"] + f["la"])))}
    log(f"RandomForest: {len(T)} Trainingsspiele, {len(modelle)} Märkte, {len(out)} Prognosen")
    return out


def rueckblick(spiele, ab, log=print):
    """Ehrlicher Test: Training vor 'ab', Prognose ab 'ab'."""
    vor, _ = X._features(spiele)
    ligen = {d: i for i, d in enumerate(sorted(spiele["Div"].unique()))}
    ab = pd.Timestamp(ab); tr, te = [], []
    for r in spiele.dropna(subset=["HG", "AG"]).itertuples(index=False):
        f = vor.get(X._key(r.Div, r.Date, r.Home, r.Away))
        if f:
            z = dict(f, liga=ligen.get(r.Div, -1), **_ziele(r), HG=int(r.HG), AG=int(r.AG))
            (tr if r.Date < ab else te).append(z)
    T, E = pd.DataFrame(tr), pd.DataFrame(te)
    m1, mo = _modell(), _modell()
    m1.fit(T[FEAT], T["y1"]); mo.fit(T[FEAT], T["yo"])
    return E, m1.predict_proba(E[FEAT]), mo.predict_proba(E[FEAT])
