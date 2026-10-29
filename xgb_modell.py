# ===== MODELL X · XGBOOST (mit Quote) =====
# Gradient-Boosting-Baeume lernen direkt jeden Markt (1X2, OVER/UNDER 2,5, BTTS, 1. und 2. Halbzeit) aus:
#   - fairen Markt-Wahrscheinlichkeiten (Pinnacle, sonst Durchschnitt; Marge entfernt)
#   - Schuss-Staerken beider Teams (Schuesse aufs Tor + Tore, laufend nachgefuehrt)
#   - reiner Tor-Staerke, Form (letzte 5 Spiele), Ruhetage, Liga
# Getestet 2024-26 auf 10 Ligen: in jeder Liga genauer als alle bisherigen Modelle.
# Trainiert wird nur mit Spielen VOR der angezeigten Spielwoche (kein Blick in die Zukunft).
# Nutzt xgboost, falls installiert, sonst das gleichwertige HistGradientBoosting aus scikit-learn.
import numpy as np
import pandas as pd

import kern
import markt

INFO = {"id": "xgb", "name": "XGBoost · Markt + Schüsse", "kurz": "XGBoost", "icon": "X",
        "farbe1": "#ffd84d", "farbe2": "#ff7a45",
        "beschreibung": "Gradient Boosting aus fairer Quote, Schuss-Stärken, Form und Ruhetagen"}

K = 0.03
FEAT = ["m1", "mx", "m2", "mo", "s1", "sx", "s2", "so", "sb", "lh", "la", "g1", "gx", "g2", "go",
        "Ah", "Dh", "Aa", "Da", "f_h", "f_a", "rest_h", "rest_a", "liga"]


def _modell(n_klassen):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=40,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbosity=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=4,
                                              min_samples_leaf=80, l2_regularization=1.0, random_state=1)


def _kurz_maerkte(lh, la):
    M = kern.matrix_unabhaengig(lh, la, g=10)
    idx = np.add.outer(np.arange(11), np.arange(11))
    return (*kern._1x2(M), float(1 - M[idx <= 2].sum()), float(1 - M[0].sum() - M[:, 0].sum() + M[0, 0]))


def _key(div, date, h, a):
    return (div, pd.Timestamp(date).strftime("%Y-%m-%d"), h, a)


def _features(spiele):
    """Walk-forward ueber alle Spiele je Land: Merkmale VOR jedem Spiel + Endzustand je Liga-Gruppe."""
    vor, ende = {}, {}
    for div, df in spiele.groupby("Div"):
        df = df.sort_values("Date")
        A, D, A0, D0 = {}, {}, {}, {}
        mu, h, conv = np.log(1.35), 0.12, 0.32
        mu0, h0 = mu, h
        form, last = {}, {}
        for r in df.itertuples(index=False):
            a, b = r.Home, r.Away
            lh = np.exp(mu + h + A.get(a, 0) - D.get(b, 0)); la = np.exp(mu - h + A.get(b, 0) - D.get(a, 0))
            gh = np.exp(mu0 + h0 + A0.get(a, 0) - D0.get(b, 0)); ga = np.exp(mu0 - h0 + A0.get(b, 0) - D0.get(a, 0))
            vor[_key(div, r.Date, a, b)] = _zeile(lh, la, gh, ga, A, D, a, b, form, last, r.Date)
            hg, ag = float(r.HG), float(r.AG)
            sh, sa = r.HST, r.AST
            if pd.isna(sh) or pd.isna(sa):
                th, ta = hg, ag
            else:
                th, ta = 0.6 * sh * conv + 0.4 * hg, 0.6 * sa * conv + 0.4 * ag
                if sh + sa > 0:
                    conv = 0.995 * conv + 0.005 * ((hg + ag) / (sh + sa))
            eh, ea = th - lh, ta - la
            A[a] = A.get(a, 0) + K * eh; D[b] = D.get(b, 0) - K * eh; A[b] = A.get(b, 0) + K * ea; D[a] = D.get(a, 0) - K * ea
            mu += 0.002 * (eh + ea); h += 0.002 * (eh - ea)
            e0, e1 = hg - gh, ag - ga
            A0[a] = A0.get(a, 0) + K * e0; D0[b] = D0.get(b, 0) - K * e0; A0[b] = A0.get(b, 0) + K * e1; D0[a] = D0.get(a, 0) - K * e1
            mu0 += 0.002 * (e0 + e1); h0 += 0.002 * (e0 - e1)
            form.setdefault(a, []).append(hg - ag); form.setdefault(b, []).append(ag - hg)
            last[a] = last[b] = r.Date
        ende[div] = dict(A=A, D=D, A0=A0, D0=D0, mu=mu, h=h, mu0=mu0, h0=h0, form=form, last=last)
    return vor, ende


def _zeile(lh, la, gh, ga, A, D, a, b, form, last, datum):
    s, g = _kurz_maerkte(lh, la), _kurz_maerkte(gh, ga)
    fa, fb = form.get(a, []), form.get(b, [])
    rest = lambda t: min((pd.Timestamp(datum) - last[t]).days, 30) if t in last else 7
    return dict(s1=s[0], sx=s[1], s2=s[2], so=s[3], sb=s[4], lh=lh, la=la, g1=g[0], gx=g[1], g2=g[2], go=g[3],
                Ah=A.get(a, 0), Dh=D.get(a, 0), Aa=A.get(b, 0), Da=D.get(b, 0),
                f_h=float(np.mean(fa[-5:])) if fa else 0.0, f_a=float(np.mean(fb[-5:])) if fb else 0.0,
                rest_h=rest(a), rest_a=rest(b))


def _zukunft(e, div, datum, a, b):
    lh = np.exp(e["mu"] + e["h"] + e["A"].get(a, 0) - e["D"].get(b, 0)); la = np.exp(e["mu"] - e["h"] + e["A"].get(b, 0) - e["D"].get(a, 0))
    gh = np.exp(e["mu0"] + e["h0"] + e["A0"].get(a, 0) - e["D0"].get(b, 0)); ga = np.exp(e["mu0"] - e["h0"] + e["A0"].get(b, 0) - e["D0"].get(a, 0))
    return _zeile(lh, la, gh, ga, e["A"], e["D"], a, b, e["form"], e["last"], datum)


def _markt(row):
    if row is None:
        return None
    p, _ = markt._hole(row, markt.QUELLEN_1X2, ["H", "D", "A"])
    if p is None:
        return None
    o, _ = markt._hole(row, markt.QUELLEN_OU, [">2.5", "<2.5"])
    return dict(m1=p[0], mx=p[1], m2=p[2], mo=o[0] if o else np.nan)


def berechne(spiele, q, liste, zeilen, log=print):
    """-> {spiel_index: prognose} im Format der anderen Modelle."""
    vor, ende = _features(spiele)
    ligen = {d: i for i, d in enumerate(sorted(spiele["Div"].unique()))}
    stichtag = min(pd.Timestamp(e["datum"]) for e in liste) if liste else pd.Timestamp.today()

    rows = []
    for r in q.itertuples(index=False):
        if r.Date >= stichtag:
            continue
        k = _key(r.Div, r.Date, r.Home, r.Away)
        f, m = vor.get(k), _markt(zeilen.get(k))
        if not f or not m:
            continue
        hg, ag = int(r.HG), int(r.AG)
        z = dict(f, **m, liga=ligen.get(r.Div, -1),
                 y1=0 if hg > ag else 1 if hg == ag else 2, yo=int(hg + ag > 2), yb=int(hg > 0 and ag > 0))
        if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)):
            h1, a1 = int(r.HTHG), int(r.HTAG)
            z["yh"] = 0 if h1 > a1 else 1 if h1 == a1 else 2
            z["yh2"] = 0 if hg - h1 > ag - a1 else 1 if hg - h1 == ag - a1 else 2
        rows.append(z)
    T = pd.DataFrame(rows)
    if len(T) < 1000:
        log(f"XGBoost: zu wenig Trainingsspiele ({len(T)})")
        return {}
    modelle = {}
    for ziel in ("y1", "yo", "yb", "yh", "yh2"):
        t = T.dropna(subset=[ziel]) if ziel in T else None
        if t is None or len(t) < 1000 or t[ziel].nunique() < 2:
            continue
        mod = _modell(t[ziel].nunique())
        mod.fit(t[FEAT], t[ziel].astype(int))
        modelle[ziel] = mod

    out = {}
    for i, e in enumerate(liste):
        k = (e["liga"], e["datum"], e["heim"], e["gast"])
        m = _markt(zeilen.get(k))
        if not m:
            continue
        f = vor.get(k) if e.get("ergebnis") else None
        if f is None:
            if e["liga"] not in ende:
                continue
            f = _zukunft(ende[e["liga"]], e["liga"], e["datum"], e["heim"], e["gast"])
        X = pd.DataFrame([dict(f, **m, liga=ligen.get(e["liga"], -1))])[FEAT]
        pr = lambda z: modelle[z].predict_proba(X)[0] if z in modelle else None
        p1, po, pb, ph, ph2 = pr("y1"), pr("yo"), pr("yb"), pr("yh"), pr("yh2")
        if p1 is None or po is None:
            continue
        M = kern.matrix_unabhaengig(f["lh"], f["la"])
        a, b = np.unravel_index(M.argmax(), M.shape)
        r0 = lambda x: round(float(x) * 100, 1)
        out[i] = {"p1": r0(p1[0]), "px": r0(p1[1]), "p2": r0(p1[2]), "ue25": r0(po[1]),
                  "bts": r0(pb[1]) if pb is not None else r0(f["sb"]),
                  "ht": [r0(x) for x in ph] if ph is not None else None,
                  "ht2": [r0(x) for x in ph2] if ph2 is not None else None,
                  "tipp": f"{a}:{b}", "xg": [round(f["lh"], 2), round(f["la"], 2)], "value": []}
    log(f"XGBoost: {len(T)} Trainingsspiele, {len(modelle)} Märkte, {len(out)} Prognosen")
    return out


def rueckblick(spiele, q, zeilen, ab, log=print):
    """Ehrlicher Rueckblick fuer das Modell Linie: Training nur mit Spielen VOR 'ab',
    Prognose fuer alle Spiele ab 'ab' -> Liste wie die Backtest-Spiele der anderen Modelle."""
    vor, _ = _features(spiele)
    ligen = {d: i for i, d in enumerate(sorted(spiele["Div"].unique()))}
    ab = pd.Timestamp(ab)
    tr, te = [], []
    for r in q.itertuples(index=False):
        k = _key(r.Div, r.Date, r.Home, r.Away)
        f, m = vor.get(k), _markt(zeilen.get(k))
        if not f or not m:
            continue
        hg, ag = int(r.HG), int(r.AG)
        z = dict(f, **m, liga=ligen.get(r.Div, -1), y1=0 if hg > ag else 1 if hg == ag else 2, yo=int(hg + ag > 2),
                 Div=r.Div, Date=r.Date, Home=r.Home, Away=r.Away, HG=hg, AG=ag)
        (tr if r.Date < ab else te).append(z)
    if len(tr) < 1000 or not te:
        return []
    T, E = pd.DataFrame(tr), pd.DataFrame(te)
    m1, mo = _modell(3), _modell(2)
    m1.fit(T[FEAT], T["y1"]); mo.fit(T[FEAT], T["yo"])
    p1, po = m1.predict_proba(E[FEAT]), mo.predict_proba(E[FEAT])
    out = [{"Div": r.Div, "Date": r.Date, "Home": r.Home, "Away": r.Away, "HG": r.HG, "AG": r.AG,
            "p": {"1": a[0], "X": a[1], "2": a[2], "\u00dc2.5": b[1], "U2.5": b[0]}}
           for r, a, b in zip(E.itertuples(index=False), p1, po)]
    log(f"XGBoost-Rückblick: {len(T)} Trainings-, {len(out)} Testspiele")
    return out
