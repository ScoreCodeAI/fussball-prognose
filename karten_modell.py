# ===== MODELL K · KARTEN (aus ecken_modell abgeleitet) =====
# Karten je Team = Gelb + Rot (football-data HY/AY/HR/AR, aktuell ergaenzt durch SportsAPI).
# Jedes Team hat eine Ecken-Staerke (Ecken herausholen) und eine Ecken-Abwehr (Ecken zulassen),
# laufend nachgefuehrt. Lernziel: 70 % echte Ecken + 30 % Schuss-Erwartung (Schuesse x Ecken je Schuss).
# Test 2024-26, 10 Ligen:
#   - "Wer bekommt mehr Ecken": 66 % Treffer (immer "Heim" waere 59,5 %)
#   - Ecken je Team: deutlich genauer als der Liga-Durchschnitt
#   - Gesamt-Ecken: kaum genauer als der Liga-Durchschnitt (wird trotzdem angezeigt)
# NEU Fouls (HF/AF): je Team Foul-Neigung (begeht Fouls) und Foul-Provokation (zieht Fouls),
#   erwartete Fouls x Karten je Foul (laufend je Liga) = zweite Karten-Erwartung.
#   Beide werden gemischt (geometrisch, Gewicht W je Liga automatisch aus dem Gitter, ehrlich vorwaerts gemessen).
#   Test 2019-26, 16 Ligen: Fouls verbessern die Karten-Prognose in ALLEN Ligen.
import numpy as np
import pandas as pd
from scipy.stats import poisson

INFO = {"id": "karten", "typ": "karten", "name": "Karten · Prognose", "kurz": "Karten", "icon": "🟨",
        "farbe1": "#ffe04d", "farbe2": "#ff4d7d",
        "beschreibung": "Karten (Gelb + Rot) je Team und gesamt"}

K, SCHUSS, PRO_SCHUSS = 0.04, 0.0, 1.0
KF = 0.03
GITTER_W = [0.0, 0.2, 0.35, 0.5, 0.65]
G = 25
LINIEN_GESAMT = [3.5, 4.5, 5.5]
LINIEN_TEAM = [1.5, 2.5]


def _key(div, date, h, a):
    return (div, pd.Timestamp(date).strftime("%Y-%m-%d"), h, a)


def _liga(df, W):
    A, D, TF, PF = {}, {}, {}, {}
    mu, h, muf, hf, r = np.log(2.1), -0.05, np.log(12), -0.03, 0.17
    vor, echt, ll = {}, {}, 0.0
    for i, x in enumerate(df.itertuples(index=False)):
        a, b = x.Home, x.Away
        ch, ca = x.HC, x.AC
        lh = np.exp(mu + h + A.get(a, 0) - D.get(b, 0)); la = np.exp(mu - h + A.get(b, 0) - D.get(a, 0))
        fh = np.exp(muf + hf + TF.get(a, 0) + PF.get(b, 0)); fa = np.exp(muf - hf + TF.get(b, 0) + PF.get(a, 0))
        ph, pa = lh ** (1 - W) * (fh * r) ** W, la ** (1 - W) * (fa * r) ** W
        k = _key(x.Div, x.Date, a, b)
        vor[k] = (ph, pa); echt[k] = (int(ch), int(ca))
        if i > 300:
            ll += poisson.logpmf(ch, ph) + poisson.logpmf(ca, pa)
        eh, ea = (ch - lh) / lh, (ca - la) / la
        A[a] = A.get(a, 0) + K * eh; D[b] = D.get(b, 0) - K * eh
        A[b] = A.get(b, 0) + K * ea; D[a] = D.get(a, 0) - K * ea
        mu += 0.003 * (eh + ea); h += 0.003 * (eh - ea)
        if not (pd.isna(x.HF) or pd.isna(x.AF)):
            gh, ga = (x.HF - fh) / fh, (x.AF - fa) / fa
            TF[a] = TF.get(a, 0) + KF * gh; PF[b] = PF.get(b, 0) + KF * gh
            TF[b] = TF.get(b, 0) + KF * ga; PF[a] = PF.get(a, 0) + KF * ga
            muf += 0.003 * (gh + ga); hf += 0.003 * (gh - ga)
            if x.HF + x.AF > 0:
                r = 0.995 * r + 0.005 * (ch + ca) / (x.HF + x.AF)
    return vor, echt, ll, (A, D, mu, h, TF, PF, muf, hf, r, W)


def _lauf(spiele, log=None):
    vor, ende, echt, wahl = {}, {}, {}, {}
    for div, df in spiele.groupby("Div"):
        df = df.dropna(subset=["HY", "AY"]).sort_values("Date")
        if len(df) < 300:
            continue
        for c in ("HF", "AF", "HR", "AR"):
            if c not in df:
                df[c] = np.nan
        df = df.assign(HC=df["HY"] + df["HR"].fillna(0), AC=df["AY"] + df["AR"].fillna(0))
        gitter = GITTER_W if df["HF"].notna().mean() > 0.3 else [0.0]
        best = max((_liga(df, W) for W in gitter), key=lambda x: x[2])
        v, e, ll, en = best
        vor.update(v); echt.update(e); ende[div] = en; wahl[div] = en[-1]
    if log:
        log("Karten: Foul-Gewicht je Liga " + ", ".join(f"{d}={w:.2f}" for d, w in sorted(wahl.items())))
    return vor, ende, echt


def prognose(lh, la):
    k = np.arange(G + 1)
    ph, pa = poisson.pmf(k, lh), poisson.pmf(k, la)
    M = np.outer(ph, pa)
    M /= M.sum()
    idx = np.add.outer(k, k)
    r = lambda x: round(float(x) * 100, 1)
    return {"xh": round(lh, 1), "xa": round(la, 1), "xt": round(lh + la, 1),
            "mehr": [r(np.tril(M, -1).sum()), r(np.trace(M)), r(np.triu(M, 1).sum())],
            "gesamt": {str(l): r(M[idx > l].sum()) for l in LINIEN_GESAMT},
            "heim": {str(l): r(ph[k > l].sum()) for l in LINIEN_TEAM},
            "gast": {str(l): r(pa[k > l].sum()) for l in LINIEN_TEAM}}


def berechne(spiele, liste, log=print):
    vor, ende, echt = _lauf(spiele, log)
    n = 0
    for e in liste:
        k = (e["liga"], e["datum"], e["heim"], e["gast"])
        if e.get("ergebnis") and k in vor:
            lh, la = vor[k]
            if k in echt:
                e["karten"] = list(echt[k])
        elif e["liga"] in ende:
            A, D, mu, h, TF, PF, muf, hf, r, W = ende[e["liga"]]
            a, b = e["heim"], e["gast"]
            lh = np.exp(mu + h + A.get(a, 0) - D.get(b, 0)); la = np.exp(mu - h + A.get(b, 0) - D.get(a, 0))
            fh = np.exp(muf + hf + TF.get(a, 0) + PF.get(b, 0)); fa = np.exp(muf - hf + TF.get(b, 0) + PF.get(a, 0))
            lh, la = lh ** (1 - W) * (fh * r) ** W, la ** (1 - W) * (fa * r) ** W
        else:
            e["prog"]["karten"] = None
            continue
        e["prog"]["karten"] = prognose(float(lh), float(la))
        n += 1
    log(f"Karten-Prognose: {n} von {len(liste)} Spielen")
