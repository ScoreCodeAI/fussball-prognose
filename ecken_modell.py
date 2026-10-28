# ===== MODELL E · ECKEN =====
# Jedes Team hat eine Ecken-Staerke (Ecken herausholen) und eine Ecken-Abwehr (Ecken zulassen),
# laufend nachgefuehrt. Lernziel: 70 % echte Ecken + 30 % Schuss-Erwartung (Schuesse x Ecken je Schuss).
# Test 2024-26, 10 Ligen:
#   - "Wer bekommt mehr Ecken": 66 % Treffer (immer "Heim" waere 59,5 %)
#   - Ecken je Team: deutlich genauer als der Liga-Durchschnitt
#   - Gesamt-Ecken: kaum genauer als der Liga-Durchschnitt (wird trotzdem angezeigt)
# Nur Ligen mit Ecken-Daten. Keine Quoten fuer Ecken vorhanden -> reine Prognose.
import numpy as np
import pandas as pd
from scipy.stats import poisson

INFO = {"id": "ecken", "typ": "ecken", "name": "Ecken · Prognose", "kurz": "Ecken", "icon": "🚩",
        "farbe1": "#ffb02b", "farbe2": "#ff4d7d",
        "beschreibung": "Ecken je Team und gesamt, wer mehr Ecken bekommt"}

K, SCHUSS, PRO_SCHUSS = 0.05, 0.3, 0.38
G = 25
LINIEN_GESAMT = [8.5, 9.5, 10.5]
LINIEN_TEAM = [5.5]


def _key(div, date, h, a):
    return (div, pd.Timestamp(date).strftime("%Y-%m-%d"), h, a)


def _lauf(spiele):
    vor, ende, echt = {}, {}, {}
    for div, df in spiele.groupby("Div"):
        df = df.dropna(subset=["HC", "AC"]).sort_values("Date")
        if len(df) < 300:
            continue
        A, D = {}, {}
        mu, h = np.log(5.0), 0.1
        for r in df.itertuples(index=False):
            a, b = r.Home, r.Away
            lh = np.exp(mu + h + A.get(a, 0) - D.get(b, 0)); la = np.exp(mu - h + A.get(b, 0) - D.get(a, 0))
            k = _key(div, r.Date, a, b)
            vor[k] = (lh, la)
            echt[k] = (int(r.HC), int(r.AC))
            sh = r.HS if not pd.isna(r.HS) else r.HC / PRO_SCHUSS
            sa = r.AS if not pd.isna(r.AS) else r.AC / PRO_SCHUSS
            th = (1 - SCHUSS) * r.HC + SCHUSS * sh * PRO_SCHUSS
            ta = (1 - SCHUSS) * r.AC + SCHUSS * sa * PRO_SCHUSS
            eh, ea = (th - lh) / lh, (ta - la) / la
            A[a] = A.get(a, 0) + K * eh; D[b] = D.get(b, 0) - K * eh
            A[b] = A.get(b, 0) + K * ea; D[a] = D.get(a, 0) - K * ea
            mu += 0.003 * (eh + ea); h += 0.003 * (eh - ea)
        ende[div] = (A, D, mu, h)
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
    vor, ende, echt = _lauf(spiele)
    n = 0
    for e in liste:
        k = (e["liga"], e["datum"], e["heim"], e["gast"])
        if e.get("ergebnis") and k in vor:
            lh, la = vor[k]
            if k in echt:
                e["ecken"] = list(echt[k])
        elif e["liga"] in ende:
            A, D, mu, h = ende[e["liga"]]
            lh = np.exp(mu + h + A.get(e["heim"], 0) - D.get(e["gast"], 0))
            la = np.exp(mu - h + A.get(e["gast"], 0) - D.get(e["heim"], 0))
        else:
            e["prog"]["ecken"] = None
            continue
        e["prog"]["ecken"] = prognose(float(lh), float(la))
        n += 1
    log(f"Ecken-Prognose: {n} von {len(liste)} Spielen")
