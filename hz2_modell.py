# ===== MODELL HZ2 · TORE IN DER 2. HALBZEIT =====
# Eigener Analyse-Prozess NUR aus Daten der 2. Halbzeit:
#   Tore 2. HZ = Endstand minus Halbzeitstand (nur Spiele mit Halbzeitstand).
# Jedes Team hat Angriff/Abwehr fuer die 2. Halbzeit (log-Skala), nach jedem Spiel nachgefuehrt.
# Lernrate je Liga automatisch gewaehlt (Vorhersage-Guete auf den Spielen davor, also ehrlich).
# Ausgabe: erwartete Tore 2. HZ je Team, P(mind. 2 Tore in der 2. HZ) = OVER 1,5, dazu OVER 0,5 / 2,5.
import numpy as np
import pandas as pd
from scipy.stats import poisson

INFO = {"id": "hz2", "typ": "hz2", "name": "2. Halbzeit · Tore", "kurz": "HZ2", "icon": "⏱",
        "farbe1": "#7ff0ff", "farbe2": "#b388ff",
        "beschreibung": "Eigenes Modell nur aus Toren der 2. Halbzeit"}
GITTER_K = [0.02, 0.035, 0.05]
G = 12


def _key(div, date, h, a):
    return (div, pd.Timestamp(date).strftime("%Y-%m-%d"), h, a)


def _lauf(df, K):
    A, D = {}, {}
    mu, h = np.log(0.75), 0.08
    vor, ll, n = {}, 0.0, 0
    for r in df.itertuples(index=False):
        a, b = r.Home, r.Away
        lh = np.exp(mu + h + A.get(a, 0) - D.get(b, 0)); la = np.exp(mu - h + A.get(b, 0) - D.get(a, 0))
        vor[_key(r.Div, r.Date, a, b)] = (lh, la)
        gh, ga = float(r.HG - r.HTHG), float(r.AG - r.HTAG)
        if n > 200:
            ll += gh * np.log(lh) - lh + ga * np.log(la) - la
        n += 1
        eh, ea = gh - lh, ga - la
        A[a] = A.get(a, 0) + K * eh; D[b] = D.get(b, 0) - K * eh
        A[b] = A.get(b, 0) + K * ea; D[a] = D.get(a, 0) - K * ea
        mu += 0.002 * (eh + ea); h += 0.002 * (eh - ea)
    return vor, (A, D, mu, h), ll


def prognose(lh, la):
    k = np.arange(G + 1)
    M = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la)); M /= M.sum()
    idx = np.add.outer(k, k)
    r = lambda x: round(float(x) * 100, 2)
    return {"xh": round(lh, 2), "xa": round(la, 2), "xt": round(lh + la, 2),
            "o05": r(M[idx > 0].sum()), "o15": r(M[idx > 1].sum()), "o25": r(M[idx > 2].sum())}


def berechne(spiele, liste, log=print):
    vor, ende, wahl = {}, {}, {}
    for div, df in spiele.groupby("Div"):
        df = df.dropna(subset=["HG", "AG", "HTHG", "HTAG"]).sort_values("Date")
        if len(df) < 400:
            continue
        best = max(((K,) + _lauf(df, K) for K in GITTER_K), key=lambda x: x[3])
        K, v, e, _ = best
        vor.update(v); ende[div] = e; wahl[div] = K
    n = t = w = 0
    for e in liste:
        k = (e["liga"], e["datum"], e["heim"], e["gast"])
        if e.get("ergebnis") and k in vor:
            lh, la = vor[k]
        elif e["liga"] in ende:
            A, D, mu, h = ende[e["liga"]]
            lh = np.exp(mu + h + A.get(e["heim"], 0) - D.get(e["gast"], 0))
            la = np.exp(mu - h + A.get(e["gast"], 0) - D.get(e["heim"], 0))
        else:
            e["prog"]["hz2"] = None
            continue
        e["prog"]["hz2"] = prognose(float(lh), float(la))
        n += 1
    # Rueckblick: wie oft trat OVER 1,5 in der 2. HZ ein, wenn das Modell >= 60 % sagte?
    for div, df in spiele.groupby("Div"):
        pass
    tr = [(p, (r.HG - r.HTHG + r.AG - r.HTAG) > 1) for r in spiele.dropna(subset=["HTHG", "HTAG"]).itertuples(index=False)
          for p in [vor.get(_key(r.Div, r.Date, r.Home, r.Away))] if p]
    if tr:
        ps = np.array([prognose(*p)["o15"] for p, _ in tr[-20000:]]); ys = np.array([y for _, y in tr[-20000:]])
        m = ps >= 60
        log(f"HZ2-Prognose: {n} von {len(liste)} Spielen | Rückblick OVER 1,5 2.HZ: Modell Ø {ps.mean():.1f} % vs echt "
            f"{ys.mean() * 100:.1f} % | bei ≥60 %: {ys[m].mean() * 100 if m.any() else 0:.1f} % Treffer ({int(m.sum())} Spiele)")
    else:
        log(f"HZ2-Prognose: {n} von {len(liste)} Spielen")
