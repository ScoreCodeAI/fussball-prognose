# ===== MODELL V · VERLAUF (Halbzeit-Elo + Spielverlauf) =====
# Idee 1 - Halbzeit-Elo: jedes Team hat getrennte Staerken fuer die 1. und die 2. Halbzeit
#   (Angriff/Abwehr je Halbzeit). Nach jedem Spiel werden sie wie bei Elo laufend nachgefuehrt:
#   mehr Tore als erwartet -> Angriff hoch, Gegner-Abwehr runter. Juengste Spiele zaehlen mehr.
# Idee 2 - Spielverlauf: wie reagiert ein Team in der 2. Halbzeit auf den Halbzeitstand?
#   (Fuehrung, Remis, Rueckstand) - global fuer die Liga und je Team (zur Liga hin gezogen).
# Prognose: erst die 1. Halbzeit, dann die 2. Halbzeit ABHAENGIG vom Halbzeitstand
#   -> Endstand-Matrix und Halbzeit-Matrix, daraus alle Maerkte (konsistent).
import numpy as np
import pandas as pd
from scipy.stats import poisson

NAME = "Verlauf · Halbzeit-Elo"
KURZ = "Verlauf"
ICON = "V"
FARBE1 = "#22e4ff"
FARBE2 = "#2bf5a0"
BESCHREIBUNG = "Getrennte Teamstärken für 1. und 2. Halbzeit plus Reaktion auf den Halbzeitstand"

JAHRE = 4             # Trainingsfenster
K = 0.03              # Lernrate Halbzeit-Elo (im Test 2023-25 am besten fuer 1X2)
K_GLOBAL = 0.004      # Lernrate Liga-Spielverlauf
K_TEAM = 0.02         # Lernrate Team-Spielverlauf
ZUG = 0.985           # Team-Spielverlauf wird zur Liga hin gezogen (Shrinkage)
G_MAX = 9             # Tore je Halbzeit in der Matrix
ZUSTAENDE = (-1, 0, 1)  # Rueckstand, Remis, Fuehrung (aus Sicht des Teams)


def _mittel(df, spalte_h, spalte_a):
    h, a = df[spalte_h].astype(float).mean(), df[spalte_a].astype(float).mean()
    h, a = max(h, 0.15), max(a, 0.12)
    return float(np.log(np.sqrt(h * a))), float(np.log(h / a) / 2)


def trainiere(df, heute):
    d = df[df["Date"] >= pd.Timestamp(heute) - pd.DateOffset(years=JAHRE)].sort_values("Date")
    d = d.dropna(subset=["HG", "AG"])
    mit_ht = d.dropna(subset=["HTHG", "HTAG"])
    if len(mit_ht) > 200:
        mu1, h1 = _mittel(mit_ht, "HTHG", "HTAG")
        z = mit_ht.assign(H2=mit_ht["HG"] - mit_ht["HTHG"], A2=mit_ht["AG"] - mit_ht["HTAG"])
        mu2, h2 = _mittel(z, "H2", "A2")
    else:   # keine Halbzeitdaten (z. B. Skandinavien): Endstand aufteilen
        mu, h = _mittel(d, "HG", "AG")
        mu1, h1, mu2, h2 = mu + np.log(0.45), h, mu + np.log(0.55), h

    A1, D1, A2, D2 = {}, {}, {}, {}
    G = {s: 0.0 for s in ZUSTAENDE}
    R = {}

    for r in d.itertuples(index=False):
        hm, aw = r.Home, r.Away
        lh1 = np.exp(mu1 + h1 + A1.get(hm, 0) - D1.get(aw, 0))
        la1 = np.exp(mu1 - h1 + A1.get(aw, 0) - D1.get(hm, 0))
        bh2 = np.exp(mu2 + h2 + A2.get(hm, 0) - D2.get(aw, 0))
        ba2 = np.exp(mu2 - h2 + A2.get(aw, 0) - D2.get(hm, 0))
        hg, ag = float(r.HG), float(r.AG)
        ht_ok = not (pd.isna(r.HTHG) or pd.isna(r.HTAG))
        if ht_ok:
            gh1, ga1 = float(r.HTHG), float(r.HTAG)
            s = int(np.sign(gh1 - ga1))
        else:
            gh1 = hg * lh1 / (lh1 + bh2)
            ga1 = ag * la1 / (la1 + ba2)
            s = 0
        gh2, ga2 = hg - gh1, ag - ga1
        rh, ra = R.setdefault(hm, {x: 0.0 for x in ZUSTAENDE}), R.setdefault(aw, {x: 0.0 for x in ZUSTAENDE})
        lh2 = bh2 * np.exp(G[s] + rh[s])
        la2 = ba2 * np.exp(G[-s] + ra[-s])

        # Halbzeit-Elo (Gradient der Poisson-Likelihood auf der log-Rate)
        e = gh1 - lh1; A1[hm] = A1.get(hm, 0) + K * e; D1[aw] = D1.get(aw, 0) - K * e
        e = ga1 - la1; A1[aw] = A1.get(aw, 0) + K * e; D1[hm] = D1.get(hm, 0) - K * e
        eh, ea = gh2 - lh2, ga2 - la2
        A2[hm] = A2.get(hm, 0) + K * eh; D2[aw] = D2.get(aw, 0) - K * eh
        A2[aw] = A2.get(aw, 0) + K * ea; D2[hm] = D2.get(hm, 0) - K * ea
        # Spielverlauf: nur mit echtem Halbzeitstand
        if ht_ok:
            G[s] += K_GLOBAL * eh
            G[-s] += K_GLOBAL * ea
            for team, zust, err in ((rh, s, eh), (ra, -s, ea)):
                for x in ZUSTAENDE:
                    team[x] *= ZUG
                team[zust] += K_TEAM * err

    return {"mu1": mu1, "h1": h1, "mu2": mu2, "h2": h2, "A1": A1, "D1": D1, "A2": A2, "D2": D2,
            "G": G, "R": R, "n": len(d), "ht": len(mit_ht)}


def vorhersage(m, heim, gast):
    lh1 = np.exp(m["mu1"] + m["h1"] + m["A1"].get(heim, 0) - m["D1"].get(gast, 0))
    la1 = np.exp(m["mu1"] - m["h1"] + m["A1"].get(gast, 0) - m["D1"].get(heim, 0))
    bh2 = np.exp(m["mu2"] + m["h2"] + m["A2"].get(heim, 0) - m["D2"].get(gast, 0))
    ba2 = np.exp(m["mu2"] - m["h2"] + m["A2"].get(gast, 0) - m["D2"].get(heim, 0))
    rh = m["R"].get(heim, {x: 0.0 for x in ZUSTAENDE})
    ra = m["R"].get(gast, {x: 0.0 for x in ZUSTAENDE})

    k = np.arange(G_MAX + 1)
    ht = np.outer(poisson.pmf(k, lh1), poisson.pmf(k, la1))
    ht /= ht.sum()
    zweite = {}
    for s in ZUSTAENDE:
        lh2 = bh2 * np.exp(m["G"][s] + rh[s])
        la2 = ba2 * np.exp(m["G"][-s] + ra[-s])
        zweite[s] = np.outer(poisson.pmf(k, lh2), poisson.pmf(k, la2))
    n = 2 * G_MAX + 1
    ft = np.zeros((n, n))
    for i in range(G_MAX + 1):
        for j in range(G_MAX + 1):
            ft[i:i + G_MAX + 1, j:j + G_MAX + 1] += ht[i, j] * zweite[int(np.sign(i - j))]
    ft = ft[:15, :15]
    ft /= ft.sum()
    xh = float(ft.sum(axis=1) @ np.arange(15))
    xa = float(ft.sum(axis=0) @ np.arange(15))
    return {"ft": ft, "ht": ht, "xg": [xh, xa]}


def diagnose(m):
    g = m["G"]
    pc = lambda x: f"{(np.exp(x) - 1) * 100:+.0f}%"
    return f"Spiele {m['n']} (mit Halbzeit {m['ht']}) | 2. HZ bei Führung {pc(g[1])}, Remis {pc(g[0])}, Rückstand {pc(g[-1])}"
