# ===== MODELL E · ENSEMBLE (nur Tore, eigenstaendig) =====
# Fuenf eigene Bausteine, alle nur aus Toren und Halbzeitstaenden (keine Quoten, keine Schuesse):
#   P  Poisson      - EINE Angriffs-/Abwehrstaerke je Team, gegenseitig gegnerbereinigt (Maher), Heimfaktor,
#                     Zeitgewichtung, Glaettung. (Getrennte Heim/Auswaerts-Staerken waren im Test zu wackelig.)
#   DC Dixon-Coles  - gleiche Staerken, Korrektur fuer 0:0, 1:0, 0:1, 1:1 (Parameter aus den Daten geschaetzt)
#   H  Haeufigkeit  - echte Torverteilungen der Teams (wie oft 0, 1, 2, 3 ... Tore), geglaettet
#   F  Form         - letzte 5 / 10 / 15 Spiele, gegnerbereinigt: Tore im Verhaeltnis zu den ERWARTETEN Toren
#   Z  Zwillinge    - echte Ergebnisse frueherer Spiele mit aehnlichen erwarteten Toren
# Ensemble: Gewichte je Markt (Correct Score, 1X2, OVER/UNDER, BTTS) je Land im Rueckblick gelernt.
# Test 5 Top-Ligen ab 07/2024: 1X2 0,998 -> 0,991, O/U 0,682 -> 0,680, BTTS 0,690 -> 0,688.
# Halbzeit: dieselben Bausteine auf Halbzeit-Toren, nur Spiele mit echtem Halbzeitstand.
# Zusatzmaerkte (kern.extra): Halbzeit/Endstand (9), Correct Score Top 10.
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

import kern

NAME = "Ensemble · Tore"
KURZ = "Ensemble"
ICON = "E"
FARBE1 = "#c084fc"
FARBE2 = "#ff4d7d"
BESCHREIBUNG = "Gegnerbereinigte Stärke, Dixon-Coles, Häufigkeit, Form und Zwillinge – Gewichte je Markt gelernt, nur aus Toren"
EXTRA = True            # Engine berechnet Zusatzmaerkte (HZ/ES, Correct Score, weitere O/U)

HALBWERT = 365          # Tage: aeltere Spiele zaehlen weniger
JAHRE = 4
PRIOR = 5.0             # Glaettung zum Ligamittel (in "Spielen")
G = 10                  # Matrix 0..10 Tore, Rest in letzter Zeile/Spalte
FORM_N = (5, 10, 15)
FORM_W = (0.5, 0.3, 0.2)
VALID_TAGE = 365
SCHRITT = 0.1
KOMP = ("P", "DC", "H", "F", "Z")
K_ZW = 150
_k = np.arange(G + 1)


def _pois(lam):
    p = poisson.pmf(_k, max(lam, 0.03))
    p[-1] += max(0.0, 1 - p.sum())
    return p


def _dc(M, lh, la, rho):
    M = M.copy()
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho; M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    return M / M.sum()


def _verteilung(werte, gew, prior_p, k=PRIOR):
    """Geglaettete Torverteilung 0..G aus gewichteten Beobachtungen."""
    v = np.zeros(G + 1)
    for x, w in zip(werte, gew):
        v[min(int(x), G)] += w
    v = v + k * prior_p
    return v / v.sum()



class _Teil:
    """Bausteine fuer einen Spielabschnitt (Endstand oder Halbzeit)."""

    def __init__(s, d, hs, as_, heute):
        d = d.dropna(subset=[hs, as_]).sort_values("Date")
        s.ok = len(d) >= 100
        if not s.ok:
            return
        w = (0.5 ** ((heute - d["Date"]).dt.days.clip(lower=0) / HALBWERT)).to_numpy()
        hg, ag = d[hs].to_numpy(float), d[as_].to_numpy(float)
        H, A = d["Home"].to_numpy(), d["Away"].to_numpy()
        s.mh = max(float(np.average(hg, weights=w)), 0.15)
        s.ma = max(float(np.average(ag, weights=w)), 0.15)
        s.ph = _verteilung(hg, w, np.zeros(G + 1), 0); s.pa = _verteilung(ag, w, np.zeros(G + 1), 0)
        teams = sorted(set(H) | set(A)); s.ix = {t: i for i, t in enumerate(teams)}
        hi = np.array([s.ix[t] for t in H]); ai = np.array([s.ix[t] for t in A]); n = len(teams)
        At, Dt = np.ones(n), np.ones(n)
        for _ in range(30):     # Angriff/Abwehr gegenseitig gegnerbereinigt
            At = (np.bincount(hi, w * hg, n) + np.bincount(ai, w * ag, n) + PRIOR) / \
                 (np.bincount(hi, w * s.mh * Dt[ai], n) + np.bincount(ai, w * s.ma * Dt[hi], n) + PRIOR)
            Dt = (np.bincount(ai, w * hg, n) + np.bincount(hi, w * ag, n) + PRIOR) / \
                 (np.bincount(ai, w * s.mh * At[hi], n) + np.bincount(hi, w * s.ma * At[ai], n) + PRIOR)
        s.At, s.Dt = At, Dt
        lh, la = s.mh * At[hi] * Dt[ai], s.ma * At[ai] * Dt[hi]
        t = pd.DataFrame({"H": H, "A": A, "hg": hg, "ag": ag, "w": w, "lh": lh, "la": la})
        s.vh, s.va, s.vhc, s.vac = {}, {}, {}, {}
        for team, g in t.groupby("H"):
            s.vh[team] = _verteilung(g.hg, g.w, s.ph); s.vhc[team] = _verteilung(g.ag, g.w, s.pa)
        for team, g in t.groupby("A"):
            s.va[team] = _verteilung(g.ag, g.w, s.pa); s.vac[team] = _verteilung(g.hg, g.w, s.ph)
        # Form gegnerbereinigt: echte Tore / erwartete Tore (Heimteam aus Heimspielen, Gast aus Auswaertsspielen)
        s.fh = {team: (g.hg.to_numpy() / g.lh.to_numpy(), g.ag.to_numpy() / g.la.to_numpy()) for team, g in t.groupby("H")}
        s.fa = {team: (g.ag.to_numpy() / g.la.to_numpy(), g.hg.to_numpy() / g.lh.to_numpy()) for team, g in t.groupby("A")}
        # Zwillinge
        s.z_l = np.c_[np.log(lh), np.log(la)]
        s.z_hg, s.z_ag, s.z_w = np.minimum(hg, G).astype(int), np.minimum(ag, G).astype(int), w
        x, y = hg.astype(int), ag.astype(int)

        def nll(r):
            tau = np.ones_like(lh)
            tau = np.where((x == 0) & (y == 0), 1 - lh * la * r, tau)
            tau = np.where((x == 0) & (y == 1), 1 + lh * r, tau)
            tau = np.where((x == 1) & (y == 0), 1 + la * r, tau)
            tau = np.where((x == 1) & (y == 1), 1 - r, tau)
            return -np.sum(w * np.log(np.clip(tau, 1e-9, None)))
        s.rho = float(minimize_scalar(nll, bounds=(-0.2, 0.15), method="bounded").x)

    def _lam(s, a, b):
        i, j = s.ix.get(a), s.ix.get(b)
        A = lambda k: s.At[k] if k is not None else 1.0
        D = lambda k: s.Dt[k] if k is not None else 1.0
        return float(np.clip(s.mh * A(i) * D(j), 0.1, 5)), float(np.clip(s.ma * A(j) * D(i), 0.1, 5))

    def _form(s, a, b):
        lh, la = s._lam(a, b)

        def f(r):
            if r is None or len(r[0]) == 0:
                return 1.0, 1.0
            at = sum(w * np.mean(r[0][-n:]) for n, w in zip(FORM_N, FORM_W))
            df = sum(w * np.mean(r[1][-n:]) for n, w in zip(FORM_N, FORM_W))
            return 0.5 + 0.5 * at, 0.5 + 0.5 * df       # halb zum Erwarteten gezogen
        ah, dh = f(s.fh.get(a)); aa, da = f(s.fa.get(b))
        return float(np.clip(lh * ah * da, 0.1, 5)), float(np.clip(la * aa * dh, 0.1, 5))

    def matrizen(s, a, b):
        lh, la = s._lam(a, b)
        P = np.outer(_pois(lh), _pois(la))
        hv = 0.5 * (s.vh.get(a, s.ph) + s.vac.get(b, s.ph))
        av = 0.5 * (s.va.get(b, s.pa) + s.vhc.get(a, s.pa))
        fh, fa = s._form(a, b)
        dd = ((s.z_l - np.log([lh, la])) ** 2).sum(1)
        k = np.argsort(dd)[:K_ZW]
        Z = np.zeros((G + 1, G + 1)); np.add.at(Z, (s.z_hg[k], s.z_ag[k]), s.z_w[k])
        Z = 0.7 * Z / Z.sum() + 0.3 * P
        return {"P": P, "DC": _dc(P, lh, la, s.rho), "H": np.outer(hv, av), "F": np.outer(_pois(fh), _pois(fa)), "Z": Z}


def _gitter():
    st = int(round(1 / SCHRITT))
    for c in itertools.product(range(st + 1), repeat=len(KOMP)):
        if sum(c) == st:
            yield np.array(c) / st


_idx = np.add.outer(np.arange(G + 1), np.arange(G + 1))


def _gewichte(d, heute):
    gleich = np.ones(len(KOMP)) / len(KOMP)
    cut = heute - pd.Timedelta(days=VALID_TAGE)
    alt, neu = d[d["Date"] < cut], d[d["Date"] >= cut]
    if len(alt) < 600 or len(neu) < 80:
        return {k: gleich for k in ("cs", "x", "ou", "bts")}, "zu wenig Daten – gleiche Gewichte"
    t = _Teil(alt, "HG", "AG", cut)
    L, X1, XO, XB, yo, yb = [], [], [], [], [], []
    for r in neu.itertuples(index=False):
        m = t.matrizen(r.Home, r.Away)
        i, j = min(int(r.HG), G), min(int(r.AG), G)
        o = 0 if i > j else 1 if i == j else 2
        L.append([m[k][i, j] for k in KOMP])
        X1.append([[np.tril(m[k], -1).sum(), np.trace(m[k]), np.triu(m[k], 1).sum()][o] for k in KOMP])
        XO.append([m[k][_idx > 2].sum() for k in KOMP])
        XB.append([1 - m[k][0].sum() - m[k][:, 0].sum() + m[k][0, 0] for k in KOMP])
        yo.append(i + j > 2); yb.append(i > 0 and j > 0)
    L, X1, XO, XB, yo, yb = map(np.array, (L, X1, XO, XB, yo, yb))
    gi = list(_gitter())
    nl = lambda v: -np.log(np.clip(v, 1e-12, None)).mean()
    W = {"cs": min(gi, key=lambda w: nl(L @ w)),
         "x": min(gi, key=lambda w: nl(X1 @ w)),
         "ou": min(gi, key=lambda w: nl(np.where(yo, XO @ w, 1 - XO @ w))),
         "bts": min(gi, key=lambda w: nl(np.where(yb, XB @ w, 1 - XB @ w)))}
    return W, f"gelernt auf {len(neu)} Spielen"


def trainiere(df, heute):
    heute = pd.Timestamp(heute)
    d = df[(df["Date"] < heute) & (df["Date"] >= heute - pd.DateOffset(years=JAHRE))].sort_values("Date")
    if len(d) < 300:
        raise ValueError("zu wenig Daten")
    W, info = _gewichte(d, heute)
    ft = _Teil(d, "HG", "AG", heute)
    ht = _Teil(d, "HTHG", "HTAG", heute)
    return {"W": W, "info": info, "ft": ft, "ht": ht if ht.ok else None, "n": len(d)}


def _mix(m, w):
    M = sum(wi * m[k] for wi, k in zip(w, KOMP))
    return M / M.sum()


def vorhersage(m, heim, gast):
    mm = m["ft"].matrizen(heim, gast)
    ft = _mix(mm, m["W"]["cs"])
    if m["ht"] is not None:
        ht = _mix(m["ht"].matrizen(heim, gast), m["W"]["cs"])
    else:
        xh, xa = kern.erwartete_tore(ft)
        ht = np.outer(_pois(xh * kern.HZ_ANTEIL), _pois(xa * kern.HZ_ANTEIL))
    # Maerkte mit eigenen Gewichten (1X2, OVER/UNDER, BTTS)
    M1, MO, MB = _mix(mm, m["W"]["x"]), _mix(mm, m["W"]["ou"]), _mix(mm, m["W"]["bts"])
    ue = float(MO[_idx > 2].sum())
    markt = {"1": float(np.tril(M1, -1).sum()), "X": float(np.trace(M1)), "2": float(np.triu(M1, 1).sum()),
             "Ü2.5": ue, "U2.5": 1 - ue, "BTS": float(1 - MB[0].sum() - MB[:, 0].sum() + MB[0, 0])}
    return {"ft": ft, "ht": ht, "xg": list(kern.erwartete_tore(ft)), "markt": markt}


def diagnose(m):
    w = lambda v: "/".join(f"{x:.0%}" for x in v)
    return (f"Gewichte P/DC/H/F/Z – CS {w(m['W']['cs'])} · 1X2 {w(m['W']['x'])} · O/U {w(m['W']['ou'])} · BTTS {w(m['W']['bts'])} "
            f"({m['info']}) | Halbzeit {'echt' if m['ht'] is not None else 'Ersatz 45 %'} | {m['n']} Spiele")
