# ===== KERN: gemeinsame Bausteine fuer alle Modelle =====
# - Ergebnis-Matrizen (Correct Score) mit automatischer Obergrenze
# - Abhaengigkeit Heim/Gast: unabhaengig, Dixon-Coles oder Frank-Copula
# - alle Maerkte werden aus genau einer Matrix abgeleitet (konsistent)
# - Qualitaetsmasse (Log-Loss, RPS, Brier, NLL)
# - Poisson-Grundmodell mit getrennter Heim-/Auswaertsstaerke (Partial Pooling)
# Nur numpy/scipy - keine weiteren Pakete noetig.
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

REST_TOLERANZ = 1e-6     # erlaubte Restmasse ausserhalb der Matrix (wird danach normiert)
MIN_TORE, MAX_TORE = 6, 15


# ---------- Ergebnis-Matrizen ----------
def obergrenze(lh, la, tol=REST_TOLERANZ):
    """Kleinste Torzahl, ab der die Restwahrscheinlichkeit unter tol liegt (keine feste Grenze)."""
    g = int(poisson.ppf(1 - tol, max(lh, la, 0.05))) + 1
    return int(min(max(g, MIN_TORE), MAX_TORE))


def _normiere(M):
    M = np.clip(M, 0, None)
    s = M.sum()
    return M / s if s > 0 else M


def matrix_unabhaengig(lh, la, g=None):
    g = g or obergrenze(lh, la)
    k = np.arange(g + 1)
    return _normiere(np.outer(poisson.pmf(k, lh), poisson.pmf(k, la)))


def _dc_tau(x, y, lh, la, rho):
    t = np.ones(np.broadcast(x, y, lh, la).shape)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.clip(t, 1e-9, None)


def matrix_dixon_coles(lh, la, rho, g=None):
    g = g or obergrenze(lh, la)
    k = np.arange(g + 1)
    M = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    x, y = np.meshgrid(k, k, indexing="ij")
    return _normiere(M * _dc_tau(x, y, lh, la, rho))


def _frank_C(u, v, th):
    if abs(th) < 1e-6:
        return u * v
    return -np.log1p(np.expm1(-th * u) * np.expm1(-th * v) / np.expm1(-th)) / th


def matrix_frank(lh, la, th, g=None):
    g = g or obergrenze(lh, la)
    k = np.arange(g + 1)
    F = np.concatenate([[0.0], poisson.cdf(k, lh)])
    G = np.concatenate([[0.0], poisson.cdf(k, la)])
    C = _frank_C(F[:, None], G[None, :], th)
    return _normiere(C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1])


def matrix(lh, la, abh=("unabhaengig", 0.0)):
    typ, par = abh
    if typ == "dixon_coles":
        return matrix_dixon_coles(lh, la, par)
    if typ == "frank":
        return matrix_frank(lh, la, par)
    return matrix_unabhaengig(lh, la)


# ---------- Abhaengigkeit automatisch waehlen ----------
def _ll_unabh(x, y, lh, la, w):
    return float(np.sum(w * (poisson.logpmf(x, lh) + poisson.logpmf(y, la))))


def _ll_frank(x, y, lh, la, w, th):
    Fx, Fx1 = poisson.cdf(x, lh), poisson.cdf(x - 1, lh)
    Gy, Gy1 = poisson.cdf(y, la), poisson.cdf(y - 1, la)
    p = _frank_C(Fx, Gy, th) - _frank_C(Fx1, Gy, th) - _frank_C(Fx, Gy1, th) + _frank_C(Fx1, Gy1, th)
    return float(np.sum(w * np.log(np.clip(p, 1e-12, None))))


def waehle_abhaengigkeit(x, y, lh, la, w):
    """Vergleicht unabhaengig, Dixon-Coles (rho) und Frank-Copula (theta) per Likelihood
    mit Strafterm fuer den Zusatzparameter und gibt (typ, parameter) zurueck."""
    x, y, lh, la, w = map(np.asarray, (x, y, lh, la, w))
    basis = _ll_unabh(x, y, lh, la, w)
    kand = [("unabhaengig", 0.0, basis)]
    try:
        f = lambda r: -(basis + float(np.sum(w * np.log(_dc_tau(x, y, lh, la, r)))))
        res = minimize_scalar(f, bounds=(-0.25, 0.25), method="bounded")
        kand.append(("dixon_coles", float(res.x), -res.fun - 1))
    except Exception:
        pass
    try:
        f = lambda t: -_ll_frank(x, y, lh, la, w, t)
        res = minimize_scalar(f, bounds=(-8, 8), method="bounded")
        kand.append(("frank", float(res.x), -res.fun - 1))
    except Exception:
        pass
    typ, par, _ = max(kand, key=lambda k: k[2])
    return typ, par


# ---------- Maerkte aus der Matrix ----------
HTFT_LABELS = ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"]
HZ_ANTEIL = 0.45          # Anteil der Tore in Halbzeit 1, falls ein Modell keine Halbzeit liefert
CSG_MAX = 5               # Correct-Score-Raster 0..5 fuer den Markt-Vergleich


def _diff_vert(M):
    """Verteilung der Tordifferenz (Heim - Gast) einer Ergebnis-Matrix -> {diff: p}"""
    out = {}
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            out[a - b] = out.get(a - b, 0.0) + float(M[a, b])
    return out


def htft(ft, ht=None):
    """Halbzeit/Endstand (9 Varianten). Halbzeit 1 aus der HT-Matrix, Halbzeit 2 als Poisson
    mit den restlichen erwarteten Toren (Endstand minus Halbzeit)."""
    xh, xa = erwartete_tore(ft)
    if ht is None:
        ht = matrix_unabhaengig(max(xh * HZ_ANTEIL, 0.02), max(xa * HZ_ANTEIL, 0.02))
    hh, ha = erwartete_tore(ht)
    zweite = matrix_unabhaengig(max(xh - hh, 0.05), max(xa - ha, 0.05))
    d1, d2 = _diff_vert(ht), _diff_vert(zweite)
    erg = {k: 0.0 for k in HTFT_LABELS}
    for a, pa in d1.items():
        h = "1" if a > 0 else "2" if a < 0 else "X"
        for b, pb in d2.items():
            e = a + b
            erg[f"{h}/{'1' if e > 0 else '2' if e < 0 else 'X'}"] += pa * pb
    s = sum(erg.values()) or 1.0
    return [erg[k] / s for k in HTFT_LABELS]


def maerkte(ft, ht=None):
    M = ft
    idx = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    unter = M[idx <= 2].sum()
    i, j = np.unravel_index(M.argmax(), M.shape)
    zellen = sorted(((M[a, b], f"{a}:{b}") for a in range(M.shape[0]) for b in range(M.shape[1])), reverse=True)[:5]
    g = min(CSG_MAX, M.shape[0] - 1, M.shape[1] - 1)
    return {"1": float(np.tril(M, -1).sum()), "X": float(np.trace(M)), "2": float(np.triu(M, 1).sum()),
            "Ü2.5": float(1 - unter), "U2.5": float(unter),
            "BTS": float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0]), "tipp": f"{i}:{j}",
            "HT": None if ht is None else [float(np.tril(ht, -1).sum()), float(np.trace(ht)), float(np.triu(ht, 1).sum())],
            "HTFT": htft(ft, ht),
            "CS": [[k, float(v)] for v, k in zellen],
            "CSG": [[float(M[a, b]) for b in range(g + 1)] for a in range(g + 1)]}


def erwartete_tore(M):
    k0, k1 = np.arange(M.shape[0]), np.arange(M.shape[1])
    return float(M.sum(axis=1) @ k0), float(M.sum(axis=0) @ k1)


# ---------- Qualitaetsmasse ----------
def qualitaet(p, ft, ht, hg, ag, hth=None, hta=None):
    eps = 1e-12
    o = 0 if hg > ag else 1 if hg == ag else 2
    P = np.array([p["1"], p["X"], p["2"]])
    O = np.eye(3)[o]
    q = {"logloss_1x2": -np.log(max(P[o], eps)),
         "rps": float(((P[0] - O[0]) ** 2 + (P[0] + P[1] - O[0] - O[1]) ** 2) / 2),
         "brier": float(np.sum((P - O) ** 2)),
         "logloss_ou": -np.log(max(p["Ü2.5"] if hg + ag > 2 else p["U2.5"], eps)),
         "logloss_bts": -np.log(max(p["BTS"] if (hg > 0 and ag > 0) else 1 - p["BTS"], eps)),
         "nll_cs": -np.log(max(ft[hg, ag] if hg < ft.shape[0] and ag < ft.shape[1] else 0.0, 1e-9))}
    if ht is not None and hth is not None and hta is not None:
        oh = 0 if hth > hta else 1 if hth == hta else 2
        q["logloss_ht"] = -np.log(max(p["HT"][oh], eps))
    return {k: float(v) for k, v in q.items()}


# ---------- Poisson-Grundmodell (Heim/Auswaerts getrennt, Partial Pooling) ----------
# log lambda_heim = mu + heimvorteil + A[i] + AH[i] + C[j] + CA[j]
# log lambda_gast = mu + A[j] + AA[j] + C[i] + CH[i]
# A = Angriff, C = Gegentor-Neigung; AH/AA/CH/CA = Heim-/Auswaerts-Abweichungen.
# Die Abweichungen werden zur Team-Grundstaerke hin "gezogen" (Partial Pooling, Staerke lam_split),
# damit Heim und Auswaerts getrennt geschaetzt werden, ohne bei wenig Daten zu wild zu werden.
def poisson_fit(i, j, yh, ya, w, n, lam_basis, lam_split, x0=None):
    i, j = np.asarray(i), np.asarray(j)
    yh, ya, w = np.asarray(yh, float), np.asarray(ya, float), np.asarray(w, float)

    def teile(p):
        return p[0], p[1], p[2:2 + n], p[2 + n:2 + 2 * n], p[2 + 2 * n:2 + 3 * n], \
            p[2 + 3 * n:2 + 4 * n], p[2 + 4 * n:2 + 5 * n], p[2 + 5 * n:2 + 6 * n]

    def f(p):
        mu, h, A, C, AH, AA, CH, CA = teile(p)
        eh = mu + h + A[i] + AH[i] + C[j] + CA[j]
        ea = mu + A[j] + AA[j] + C[i] + CH[i]
        lh, la = np.exp(eh), np.exp(ea)
        nll = np.sum(w * (lh - yh * eh)) + np.sum(w * (la - ya * ea))
        pen = lam_basis * (A @ A + C @ C) + lam_split * (AH @ AH + AA @ AA + CH @ CH + CA @ CA)
        gh, ga = w * (lh - yh), w * (la - ya)
        bc = lambda idx, v: np.bincount(idx, weights=v, minlength=n)
        g = np.concatenate([[gh.sum() + ga.sum(), gh.sum()],
                            bc(i, gh) + bc(j, ga) + 2 * lam_basis * A,
                            bc(j, gh) + bc(i, ga) + 2 * lam_basis * C,
                            bc(i, gh) + 2 * lam_split * AH,
                            bc(j, ga) + 2 * lam_split * AA,
                            bc(i, ga) + 2 * lam_split * CH,
                            bc(j, gh) + 2 * lam_split * CA])
        return nll + pen, g

    if x0 is None:
        m = max(np.average(np.concatenate([yh, ya]), weights=np.concatenate([w, w])), 0.05)
        x0 = np.zeros(2 + 6 * n)
        x0[0], x0[1] = np.log(m), 0.2
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


def poisson_lambdas(p, n, i, j):
    """Erwartete Tore fuer Index-Paare; Index -1 = unbekanntes Team (Liga-Durchschnitt)."""
    mu, h = p[0], p[1]
    blk = [np.append(p[2 + k * n:2 + (k + 1) * n], 0.0) for k in range(6)]  # letzte Stelle = unbekannt
    A, C, AH, AA, CH, CA = blk
    i, j = np.asarray(i), np.asarray(j)
    return np.exp(mu + h + A[i] + AH[i] + C[j] + CA[j]), np.exp(mu + A[j] + AA[j] + C[i] + CH[i])
