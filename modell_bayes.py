# ===== MODELL: Bayes · Hierarchisch (Engine 2) =====
# Hierarchie der Teamstaerke (je Ziel Vollzeit / Halbzeit getrennt):
#   Liga-Grundniveau + Heimvorteil
#     -> Team-Grundstaerke (Angriff, Gegentore)                    Streuung sigma_A, sigma_C
#        -> Saison-Abweichung des Teams (aktuelle Saison vs. sonst)  Streuung sigma_S
#        -> Heim/Auswaerts-Abweichung des Teams                      Streuung sigma_D
# Partial Pooling: Jede Ebene wird zur naechsthoeheren hin "gezogen", umso staerker,
# je weniger Daten es gibt (z. B. Aufsteiger, Saisonbeginn, kleine Ligen).
#
# Die Streuungen (Hyperparameter) werden AUTOMATISCH geschaetzt: Empirisches Bayes ueber die
# Laplace-Naeherung der Marginal-Likelihood (wie wahrscheinlich sind die beobachteten Ergebnisse
# unter diesen Streuungen, ueber alle moeglichen Teamstaerken gemittelt). Neu bestimmt jedes
# Halbjahr, nur mit Daten VOR diesem Zeitpunkt.
# Warum keine volle MCMC-Simulation (NUTS/HMC): fuer 20 Ligen mit woechentlichem Backtest jeden
# Morgen auf einem Gratis-Rechner zu langsam. Die Laplace-Naeherung liefert Modus und
# Unsicherheit (Posterior-Kovarianz) in Sekunden.
# Die Posterior-Unsicherheit fliesst in die Prognose ein (Lognormal-Korrektur der Torerwartung).
# Zeitliche Veraenderung: ueber die Saison-Ebene (kein Abklingen per fester Formel).
# Daten: aktuelle + 3 vorherige Saisons. Skandinavien (ausser Daenemark): Saison = Kalenderjahr.
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

import kern

NAME = "Bayes · Hierarchisch"
KURZ = "Bayes"
ICON = "B"
FARBE1 = "#4fa8ff"
FARBE2 = "#2b5cff"
BESCHREIBUNG = "Liga, Team, Saison und Heim/Auswärts als Ebenen, Unsicherheit wird mitgerechnet"

SAISONS = 4
KALENDER_LIGEN = {"NOR", "SWE", "FIN"}
START_SIGMA = np.log([0.30, 0.25, 0.10, 0.15])     # sigma_A, sigma_C, sigma_D, sigma_S (Startwerte)
_TUNING = {}


def _spalten(ziel):
    return ("HG", "AG") if ziel == "ft" else ("HTHG", "HTAG")


def _saison(datum, div):
    if div in KALENDER_LIGEN:
        return datum.dt.year if hasattr(datum, "dt") else datum.year
    if hasattr(datum, "dt"):
        return datum.dt.year - (datum.dt.month < 7)
    return datum.year - (datum.month < 7)


def _daten(df, heute, ziel):
    sh, sa = _spalten(ziel)
    d = df[df["Date"] < heute].dropna(subset=[sh, sa]).copy()
    if d.empty:
        return d
    d["S"] = _saison(d["Date"], d["Div"].iloc[0]) if d["Div"].nunique() == 1 else \
        np.where(d["Div"].isin(KALENDER_LIGEN), d["Date"].dt.year, d["Date"].dt.year - (d["Date"].dt.month < 7))
    akt = int(d["S"].max())
    return d[d["S"] > akt - SAISONS]


class _Struktur:
    """Parameter-Layout und Designmatrix."""
    def __init__(self, d, ziel):
        sh, sa = _spalten(ziel)
        self.teams = sorted(set(d["Home"]) | set(d["Away"]))
        self.ti = {t: k for k, t in enumerate(self.teams)}
        n = self.n = len(self.teams)
        paare = sorted(set(zip(d["Home"], d["S"])) | set(zip(d["Away"], d["S"])))
        self.si = {p: k for k, p in enumerate(paare)}
        self.oA, self.oC, self.odA, self.odC = 2, 2 + n, 2 + 2 * n, 2 + 3 * n
        self.oa = 2 + 4 * n
        self.oc = self.oa + len(paare)
        self.P = self.oc + len(paare)
        i = d["Home"].map(self.ti).to_numpy(); j = d["Away"].map(self.ti).to_numpy()
        si = np.array([self.si[(h, s)] for h, s in zip(d["Home"], d["S"])])
        sj = np.array([self.si[(a, s)] for a, s in zip(d["Away"], d["S"])])
        N = len(d)
        r = np.arange(N)
        rows, cols, vals = [], [], []
        def add(rr, cc, vv):
            rows.append(rr); cols.append(cc); vals.append(np.full(len(rr), vv, float) if np.isscalar(vv) else vv)
        # Heimtore (Zeilen 0..N-1)
        add(r, np.zeros(N, int), 1); add(r, np.ones(N, int), 1)
        add(r, self.oA + i, 1); add(r, self.odA + i, 1); add(r, self.oa + si, 1)
        add(r, self.oC + j, 1); add(r, self.odC + j, -1); add(r, self.oc + sj, 1)
        # Auswaertstore (Zeilen N..2N-1)
        add(N + r, np.zeros(N, int), 1)
        add(N + r, self.oA + j, 1); add(N + r, self.odA + j, -1); add(N + r, self.oa + sj, 1)
        add(N + r, self.oC + i, 1); add(N + r, self.odC + i, 1); add(N + r, self.oc + si, 1)
        self.X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * N, self.P))
        self.y = np.concatenate([d[sh].to_numpy(float), d[sa].to_numpy(float)])
        self.N = N

    def praezision(self, log_sig):
        sA, sC, sD, sS = np.exp(log_sig)
        p = np.empty(self.P)
        p[:2] = 0.1
        p[self.oA:self.oC] = 1 / sA ** 2; p[self.oC:self.odA] = 1 / sC ** 2
        p[self.odA:self.oa] = 1 / sD ** 2; p[self.oa:] = 1 / sS ** 2
        return p


def _map(st, prec, x0=None):
    X, y = st.X, st.y
    def f(t):
        eta = X @ t
        lam = np.exp(eta)
        return float(np.sum(lam - y * eta) + 0.5 * np.sum(prec * t * t)), X.T @ (lam - y) + prec * t
    if x0 is None:
        x0 = np.zeros(st.P); x0[0] = np.log(max(y.mean(), 0.05))
    return minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 400}).x


def _hesse(st, t, prec):
    lam = np.exp(st.X @ t)
    Hs = (st.X.T @ st.X.multiply(lam[:, None])).toarray()
    Hs[np.diag_indices_from(Hs)] += prec
    return Hs


def _log_marginal(st, log_sig, cache):
    prec = st.praezision(log_sig)
    t = _map(st, prec, cache.get("t"))
    cache["t"] = t
    eta = st.X @ t
    H = _hesse(st, t, prec)
    sign, logdet = np.linalg.slogdet(H)
    return float(np.sum(st.y * eta - np.exp(eta)) - 0.5 * np.sum(prec * t * t) + 0.5 * np.sum(np.log(prec[2:])) - 0.5 * logdet)


def _stichtag(heute):
    return pd.Timestamp(heute.year, 1 if heute.month <= 6 else 7, 1)


def _tune(df, heute, ziel):
    stich = _stichtag(heute)
    key = (tuple(sorted(df["Div"].unique())), ziel, stich)
    if key in _TUNING:
        return _TUNING[key]
    d = _daten(df, stich, ziel)
    best = START_SIGMA.copy()
    if len(d) >= 400:
        st, cache = _Struktur(d, ziel), {}
        res = minimize(lambda s: -_log_marginal(st, np.clip(s, -5, 1), cache), START_SIGMA, method="Nelder-Mead",
                       options={"maxiter": 45, "xatol": 0.05, "fatol": 0.5})
        best = np.clip(res.x, -5, 1)
    _TUNING[key] = best
    return best


def _fit(df, heute, ziel):
    d = _daten(df, heute, ziel)
    if len(d) < 200:
        return None
    log_sig = _tune(df, heute, ziel)
    st = _Struktur(d, ziel)
    prec = st.praezision(log_sig)
    t = _map(st, prec)
    cov = np.linalg.inv(_hesse(st, t, prec))
    eta = st.X @ t
    lam = np.exp(eta)
    abh = kern.waehle_abhaengigkeit(st.y[:st.N], st.y[st.N:], lam[:st.N], lam[st.N:], np.ones(st.N))
    akt = int(d["S"].max())
    return {"st": st, "t": t, "cov": cov, "sig": np.exp(log_sig), "abh": abh, "saison": akt}


def trainiere(df, heute):
    m = {"ft": _fit(df, heute, "ft"), "ht": _fit(df, heute, "ht")}
    if m["ft"] is None:
        raise ValueError("zu wenig Daten")
    return m


def _lambdas(f, heim, gast):
    st, t, cov = f["st"], f["t"], f["cov"]
    sA, sC, sD, sS = f["sig"]
    def zeile(eintraege):
        x = np.zeros(st.P); extra = 0.0
        for art, team, w in eintraege:
            k = st.ti.get(team)
            if art in ("a", "c"):
                s = st.si.get((team, f["saison"]))
                if s is None:
                    extra += sS ** 2           # Saison-Abweichung unbekannt -> nur Unsicherheit
                else:
                    x[(st.oa if art == "a" else st.oc) + s] += w
            elif k is None:
                extra += {"A": sA, "C": sC, "dA": sD, "dC": sD}[art] ** 2   # unbekanntes Team
            else:
                x[{"A": st.oA, "C": st.oC, "dA": st.odA, "dC": st.odC}[art] + k] += w
        return x, extra
    xh, eh = zeile([("A", heim, 1), ("dA", heim, 1), ("a", heim, 1), ("C", gast, 1), ("dC", gast, -1), ("c", gast, 1)])
    xa, ea = zeile([("A", gast, 1), ("dA", gast, -1), ("a", gast, 1), ("C", heim, 1), ("dC", heim, 1), ("c", heim, 1)])
    xh[0] = xh[1] = 1; xa[0] = 1
    lh = np.exp(xh @ t + (xh @ cov @ xh + eh) / 2)
    la = np.exp(xa @ t + (xa @ cov @ xa + ea) / 2)
    return float(lh), float(la)


def vorhersage(m, heim, gast):
    lh, la = _lambdas(m["ft"], heim, gast)
    out = {"ft": kern.matrix(lh, la, m["ft"]["abh"]), "ht": None, "xg": [lh, la]}
    if m["ht"] is not None:
        hh, ha = _lambdas(m["ht"], heim, gast)
        out["ht"] = kern.matrix(hh, ha, m["ht"]["abh"])
    return out


def diagnose(m):
    t = []
    for ziel, name in (("ft", "Vollzeit"), ("ht", "Halbzeit")):
        f = m.get(ziel)
        if f is None:
            t.append(f"{name}: keine Daten")
        else:
            sA, sC, sD, sS = f["sig"]
            t.append(f"{name}: Streuung Angriff {sA:.2f}, Abwehr {sC:.2f}, Heim/Ausw. {sD:.2f}, Saison {sS:.2f}, "
                     f"Abhängigkeit {f['abh'][0]} ({f['abh'][1]:+.3f}), Heimvorteil {np.exp(f['t'][1]):.2f}x")
    return " | ".join(t)
