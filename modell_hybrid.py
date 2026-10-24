# ===== MODELL: Hybrid · Bayes-Dynamik (Engine 6, adaptiver Bayes-State-Space-Hybrid) =====
# Eigenstaendiges Modell, das die Ideen von "Dynamik" (sich veraendernde Teamstaerken) und
# "Bayes" (Hierarchie, Pooling) verbindet:
#  - Teamzustand wie bei Dynamik: Angriff, Gegentore, Heim/Auswaerts-Abweichungen (Random Walk)
#  - NEU: auch das Liga-Niveau (Torniveau + Heimvorteil) ist ein sich veraendernder Zustand
#    je Liga -> erkennt z. B. torreichere Saisons oder einen schwaecheren Heimvorteil
#  - NEU (Hierarchie): neue Teams starten nicht bei 0, sondern beim aktuellen Durchschnitt ihrer
#    Liga plus einem "Neuling-Abstand", der aus den Daten geschaetzt wird
#  - Alle Hyperparameter (Veraenderung der Teams, Veraenderung des Liga-Niveaus, Neuling-Abstand
#    Angriff/Abwehr, Startunsicherheit) werden gemeinsam und stufenlos optimiert
#    (One-Step-Ahead-Likelihood, Nelder-Mead), einmal pro Jahr, nur mit Daten VOR dem Stichtag
#  - Zustands-Unsicherheit (Team UND Liga) fliesst in die Prognose ein
#  - Vollzeit und Halbzeit: getrennte Filter mit eigenen Parametern
# Ehrliche Grenze: Unsicherheit der Hyperparameter selbst wird nicht gemittelt (nur Zustaende).
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import kern

NAME = "Hybrid · Bayes-Dynamik"
KURZ = "Hybrid"
ICON = "H"
FARBE1 = "#e8ecff"
FARBE2 = "#8d99c9"
BESCHREIBUNG = "Dynamische Teamstärken plus Liga-Niveau und Neulings-Startwerte aus der Hierarchie"

FILTER_JAHRE = 3
EINLAUF_TAGE = 365
RATIO = 0.1                                    # Heim/Auswaerts-Abweichung veraendert sich langsamer
START = np.array([np.log(2e-5), np.log(2e-6), -0.10, 0.10, np.log(1.0)])   # log q, log qL, dA, dC, log s0
P0_BASIS = np.array([0.09, 0.09, 0.01, 0.01])
PL0 = np.diag([0.01, 0.01])
HJ = np.array([[1., 0, 1, 0, 0, 1, 0, -1, 1, 1],
               [0., 1, 0, 1, 1, 0, -1, 0, 1, 0]])
_TUNING, _CACHE = {}, {}


def _spalten(ziel):
    return ("HG", "AG") if ziel == "ft" else ("HTHG", "HTAG")


def _stichtag(heute):
    return pd.Timestamp(heute.year if heute.month >= 7 else heute.year - 1, 7, 1)


def _zeilen(df, von, bis, ziel):
    sh, sa = _spalten(ziel)
    d = df[(df["Date"] >= von) & (df["Date"] < bis)].dropna(subset=[sh, sa])
    return d.sort_values(["Date", "Home"], kind="mergesort")


def _tag(ts):
    return int(np.datetime64(pd.Timestamp(ts).date(), "D").astype(np.int64))


def _basis(df, von, ziel):
    sh, sa = _spalten(ziel)
    d = _zeilen(df, von - pd.DateOffset(years=2), von, ziel)
    if len(d) < 200:
        d = _zeilen(df, von, von + pd.DateOffset(years=1), ziel).head(300)
    mh, ma = max(d[sh].mean(), 0.05), max(d[sa].mean(), 0.05)
    return np.log(ma), np.log(mh / ma)


class _Filter:
    def __init__(self, theta, mu, h):
        lq, lql, dA, dC, ls0 = theta
        q, ql = np.exp(lq), np.exp(lql)
        self.Q = np.array([q, q, q * RATIO, q * RATIO])
        self.QL = np.array([ql, ql])
        self.neu = np.array([dA, dC, 0.0, 0.0])
        self.P0 = np.diag(P0_BASIS * np.exp(ls0))
        self.mu0, self.h0 = mu, h
        self.idx, self.M, self.P, self.last, self.liga_von = {}, [], [], [], []
        self.lig, self.LM, self.LP, self.Llast = {}, [], [], []
        self.n, self.bis, self.ll, self.paare = 0, None, 0.0, []

    def _liga(self, div, tag):
        k = self.lig.get(div)
        if k is None:
            k = self.lig[div] = len(self.LM)
            self.LM.append(np.array([self.mu0, self.h0])); self.LP.append(PL0.copy()); self.Llast.append(tag)
        return k

    def _team(self, t, div, tag):
        k = self.idx.get(t)
        if k is None:
            mitglieder = [i for i, d in enumerate(self.liga_von) if d == div and tag - self.last[i] < 365]
            basis = np.mean([self.M[i] for i in mitglieder], axis=0) if mitglieder else np.zeros(4)
            basis = basis.copy(); basis[2:] = 0
            k = self.idx[t] = len(self.M)
            self.M.append(basis + self.neu); self.P.append(self.P0.copy()); self.last.append(tag); self.liga_von.append(div)
        self.liga_von[k] = div
        return k

    def _vor_team(self, k, tag):
        return self.M[k], self.P[k] + np.diag(self.Q * max(tag - self.last[k], 0))

    def _vor_liga(self, k, tag):
        return self.LM[k], self.LP[k] + np.diag(self.QL * max(tag - self.Llast[k], 0))

    def _joint(self, i, j, l, tag):
        mi, Pi = self._vor_team(i, tag); mj, Pj = self._vor_team(j, tag); ml, Pl = self._vor_liga(l, tag)
        m = np.concatenate([mi, mj, ml])
        S = np.zeros((10, 10)); S[:4, :4] = Pi; S[4:8, 4:8] = Pj; S[8:, 8:] = Pl
        return m, S

    def schritt(self, tag, div, heim, gast, yh, ya, zaehlen):
        l = self._liga(div, tag)
        i, j = self._team(heim, div, tag), self._team(gast, div, tag)
        m, S = self._joint(i, j, l, tag)
        eta = HJ @ m
        v = np.einsum("ij,jk,ik->i", HJ, S, HJ)
        lam = np.exp(eta + v / 2)
        y = np.array([yh, ya], float)
        if zaehlen:
            self.ll += float(np.sum(y * np.log(lam) - lam))
            self.paare.append((tag, yh, ya, lam[0], lam[1]))
        Sinv = np.linalg.inv(S)
        x = m.copy()
        for _ in range(3):
            lx = np.exp(HJ @ x)
            A = Sinv + HJ.T @ (lx[:, None] * HJ)
            x = x + np.linalg.solve(A, HJ.T @ (y - lx) - Sinv @ (x - m))
        lx = np.exp(HJ @ x)
        Pn = np.linalg.inv(Sinv + HJ.T @ (lx[:, None] * HJ))
        self.M[i], self.M[j], self.LM[l] = x[:4], x[4:8], x[8:]
        self.P[i], self.P[j], self.LP[l] = Pn[:4, :4], Pn[4:8, 4:8], Pn[8:, 8:]
        self.last[i] = self.last[j] = self.Llast[l] = tag

    def lauf(self, d, ziel, ab):
        sh, sa = _spalten(ziel)
        tage = d["Date"].to_numpy("datetime64[D]").astype(np.int64)
        for t, dv, h, a, x, y in zip(tage, d["Div"].to_numpy(), d["Home"].to_numpy(), d["Away"].to_numpy(),
                                      d[sh].to_numpy(float), d[sa].to_numpy(float)):
            self.schritt(int(t), dv, h, a, x, y, int(t) >= ab)
        self.n += len(d)
        if len(d):
            self.bis = d["Date"].max()

    def kopie(self):
        f = _Filter.__new__(_Filter)
        f.__dict__.update({k: v for k, v in self.__dict__.items()})
        f.idx, f.lig = dict(self.idx), dict(self.lig)
        f.M, f.P, f.LM, f.LP = [x.copy() for x in self.M], [x.copy() for x in self.P], [x.copy() for x in self.LM], [x.copy() for x in self.LP]
        f.last, f.Llast, f.liga_von, f.paare = list(self.last), list(self.Llast), list(self.liga_von), list(self.paare)
        return f


def _tune(df, stich, ziel):
    key = (tuple(sorted(df["Div"].unique())), ziel, stich)
    if key in _TUNING:
        return _TUNING[key]
    von = stich - pd.DateOffset(years=FILTER_JAHRE)
    d = _zeilen(df, von, stich, ziel)
    best = START.copy()
    if len(d) >= 400:
        mu, h = _basis(df, von, ziel)
        ab = _tag(von + pd.Timedelta(days=EINLAUF_TAGE))
        def ziel_fn(th):
            th = np.clip(th, [-14, -18, -0.8, -0.8, -3], [-6, -8, 0.8, 0.8, 2])
            f = _Filter(th, mu, h); f.lauf(d, ziel, ab)
            return -f.ll
        res = minimize(ziel_fn, START, method="Nelder-Mead", options={"maxiter": 40, "xatol": 0.05, "fatol": 0.5})
        best = np.clip(res.x, [-14, -18, -0.8, -0.8, -3], [-6, -8, 0.8, 0.8, 2])
    _TUNING[key] = best
    return best


def _filter_bis(df, heute, ziel):
    stich = _stichtag(heute)
    von = stich - pd.DateOffset(years=FILTER_JAHRE)
    d = _zeilen(df, von, heute, ziel)
    if len(d) < 200:
        return None
    th = _tune(df, stich, ziel)
    key = (tuple(sorted(df["Div"].unique())), ziel, stich, tuple(np.round(th, 6)))
    ab = _tag(von + pd.Timedelta(days=EINLAUF_TAGE))
    f = _CACHE.get(key)
    if f is None or f.bis is None or (d["Date"] <= f.bis).sum() != f.n or d["Date"].max() < f.bis:
        mu, h = _basis(df, von, ziel)
        f = _Filter(th, mu, h)
        f.lauf(d, ziel, ab)
    else:
        f.lauf(d[d["Date"] > f.bis], ziel, ab)
    _CACHE[key] = f
    s = f.kopie()
    grenze = _tag(heute - pd.DateOffset(years=2))
    p = np.array([x for x in s.paare if x[0] >= grenze]) if s.paare else np.zeros((0, 5))
    s.abh = kern.waehle_abhaengigkeit(p[:, 1], p[:, 2], p[:, 3], p[:, 4], np.ones(len(p))) if len(p) > 200 else ("unabhaengig", 0.0)
    s.heute, s.theta = _tag(heute), th
    return s


def trainiere(df, heute):
    m = {"ft": _filter_bis(df, heute, "ft"), "ht": _filter_bis(df, heute, "ht")}
    if m["ft"] is None:
        raise ValueError("zu wenig Daten")
    return m


def _lambdas(f, heim, gast):
    def team(t):
        k = f.idx.get(t)
        if k is None:   # unbekannt: Durchschnitt der groessten Liga + Neuling-Abstand
            return np.zeros(4) + f.neu, f.P0.copy()
        return f.M[k], f.P[k] + np.diag(f.Q * max(f.heute - f.last[k], 0))
    k_liga = f.lig.get(f.liga_von[f.idx[heim]]) if heim in f.idx else (f.lig.get(f.liga_von[f.idx[gast]]) if gast in f.idx else 0)
    mi, Pi = team(heim); mj, Pj = team(gast)
    ml = f.LM[k_liga]; Pl = f.LP[k_liga] + np.diag(f.QL * max(f.heute - f.Llast[k_liga], 0))
    m = np.concatenate([mi, mj, ml])
    S = np.zeros((10, 10)); S[:4, :4] = Pi; S[4:8, 4:8] = Pj; S[8:, 8:] = Pl
    eta = HJ @ m
    v = np.einsum("ij,jk,ik->i", HJ, S, HJ)
    lam = np.exp(eta + v / 2)
    return float(lam[0]), float(lam[1])


def vorhersage(m, heim, gast):
    lh, la = _lambdas(m["ft"], heim, gast)
    out = {"ft": kern.matrix(lh, la, m["ft"].abh), "ht": None, "xg": [lh, la]}
    if m["ht"] is not None:
        hh, ha = _lambdas(m["ht"], heim, gast)
        out["ht"] = kern.matrix(hh, ha, m["ht"].abh)
    return out


def diagnose(m):
    t = []
    for ziel, name in (("ft", "Vollzeit"), ("ht", "Halbzeit")):
        f = m.get(ziel)
        if f is None:
            t.append(f"{name}: keine Daten")
        else:
            lq, lql, dA, dC, ls0 = f.theta
            t.append(f"{name}: Team-Veränderung/Jahr ±{np.sqrt(365 * np.exp(lq)):.2f}, Liga ±{np.sqrt(365 * np.exp(lql)):.3f}, "
                     f"Neuling Angriff {dA:+.2f} Abwehr {dC:+.2f}, Abhängigkeit {f.abh[0]} ({f.abh[1]:+.3f})")
    return " | ".join(t)
