# ===== MODELL: Dynamik · Teamstärke (Engine 1, adaptives State-Space-Modell) =====
# Jedes Team hat einen versteckten Zustand: [Angriff, Gegentor-Neigung, Heim/Auswaerts-Abweichung
# im Angriff, Heim/Auswaerts-Abweichung bei Gegentoren]. Heim- und Auswaertsstaerke sind dadurch
# getrennt (Heim = Grundwert + Abweichung, Auswaerts = Grundwert - Abweichung).
#
# - Zwischen zwei Spielen darf sich der Zustand veraendern (Random Walk). WIE STARK er sich pro Tag
#   veraendern darf (Prozessrauschen), wird AUTOMATISCH geschaetzt: das Rauschen, mit dem die
#   Vorhersagen der jeweils naechsten Spiele am besten waren (One-Step-Ahead-Likelihood),
#   neu bestimmt jedes Halbjahr nur mit Daten VOR diesem Zeitpunkt.
# - Nach jedem Spiel wird der Zustand mit dem Ergebnis aktualisiert. Weil Tore Poisson-verteilt
#   sind (nicht normalverteilt), wird statt des klassischen Kalman-Filters eine Laplace-Naeherung
#   (iterierter Newton-Schritt) verwendet.
# - Die Unsicherheit des Zustands fliesst in die Prognose ein (unsichere Teams -> breitere Verteilung).
# - Kein festes Formfenster: alte Spiele verlieren automatisch an Einfluss, so schnell wie die
#   Daten es nahelegen.
# - Vollzeit und Halbzeit haben getrennte Filter mit eigenen Parametern.
# - Abhaengigkeit Heim/Gast (unabhaengig / Dixon-Coles / Frank) aus den echten Vorhersagefehlern.
#
# Feste Annahmen: Startunsicherheit neuer Teams, Liga-Grundniveau und Heimvorteil aus den
# 2 Jahren vor Filterbeginn, Filter startet 3 Jahre vor dem Halbjahres-Stichtag (Rechenzeit).
import numpy as np
import pandas as pd

import kern

NAME = "Dynamik · Teamstärke"
KURZ = "Dynamik"
ICON = "D"
FARBE1 = "#2bf5a0"
FARBE2 = "#00b3a4"
BESCHREIBUNG = "Teamstärken verändern sich nach jedem Spiel, wie schnell lernt das Modell selbst"

FILTER_JAHRE = 3
EINLAUF_TAGE = 365                     # erstes Jahr nur Einlaufen, zaehlt nicht fuer die Schaetzung
GITTER_Q = [5e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4, 3.2e-4]   # Veraenderung pro Tag (Jahres-Std. ~0,04 bis ~0,34)
GITTER_RATIO = [0.1, 0.5]              # Heim/Auswaerts-Abweichung veraendert sich langsamer
P0 = np.diag([0.09, 0.09, 0.01, 0.01])  # Startunsicherheit neuer Teams
H_HEIM = np.array([1., 0, 1, 0, 0, 1, 0, -1])
H_GAST = np.array([0., 1, 0, 1, 1, 0, -1, 0])
H = np.vstack([H_HEIM, H_GAST])
_TUNING, _CACHE = {}, {}


def _spalten(ziel):
    return ("HG", "AG") if ziel == "ft" else ("HTHG", "HTAG")


def _stichtag(heute):
    return pd.Timestamp(heute.year, 1 if heute.month <= 6 else 7, 1)


def _zeilen(df, von, bis, ziel):
    sh, sa = _spalten(ziel)
    d = df[(df["Date"] >= von) & (df["Date"] < bis)].dropna(subset=[sh, sa])
    return d.sort_values(["Date", "Home"], kind="mergesort")


def _basis(df, von, ziel):
    """Liga-Grundniveau und Heimvorteil aus den 2 Jahren VOR Filterbeginn."""
    sh, sa = _spalten(ziel)
    d = _zeilen(df, von - pd.DateOffset(years=2), von, ziel)
    if len(d) < 200:
        d = _zeilen(df, von, von + pd.DateOffset(years=1), ziel).head(300)
    mh, ma = max(d[sh].mean(), 0.05), max(d[sa].mean(), 0.05)
    return np.log(ma), np.log(mh / ma)


class _Filter:
    def __init__(self, q, ratio, mu, h):
        self.Q = np.array([q, q, q * ratio, q * ratio])
        self.mu, self.h = mu, h
        self.idx, self.M, self.P, self.last = {}, [], [], []
        self.n, self.bis, self.ll, self.paare = 0, None, 0.0, []

    def _team(self, t, tag):
        k = self.idx.get(t)
        if k is None:
            k = self.idx[t] = len(self.M)
            self.M.append(np.zeros(4)); self.P.append(P0.copy()); self.last.append(tag)
        return k

    def _vorher(self, k, tag):
        P = self.P[k] + np.diag(self.Q * max(tag - self.last[k], 0))
        return self.M[k], P

    def schritt(self, tag, heim, gast, yh, ya, zaehlen):
        i, j = self._team(heim, tag), self._team(gast, tag)
        mi, Pi = self._vorher(i, tag)
        mj, Pj = self._vorher(j, tag)
        m = np.concatenate([mi, mj])
        S = np.zeros((8, 8)); S[:4, :4] = Pi; S[4:, 4:] = Pj
        c = np.array([self.mu + self.h, self.mu])
        eta = c + H @ m
        v = np.einsum("ij,jk,ik->i", H, S, H)
        lam = np.exp(eta + v / 2)
        y = np.array([yh, ya], float)
        if zaehlen:
            self.ll += float(np.sum(y * np.log(lam) - lam))
            self.paare.append((tag, yh, ya, lam[0], lam[1]))
        Sinv = np.linalg.inv(S)
        x = m.copy()
        for _ in range(3):
            l = np.exp(c + H @ x)
            A = Sinv + H.T @ (l[:, None] * H)
            x = x + np.linalg.solve(A, H.T @ (y - l) - Sinv @ (x - m))
        l = np.exp(c + H @ x)
        Pn = np.linalg.inv(Sinv + H.T @ (l[:, None] * H))
        self.M[i], self.M[j] = x[:4], x[4:]
        self.P[i], self.P[j] = Pn[:4, :4], Pn[4:, 4:]
        self.last[i] = self.last[j] = tag

    def lauf(self, d, ziel, ab_tag_zaehlen):
        sh, sa = _spalten(ziel)
        tage = (d["Date"].to_numpy("datetime64[D]").astype(np.int64))
        for t, h, a, x, y in zip(tage, d["Home"].to_numpy(), d["Away"].to_numpy(), d[sh].to_numpy(float), d[sa].to_numpy(float)):
            self.schritt(int(t), h, a, x, y, int(t) >= ab_tag_zaehlen)
        self.n += len(d)
        if len(d):
            self.bis = d["Date"].max()

    def kopie(self):
        f = _Filter.__new__(_Filter)
        f.Q, f.mu, f.h = self.Q, self.mu, self.h
        f.idx, f.M, f.P, f.last = dict(self.idx), [m.copy() for m in self.M], [p.copy() for p in self.P], list(self.last)
        f.n, f.bis, f.ll, f.paare = self.n, self.bis, self.ll, list(self.paare)
        return f


def _tag(ts):
    return int(np.datetime64(pd.Timestamp(ts).date(), "D").astype(np.int64))


def _tune(df, stich, ziel):
    key = (tuple(sorted(df["Div"].unique())), ziel, stich)
    if key in _TUNING:
        return _TUNING[key]
    von = stich - pd.DateOffset(years=FILTER_JAHRE)
    d = _zeilen(df, von, stich, ziel)
    mu, h = _basis(df, von, ziel)
    best, best_ll = (GITTER_Q[1], GITTER_RATIO[0]), -np.inf
    if len(d) >= 400:
        for q in GITTER_Q:
            for ratio in GITTER_RATIO:
                f = _Filter(q, ratio, mu, h)
                f.lauf(d, ziel, _tag(von + pd.Timedelta(days=EINLAUF_TAGE)))
                if f.ll > best_ll:
                    best, best_ll = (q, ratio), f.ll
    _TUNING[key] = best
    return best


def _filter_bis(df, heute, ziel):
    stich = _stichtag(heute)
    von = stich - pd.DateOffset(years=FILTER_JAHRE)
    d = _zeilen(df, von, heute, ziel)
    if len(d) < 200:
        return None
    q, ratio = _tune(df, stich, ziel)
    key = (tuple(sorted(df["Div"].unique())), ziel, stich, q, ratio)
    f = _CACHE.get(key)
    # Fortsetzen, wenn die bisher verarbeiteten Spiele genau der Anfang der neuen Daten sind
    if f is None or f.bis is None or (d["Date"] <= f.bis).sum() != f.n or d["Date"].max() < f.bis:
        mu, h = _basis(df, von, ziel)
        f = _Filter(q, ratio, mu, h)
        f.lauf(d, ziel, _tag(von + pd.Timedelta(days=EINLAUF_TAGE)))
    else:
        f.lauf(d[d["Date"] > f.bis], ziel, _tag(von + pd.Timedelta(days=EINLAUF_TAGE)))
    _CACHE[key] = f
    snap = f.kopie()
    grenze = _tag(heute - pd.DateOffset(years=2))
    p = np.array([x for x in snap.paare if x[0] >= grenze]) if snap.paare else np.zeros((0, 5))
    snap.abh = kern.waehle_abhaengigkeit(p[:, 1], p[:, 2], p[:, 3], p[:, 4], np.ones(len(p))) if len(p) > 200 else ("unabhaengig", 0.0)
    snap.heute = _tag(heute)
    snap.q, snap.ratio = q, ratio
    return snap


def trainiere(df, heute):
    m = {"ft": _filter_bis(df, heute, "ft"), "ht": _filter_bis(df, heute, "ht")}
    if m["ft"] is None:
        raise ValueError("zu wenig Daten")
    return m


def _lambdas(f, heim, gast):
    def zust(t):
        k = f.idx.get(t)
        if k is None:
            return np.zeros(4), P0.copy()
        return f.M[k], f.P[k] + np.diag(f.Q * max(f.heute - f.last[k], 0))
    mi, Pi = zust(heim)
    mj, Pj = zust(gast)
    m = np.concatenate([mi, mj])
    S = np.zeros((8, 8)); S[:4, :4] = Pi; S[4:, 4:] = Pj
    eta = np.array([f.mu + f.h, f.mu]) + H @ m
    v = np.einsum("ij,jk,ik->i", H, S, H)
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
            t.append(f"{name}: Veränderung/Jahr ±{np.sqrt(365 * f.q):.2f}, Heim/Auswärts-Anteil {f.ratio:g}, "
                     f"Abhängigkeit {f.abh[0]} ({f.abh[1]:+.3f}), Heimvorteil {np.exp(f.h):.2f}x")
    return " | ".join(t)
