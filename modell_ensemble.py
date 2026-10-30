# ===== MODELL E · ENSEMBLE (nur Tore, eigenstaendig) =====
# Vier eigene Bausteine, alle nur aus Toren und Halbzeitstaenden (keine Quoten, keine Schuesse):
#   P  Poisson      - Angriff/Abwehr Heim und Auswaerts getrennt, Zeitgewichtung, Glaettung zum Ligamittel
#   DC Dixon-Coles  - gleiche Staerken, Korrektur fuer 0:0, 1:0, 0:1, 1:1 (Parameter aus den Daten geschaetzt)
#   H  Haeufigkeit  - echte Torverteilungen der Teams (wie oft 0, 1, 2, 3 ... Tore), geglaettet
#   F  Form         - letzte 5 / 10 / 15 Spiele, Heimteam aus Heimspielen, Gastteam aus Auswaertsspielen
# Ensemble: Mischung der vier Ergebnis-Matrizen. Die Gewichte werden je Land im Rueckblick gelernt
# (Bausteine auf aelteren Spielen trainiert, auf den letzten VALID_TAGE geprueft) - nie festgelegt.
# Halbzeit: dieselben Bausteine auf Halbzeit-Toren, nur Spiele mit echtem Halbzeitstand.
# Zusatzmaerkte (kern.extra): Halbzeit/Endstand (9), Correct Score Top 10, O/U 1,5 / 3,5, HZ O/U 0,5 / 1,5.
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
BESCHREIBUNG = "Poisson, Dixon-Coles, Häufigkeit und Form – Gewichte im Rückblick gelernt, nur aus Toren"
EXTRA = True            # Engine berechnet Zusatzmaerkte (HZ/ES, Correct Score, weitere O/U)

HALBWERT = 365          # Tage: aeltere Spiele zaehlen weniger
JAHRE = 4
PRIOR = 5.0             # Glaettung zum Ligamittel (in "Spielen")
G = 10                  # Matrix 0..10 Tore, Rest in letzter Zeile/Spalte
FORM_N = (5, 10, 15)
FORM_W = (0.5, 0.3, 0.2)
VALID_TAGE = 365
SCHRITT = 0.1
KOMP = ("P", "DC", "H", "F")
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

    def __init__(self, d, hs, as_, heute):
        d = d.dropna(subset=[hs, as_])
        self.ok = len(d) >= 100
        if not self.ok:
            return
        w = 0.5 ** ((heute - d["Date"]).dt.days.clip(lower=0) / HALBWERT)
        hg, ag = d[hs].to_numpy(float), d[as_].to_numpy(float)
        self.mh = max(float(np.average(hg, weights=w)), 0.15)
        self.ma = max(float(np.average(ag, weights=w)), 0.15)
        self.ph = _verteilung(hg, w, np.zeros(G + 1), 0); self.pa = _verteilung(ag, w, np.zeros(G + 1), 0)
        t = pd.DataFrame({"H": d["Home"].to_numpy(), "A": d["Away"].to_numpy(), "hg": hg, "ag": ag, "w": w.to_numpy()})
        t["whg"], t["wag"] = t.w * t.hg, t.w * t.ag
        H = t.groupby("H")[["w", "whg", "wag"]].sum()
        A = t.groupby("A")[["w", "whg", "wag"]].sum()
        # Heimteam: geschossen/kassiert daheim; Gastteam: geschossen/kassiert auswaerts (geglaettet)
        self.hs = ((H.whg + PRIOR * self.mh) / (H.w + PRIOR)).to_dict()
        self.hc = ((H.wag + PRIOR * self.ma) / (H.w + PRIOR)).to_dict()
        self.as_ = ((A.wag + PRIOR * self.ma) / (A.w + PRIOR)).to_dict()
        self.ac = ((A.whg + PRIOR * self.mh) / (A.w + PRIOR)).to_dict()
        # Haeufigkeit: Torverteilungen je Team
        self.vh, self.va, self.vhc, self.vac = {}, {}, {}, {}
        for team, g in t.groupby("H"):
            self.vh[team] = _verteilung(g.hg, g.w, self.ph); self.vhc[team] = _verteilung(g.ag, g.w, self.pa)
        for team, g in t.groupby("A"):
            self.va[team] = _verteilung(g.ag, g.w, self.pa); self.vac[team] = _verteilung(g.hg, g.w, self.ph)
        # Form: letzte Heim- bzw. Auswaertsspiele
        t["D"] = d["Date"].to_numpy()
        t = t.sort_values("D")
        self.fh = {team: (g.hg.to_numpy(), g.ag.to_numpy()) for team, g in t.groupby("H")}
        self.fa = {team: (g.ag.to_numpy(), g.hg.to_numpy()) for team, g in t.groupby("A")}
        # Dixon-Coles rho aus den Daten (gewichtete Likelihood)
        lh = np.array([self._lam(a, b)[0] for a, b in zip(t.H, t.A)])
        la = np.array([self._lam(a, b)[1] for a, b in zip(t.H, t.A)])
        x, y, ww = t.hg.to_numpy().astype(int), t.ag.to_numpy().astype(int), t.w.to_numpy()

        def nll(r):
            tau = np.ones_like(lh)
            tau = np.where((x == 0) & (y == 0), 1 - lh * la * r, tau)
            tau = np.where((x == 0) & (y == 1), 1 + lh * r, tau)
            tau = np.where((x == 1) & (y == 0), 1 + la * r, tau)
            tau = np.where((x == 1) & (y == 1), 1 - r, tau)
            return -np.sum(ww * np.log(np.clip(tau, 1e-9, None)))
        self.rho = float(minimize_scalar(nll, bounds=(-0.2, 0.15), method="bounded").x)

    def _lam(self, a, b):
        lh = self.hs.get(a, self.mh) * self.ac.get(b, self.mh) / self.mh
        la = self.as_.get(b, self.ma) * self.hc.get(a, self.ma) / self.ma
        return float(np.clip(lh, 0.1, 5)), float(np.clip(la, 0.1, 5))

    def _form(self, a, b):
        def f(reihe, mittel):
            if reihe is None or len(reihe[0]) == 0:
                return 1.0, 1.0
            s, c = reihe
            att = sum(w * np.mean(s[-n:]) for n, w in zip(FORM_N, FORM_W)) / mittel[0]
            dfn = sum(w * np.mean(c[-n:]) for n, w in zip(FORM_N, FORM_W)) / mittel[1]
            return 0.5 + 0.5 * att, 0.5 + 0.5 * dfn      # halb zum Mittel gezogen
        ah, dh = f(self.fh.get(a), (self.mh, self.ma))
        aa, da = f(self.fa.get(b), (self.ma, self.mh))
        return float(np.clip(self.mh * ah * da, 0.1, 5)), float(np.clip(self.ma * aa * dh, 0.1, 5))

    def matrizen(self, a, b):
        lh, la = self._lam(a, b)
        P = np.outer(_pois(lh), _pois(la))
        hv = 0.5 * (self.vh.get(a, self.ph) + self.vac.get(b, self.ph))
        av = 0.5 * (self.va.get(b, self.pa) + self.vhc.get(a, self.pa))
        fh, fa = self._form(a, b)
        return {"P": P, "DC": _dc(P, lh, la, self.rho), "H": np.outer(hv, av), "F": np.outer(_pois(fh), _pois(fa))}


def _gitter():
    st = int(round(1 / SCHRITT))
    for c in itertools.product(range(st + 1), repeat=len(KOMP)):
        if sum(c) == st:
            yield np.array(c) / st


def _gewichte(d, heute):
    cut = heute - pd.Timedelta(days=VALID_TAGE)
    alt, neu = d[d["Date"] < cut], d[d["Date"] >= cut]
    if len(alt) < 600 or len(neu) < 80:
        return np.array([0.25, 0.25, 0.25, 0.25]), "zu wenig Daten – gleiche Gewichte"
    t = _Teil(alt, "HG", "AG", cut)
    L = []
    for r in neu.itertuples(index=False):
        m = t.matrizen(r.Home, r.Away)
        i, j = min(int(r.HG), G), min(int(r.AG), G)
        L.append([m[k][i, j] for k in KOMP])
    L = np.array(L)
    best = min(_gitter(), key=lambda w: -np.log(np.clip(L @ w, 1e-12, None)).mean())
    return best, f"gelernt auf {len(neu)} Spielen"


def trainiere(df, heute):
    heute = pd.Timestamp(heute)
    d = df[(df["Date"] < heute) & (df["Date"] >= heute - pd.DateOffset(years=JAHRE))].sort_values("Date")
    if len(d) < 300:
        raise ValueError("zu wenig Daten")
    w, info = _gewichte(d, heute)
    ft = _Teil(d, "HG", "AG", heute)
    ht = _Teil(d, "HTHG", "HTAG", heute)
    return {"w": w, "info": info, "ft": ft, "ht": ht if ht.ok else None, "n": len(d)}


def _mix(teil, w, a, b):
    m = teil.matrizen(a, b)
    M = sum(wi * m[k] for wi, k in zip(w, KOMP))
    return M / M.sum()


def vorhersage(m, heim, gast):
    ft = _mix(m["ft"], m["w"], heim, gast)
    if m["ht"] is not None:
        ht = _mix(m["ht"], m["w"], heim, gast)
    else:
        xh, xa = kern.erwartete_tore(ft)
        ht = np.outer(_pois(xh * kern.HZ_ANTEIL), _pois(xa * kern.HZ_ANTEIL))
    return {"ft": ft, "ht": ht, "xg": list(kern.erwartete_tore(ft))}


def diagnose(m):
    w = " · ".join(f"{k} {x:.0%}" for k, x in zip(KOMP, m["w"]))
    return f"Gewichte {w} ({m['info']}) | Halbzeit {'echt' if m['ht'] is not None else 'Ersatz 45 %'} | {m['n']} Spiele"
