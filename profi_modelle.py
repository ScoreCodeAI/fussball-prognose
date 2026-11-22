# ===== PROFI-MODELLE · LightGBM, Residual-Modell und gegner-bereinigte Form (eigenstaendig) =====
# Laeuft NACH "Neue Modelle" (eigener Workflow). Liest docs/daten.json und die Historie
# (daten.alle_spiele + Quoten der Historie) und ergaenzt NUR neue Eintraege:
#   spiel["prog"]["lgbm"]   LightGBM: lernt erwartete Tore aus Form, Elo und GEGNER-BEREINIGTER Form (ohne Markt)
#   spiel["prog"]["resid"]  Residual: startet bei den Markt-Toren (aus den Quoten) und lernt nur die Abweichung
#   spiel["gform"]          gegner-bereinigte Form beider Teams (Angriff/Abwehr, Tore und Schuesse aufs Tor)
#   daten["profi"]          ehrliche Pruefung: beide Modelle gegen den Markt auf Spielen, die sie nie gesehen haben
# Ausgabe im selben Format wie SVR/TCN (alle Maerkte aus erwarteten Toren -> 1X2, DC, O/U, BTTS, Halbzeiten).
# Bestehende Eintraege, Modelle und Berechnungen werden nicht veraendert.
# Ohne Daten-Leck: Merkmale je Spiel nur aus Spielen VOR dessen Datum, Training nur mit Spielen vor dem
# fruehesten Spiel in daten.json, Pruefung zeitlich getrennt (Training bis TEST_AB, Pruefung danach).
import json
import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HIER)
PFAD = os.path.join(HIER, "docs", "daten.json")
AB = "2016-07-01"            # Historie fuer Merkmale (Form braucht Vorlauf)
TRAIN_AB = "2018-07-01"      # erste Trainingszeile
TEST_AB = "2025-07-01"       # ehrliche Pruefung: alles ab hier hat das Pruef-Modell nie gesehen
MIN_SPIELE = 6               # so viele Spiele braucht ein Team, bevor es Merkmale bekommt
BOOT = 400
G = np.arange(11)


def log(*a):
    print("Profi-Modelle:", *a, flush=True)


# ---------------- Poisson-Hilfen (vektorisiert) ----------------
def _pmf(lam):
    lam = np.asarray(lam, dtype=float)[..., None]
    lg = np.array([math.lgamma(k + 1) for k in G])
    return np.exp(-lam + G * np.log(np.maximum(lam, 1e-9)) - lg)


def wk(lh, la):
    """1X2 und OVER 2,5 aus erwarteten Toren (fuer viele Spiele gleichzeitig)."""
    ph, pa = _pmf(lh), _pmf(la)
    m = ph[..., :, None] * pa[..., None, :]
    i, j = np.indices((11, 11))
    p1 = (m * (i > j)).sum((-1, -2))
    px = (m * (i == j)).sum((-1, -2))
    p2 = (m * (i < j)).sum((-1, -2))
    po = (m * (i + j > 2)).sum((-1, -2))
    s = p1 + px + p2
    return np.stack([p1 / s, px / s, p2 / s], -1), po / m.sum((-1, -2))


_GRID = None


def markt_lam(p1x2, po):
    """Erwartete Tore, die zu den Markt-% passen (Suche im Raster, Gewicht 1X2 und O/U)."""
    global _GRID
    if _GRID is None:
        v = np.arange(0.15, 4.51, 0.05)
        lh, la = np.meshgrid(v, v, indexing="ij")
        x, o = wk(lh.ravel(), la.ravel())
        _GRID = (lh.ravel(), la.ravel(), x, o)
    gh, ga, gx, go = _GRID
    out = np.full((len(p1x2), 2), np.nan)
    for a in range(0, len(p1x2), 1500):
        P, O = p1x2[a:a + 1500], po[a:a + 1500]
        d = ((P[:, None, :] - gx[None]) ** 2).sum(-1)
        oo = np.where(np.isnan(O)[:, None], 0.0, (O[:, None] - go[None]) ** 2)
        k = np.argmin(d + oo, axis=1)
        out[a:a + len(P)] = np.stack([gh[k], ga[k]], -1)
    out[np.isnan(p1x2).any(1)] = np.nan
    return out


def fair(q):
    q = np.asarray(q, dtype=float)
    inv = 1 / q
    return inv / inv.sum(-1, keepdims=True)


# ---------------- Verlauf: Form, Elo und gegner-bereinigte Staerke ----------------
class Verlauf:
    """Geht die Historie Spiel fuer Spiel durch. Alles, was ein Spiel 'weiss', stammt aus frueheren Spielen."""

    def __init__(self):
        self.liga = defaultdict(lambda: [1.48, 1.16, 4.6, 3.8])   # Heim-/Auswaertstore, Heim-/Auswaerts-Schuesse aufs Tor
        self.elo = defaultdict(lambda: 1500.0)
        self.r = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])         # Angriff, Abwehr (Tore) | Angriff, Abwehr (Schuesse aufs Tor)
        self.f = defaultdict(lambda: [None] * 7)                    # EWMA: Tore, Gegentore, SaT, SaT gegen, Punkte, Heim-Tore, Ausw.-Tore
        self.n = defaultdict(int)
        self.zuletzt = {}

    @staticmethod
    def _ew(alt, neu, a=0.25):
        return neu if alt is None else alt + a * (neu - alt)

    def _pause(self, t, d):
        z = self.zuletzt.get(t)
        if z is not None and (d - z).days > 60:                    # Sommerpause: Staerken Richtung Mittel
            self.r[t] = [x * 0.75 for x in self.r[t]]
        return min(30, (d - z).days) if z is not None else 14

    def merkmale(self, div, h, a, d):
        if self.n[h] < MIN_SPIELE or self.n[a] < MIN_SPIELE:
            return None
        L = self.liga[div]
        rh, ra = self.r[h], self.r[a]
        fh, fa = self.f[h], self.f[a]
        lam_h = L[0] * math.exp(rh[0] - ra[1])                    # gegner-bereinigte Tor-Erwartung
        lam_a = L[1] * math.exp(ra[0] - rh[1])
        slam_h = L[2] * math.exp(rh[2] - ra[3])
        slam_a = L[3] * math.exp(ra[2] - rh[3])
        zh = self.zuletzt.get(h)
        za = self.zuletzt.get(a)
        rest_h = min(30, (d - zh).days) if zh is not None else 14
        rest_a = min(30, (d - za).days) if za is not None else 14
        return [rh[0], rh[1], ra[0], ra[1], rh[2], rh[3], ra[2], ra[3], lam_h, lam_a, slam_h, slam_a,
                self.elo[h] - self.elo[a], self.elo[h], self.elo[a],
                *[x if x is not None else np.nan for x in fh], *[x if x is not None else np.nan for x in fa],
                rest_h, rest_a, min(self.n[h], 60), min(self.n[a], 60), L[0], L[1]]

    def gform(self, h, a):
        if self.n[h] < MIN_SPIELE or self.n[a] < MIN_SPIELE:
            return None
        k = lambda t: {"ang": round(self.r[t][0], 3), "abw": round(self.r[t][1], 3),
                       "sang": round(self.r[t][2], 3), "sabw": round(self.r[t][3], 3)}
        return {"h": k(h), "a": k(a)}

    def eintragen(self, div, h, a, d, hg, ag, hst, ast):
        self._pause(h, d)
        self._pause(a, d)
        L = self.liga[div]
        rh, ra = self.r[h], self.r[a]
        # gegner-bereinigte Staerke: Poisson-Gradient (Tore lr 0,05, Schuesse aufs Tor lr 0,03)
        eh, ea = L[0] * math.exp(rh[0] - ra[1]), L[1] * math.exp(ra[0] - rh[1])
        gh, ga = (hg - eh) / max(eh, 0.3), (ag - ea) / max(ea, 0.3)
        rh[0] += 0.05 * gh; ra[1] -= 0.05 * gh
        ra[0] += 0.05 * ga; rh[1] -= 0.05 * ga
        sot = not (hst is None or ast is None or math.isnan(hst) or math.isnan(ast))
        if sot:
            sh, sa = L[2] * math.exp(rh[2] - ra[3]), L[3] * math.exp(ra[2] - rh[3])
            dh, da = (hst - sh) / max(sh, 1), (ast - sa) / max(sa, 1)
            rh[2] += 0.03 * dh; ra[3] -= 0.03 * dh
            ra[2] += 0.03 * da; rh[3] -= 0.03 * da
            L[2] += 0.005 * (hst - L[2]); L[3] += 0.005 * (ast - L[3])
        for i in range(4):
            rh[i] = max(-1.5, min(1.5, rh[i])); ra[i] = max(-1.5, min(1.5, ra[i]))
        L[0] += 0.005 * (hg - L[0]); L[1] += 0.005 * (ag - L[1])
        # Elo (Heimvorteil 60, Tordifferenz-Faktor)
        e = 1 / (1 + 10 ** ((self.elo[a] - self.elo[h] - 60) / 400))
        s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        k = 20 * (1 + abs(hg - ag)) ** 0.5
        self.elo[h] += k * (s - e); self.elo[a] -= k * (s - e)
        # einfache Form (EWMA)
        pth, pta = (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
        fh, fa = self.f[h], self.f[a]
        fh[0] = self._ew(fh[0], hg); fh[1] = self._ew(fh[1], ag); fa[0] = self._ew(fa[0], ag); fa[1] = self._ew(fa[1], hg)
        if sot:
            fh[2] = self._ew(fh[2], hst); fh[3] = self._ew(fh[3], ast); fa[2] = self._ew(fa[2], ast); fa[3] = self._ew(fa[3], hst)
        fh[4] = self._ew(fh[4], pth); fa[4] = self._ew(fa[4], pta)
        fh[5] = self._ew(fh[5], hg); fa[6] = self._ew(fa[6], ag)
        self.n[h] += 1; self.n[a] += 1
        self.zuletzt[h] = d; self.zuletzt[a] = d


NAMEN = ["ang_h", "abw_h", "ang_a", "abw_a", "sang_h", "sabw_h", "sang_a", "sabw_a", "lam_h", "lam_a", "slam_h", "slam_a",
         "elo_diff", "elo_h", "elo_a"] + [f"f{i}_h" for i in range(7)] + [f"f{i}_a" for i in range(7)] + \
        ["rest_h", "rest_a", "n_h", "n_a", "liga_h", "liga_a", "liga"]


def durchlauf(H, ziele, ligen):
    """Trainingszeilen (Merkmale + Tore + Markt) und Merkmale je Ziel-Spiel zu dessen Datum."""
    ziele = sorted(ziele, key=lambda z: z["datum"])
    V = Verlauf()
    X, Y, D, M = [], [], [], []
    zf, zg = {}, {}
    zi = 0

    def ziel(z):
        dz = pd.Timestamp(z["datum"])
        f = V.merkmale(z["liga"], z["heim"], z["gast"], dz)
        if f:
            zf[z["key"]] = f + [ligen.get(z["liga"], -1)]
            zg[z["key"]] = V.gform(z["heim"], z["gast"])

    for row in H.itertuples(index=False):
        d = row.Date
        ds = d.strftime("%Y-%m-%d")
        while zi < len(ziele) and ziele[zi]["datum"] <= ds:
            ziel(ziele[zi]); zi += 1
        if ds >= TRAIN_AB:
            f = V.merkmale(row.Div, row.Home, row.Away, d)
            if f:
                X.append(f + [ligen.get(row.Div, -1)]); Y.append((row.HG, row.AG)); D.append(ds)
                M.append((row.OD1, row.ODX, row.OD2, row.OO, row.OU, row.MX1, row.MX2))
        V.eintragen(row.Div, row.Home, row.Away, d, float(row.HG), float(row.AG), row.HST, row.AST)
    while zi < len(ziele):
        ziel(ziele[zi]); zi += 1
    return np.array(X, dtype=float), np.array(Y, dtype=float), np.array(D), np.array(M, dtype=float), zf, zg


def historie():
    """Historie mit Quoten (Durchschnitts- und Hoechstquoten der Buchmacher)."""
    import daten
    H = daten.alle_spiele()
    H = H[H["Date"] >= pd.Timestamp(AB)].copy()
    try:
        nm = {"Division": "Div", "MatchDate": "Date", "HomeTeam": "Home", "AwayTeam": "Away", "OddHome": "OD1",
              "OddDraw": "ODX", "OddAway": "OD2", "Over25": "OO", "Under25": "OU", "MaxHome": "MX1", "MaxAway": "MX2"}
        q = pd.read_csv(daten._xgabora_url(), low_memory=False, usecols=list(nm)).rename(columns=nm)   # Namen, nicht Reihenfolge
        q["Date"] = pd.to_datetime(q["Date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
        H["Date"] = pd.to_datetime(H["Date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
        H = H.merge(q.drop_duplicates(subset=["Div", "Date", "Home", "Away"]), on=["Div", "Date", "Home", "Away"], how="left")
    except Exception as ex:
        log("Quoten der Historie nicht geladen:", ex)
    for c in ("OD1", "ODX", "OD2", "OO", "OU", "MX1", "MX2", "HST", "AST"):
        H[c] = pd.to_numeric(H[c], errors="coerce") if c in H else np.nan
    return H.sort_values("Date").reset_index(drop=True)


def markt_merkmale(M):
    """Markt-% (Marge raus), Markt-Tore und Buchmacher-Streuung aus den Quoten."""
    ok = np.all(M[:, :3] > 1, axis=1)
    p = np.full((len(M), 3), np.nan)
    p[ok] = fair(M[ok, :3])
    po = np.full(len(M), np.nan)
    oo = np.all(M[:, 3:5] > 1, axis=1)
    po[oo] = fair(M[oo, 3:5])[:, 0]
    lam = markt_lam(p, po)
    spread = np.where((M[:, 5] > 1) & ok, M[:, 5] / M[:, 0], np.nan), np.where((M[:, 6] > 1) & ok, M[:, 6] / M[:, 2], np.nan)
    return p, po, lam, np.stack(spread, -1)


# ---------------- Modelle ----------------
P_LGBM = dict(objective="poisson", learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_estimators=2000, verbose=-1, random_state=7)
P_RES = dict(objective="poisson", learning_rate=0.02, num_leaves=7, min_child_samples=400, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, n_estimators=1500, verbose=-1, random_state=7)


def _fit(params, X, y, init=None, kat=None):
    """Training mit Early Stopping auf den juengsten 15 % (zeitlich, nie zufaellig gemischt)."""
    import lightgbm as lgb
    n = len(X)
    k = int(n * 0.85)
    m = lgb.LGBMRegressor(**params)
    kw = {}
    if init is not None:
        kw = dict(init_score=init[:k], eval_init_score=[init[k:]])
    m.fit(X[:k], y[:k], eval_set=[(X[k:], y[k:])], categorical_feature=kat or "auto",
          callbacks=[lgb.early_stopping(100, verbose=False)], **kw)
    best = m.best_iteration_ or params["n_estimators"]
    p2 = dict(params, n_estimators=max(50, int(best * 1.1)))  # mit allen Daten neu, gleiche Baumzahl
    m2 = lgb.LGBMRegressor(**p2)
    m2.fit(X, y, init_score=init, categorical_feature=kat or "auto")
    return m2, best


class Modelle:
    def __init__(self, X, Y, MK, bis):
        """X Merkmale, Y Tore, MK = (markt_p, markt_po, markt_lam, spread). Training nur mit Zeilen in 'bis'."""
        kat = [len(NAMEN) - 1]
        Xb, Yb = X[bis], Y[bis]
        self.lh, b1 = _fit(P_LGBM, Xb, Yb[:, 0], kat=kat)
        self.la, b2 = _fit(P_LGBM, Xb, Yb[:, 1], kat=kat)
        lam, sp = MK[2][bis], MK[3][bis]
        okm = ~np.isnan(lam).any(1)
        XR = np.hstack([Xb, lam, sp])[okm]
        self.rh, b3 = _fit(P_RES, XR, Yb[okm, 0], init=np.log(lam[okm, 0]), kat=kat)
        self.ra, b4 = _fit(P_RES, XR, Yb[okm, 1], init=np.log(lam[okm, 1]), kat=kat)
        self.info = {"n_lgbm": int(len(Xb)), "n_resid": int(okm.sum()), "baeume": [b1, b2, b3, b4]}

    def lgbm(self, X):
        return np.clip(self.lh.predict(X), 0.1, 5), np.clip(self.la.predict(X), 0.1, 5)

    def resid(self, X, lam, sp):
        XR = np.hstack([X, lam, sp])
        return np.clip(lam[:, 0] * self.rh.predict(XR), 0.1, 5), np.clip(lam[:, 1] * self.ra.predict(XR), 0.1, 5)


def importance(m):
    imp = m.lh.booster_.feature_importance("gain") + m.la.booster_.feature_importance("gain")
    s = imp.sum() or 1
    return sorted(((NAMEN[i], round(float(v / s * 100), 1)) for i, v in enumerate(imp)), key=lambda x: -x[1])[:8]


# ---------------- ehrliche Pruefung ----------------
def _ll(p, y):
    return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-9))


def _urteil(llm, llk):
    d = llk - llm
    idx = np.random.default_rng(7).integers(0, len(d), (BOOT, len(d)))
    b = np.sort(d[idx].mean(1))
    lo, hi = b[int(BOOT * .025)] / llk.mean() * 100, b[int(BOOT * .975)] / llk.mean() * 100
    v = (llk.mean() - llm.mean()) / llk.mean() * 100
    return {"vorsprung": round(float(v), 2), "bereich": [round(float(lo), 2), round(float(hi), 2)],
            "urteil": "besser" if lo > 0 else "schlechter" if hi < 0 else "gleich"}


def pruefen(X, Y, D, MK):
    bis = D < TEST_AB
    test = (D >= TEST_AB) & ~np.isnan(MK[0]).any(1) & ~np.isnan(MK[2]).any(1)
    if bis.sum() < 5000 or test.sum() < 500:
        log("Pruefung: zu wenig Spiele", int(bis.sum()), int(test.sum()))
        return None
    m = Modelle(X, Y, MK, bis)
    Xt, Yt = X[test], Y[test].astype(int)
    y1 = np.where(Yt[:, 0] > Yt[:, 1], 0, np.where(Yt[:, 0] == Yt[:, 1], 1, 2))
    yo = (Yt.sum(1) > 2).astype(int)
    pk = MK[0][test]
    pok = MK[1][test]
    out = {"test_von": str(min(D[test])), "test_bis": str(max(D[test])), "n": int(test.sum()),
           "training_bis": TEST_AB, "wichtig": importance(m)}
    for nm, (lh, la) in (("lgbm", m.lgbm(Xt)), ("resid", m.resid(Xt, MK[2][test], MK[3][test]))):
        p, po = wk(lh, la)
        e = {"x": _urteil(_ll(p, y1), _ll(pk, y1))}
        e["x"]["treffer"] = round(float((p.argmax(1) == y1).mean() * 100), 1)
        e["x"]["treffer_markt"] = round(float((pk.argmax(1) == y1).mean() * 100), 1)
        oo = ~np.isnan(pok)
        if oo.sum() > 300:
            P2 = np.stack([po[oo], 1 - po[oo]], -1)
            K2 = np.stack([pok[oo], 1 - pok[oo]], -1)
            e["ou"] = _urteil(_ll(P2, 1 - yo[oo]), _ll(K2, 1 - yo[oo]))
        out[nm] = e
        log(f"Pruefung {nm}: 1X2 {e['x']['vorsprung']:+.2f} % {e['x']['bereich']} -> {e['x']['urteil']}"
            + (f" | O/U {e['ou']['vorsprung']:+.2f} % -> {e['ou']['urteil']}" if "ou" in e else "") + f" ({out['n']} Spiele)")
    return out


# ---------------- gelernte Remis-Ampel (fuer den Master-Bot) ----------------
# Keine feste Grenze: Aus der Historie wird gelernt, in welchen Spiel-Typen ein Remis WIRKLICH oefter kam als der
# Favoriten-Sieg. Spiel-Typ = Remis-% des Marktes (angepasst an die Remis-Neigung der Liga) und Abstand 1 zu 2.
# Ein Typ wird nur "Remis" wenn: mind. 150 Spiele und Remis-Quote signifikant ueber der Favoriten-Quote.
def _remis_zellen(Hm, lf):
    pdx = Hm["pD"] * Hm["Div"].map(lf).fillna(1.0)
    return np.floor(pdx * 50) * 2, np.minimum(np.floor((Hm["pH"] - Hm["pA"]).abs() * 20) * 5, 40)


def remis_lernen(H, bis):
    ok = (H["OD1"] > 1) & (H["ODX"] > 1) & (H["OD2"] > 1)
    Hm = H[ok].copy()
    p = fair(Hm[["OD1", "ODX", "OD2"]].values)
    Hm["pH"], Hm["pD"], Hm["pA"] = p[:, 0], p[:, 1], p[:, 2]
    Hm["y"] = np.where(Hm["HG"] > Hm["AG"], 0, np.where(Hm["HG"] == Hm["AG"], 1, 2))
    Hm["fav"] = np.where(Hm["pH"] >= Hm["pA"], Hm["y"] == 0, Hm["y"] == 2)
    tr = Hm[Hm["Date"] < pd.Timestamp(bis)]
    lf = (tr.groupby("Div")["y"].apply(lambda v: (v == 1).mean()) / max((tr["y"] == 1).mean(), 1e-6)).to_dict()
    a, b = _remis_zellen(tr, lf)
    g = tr.assign(cd=a, cg=b).groupby(["cd", "cg"]).agg(n=("y", "size"), dr=("y", lambda v: (v == 1).mean()), fv=("fav", "mean"))
    g["se"] = np.sqrt(g["dr"] * (1 - g["dr"]) / g["n"] + g["fv"] * (1 - g["fv"]) / g["n"])
    g["x"] = (g["n"] >= 150) & (g["dr"] - g["fv"] > g["se"])
    return {"lf": lf, "g": g, "Hm": Hm}


def remis_pruefen(H):
    L = remis_lernen(H, TEST_AB)
    te = L["Hm"][L["Hm"]["Date"] >= pd.Timestamp(TEST_AB)]
    a, b = _remis_zellen(te, L["lf"])
    t = te.assign(cd=a, cg=b).join(L["g"][["x"]], on=["cd", "cg"])
    t = t[t["x"] == True]
    out = {"zellen": int(L["g"]["x"].sum()), "tipps": int(len(t))}
    if len(t):
        out.update(remis=round(float((t["y"] == 1).mean() * 100), 1), fav=round(float(t["fav"].mean() * 100), 1),
                   markt=round(float(t["pD"].mean() * 100), 1))
    log("Remis-Ampel Pruefung:", out)
    return out


def remis_live(L, liga, p):
    pdx = p[1] * L["lf"].get(liga, 1.0)
    k = (math.floor(pdx * 50) * 2, min(math.floor(abs(p[0] - p[2]) * 20) * 5, 40))
    if k not in L["g"].index:
        return None
    r = L["g"].loc[k]
    return {"x": bool(r["x"]), "dr": round(float(r["dr"]) * 100, 1), "fav": round(float(r["fv"]) * 100, 1), "n": int(r["n"])}


# ---------------- Markt der kommenden Spiele ----------------
def live_markt(s):
    sv = (s.get("prog") or {}).get("sv") or {}
    q = s.get("quoten") or {}
    x = (sv.get("x") or {})
    p = x.get("shin") or x.get("fair")
    if p and len(p) == 3:
        p = np.array(p, dtype=float) / 100
    elif all(q.get(k) for k in ("1", "X", "2")):
        p = fair([q["1"], q["X"], q["2"]])
    else:
        return None
    o = (sv.get("ou") or {})
    po = o.get("shin") or o.get("fair")
    if po:
        po = po[0] / 100
    elif q.get("Ü2.5") and q.get("U2.5"):
        po = fair([q["Ü2.5"], q["U2.5"]])[0]
    else:
        po = np.nan
    best, avg = x.get("q"), x.get("avg_q")
    sp = [best[0] / avg[0], best[2] / avg[2]] if best and avg and len(best) == 3 and len(avg) == 3 else [np.nan, np.nan]
    return p, po, sp


def main():
    d = json.load(open(PFAD, encoding="utf-8"))
    spiele = d.get("spiele", [])
    if not spiele:
        log("keine Spiele in daten.json")
        return
    import neue_modelle as nm
    H = historie()
    ligen = {l: i for i, l in enumerate(sorted(H["Div"].unique()))}
    log(f"Historie: {len(H)} Spiele, {H['OD1'].notna().sum()} mit Quoten, {H['HST'].notna().sum()} mit Schuessen aufs Tor")
    ziele = [{"key": i, "liga": s["liga"], "datum": s["datum"], "heim": s["heim"], "gast": s["gast"]} for i, s in enumerate(spiele)]
    grenze = min(s["datum"] for s in spiele)
    X, Y, D, M, zf, zg = durchlauf(H, ziele, ligen)
    MK = markt_merkmale(M)
    log(f"{len(X)} Trainingszeilen, Merkmale fuer {len(zf)} von {len(spiele)} Spielen")

    info = {"stand": pd.Timestamp.today().strftime("%Y-%m-%d")}
    try:
        info["pruefung"] = pruefen(X, Y, D, MK)
    except Exception as ex:
        log("Pruefung nicht moeglich:", ex)

    try:
        info["remis"] = remis_pruefen(H)
        RL = remis_lernen(H, grenze)
    except Exception as ex:
        RL = None
        log("Remis-Ampel nicht moeglich:", ex)

    share = {}
    for div, g in H.dropna(subset=["HTHG", "HTAG"]).groupby("Div"):
        tot = (g["HG"] + g["AG"]).sum()
        share[div] = float((g["HTHG"] + g["HTAG"]).sum() / tot) if tot else 0.44

    m = Modelle(X, Y, MK, D < grenze)
    info["live"] = m.info
    n_l = n_r = n_g = 0
    keys = sorted(zf)
    if keys:
        XL = np.array([zf[k] for k in keys], dtype=float)
        lh, la = m.lgbm(XL)
        for k, a, b in zip(keys, lh, la):
            s = spiele[k]
            try:
                s.setdefault("prog", {})["lgbm"] = nm.maerkte(float(a), float(b), share.get(s["liga"], 0.44))
                n_l += 1
            except Exception:
                pass
            if zg.get(k):
                s["gform"] = zg[k]
                n_g += 1
        LM = [(k, live_markt(spiele[k])) for k in keys]
        LM = [(k, v) for k, v in LM if v]
        if RL:
            for k, v in LM:
                try:
                    rm = remis_live(RL, spiele[k]["liga"], v[0])
                    if rm:
                        spiele[k]["remis"] = rm
                except Exception:
                    pass
        if LM:
            P = np.array([v[0] for _, v in LM]); PO = np.array([v[1] for _, v in LM], dtype=float)
            SP = np.array([v[2] for _, v in LM], dtype=float)
            lam = markt_lam(P, PO)
            XR = np.array([zf[k] for k, _ in LM], dtype=float)
            ok = ~np.isnan(lam).any(1)
            if ok.any():
                rh, ra = m.resid(XR[ok], lam[ok], SP[ok])
                for (k, _), a, b in zip([x for x, o in zip(LM, ok) if o], rh, ra):
                    s = spiele[k]
                    try:
                        s["prog"]["resid"] = nm.maerkte(float(a), float(b), share.get(s["liga"], 0.44))
                        n_r += 1
                    except Exception:
                        pass
    info["anzahl"] = {"lgbm": n_l, "resid": n_r, "gform": n_g, "spiele": len(spiele)}
    d["profi"] = info
    with open(PFAD, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
    log(f"LightGBM {n_l} · Residual {n_r} · gegner-bereinigte Form {n_g} von {len(spiele)} Spielen")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        import traceback
        log("Fehler (daten.json unveraendert):", ex, traceback.format_exc()[-800:])
    sys.exit(0)
