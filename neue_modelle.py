# ===== NEUE MODELLE · SVR, TCN und Konsens 2 (eigenstaendig) =====
# Laeuft NACH dem Morgenlauf (eigener Workflow). Liest docs/daten.json und die Historie
# (daten.alle_spiele), rechnet je Spiel und ergaenzt NUR neue Eintraege in spiel["prog"]:
#   svr, tcn                  -> eigene Tabs in Analyse 1
#   remis                     -> Remis-Radar (eigenes 1X2-Modell mit Remis-Merkmalen, ohne Liga-Merkmal)
#   h2m                       -> 2.-HZ-Modell: lernt nur das Ergebnis der 2. Halbzeit (fuer den Markt 2. HZ 1X2)
#   h1m                       -> 1.-HZ-Modell: lernt nur das Ergebnis der 1. Halbzeit (fuer den Markt 1. HZ 1X2)
#   k2_korr, k2_rating, k2_hz, k2_reines, k2_xg -> Bausteine fuer Konsens 2
# Bestehende Eintraege, Modelle und Berechnungen werden nicht veraendert.
# Ohne Daten-Leck: Training nur mit Spielen VOR dem fruehesten Spiel in daten.json,
# Merkmale je Spiel nur aus Spielen vor dessen Datum.
import gzip
import json
import math
import os
import sys
import unicodedata
import difflib
from collections import defaultdict, deque

import numpy as np
import pandas as pd

HIER = os.path.dirname(os.path.abspath(__file__))
PFAD = os.path.join(HIER, "docs", "daten.json")
ARCHIV = os.path.join(HIER, "archiv", "sportsapi.json.gz")
AB = "2019-07-01"          # Historie fuer Merkmale/Training
SAISON_START = "2026-07-01"
N_SVR = 6000               # juengste Trainingsspiele fuer SVR (RBF skaliert quadratisch)
N_TCN = 25000              # juengste Trainingsspiele fuer TCN
SEQ = 10                   # Laenge der Spielfolge je Team (TCN)
MAXG = 10


def log(*a):
    print("Neue Modelle:", *a, flush=True)


# ---------------- gemeinsame Poisson-Maerkte ----------------
def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k) if lam > 0 else (1.0 if k == 0 else 0.0)


def matrix(lh, la, rho=None):
    m = np.array([[pois(i, lh) * pois(j, la) for j in range(MAXG + 1)] for i in range(MAXG + 1)])
    if rho is not None:   # Dixon-Coles fuer niedrige Ergebnisse
        m[0, 0] *= 1 - lh * la * rho
        m[0, 1] *= 1 + lh * rho
        m[1, 0] *= 1 + la * rho
        m[1, 1] *= 1 - rho
    return m / m.sum()


def maerkte(lh, la, share, rho=None, remis_boost=False):
    """Alle Werte im Format der bestehenden Modelle (Prozent)."""
    lh, la = max(lh, 0.05), max(la, 0.05)
    m = matrix(lh, la, rho)
    i, j = np.indices(m.shape)
    p1, px, p2 = m[i > j].sum(), m[i == j].sum(), m[i < j].sum()
    if remis_boost and abs(lh - la) < 0.4:
        px *= 1.15
        t = p1 + px + p2
        p1, px, p2 = p1 / t, px / t, p2 / t
    ue25 = m[i + j > 2].sum()
    bts = m[(i > 0) & (j > 0)].sum()
    h1 = matrix(lh * share, la * share)
    hi, hj = np.indices(h1.shape)
    ht = [h1[hi > hj].sum(), h1[hi == hj].sum(), h1[hi < hj].sum()]
    h2 = matrix(lh * (1 - share), la * (1 - share))
    ht2 = [h2[hi > hj].sum(), h2[hi == hj].sum(), h2[hi < hj].sum()]
    h1o05 = 1 - math.exp(-(lh + la) * share)
    h2o05 = 1 - math.exp(-(lh + la) * (1 - share))
    r = lambda x: round(float(x) * 100, 2)
    return {"p1": r(p1), "px": r(px), "p2": r(p2), "ue25": r(ue25), "bts": r(bts),
            "ht": [r(x) for x in ht], "ht2": [r(x) for x in ht2], "h1o05": r(h1o05), "h2o05": r(h2o05),
            "xg": [round(lh, 2), round(la, 2)], "tipp": None}


# ---------------- Merkmale aus der Historie ----------------
class Verlauf:
    """Geht die Historie zeitlich durch und haelt je Team die letzten Spiele."""

    def __init__(self):
        self.team = defaultdict(lambda: deque(maxlen=SEQ))   # (gf, ga, heim)
        self.liga = defaultdict(lambda: deque(maxlen=300))   # (hg, ag)
        self.elo = defaultdict(lambda: 1500.0)               # eigene Elo, fuer alle Ligen gleich gerechnet
        self._ht = None
        self.sot = defaultdict(lambda: deque(maxlen=SEQ))    # (Schuesse aufs Tor fuer, gegen)
        self.h2 = defaultdict(lambda: deque(maxlen=SEQ))     # (Tore fuer, gegen, HZ fuer, HZ gegen) nur Spiele mit Halbzeitstand

    def merkmale(self, div, h, a):
        th, ta, lg = self.team[(div, h)], self.team[(div, a)], self.liga[div]
        if len(th) < 5 or len(ta) < 5 or len(lg) < 30:
            return None
        def m(q, k, n):
            v = list(q)[-n:]
            return sum(x[k] for x in v) / len(v)
        lh = sum(x[0] for x in lg) / len(lg)
        la = sum(x[1] for x in lg) / len(lg)
        return [m(th, 0, 10), m(th, 1, 10), m(ta, 0, 10), m(ta, 1, 10),
                m(th, 0, 5), m(th, 1, 5), m(ta, 0, 5), m(ta, 1, 5), lh, la]

    def remis(self, div, h, a, quoten=None):
        th, ta = list(self.team[(div, h)]), list(self.team[(div, a)])
        if len(th) < 5 or len(ta) < 5:
            return None
        eh, ea = self.elo[(div, h)], self.elo[(div, a)]
        sh, sa = list(self.sot[(div, h)]), list(self.sot[(div, a)])
        if len(sh) >= 3 and len(sa) >= 3:
            bal = abs((sum(x[0] - x[1] for x in sh) / len(sh)) - (sum(x[0] - x[1] for x in sa) / len(sa)))
            s_ok = 1.0
        else:
            bal, s_ok = 0.0, 0.0
        mx, m_ok = 0.0, 0.0
        try:
            q1, qx, q2 = (float(quoten[k]) for k in ("1", "X", "2"))
            if min(q1, qx, q2) > 1:
                mx = (1 / qx) / (1 / q1 + 1 / qx + 1 / q2)
                m_ok = 1.0
        except Exception:
            pass
        m = lambda v, k: sum(x[k] for x in v) / len(v)
        gfh, gah, gfa, gaa = m(th, 0), m(th, 1), m(ta, 0), m(ta, 1)
        diff = (gfh - gah) - (gfa - gaa)
        rem = lambda v: sum(1 for x in v if x[0] == x[1]) / len(v)
        arm = lambda v: sum(1 for x in v if x[0] + x[1] <= 2) / len(v)
        return [gfh, gah, gfa, gaa, diff, abs(diff), (gfh + gaa) / 2 + (gfa + gah) / 2,
                rem(th), rem(ta), arm(th), arm(ta),
                (eh - ea) / 100, abs(eh - ea) / 100, bal, s_ok, mx, m_ok]

    def hz2(self, div, h, a, quoten=None, halb=2):
        """Merkmale fuer das HZ-Modell: Tore/Ergebnisse der Halbzeit (halb=1 oder 2) je Team, dazu Vollzeit, Elo, Quoten."""
        t1, t2 = list(self.h2[(div, h)]), list(self.h2[(div, a)])
        if len(t1) < 8 or len(t2) < 8:
            return None
        m = lambda q, f: sum(f(x) for x in q) / len(q)
        if halb == 1:
            f2 = lambda x: x[2]
            g2 = lambda x: x[3]
        else:
            f2 = lambda x: x[0] - x[2]
            g2 = lambda x: x[1] - x[3]
        out = [m(t1, f2), m(t1, g2), m(t2, f2), m(t2, g2),
               m(t1, lambda x: 1 if f2(x) > g2(x) else 0), m(t1, lambda x: 1 if f2(x) == g2(x) else 0),
               m(t2, lambda x: 1 if f2(x) > g2(x) else 0), m(t2, lambda x: 1 if f2(x) == g2(x) else 0),
               m(t1, lambda x: x[0]), m(t1, lambda x: x[1]), m(t2, lambda x: x[0]), m(t2, lambda x: x[1]),
               (self.elo[(div, h)] - self.elo[(div, a)]) / 100]
        mq = [0.0, 0.0, 0.0, 0.0]
        try:
            q = [float(quoten[k]) for k in ("1", "X", "2")]
            if min(q) > 1:
                inv = [1 / v for v in q]
                mq = [v / sum(inv) for v in inv] + [1.0]
        except Exception:
            pass
        return out + mq

    @staticmethod
    def ou_markt(quoten):
        try:
            o, u = float(quoten["Ü2.5"]), float(quoten["U2.5"])
            if o > 1 and u > 1:
                return [(1 / o) / (1 / o + 1 / u), 1.0]
        except Exception:
            pass
        return [0.0, 0.0]

    def folge(self, div, h, a):
        def f(q):
            v = list(q)
            pad = [(0.0, 0.0, 0.0)] * (SEQ - len(v))
            return pad + v
        return f(self.team[(div, h)]), f(self.team[(div, a)])

    def eintragen(self, div, h, a, hg, ag, hst=None, ast=None):
        self.team[(div, h)].append((hg, ag, 1.0))
        self.team[(div, a)].append((ag, hg, 0.0))
        self.liga[div].append((hg, ag))
        rh, ra = self.elo[(div, h)], self.elo[(div, a)]
        e = 1 / (1 + 10 ** ((ra - rh - 65) / 400))
        erg = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        self.elo[(div, h)] = rh + 20 * (erg - e)
        self.elo[(div, a)] = ra - 20 * (erg - e)
        if self._ht is not None:
            hh, ha = self._ht
            self.h2[(div, h)].append((hg, ag, hh, ha))
            self.h2[(div, a)].append((ag, hg, ha, hh))
        if hst is not None and ast is not None and hst == hst and ast == ast:
            self.sot[(div, h)].append((float(hst), float(ast)))
            self.sot[(div, a)].append((float(ast), float(hst)))


def durchlauf(H, ziele):
    """Liefert Trainingszeilen (Merkmale, Folgen, Tore) und je Ziel-Spiel die Merkmale zu dessen Datum."""
    global DL_LIGEN
    DL_LIGEN = []
    ziele = sorted(ziele, key=lambda z: z["datum"])
    V = Verlauf()
    X, S1, S2, Y, D = [], [], [], [], []
    RX, RY, RD = [], [], []
    HX, HY, HD = [], [], []
    H1X, H1Y = [], []
    TX, T1, T2 = [], [], []
    zt = {}
    zf, zr, zh, zh1 = {}, {}, {}, {}
    zi = 0
    for row in H.itertuples(index=False):
        d = row.Date.strftime("%Y-%m-%d")
        while zi < len(ziele) and ziele[zi]["datum"] <= d:
            z = ziele[zi]
            f = V.merkmale(z["liga"], z["heim"], z["gast"])
            if f:
                zf[z["key"]] = (f, *V.folge(z["liga"], z["heim"], z["gast"]))
            r = V.remis(z["liga"], z["heim"], z["gast"], z.get("quoten"))
            if r:
                zr[z["key"]] = r
            h2 = V.hz2(z["liga"], z["heim"], z["gast"], z.get("quoten"))
            if h2:
                zh[z["key"]] = h2
            h1 = V.hz2(z["liga"], z["heim"], z["gast"], z.get("quoten"), halb=1)
            if h1:
                zh1[z["key"]] = h1
            if h2 and h1:
                zt[z["key"]] = h1 + h2[:4] + V.ou_markt(z.get("quoten") or {})
            zi += 1
        ht_ok = row.HTHG == row.HTHG and row.HTAG == row.HTAG
        if ht_ok:
            h2 = V.hz2(row.Div, row.Home, row.Away, {"1": row.OD1, "X": row.ODX, "2": row.OD2})
            if h2:
                g2h, g2a = row.HG - row.HTHG, row.AG - row.HTAG
                HX.append(h2); HY.append(0 if g2h > g2a else (1 if g2h == g2a else 2)); HD.append(d)
                h1 = V.hz2(row.Div, row.Home, row.Away, {"1": row.OD1, "X": row.ODX, "2": row.OD2}, halb=1)
                H1X.append(h1); H1Y.append(0 if row.HTHG > row.HTAG else (1 if row.HTHG == row.HTAG else 2))
                TX.append(h1 + h2[:4] + V.ou_markt({"Ü2.5": row.OO, "U2.5": row.OU}))
                T1.append(1 if row.HTHG + row.HTAG >= 2 else 0); T2.append(1 if g2h + g2a >= 2 else 0)
        V._ht = (float(row.HTHG), float(row.HTAG)) if ht_ok else None
        r = V.remis(row.Div, row.Home, row.Away, {"1": row.OD1, "X": row.ODX, "2": row.OD2})
        if r:
            RX.append(r); RY.append(0 if row.HG > row.AG else (1 if row.HG == row.AG else 2)); RD.append(d)
        f = V.merkmale(row.Div, row.Home, row.Away)
        if f:
            s1, s2 = V.folge(row.Div, row.Home, row.Away)
            X.append(f); S1.append(s1); S2.append(s2); Y.append((row.HG, row.AG)); D.append(d); DL_LIGEN.append(row.Div)
        V.eintragen(row.Div, row.Home, row.Away, float(row.HG), float(row.AG), row.HST, row.AST)
    while zi < len(ziele):
        z = ziele[zi]
        f = V.merkmale(z["liga"], z["heim"], z["gast"])
        if f:
            zf[z["key"]] = (f, *V.folge(z["liga"], z["heim"], z["gast"]))
        r = V.remis(z["liga"], z["heim"], z["gast"], z.get("quoten"))
        if r:
            zr[z["key"]] = r
        h2 = V.hz2(z["liga"], z["heim"], z["gast"], z.get("quoten"))
        if h2:
            zh[z["key"]] = h2
        h1 = V.hz2(z["liga"], z["heim"], z["gast"], z.get("quoten"), halb=1)
        if h1:
            zh1[z["key"]] = h1
        if h2 and h1:
            zt[z["key"]] = h1 + h2[:4] + V.ou_markt(z.get("quoten") or {})
        zi += 1
    return (np.array(X), np.array(S1), np.array(S2), np.array(Y), np.array(D), zf,
            (np.array(RX), np.array(RY), np.array(RD), zr),
            (np.array(HX), np.array(HY), np.array(HD), zh),
            (np.array(H1X), np.array(H1Y), zh1),
            (np.array(TX), np.array(T1), np.array(T2), zt))


# ---------------- SVR ----------------
def mit_quoten(H):
    """Haengt 1X2-Quoten der Historie an (xgabora OddHome/OddDraw/OddAway). Ohne Quoten bleibt die Spalte leer."""
    import daten
    H = H.copy()
    try:
        q = pd.read_csv(daten._xgabora_url(), usecols=["Division", "MatchDate", "HomeTeam", "AwayTeam", "OddHome", "OddDraw", "OddAway", "Over25", "Under25"], low_memory=False)
        q.columns = ["Div", "Date", "Home", "Away", "OD1", "ODX", "OD2", "OO", "OU"]
        q["Date"] = pd.to_datetime(q["Date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
        H["Date"] = pd.to_datetime(H["Date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
        q = q.drop_duplicates(subset=["Div", "Date", "Home", "Away"])
        H = H.merge(q, on=["Div", "Date", "Home", "Away"], how="left")
    except Exception as ex:
        log("Quoten der Historie nicht geladen:", ex)
    for c in ("OD1", "ODX", "OD2", "OO", "OU"):
        if c not in H:
            H[c] = np.nan
    for c in ("HST", "AST"):
        H[c] = pd.to_numeric(H[c], errors="coerce")
    return H.sort_values("Date").reset_index(drop=True)


# ---------------- Remis-Korrektur je Liga aus den eigenen Erwartungen gelernt (kein fester Wert) ----------------
DL_LIGEN = []
K2_RHO = {}


def rho_tabelle(Y, LH, LA, ligen, D, grenze):
    """Je Liga: welches Dixon-Coles-rho passt die echten 0:0/1:0/0:1/1:1 zu den Modell-Erwartungen am besten.
    Nur Spiele der letzten 2 Jahre VOR der Grenze. Ligen mit zu wenig Spielen nehmen den Wert ueber alle Ligen."""
    import kern
    Y, LH, LA, ligen, D = np.asarray(Y, float), np.asarray(LH, float), np.asarray(LA, float), np.asarray(ligen), np.asarray(D)
    von = (pd.Timestamp(grenze) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    m = (D >= von) & (D < grenze)
    tage = (pd.Timestamp(grenze) - pd.to_datetime(pd.Series(D[m].astype(str)))).dt.days.to_numpy()
    w = np.exp(-0.003 * tage)
    tab = {"_alle": kern.lerne_rho(Y[m, 0], Y[m, 1], LH[m], LA[m], w)}
    for lg in set(ligen[m]):
        k = ligen[m] == lg
        if k.sum() >= 300:
            tab[lg] = kern.lerne_rho(Y[m][k, 0], Y[m][k, 1], LH[m][k], LA[m][k], w[k])
    return tab


def rho_von(modell, liga):
    tab = getattr(modell, "rho", None) or {}
    return tab.get(liga, tab.get("_alle", 0.0))


def svr_modell(X, Y, D, grenze):
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    ok = D < grenze
    Xt, Yt = X[ok][-N_SVR:], Y[ok][-N_SVR:]
    sk = StandardScaler().fit(Xt)
    mh = SVR(kernel="rbf", C=1.0, epsilon=0.1).fit(sk.transform(Xt), Yt[:, 0])
    ma = SVR(kernel="rbf", C=1.0, epsilon=0.1).fit(sk.transform(Xt), Yt[:, 1])
    log(f"SVR trainiert mit {len(Xt)} Spielen (bis {grenze})")
    fn = lambda f: (float(mh.predict(sk.transform([f]))[0]), float(ma.predict(sk.transform([f]))[0]))
    try:
        Lg = np.asarray(DL_LIGEN)[ok][-N_SVR:]
        Z = sk.transform(Xt)
        fn.rho = rho_tabelle(Yt, np.clip(mh.predict(Z), 0.05, 6), np.clip(ma.predict(Z), 0.05, 6), Lg, D[ok][-N_SVR:], grenze)
        log(f"SVR Remis-Korrektur gelernt: {fn.rho.get('_alle', 0):+.3f} (alle Ligen), {len(fn.rho) - 1} Ligen einzeln")
    except Exception as ex:
        log("SVR Remis-Korrektur nicht moeglich:", ex)
    return fn


# ---------------- REMIS-RADAR ----------------
N_REMIS = 40000


def remis_modell(RX, RY, RD, grenze):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ok = RD < grenze
    Xt, Yt = RX[ok][-N_REMIS:], RY[ok][-N_REMIS:]
    sk = StandardScaler().fit(Xt)
    lr = LogisticRegression(max_iter=1000).fit(sk.transform(Xt), Yt)
    log(f"Remis-Radar trainiert mit {len(Xt)} Spielen (bis {grenze}), Remis-Anteil {np.mean(Yt == 1):.1%}")
    def vorhersage(r):
        p = lr.predict_proba(sk.transform([r]))[0]
        return {"p1": round(float(p[0]) * 100, 2), "px": round(float(p[1]) * 100, 2), "p2": round(float(p[2]) * 100, 2)}
    return vorhersage


# ---------------- 2.-HZ-MODELL ----------------
def h2_modell(HX, HY, HD, grenze, schluessel="ht2", name="2.-HZ-Modell"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ok = HD < grenze
    Xt, Yt = HX[ok], HY[ok]
    sk = StandardScaler().fit(Xt)
    lr = LogisticRegression(max_iter=2000).fit(sk.transform(Xt), Yt)
    log(f"{name} trainiert mit {len(Xt)} Spielen (bis {grenze})")
    def vorhersage(f):
        p = [round(float(x) * 100, 2) for x in lr.predict_proba(sk.transform([f]))[0]]
        return {"p1": p[0], "px": p[1], "p2": p[2], schluessel: p}
    return vorhersage


# ---------------- TORE JE HALBZEIT: 0-1 oder 2+ ----------------
def tor_modell(TX, T, HD, grenze, name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ok = HD < grenze
    sk = StandardScaler().fit(TX[ok])
    lr = LogisticRegression(max_iter=3000).fit(sk.transform(TX[ok]), T[ok])
    log(f"{name} trainiert mit {ok.sum()} Spielen (2+ Tore: {T[ok].mean():.1%})")
    return lambda f: round(float(lr.predict_proba(sk.transform([f]))[0][1]) * 100, 2)


# ---------------- TCN ----------------
def tcn_modell(X, S1, S2, Y, D, grenze):
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    np.random.seed(0)
    ok = D < grenze
    Xt, A, B, Yt = X[ok][-N_TCN:], S1[ok][-N_TCN:], S2[ok][-N_TCN:], Y[ok][-N_TCN:]
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6

    class Kausal(nn.Module):
        def __init__(self, cin, cout, dil):
            super().__init__()
            self.pad = 2 * dil
            self.c = nn.Conv1d(cin, cout, 3, dilation=dil)
            self.r = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):
            y = self.c(nn.functional.pad(x, (self.pad, 0)))
            return torch.relu(y) + self.r(x)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(Kausal(3, 16, 1), Kausal(16, 16, 2), Kausal(16, 16, 4))
            self.kopf = nn.Sequential(nn.Linear(16 * 2 + 10, 32), nn.ReLU(), nn.Linear(32, 2))

        def forward(self, s1, s2, f):
            e1 = self.enc(s1)[:, :, -1]
            e2 = self.enc(s2)[:, :, -1]
            return nn.functional.softplus(self.kopf(torch.cat([e1, e2, f], 1))) + 0.05

    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    s1, s2 = t(A).permute(0, 2, 1), t(B).permute(0, 2, 1)
    f, y = t((Xt - mu) / sd), t(Yt)
    net = TCN()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(y)
    for ep in range(8):
        perm = torch.randperm(n)
        for k in range(0, n, 256):
            b = perm[k:k + 256]
            lam = net(s1[b], s2[b], f[b])
            loss = (lam - y[b] * torch.log(lam)).mean()   # Poisson-Verlust
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    log(f"TCN trainiert mit {n} Spielen (bis {grenze})")

    def vorhersage(fz, q1, q2):
        with torch.no_grad():
            lam = net(t([q1]).permute(0, 2, 1), t([q2]).permute(0, 2, 1), t([(np.array(fz) - mu) / sd]))[0]
        return float(lam[0]), float(lam[1])
    try:
        with torch.no_grad():
            L = torch.cat([net(s1[k:k + 2048], s2[k:k + 2048], f[k:k + 2048]) for k in range(0, n, 2048)]).numpy()
        Lg = np.asarray(DL_LIGEN)[ok][-N_TCN:]
        vorhersage.rho = rho_tabelle(Yt, L[:, 0], L[:, 1], Lg, D[ok][-N_TCN:], grenze)
        log(f"TCN Remis-Korrektur gelernt: {vorhersage.rho.get('_alle', 0):+.3f} (alle Ligen)")
    except Exception as ex:
        log("TCN Remis-Korrektur nicht moeglich:", ex)
    return vorhersage


# ---------------- Konsens 2: fuenf Modelle (Saison-Historie der Liga) ----------------
K = 8


def team_stat(hist, name):
    m = []
    for e in hist:
        if e["heim"] == name:
            m.append((e["ft"][0], e["ft"][1], e["ht"][0] if e["ht"] else None, e["ft"][0] > e["ft"][1], e["ft"][0] == e["ft"][1]))
        elif e["gast"] == name:
            m.append((e["ft"][1], e["ft"][0], e["ht"][1] if e["ht"] else None, e["ft"][1] > e["ft"][0], e["ft"][1] == e["ft"][0]))
    n = len(m)
    if n < 2:
        return None
    hz = [x[2] for x in m if x[2] is not None]
    return dict(n=n, off=sum(x[0] for x in m) / n, de=sum(x[1] for x in m) / n,
                hz1=(sum(hz) / len(hz)) if hz else None, W=sum(x[3] for x in m), D=sum(x[4] for x in m))


def liga_kennz(hist):
    n = len(hist)
    tg = sum(e["ft"][0] + e["ft"][1] for e in hist)
    mh = sum(e["ft"][0] for e in hist) / n
    ma = sum(e["ft"][1] for e in hist) / n
    mht = [e for e in hist if e["ht"]]
    share = (sum(e["ht"][0] + e["ht"][1] for e in mht) / max(1, sum(e["ft"][0] + e["ft"][1] for e in mht))) if mht else 0.44
    return dict(mu=tg / (2 * n), hf=math.sqrt(mh / ma) if ma > 0 else 1.0, share=share if 0.2 < share < 0.7 else 0.44)


def lam_korr(A, B, L):
    att_a = (A["n"] * (A["off"] / L["mu"]) + K) / (A["n"] + K)
    att_b = (B["n"] * (B["off"] / L["mu"]) + K) / (B["n"] + K)
    def_a = (A["n"] * (A["de"] / L["mu"]) + K) / (A["n"] + K)
    def_b = (B["n"] * (B["de"] / L["mu"]) + K) / (B["n"] + K)
    return att_a * def_b * L["mu"] * L["hf"], att_b * def_a * L["mu"] / L["hf"]


def lam_rating(A, B):
    clip = lambda x: max(0.0, min(100.0, x))
    def rating(T, td):
        return 0.40 * clip(T["off"] / 8 * 100) + 0.35 * clip((8 - T["de"]) / 7.5 * 100) + 0.25 * clip(((td / T["n"]) + 2) / 4 * 100)
    tda, tdb = (A["off"] - A["de"]) * A["n"], (B["off"] - B["de"]) * B["n"]
    ra, rb = rating(A, tda), rating(B, tdb)
    def w(T):
        tot = T["off"] + T["de"]
        wo = 0.40 + (T["off"] / tot if tot > 0 else 0.5) * 0.30
        return wo, 1 - wo
    woa, wda = w(A)
    wob, wdb = w(B)
    la = (A["off"] * woa + B["de"] * wda) * (1 + tda / 40) * (1 + (ra - rb) / 100 * 0.15)
    lb = (B["off"] * wob + A["de"] * wdb) * (1 + tdb / 40) * (1 + (rb - ra) / 100 * 0.15)
    return la, lb


def lam_hz(A, B):
    def lam(T):
        if T["hz1"] is None:
            return T["off"]
        r1, r2 = T["hz1"], T["off"] - T["hz1"]
        diff = (abs(r2 - r1) / r1) if r1 > 0 else (1.0 if r2 > 0 else 0)
        if diff < 0.3:
            w1 = 0.5
        else:
            w1 = T["hz1"] / T["off"] if T["off"] else 0.5
        return (w1 * r1 + (1 - w1) * r2) * 2
    return lam(A), lam(B)


def lam_reines(A, B):
    avg = max(0.1, (A["off"] + B["off"]) / 2)
    return A["off"] * (B["de"] / avg), B["off"] * (A["de"] / avg)


def lam_xg(Ax, Bx):
    fa, ga = Ax
    fb, gb = Bx
    d = (fa - ga) - (fb - gb)
    return (fa + gb) / 2 * max(0.8, min(1.2, 1 + d * 0.12)), (fb + ga) / 2 * max(0.8, min(1.2, 1 - d * 0.12))


def _ascii(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()


def lade_xg():
    """xG je Liga aus dem SportsAPI-Archiv (nur lesen). Namen werden unscharf zugeordnet."""
    try:
        a = json.loads(gzip.open(ARCHIV, "rt", encoding="utf-8").read())
    except Exception:
        return {}
    je = defaultdict(list)
    for e in a.get("spiele", {}).values():
        x = (e.get("stat") or {}).get("ALL", {}).get("expectedGoals")
        if e.get("fertig") and x and e.get("datum"):
            je[e["liga"]].append((e["datum"], e["heim"], e["gast"], x[0], x[1]))
    return je


def xg_team(xg_liga, name, vor):
    namen = {}
    for d, h, g, *_ in xg_liga:
        namen.setdefault(_ascii(h), h)
        namen.setdefault(_ascii(g), g)
    z = _ascii(name)
    if z not in namen:
        best = difflib.get_close_matches(z, list(namen), n=1, cutoff=0.6)
        if not best:
            return None
        z = best[0]
    echt = namen[z]
    f, g = [], []
    for d, h, a, xh, xa in xg_liga:
        if d >= vor:
            continue
        if h == echt:
            f.append(xh); g.append(xa)
        elif a == echt:
            f.append(xa); g.append(xh)
    if len(f) < 2:
        return None
    return sum(f) / len(f), sum(g) / len(g)


K2_LAM = {"korr": lambda A, B, L: lam_korr(A, B, L), "rating": lambda A, B, L: lam_rating(A, B),
          "hz": lambda A, B, L: lam_hz(A, B), "reines": lambda A, B, L: lam_reines(A, B)}


def k2_rho_lernen(H, bis):
    """Konsens 2: Remis-Korrektur je Variante und Liga aus den letzten 2 Jahren gelernt (Saison-Historie wie live)."""
    import kern
    global K2_RHO
    von = pd.Timestamp(bis) - pd.DateOffset(years=2)
    d = H[(H["Date"] >= von) & (H["Date"] < pd.Timestamp(bis))].dropna(subset=["HG", "AG"])
    samm = {v: defaultdict(list) for v in K2_LAM}
    for liga, g in d.groupby("Div"):
        saison = defaultdict(list)
        for r in g.sort_values("Date").itertuples(index=False):
            sj = r.Date.year if r.Date.month >= 7 else r.Date.year - 1
            hist = saison[sj]
            if len(hist) >= 15:
                A, B = team_stat(hist, r.Home), team_stat(hist, r.Away)
                if A and B:
                    L = liga_kennz(hist)
                    for v, fn in K2_LAM.items():
                        try:
                            lh, la = fn(A, B, L)
                            samm[v][liga].append((float(r.HG), float(r.AG), max(lh, 0.05), max(la, 0.05), r.Date))
                        except Exception:
                            pass
            ht = [int(r.HTHG), int(r.HTAG)] if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)) else None
            hist.append({"datum": r.Date.strftime("%Y-%m-%d"), "heim": r.Home, "gast": r.Away, "ft": [int(r.HG), int(r.AG)], "ht": ht})
    K2_RHO = {}
    for v in K2_LAM:
        alle = [x for l in samm[v].values() for x in l]
        tab = {"_alle": kern.rho_aus_sammlung(alle, bis)}
        for liga, l in samm[v].items():
            if len(l) >= 300:
                tab[liga] = kern.rho_aus_sammlung(l, bis)
        K2_RHO[v] = tab
    log("Konsens 2 Remis-Korrektur gelernt: " + ", ".join(f"{v} {K2_RHO[v]['_alle']:+.3f}" for v in K2_RHO))


def k2_rho(v, liga):
    tab = K2_RHO.get(v, {})
    return tab.get(liga, tab.get("_alle", 0.0))


def konsens2(s, saison_liga, xg_liga):
    hist = [e for e in saison_liga if e["datum"] < s["datum"]]
    if len(hist) < 15:
        return {}
    A, B = team_stat(hist, s["heim"]), team_stat(hist, s["gast"])
    if not A or not B:
        return {}
    L = liga_kennz(hist)
    out = {
        "k2_korr": maerkte(*lam_korr(A, B, L), L["share"], rho=k2_rho("korr", s["liga"])),
        "k2_rating": maerkte(*lam_rating(A, B), L["share"], rho=k2_rho("rating", s["liga"])),
        "k2_hz": maerkte(*lam_hz(A, B), L["share"], rho=k2_rho("hz", s["liga"])),
        "k2_reines": maerkte(*lam_reines(A, B), L["share"], rho=k2_rho("reines", s["liga"])),
    }
    Ax = xg_team(xg_liga, s["heim"], s["datum"]) if xg_liga else None
    Bx = xg_team(xg_liga, s["gast"], s["datum"]) if xg_liga else None
    if Ax and Bx:
        out["k2_xg"] = maerkte(*lam_xg(Ax, Bx), L["share"], rho=k2_rho("korr", s["liga"]))
    return out


# ---------------- MASTER: Rueckblick-Test (nur Bilanz, aendert keine Prognose) ----------------
RUECKBLICK_TAGE = 200


def rueckblick(H, share, xg, log_=None):
    """Modelle so trainieren, als waere heute vor RUECKBLICK_TAGE Tagen, und die Spiele danach tippen lassen.
    Gezaehlt wird je Quelle und Markt (gleiche Regeln wie der MASTER auf der Seite)."""
    import master_gewichte as MG
    ende = H["Date"].max()
    start = ende - pd.Timedelta(days=RUECKBLICK_TAGE)
    T = H[(H["Date"] >= start)].copy()
    ziele = []
    for i, r in enumerate(T.itertuples(index=False)):
        ziele.append({"key": i, "liga": r.Div, "datum": r.Date.strftime("%Y-%m-%d"), "heim": r.Home, "gast": r.Away,
                      "quoten": {"1": r.OD1, "X": r.ODX, "2": r.OD2, "Ü2.5": r.OO, "U2.5": r.OU},
                      "erg": (int(r.HG), int(r.AG), None if r.HTHG != r.HTHG else int(r.HTHG), None if r.HTAG != r.HTAG else int(r.HTAG))})
    g2 = start.strftime("%Y-%m-%d")
    X, S1, S2, Y, D, zf, _r, (HX, HY, HD, zh), (H1X, H1Y, zh1), (TX, T1, T2, zt) = durchlauf(H, ziele)
    svr = tcn = h1m = h2m = None
    try:
        svr = svr_modell(X, Y, D, g2)
    except Exception as ex:
        log("Rueckblick SVR:", ex)
    try:
        tcn = tcn_modell(X, S1, S2, Y, D, g2)
    except Exception as ex:
        log("Rueckblick TCN:", ex)
    try:
        h1m = h2_modell(H1X, H1Y, HD, g2, "ht", "1.-HZ-Modell")
    except Exception as ex:
        log("Rueckblick 1.-HZ:", ex)
    try:
        h2m = h2_modell(HX, HY, HD, g2)
    except Exception as ex:
        log("Rueckblick 2.-HZ:", ex)
    try:
        k2_rho_lernen(H, g2)
    except Exception as ex:
        log("Rueckblick K2-Remis-Korrektur:", ex)
    # Konsens 2: je Spiel nur Spiele derselben Saison vor dem Datum (wie live)
    sai = defaultdict(list)
    for r in H.itertuples(index=False):
        ht = [int(r.HTHG), int(r.HTAG)] if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)) else None
        sai[r.Div].append({"datum": r.Date.strftime("%Y-%m-%d"), "heim": r.Home, "gast": r.Away, "ft": [int(r.HG), int(r.AG)], "ht": ht})
    stats = {}
    for z in ziele:
        k, sh = z["key"], share.get(z["liga"], 0.44)
        try:
            if k in zf:
                f, q1, q2 = zf[k]
                if svr:
                    MG.zaehlen(stats, "svr", maerkte(*svr(f), sh, rho=rho_von(svr, z["liga"])), z["erg"])
                if tcn:
                    MG.zaehlen(stats, "tcn", maerkte(*tcn(f, q1, q2), sh, rho=rho_von(tcn, z["liga"])), z["erg"])
            if h1m and k in zh1:
                MG.zaehlen(stats, "h1m", {"ht": h1m(zh1[k])["ht"]}, z["erg"])
            if h2m and k in zh:
                MG.zaehlen(stats, "h2m", {"ht2": h2m(zh[k])["ht2"]}, z["erg"])
            y = int(z["datum"][:4]) if int(z["datum"][5:7]) >= 7 else int(z["datum"][:4]) - 1
            sstart = f"{y}-07-01"
            hist = [e for e in sai.get(z["liga"], []) if sstart <= e["datum"] < z["datum"]]
            k2 = konsens2(z, hist, xg.get(z["liga"], []))
            for q, pp in (k2 or {}).items():
                MG.zaehlen(stats, q, pp, z["erg"])
        except Exception:
            continue
    log("Rueckblick fuer MASTER:", ", ".join(f"{q} {s.get('x1', s.get('h1x', s.get('h2x', [0])))[0]}" for q, s in stats.items()))
    return stats


# ---------------- Ablauf ----------------
def main():
    d = json.load(open(PFAD, encoding="utf-8"))
    spiele = d.get("spiele", [])
    if not spiele:
        log("keine Spiele in daten.json")
        return
    import daten
    H = daten.alle_spiele()
    H = H[H["Date"] >= pd.Timestamp(AB)].copy()
    H["HTHG"] = pd.to_numeric(H["HTHG"], errors="coerce")
    H["HTAG"] = pd.to_numeric(H["HTAG"], errors="coerce")
    H = mit_quoten(H)
    log(f"Historie: {len(H)} Spiele, davon {H['ODX'].notna().sum()} mit Remis-Quote")
    ziele = [{"key": i, "liga": s["liga"], "datum": s["datum"], "heim": s["heim"], "gast": s["gast"], "quoten": s.get("quoten")} for i, s in enumerate(spiele)]
    grenze = min(s["datum"] for s in spiele)
    X, S1, S2, Y, D, zf, (RX, RY, RD, zr), (HX, HY, HD, zh), (H1X, H1Y, zh1), (TX, T1, T2, zt) = durchlauf(H, ziele)
    log(f"{len(X)} Trainingszeilen, Merkmale fuer {len(zf)} von {len(spiele)} Spielen")

    # Halbzeit-Anteil je Liga (fuer HZ-Maerkte von SVR/TCN)
    share = {}
    for div, g in H.dropna(subset=["HTHG", "HTAG"]).groupby("Div"):
        tot = (g["HG"] + g["AG"]).sum()
        share[div] = float((g["HTHG"] + g["HTAG"]).sum() / tot) if tot else 0.44

    svr = tcn = remis = h2m = h1m = t1m = t2m = None
    try:
        t1m = tor_modell(TX, T1, HD, grenze, "Tore 1. HZ")
        t2m = tor_modell(TX, T2, HD, grenze, "Tore 2. HZ")
    except Exception as ex:
        log("Tor-Modell nicht moeglich:", ex)
    try:
        h1m = h2_modell(H1X, H1Y, HD, grenze, "ht", "1.-HZ-Modell")
    except Exception as ex:
        log("1.-HZ-Modell nicht moeglich:", ex)
    try:
        h2m = h2_modell(HX, HY, HD, grenze)
    except Exception as ex:
        log("2.-HZ-Modell nicht moeglich:", ex)
    try:
        remis = remis_modell(RX, RY, RD, grenze)
    except Exception as ex:
        log("Remis-Radar nicht moeglich:", ex)
    try:
        svr = svr_modell(X, Y, D, grenze)
    except Exception as ex:
        log("SVR nicht moeglich:", ex)
    try:
        tcn = tcn_modell(X, S1, S2, Y, D, grenze)
    except Exception as ex:
        log("TCN nicht moeglich:", ex)

    try:
        k2_rho_lernen(H, grenze)
    except Exception as ex:
        log("K2-Remis-Korrektur nicht moeglich:", ex)
    # Saison-Historie je Liga fuer Konsens 2
    sai = H[H["Date"] >= pd.Timestamp(SAISON_START)]
    saison = defaultdict(list)
    for r in sai.itertuples(index=False):
        ht = [int(r.HTHG), int(r.HTAG)] if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)) else None
        saison[r.Div].append({"datum": r.Date.strftime("%Y-%m-%d"), "heim": r.Home, "gast": r.Away,
                              "ft": [int(r.HG), int(r.AG)], "ht": ht})
    xg = lade_xg()

    n_svr = n_tcn = n_k2 = n_rem = n_h2 = n_h1 = 0
    for i, s in enumerate(spiele):
        p = s.setdefault("prog", {})
        sh = share.get(s["liga"], 0.44)
        if i in zf:
            f, q1, q2 = zf[i]
            if svr:
                try:
                    p["svr"] = maerkte(*svr(f), sh, rho=rho_von(svr, s["liga"]))
                    n_svr += 1
                except Exception:
                    pass
            if tcn:
                try:
                    p["tcn"] = maerkte(*tcn(f, q1, q2), sh, rho=rho_von(tcn, s["liga"]))
                    n_tcn += 1
                except Exception:
                    pass
        if h1m and i in zh1:
            try:
                p["h1m"] = h1m(zh1[i])
                if t1m and i in zt:
                    p["h1m"]["t2p"] = t1m(zt[i])
                n_h1 += 1
            except Exception:
                pass
        if h2m and i in zh:
            try:
                p["h2m"] = h2m(zh[i])
                if t2m and i in zt:
                    p["h2m"]["t2p"] = t2m(zt[i])
                n_h2 += 1
            except Exception:
                pass
        if remis and i in zr:
            try:
                p["remis"] = remis(zr[i])
                n_rem += 1
            except Exception:
                pass
        try:
            k2 = konsens2(s, saison.get(s["liga"], []), xg.get(s["liga"], []))
            if k2:
                p.update(k2)
                n_k2 += 1
        except Exception:
            pass

    try:
        import master_gewichte
        master_gewichte.eintragen(d, rueckblick(H, share, xg))
    except Exception as ex:
        log("Rueckblick fuer MASTER nicht moeglich:", ex)

    with open(PFAD, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
    log(f"SVR {n_svr} · TCN {n_tcn} · Konsens 2 {n_k2} · Remis-Radar {n_rem} · 1.-HZ-Modell {n_h1} · 2.-HZ-Modell {n_h2} von {len(spiele)} Spielen")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log("Fehler (daten.json unveraendert):", ex)
    sys.exit(0)
