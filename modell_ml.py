# ===== MODELL: ML · Gradient Boosting (Engine 5) =====
# Gradient Boosting (scikit-learn HistGradientBoosting) mit POISSON-Zielfunktion:
# Das Modell lernt direkt die erwartete Torzahl (Lambda), daraus entsteht eine gueltige
# Ergebnis-Matrix (keine erfundenen Wahrscheinlichkeiten aus Klassen-Werten).
#
# Merkmale - alle nur aus Spielen VOR dem jeweiligen Spiel berechnet (kein Blick in die Zukunft):
#  - eigene Elo-Werte (Torunterschied-gewichtet), Ruhetage, Anzahl bisheriger Spiele
#  - Form als gleitende Mittelwerte (langsam UND schnell, das Modell lernt selbst, was zaehlt):
#    Tore, Halbzeittore, Schuesse aufs Tor, Schuesse, Ecken - jeweils erzielt und zugelassen
#  - Heim/Auswaerts getrennt: Angreifer-Form aus seinen Heim- bzw. Auswaertsspielen,
#    Verteidiger-Form aus den Spielen am jeweils anderen Ort, dazu die Gesamtform
#  - Liga-Torniveau
#  - KEINE Wettquoten (sonst wuerde das Modell nur den Markt nachahmen und Value-Tipps verfaelschen)
# Training: je Spiel zwei Zeilen (Heimteam greift an / Gastteam greift an, mit Heim-Merkmal).
# Vollzeit und Halbzeit: zwei getrennte Modelle (Halbzeit auf echten Halbzeitstaenden).
# Zeitlich korrektes Tuning: Anzahl Boosting-Runden per Early Stopping auf den letzten 20 % der
# Trainingsdaten (chronologisch), dann Neutraining auf allen Daten. Neu trainiert einmal pro Monat,
# die Merkmale (Form, Elo) werden aber vor jeder Prognose aktualisiert.
# Abhaengigkeit Heim/Gast aus den Vorhersagen auf dem zeitlich spaeteren Pruefabschnitt.
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import kern

NAME = "ML · Gradient Boosting"
KURZ = "ML"
ICON = "ML"
FARBE1 = "#e6ff4f"
FARBE2 = "#7bd81f"
BESCHREIBUNG = "Lernt aus Elo, Form, Schüssen und Ecken (Heim/Auswärts getrennt) die Torverteilung"

TRAIN_JAHRE = 6
VORLAUF_JAHRE = 2
ELO_K, ELO_HEIM = 20.0, 60.0
LIGA_ALPHA = 0.02
GBM = dict(loss="poisson", learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=60,
           l2_regularization=1.0, random_state=0)
MAX_RUNDEN = 500
# (Statistik, Heim-Spalte, Gast-Spalte, Geschwindigkeiten der gleitenden Mittelwerte)
STATS = [("g", "HG", "AG", (0.08, 0.3)), ("ht", "HTHG", "HTAG", (0.08, 0.3)),
         ("sot", "HST", "AST", (0.1,)), ("sh", "HS", "AS", (0.1,)), ("co", "HC", "AC", (0.1,))]
SLOTS = [(s, seite, a) for s, _, _, al in STATS for seite in ("f", "a") for a in al]   # f=erzielt, a=zugelassen
SLOT_SPALTEN = [[(x[1], x[2]) for x in STATS if x[0] == s][0] for s, _, _ in SLOTS]
L = len(SLOTS)
CTX = {"h": 0, "a": 1, "all": 2}
_FEAT, _MODELLE = {}, {}


def _tag(ts):
    return int(np.datetime64(pd.Timestamp(ts).date(), "D").astype(np.int64))


class _Merkmale:
    def __init__(self, start):
        self.start = start
        self.form, self.elo, self.last, self.n, self.liga_von = {}, {}, {}, {}, {}
        self.liga = {}           # div -> {stat: [heim_schnitt, gast_schnitt]}
        self.X, self.Y, self.T = [], [], []    # je Spiel: Merkmale Heim-Angriff / Gast-Angriff, Ziele, Tag
        self.n_zeilen, self.bis = 0, None

    def kopie(self):
        k = _Merkmale.__new__(_Merkmale)
        k.start = self.start
        k.form = {t: v.copy() for t, v in self.form.items()}
        k.elo, k.last, k.n, k.liga_von = dict(self.elo), dict(self.last), dict(self.n), dict(self.liga_von)
        k.liga = {d: {s: list(v) for s, v in x.items()} for d, x in self.liga.items()}
        k.X, k.Y, k.T, k.n_zeilen, k.bis = self.X, self.Y, self.T, self.n_zeilen, self.bis
        return k

    def _liga(self, div):
        if div not in self.liga:
            self.liga[div] = {"g": [1.5, 1.15], "ht": [0.68, 0.52], "sot": [np.nan, np.nan], "sh": [np.nan, np.nan], "co": [np.nan, np.nan]}
        return self.liga[div]

    def _team(self, t, div, tag):
        if t not in self.form:
            lg = self._liga(div)
            v = np.full((3, L), np.nan)
            for k, (s, seite, _) in enumerate(SLOTS):
                h, a = lg[s]
                v[0, k] = h if seite == "f" else a          # zu Hause: erzielt ~ Heimschnitt
                v[1, k] = a if seite == "f" else h          # auswaerts: erzielt ~ Gastschnitt
                v[2, k] = (h + a) / 2 if not (np.isnan(h) or np.isnan(a)) else np.nan
            self.form[t] = v
            andere = [self.elo[x] for x, d in self.liga_von.items() if d == div and tag - self.last.get(x, tag) < 365]
            self.elo[t] = float(np.mean(andere)) if andere else 1500.0
            self.n[t] = 0
        self.liga_von[t] = div

    def zeile(self, angr, vert, heim, div, tag):
        lg = self._liga(div)
        ca, cv = ("h", "a") if heim else ("a", "h")
        ea, ev = self.elo[angr], self.elo[vert]
        ruhe = lambda t: min(tag - self.last[t], 30) if t in self.last else np.nan
        kopf = [ea, ev, ea - ev + (ELO_HEIM if heim else -ELO_HEIM), ruhe(angr), ruhe(vert),
                min(self.n[angr], 60), min(self.n[vert], 60), float(heim),
                lg["g"][0 if heim else 1], lg["g"][1 if heim else 0], lg["ht"][0 if heim else 1], lg["ht"][1 if heim else 0]]
        f = self.form
        return np.concatenate([kopf, f[angr][CTX[ca]], f[angr][CTX["all"]], f[vert][CTX[cv]], f[vert][CTX["all"]]])

    def schritt(self, tag, div, h, a, werte):
        self._team(h, div, tag); self._team(a, div, tag)
        self.X.append((self.zeile(h, a, True, div, tag), self.zeile(a, h, False, div, tag)))
        self.Y.append((werte["HG"], werte["AG"], werte["HTHG"], werte["HTAG"]))
        self.T.append(tag)
        # --- nach dem Spiel aktualisieren ---
        lg = self._liga(div)
        for s, sh, sa, _ in STATS:
            xh, xa = werte[sh], werte[sa]
            if np.isnan(xh) or np.isnan(xa):
                continue
            for k, (x, i) in enumerate(((xh, 0), (xa, 1))):
                lg[s][i] = x if np.isnan(lg[s][i]) else lg[s][i] + LIGA_ALPHA * (x - lg[s][i])
        for t, ort, f_wert, g_wert in ((h, 0, "h", "a"), (a, 1, "a", "h")):
            v = self.form[t]
            for k, (s, seite, al) in enumerate(SLOTS):
                sh, sa = SLOT_SPALTEN[k]
                eigen, fremd = (werte[sh], werte[sa]) if ort == 0 else (werte[sa], werte[sh])
                x = eigen if seite == "f" else fremd
                if np.isnan(x):
                    continue
                for c in (ort, 2):
                    v[c, k] = x if np.isnan(v[c, k]) else v[c, k] + al * (x - v[c, k])
        # Elo
        eh, ea = self.elo[h], self.elo[a]
        erw = 1 / (1 + 10 ** ((ea - eh - ELO_HEIM) / 400))
        gd = werte["HG"] - werte["AG"]
        erg = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        delta = ELO_K * (np.log(abs(gd) + 1) + 1) * (erg - erw)
        self.elo[h], self.elo[a] = eh + delta, ea - delta
        self.last[h] = self.last[a] = tag
        self.n[h] += 1; self.n[a] += 1

    def lauf(self, d):
        cols = {c: d[c].to_numpy(float) if c in d else np.full(len(d), np.nan)
                for c in ("HG", "AG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC")}
        tage = d["Date"].to_numpy("datetime64[D]").astype(np.int64)
        for r, (t, dv, h, a) in enumerate(zip(tage, d["Div"].to_numpy(), d["Home"].to_numpy(), d["Away"].to_numpy())):
            self.schritt(int(t), dv, h, a, {c: v[r] for c, v in cols.items()})
        self.n_zeilen += len(d)
        if len(d):
            self.bis = d["Date"].max()


def _merkmale_bis(df, heute):
    key = tuple(sorted(df["Div"].unique()))
    m = _FEAT.get(key)
    if m is None:
        start = heute - pd.DateOffset(years=TRAIN_JAHRE + VORLAUF_JAHRE)
    else:
        start = m.start
    d = df[(df["Date"] >= start) & (df["Date"] < heute)].sort_values(["Date", "Home"], kind="mergesort")
    if m is None or m.bis is None or (d["Date"] <= m.bis).sum() != m.n_zeilen or (len(d) and d["Date"].max() < m.bis):
        m = _Merkmale(start)
        m.lauf(d)
    else:
        m.lauf(d[d["Date"] > m.bis])
    _FEAT[key] = m
    return m


def _deviance(y, mu):
    mu = np.clip(mu, 1e-6, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (t - (y - mu))))


def _trainieren(m, heute, ziel):
    grenze = _tag(heute - pd.DateOffset(years=TRAIN_JAHRE))
    T = np.array(m.T)
    sel = np.where((T >= grenze) & (T < _tag(heute)))[0]
    yi = (0, 1) if ziel == "ft" else (2, 3)
    Y = np.array(m.Y)[sel]
    gueltig = ~np.isnan(Y[:, yi[0]]) & ~np.isnan(Y[:, yi[1]])
    sel, Y = sel[gueltig], Y[gueltig]
    if len(sel) < 300:
        return None
    Xh = np.array([m.X[k][0] for k in sel]); Xa = np.array([m.X[k][1] for k in sel])
    spalten = ~np.all(np.isnan(np.vstack([Xh, Xa])), axis=0)   # komplett leere Merkmale (z. B. keine Schussdaten) weglassen
    Xh, Xa = Xh[:, spalten], Xa[:, spalten]
    yh, ya = Y[:, yi[0]], Y[:, yi[1]]
    cut = int(len(sel) * 0.8)
    X_tr, y_tr = np.vstack([Xh[:cut], Xa[:cut]]), np.concatenate([yh[:cut], ya[:cut]])
    probe = HistGradientBoostingRegressor(max_iter=MAX_RUNDEN, **GBM).fit(X_tr, y_tr)
    ph = list(probe.staged_predict(Xh[cut:])); pa = list(probe.staged_predict(Xa[cut:]))
    dev = [_deviance(np.concatenate([yh[cut:], ya[cut:]]), np.concatenate([a, b])) for a, b in zip(ph, pa)]
    runden = int(np.argmin(dev)) + 1
    abh = kern.waehle_abhaengigkeit(yh[cut:], ya[cut:], ph[runden - 1], pa[runden - 1], np.ones(len(sel) - cut))
    voll = HistGradientBoostingRegressor(max_iter=max(runden, 20), **GBM).fit(np.vstack([Xh, Xa]), np.concatenate([yh, ya]))
    return {"gbm": voll, "abh": abh, "runden": runden, "dev": dev[runden - 1], "spalten": spalten}


def trainiere(df, heute):
    feat = _merkmale_bis(df, heute)
    key = (tuple(sorted(df["Div"].unique())), heute.year, heute.month)
    mods = _MODELLE.get(key)
    if mods is None:
        mods = {"ft": _trainieren(feat, heute, "ft"), "ht": _trainieren(feat, heute, "ht")}
        _MODELLE[key] = mods
    if mods["ft"] is None:
        raise ValueError("zu wenig Daten")
    return {"feat": feat.kopie(), "mods": mods, "tag": _tag(heute)}


def _lambdas(m, mod, heim, gast):
    f = m["feat"]
    div = f.liga_von.get(heim) or f.liga_von.get(gast) or next(iter(f.liga))
    for t in (heim, gast):
        f._team(t, f.liga_von.get(t, div), m["tag"])
    X = np.vstack([f.zeile(heim, gast, True, div, m["tag"]), f.zeile(gast, heim, False, div, m["tag"])])
    lh, la = mod["gbm"].predict(X[:, mod["spalten"]])
    return float(np.clip(lh, 0.05, 6)), float(np.clip(la, 0.05, 6))


def vorhersage(m, heim, gast):
    ft = m["mods"]["ft"]
    lh, la = _lambdas(m, ft, heim, gast)
    out = {"ft": kern.matrix(lh, la, ft["abh"]), "ht": None, "xg": [lh, la]}
    ht = m["mods"]["ht"]
    if ht is not None:
        hh, ha = _lambdas(m, ht, heim, gast)
        out["ht"] = kern.matrix(min(hh, 4.0), min(ha, 4.0), ht["abh"])
    return out


def diagnose(m):
    t = []
    for ziel, name in (("ft", "Vollzeit"), ("ht", "Halbzeit")):
        x = m["mods"].get(ziel)
        t.append(f"{name}: keine Daten" if x is None else
                 f"{name}: {x['runden']} Boosting-Runden (Early Stopping), Prüf-Devianz {x['dev']:.3f}, Abhängigkeit {x['abh'][0]} ({x['abh'][1]:+.3f})")
    return " | ".join(t)
