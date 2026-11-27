# -*- coding: utf-8 -*-
"""ECHTES xG FUER ALLE MODELLE (27.11.2026)

Fuer jedes Spiel wird aus dem SportsAPI-Archiv (echtes xG je Spiel, seit 12.09.2026) die xG-Form beider Teams
berechnet (xG fuer / xG gegen, letzte Spiele, neuere zaehlen mehr, nur Spiele VOR dem Spieltag).
Daraus eine xG-Tor-Erwartung. Jedes Modell mit erwarteten Toren (alle ausser den reinen Hilfs-Feldern) wird damit
verschoben: lam_neu = (1-w)*lam_modell + w*lam_xG. Alle Maerkte des Modells (1X2, DC, O/U, Beide treffen,
1./2. HZ, Erstes/Naechstes Tor) werden im gleichen Verhaeltnis angepasst - die Eigenart jedes Modells bleibt.

w wird NICHT geraten: es wird aus den Archiv-Spielen gelernt (erste Haelfte waehlen, zweite Haelfte pruefen).
Bringt xG in der Pruefung nichts, ist w = 0 und nichts wird veraendert.
"""
import gzip
import json
import math
import os
import unicodedata
import difflib
from collections import defaultdict

import numpy as np

HIER = os.path.dirname(os.path.abspath(__file__))
ARCHIV = os.path.join(HIER, "archiv", "sportsapi.json.gz")
MIN_SPIELE = 3
N_FORM = 8
# weitere echte SportsAPI-Statistiken (fuer / gegen), aus denen die Tor-Erwartung gelernt wird
STATS = ["expectedGoals", "expectedGoalsOnTarget", "bigChanceCreated", "totalShotsInsideBox", "touchesInOppBox", "shotsOnGoal"]
OHNE = {"remis", "sv", "hz2", "karten", "ecken", "ou_best", "h1m", "h2m"}   # Hilfs-/Spezialfelder ohne Tor-Erwartung


def log(*a):
    print("xG fuer alle:", *a, flush=True)


def _ascii(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()


def lade():
    try:
        a = json.loads(gzip.open(ARCHIV, "rt", encoding="utf-8").read())
    except Exception as ex:
        log("Archiv nicht lesbar:", ex)
        return {}
    je = defaultdict(list)
    for e in a.get("spiele", {}).values():
        x = (e.get("stat") or {}).get("ALL", {}).get("expectedGoals")
        ft = e.get("ft")
        if e.get("fertig") and x and ft and e.get("datum"):
            al = (e.get("stat") or {}).get("ALL", {})
            st = []
            for k in STATS:
                v = al.get(k)
                try:
                    st.append((float(v[0]), float(v[1])))
                except Exception:
                    st.append((np.nan, np.nan))
            je[e["liga"]].append((e["datum"], e["heim"], e["gast"], float(x[0]), float(x[1]), int(ft[0]), int(ft[1]), st))
    for l in je:
        je[l].sort()
    return je


class Form:
    def __init__(self, je):
        self.je = je
        self.namen = {l: {} for l in je}
        for l, L in je.items():
            for d, h, g, *_ in L:
                self.namen[l].setdefault(_ascii(h), h)
                self.namen[l].setdefault(_ascii(g), g)

    def name(self, liga, team):
        nm = self.namen.get(liga) or {}
        z = _ascii(team)
        if z in nm:
            return nm[z]
        b = difflib.get_close_matches(z, list(nm), n=1, cutoff=0.6)
        return nm[b[0]] if b else None

    def team(self, liga, team, vor):
        """(xG fuer, xG gegen, Tore fuer, Tore gegen) gewichtet, nur Spiele vor 'vor'."""
        echt = self.name(liga, team)
        if not echt:
            return None
        R = []
        for d, h, a, xh, xa, gh, ga, st in self.je[liga]:
            if d >= vor:
                break
            if h == echt:
                R.append((xh, xa, gh, ga, *[v[0] for v in st], *[v[1] for v in st]))
            elif a == echt:
                R.append((xa, xh, ga, gh, *[v[1] for v in st], *[v[0] for v in st]))
        R = R[-N_FORM:]
        if len(R) < MIN_SPIELE:
            return None
        w = np.array([0.85 ** (len(R) - 1 - i) for i in range(len(R))])
        A = np.array(R, float)
        ok = ~np.isnan(A)
        return tuple(np.where(ok.any(0), np.nansum(A * w[:, None], 0) / np.maximum((ok * w[:, None]).sum(0), 1e-9), np.nan))

    def schnitt(self, liga, vor):
        L = [x for x in self.je.get(liga, []) if x[0] < vor][-300:]
        if len(L) < 30:
            return None
        A = np.array([x[3:7] for x in L], float)
        return A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(), A[:, 3].mean()   # xG heim, xG gast, Tore heim, Tore gast

    def erwartung(self, liga, h, a, vor):
        """xG-Erwartung und Tor-Form-Erwartung fuer das Spiel (beide auf Tor-Niveau der Liga)."""
        fh, fa, s = self.team(liga, h, vor), self.team(liga, a, vor), self.schnitt(liga, vor)
        if not (fh and fa and s):
            return None
        xgh, xga, gh, ga = s
        x = (fh[0] * fa[1] / max(xga, .2) * gh / max(xgh, .2), fa[0] * fh[1] / max(xgh, .2) * ga / max(xga, .2))
        g = (fh[2] * fa[3] / max(ga, .2), fa[2] * fh[3] / max(gh, .2))
        x = tuple(min(max(v, .1), 5) for v in x)
        g = tuple(min(max(v, .1), 5) for v in g)
        if self.reg is not None:                       # aus allen Statistiken gelernte Erwartung
            try:
                x = self.reg_erw(fh, fa)
            except Exception:
                pass
        return x, g

    reg = None
    K = len(STATS)

    def merk(self, fh, fa):
        """Merkmale Heim: eigene Offensiv-Statistik + Defensiv-Statistik des Gegners (log)."""
        K = self.K
        f = lambda v: math.log(max(v, 0.05)) if v == v else 0.0
        h = [f(fh[4 + k]) for k in range(K)] + [f(fa[4 + K + k]) for k in range(K)]
        a = [f(fa[4 + k]) for k in range(K)] + [f(fh[4 + K + k]) for k in range(K)]
        return h, a

    def reg_erw(self, fh, fa):
        h, a = self.merk(fh, fa)
        b = self.reg
        lh = math.exp(b[0] + b[1] + float(np.dot(b[2:], h)))      # b[1] = Heim-Abschnitt (gelernt, kein fester Wert)
        la = math.exp(b[0] + float(np.dot(b[2:], a)))
        return min(max(lh, .1), 5), min(max(la, .1), 5)


def _p(lh, la):
    k = np.arange(11)
    f = np.array([math.factorial(i) for i in k], float)
    m = np.outer(np.exp(-lh) * lh ** k / f, np.exp(-la) * la ** k / f)
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]), m[np.add.outer(k, k) > 2].sum()


def _ll(lh, la, gh, ga):
    p, po = _p(lh, la)
    e = 0 if gh > ga else 1 if gh == ga else 2
    o = gh + ga > 2
    return -math.log(max(p[e], 1e-9)) - math.log(max(po if o else 1 - po, 1e-9))


def lerne_regression(F):
    """Tor-Erwartung aus ALLEN Statistiken (xG, xGOT, Grosschancen, Schuesse im Strafraum, Ballkontakte im
    Strafraum, Schuesse aufs Tor; jeweils eigene + Gegner-Abwehr). Poisson-Regression, nur Spiele mit Vorgeschichte.
    Gibt Koeffizienten zurueck, oder None wenn sie in der Pruefhaelfte nicht besser als reines xG ist."""
    from scipy.optimize import minimize
    Xs, Ys, xe, D = [], [], [], []
    for liga, L in F.je.items():
        for d, h, a, xh, xa, gh, ga, st in L:
            fh, fa = F.team(liga, h, d), F.team(liga, a, d)
            e = F.erwartung(liga, h, a, d) if (fh and fa) else None
            if not e:
                continue
            mh, ma = F.merk(fh, fa)
            Xs += [[1.0, 1.0] + mh, [1.0, 0.0] + ma]
            Ys += [gh, ga]
            xe += [e[0][0], e[0][1]]
            D += [d, d]
    if len(Ys) < 400:
        return None, {"stat_spiele": len(Ys) // 2}
    Xs, Ys, xe, D = np.array(Xs), np.array(Ys, float), np.array(xe), np.array(D)
    o = np.argsort(D, kind="stable")
    Xs, Ys, xe = Xs[o], Ys[o], xe[o]
    half = len(Ys) // 2

    def nll(b, X, Y):
        lam = np.exp(np.clip(X @ b, -3, 2))
        return float(np.mean(lam - Y * np.log(lam))) + 1e-2 * float(np.sum(b[2:] ** 2))
    b = minimize(nll, np.zeros(Xs.shape[1]), args=(Xs[:half], Ys[:half]), method="L-BFGS-B").x
    lam_r = np.exp(np.clip(Xs[half:] @ b, -3, 2))
    pl = lambda lam: float(np.mean(lam - Ys[half:] * np.log(lam)))
    info = {"stat_spiele": len(Ys) // 2, "pruef_regression": round(pl(lam_r), 4), "pruef_nur_xg": round(pl(np.clip(xe[half:], .1, 5)), 4)}
    if pl(lam_r) >= pl(np.clip(xe[half:], .1, 5)):
        log("alle Statistiken: nicht besser als reines xG - bleibt bei xG", info)
        return None, info
    b = minimize(nll, b, args=(Xs, Ys), method="L-BFGS-B").x          # mit allen Spielen neu
    log("alle Statistiken: besser als reines xG", info)
    return b, info


def lerne_w(F):
    """Wie stark soll echtes xG gegenueber der Tor-Form zaehlen? Erste Haelfte waehlen, zweite pruefen."""
    Z = []
    for liga, L in F.je.items():
        for d, h, a, xh, xa, gh, ga, st in L:
            e = F.erwartung(liga, h, a, d)
            if e:
                Z.append((d, e, gh, ga))
    if len(Z) < 200:
        log(f"nur {len(Z)} pruefbare Spiele - xG bleibt aus (w = 0)")
        return 0.0, {"spiele": len(Z)}
    Z.sort(key=lambda z: z[0])
    half = len(Z) // 2
    W = [i / 10 for i in range(11)]

    def ll(T, w):
        return float(np.mean([_ll((1 - w) * g[0] + w * x[0], (1 - w) * g[1] + w * x[1], gh, ga) for d, (x, g), gh, ga in T]))
    a = {w: ll(Z[:half], w) for w in W}
    w = min(a, key=a.get)
    b0, bw = ll(Z[half:], 0.0), ll(Z[half:], w)
    info = {"spiele": len(Z), "w": w, "pruef_ll_ohne": round(b0, 4), "pruef_ll_mit": round(bw, 4)}
    if bw >= b0:
        log("xG brachte in der Pruefung nichts - w = 0", info)
        return 0.0, info
    log("gelernt:", info)
    return w, info


def _erg(s):
    try:
        a, b = str(s.get("ergebnis")).split(":")
        return int(a), int(b)
    except Exception:
        return None


def lerne_w_modelle(F, spiele, nm, share):
    """Entscheidend: Wird JEDES MODELL mit xG besser? Gemessen an den Spielen mit Ergebnis (Archiv + letzte Tage),
    nur xG-Werte von VOR dem Spieltag. w aus der ersten Haelfte, Pruefung auf der zweiten."""
    Z = []
    for s in spiele:
        r = _erg(s)
        e = F.erwartung(s.get("liga"), s.get("heim"), s.get("gast"), s.get("datum", "9999")) if r else None
        if not e:
            continue
        for mid, p in (s.get("prog") or {}).items():
            lam = p.get("xg_roh", p.get("xg")) if isinstance(p, dict) else None
            if mid in OHNE or not (isinstance(lam, list) and len(lam) == 2) or p.get("p1") is None:
                continue
            Z.append((s.get("datum"), mid, p, lam, e[0], r, share.get(s.get("liga"), 0.44)))
    if len(Z) < 300:
        return None, {"modell_tests": len(Z)}
    Z.sort(key=lambda z: z[0])
    half = len(Z) // 2

    def ll(T, w):
        L = []
        for d, mid, p, lam, (xh, xa), (gh, ga), sh in T:
            q = dict(p.get("roh_p") or {k: p.get(k) for k in ("p1", "px", "p2", "ue25")})   # immer die Original-Werte
            if w:
                _anpassen(q, nm.maerkte(lam[0], lam[1], sh), nm.maerkte((1 - w) * lam[0] + w * xh, (1 - w) * lam[1] + w * xa, sh))
            e = 0 if gh > ga else 1 if gh == ga else 2
            L.append(-math.log(max([q["p1"], q["px"], q["p2"]][e] / 100, 1e-9)))
        return float(np.mean(L))
    W = [i / 10 for i in range(9)]
    a = {w: ll(Z[:half], w) for w in W}
    w = min(a, key=a.get)
    b0, bw = ll(Z[half:], 0.0), ll(Z[half:], w)
    info = {"modell_tests": len(Z), "w": w, "modell_ll_ohne": round(b0, 4), "modell_ll_mit": round(bw, 4)}
    return (w if bw < b0 else 0.0), info


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _anpassen(p, base, neu):
    """Model-Wahrscheinlichkeiten im Verhaeltnis neu/base verschieben (Eigenart des Modells bleibt)."""
    if all(p.get(k) is not None for k in ("p1", "px", "p2")):
        v = [p["p1"] * neu["p1"] / max(base["p1"], 1e-6), p["px"] * neu["px"] / max(base["px"], 1e-6), p["p2"] * neu["p2"] / max(base["p2"], 1e-6)]
        t = sum(v) or 1
        p["p1"], p["px"], p["p2"] = (round(x / t * 100, 2) for x in v)
    for f in ("ue25", "bts", "h1o05", "h2o05"):
        if p.get(f) is not None and base.get(f) is not None and neu.get(f) is not None:
            z = _logit(p[f] / 100) + _logit(neu[f] / 100) - _logit(base[f] / 100)
            p[f] = round(100 / (1 + math.exp(-z)), 2)
    for f in ("ht", "ht2"):
        if isinstance(p.get(f), list) and len(p[f]) == 3 and base.get(f) and neu.get(f) and all(x is not None for x in p[f]):
            v = [p[f][i] * neu[f][i] / max(base[f][i], 1e-6) for i in range(3)]
            t = sum(v) or 1
            p[f] = [round(x / t * 100, 2) for x in v]


def anwenden(d, nm, share):
    je = lade()
    if not je:
        return {"aktiv": False, "grund": "kein xG-Archiv"}
    F = Form(je)
    try:
        F.reg, ri = lerne_regression(F)
    except Exception as ex:
        F.reg, ri = None, {"fehler": str(ex)}
    w0, info = lerne_w(F)
    info["statistiken"] = ri
    # zweite, entscheidende Pruefung direkt an den Modellen (Archiv + Spiele mit Ergebnis)
    try:
        A = json.load(open(os.path.join(HIER, "docs", "archiv_spiele.json"), encoding="utf-8"))
    except Exception:
        A = []
    pool = list(A) + [s for s in d.get("spiele", []) if _erg(s)]
    wm, im = lerne_w_modelle(F, pool, nm, share)
    if F.reg is not None:                 # entscheidend ist, was den MODELLEN mehr hilft: alle Statistiken oder nur xG
        reg, F.reg = F.reg, None
        wx, ix = lerne_w_modelle(F, pool, nm, share)
        if wm is not None and wx is not None and ix.get("modell_ll_mit", 9) <= im.get("modell_ll_mit", 9):
            log("Modelle profitieren mehr von reinem xG:", ix, "statt", im)
            wm, im = wx, dict(ix, quelle="nur xG")
        else:
            F.reg = reg
            im = dict(im, quelle="alle Statistiken")
    info["modelle"] = im
    if wm is not None:
        w = wm
    else:
        w = w0 / 2          # zu wenig Modell-Pruefspiele: vorsichtig nur halbe Staerke
    log("Pruefung an den Modellen:", im, "-> w =", w)
    info["w_genutzt"] = w
    info["aktiv"] = w > 0
    if w <= 0:
        return info
    n_sp = n_mod = 0
    for s in d.get("spiele", []):
        e = F.erwartung(s.get("liga"), s.get("heim"), s.get("gast"), s.get("datum", "9999"))
        if not e:
            continue
        (xh, xa), _ = e
        sh = share.get(s.get("liga"), 0.44)
        tr = False
        for mid, p in (s.get("prog") or {}).items():
            if mid in OHNE or not isinstance(p, dict) or p.get("xgk") is not None:
                continue
            lam = p.get("xg")
            if not (isinstance(lam, list) and len(lam) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in lam)):
                continue
            try:
                lh, la = (1 - w) * lam[0] + w * xh, (1 - w) * lam[1] + w * xa
                base, neu = nm.maerkte(lam[0], lam[1], sh), nm.maerkte(lh, la, sh)
                p["roh_p"] = {k: p.get(k) for k in ("p1", "px", "p2", "ue25")}
                _anpassen(p, base, neu)
                p["xg_roh"] = lam
                p["xg"] = [round(float(lh), 3), round(float(la), 3)]
                p["xgk"] = float(w)
                n_mod += 1
                tr = True
            except Exception:
                pass
        n_sp += tr
    info.update({"spiele_angepasst": n_sp, "modell_prognosen": n_mod})
    log(f"w = {w}: {n_sp} Spiele, {n_mod} Modell-Prognosen mit echtem xG angepasst")
    return info
