"""sichten.py - Neue Architektur (Tab 🧩 Modelle)

Jedes Modell rechnet seine Sichten SELBST, mit seiner eigenen Methode - nichts wird zwischen Modellen gemischt:
  Gesamt          = die normale Prognose des Modells (steht schon in prog)
  Heim/Auswaerts  = das Modell wird auf Daten trainiert, in denen jedes Team als zwei getrennte Teams vorkommt
                    ("Team@H" nur Heimspiele, "Team@A" nur Auswaertsspiele). Heimteam -> nur seine Heimspiele,
                    Gastteam -> nur seine Auswaertsspiele.
  Nur H2H         = fuer die beiden Teams zaehlen NUR ihre direkten Duelle (seit 2000), alle anderen Spiele der zwei
                    Teams werden entfernt. Statistik-Modelle lernen je Spiel neu (Rest der Liga bleibt), lernende Modelle
                    (LightGBM, Residual, Waage, SVR, RF, XGBoost) behalten ihr Gesamt-Training, bekommen aber als
                    Eingabe nur die Duell-Werte. Aktualitaet: Duelle der letzten 2 Jahre zaehlen 3-fach,
                    2-5 Jahre 2-fach, aelter 1-fach. Unter 3 Duellen gibt es keine H2H-Prognose.
Keine Buchmacher-Quote als Eingabe (XGBoost und Stacking laufen ohne Markt).
Ausserdem: X-Werte / letzte Spiele je Team (Gesamt, Heim, Auswaerts, H2H), Liga-Normalwerte je Markt,
Vergleich Modell-Erwartung vs. echtes xG je Modell und Sicht.
Training immer nur mit Spielen VOR dem ersten angezeigten Tag (kein Blick in die Zukunft).
Laeuft als letzter Schritt (nach Profi-Modelle)."""
import importlib, json, math, os, sys, time
from collections import defaultdict

import numpy as np
import pandas as pd

import daten
import kern
from ligen import LAENDER

PFAD = "docs/daten.json"
ARCHIV = "docs/archiv_spiele.json"
AUS = "docs/sichten.json"
BASIS = ["modell_bayes", "modell_copula", "modell_elo", "modell_ensemble", "modell_sarimax", "modell_schuss", "modell_xg"]
MIN_DUELLE = 3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ---------------- Daten-Sichten ----------------
def t_ha(H):
    H = H.copy()
    H["Home"] = H["Home"].astype(str) + "@H"
    H["Away"] = H["Away"].astype(str) + "@A"
    return H


def _kopien(datum, grenze):
    a = (pd.Timestamp(grenze) - pd.Timestamp(datum)).days / 365.25
    return 3 if a <= 2 else 2 if a <= 5 else 1


def t_h2h(H, paare, grenze):
    """Fuer die Teams der Paare nur ihre Duelle (nach Aktualitaet vervielfacht), alle anderen Spiele bleiben."""
    teams = {t for p in paare for t in p}
    ps = {frozenset(p) for p in paare}
    hm, am = H["Home"].isin(teams), H["Away"].isin(teams)
    duell = np.array([frozenset((h, a)) in ps for h, a in zip(H["Home"], H["Away"])]) & hm.to_numpy() & am.to_numpy()
    rest = H[~(hm | am).to_numpy()]
    d = H[duell]
    if len(d):
        c = np.array([_kopien(x, grenze) for x in d["Date"]])
        d = d.loc[d.index.repeat(c)]
    return pd.concat([rest, d]).sort_values("Date", kind="stable").reset_index(drop=True)


def duelle(HALL, h, a, vor):
    m = (((HALL["Home"] == h) & (HALL["Away"] == a)) | ((HALL["Home"] == a) & (HALL["Away"] == h))) & (HALL["Date"] < vor)
    return HALL[m]


def chargen(paare):
    """Paare so in Gruppen teilen, dass kein Team zweimal in einer Gruppe steht."""
    B = []
    for i, (h, a) in paare:
        for b in B:
            if h not in b["t"] and a not in b["t"]:
                b["t"] |= {h, a}; b["p"].append((i, h, a)); break
        else:
            B.append({"t": {h, a}, "p": [(i, h, a)]})
    return [b["p"] for b in B]


# ---------------- Prognose-Format (wie in prog) ----------------
def _r(x):
    return round(float(x) * 100, 2)


def aus_matrix(ft, ht, share):
    xh, xa = kern.erwartete_tore(ft)
    p = kern.maerkte(ft, ht)
    out = {"p1": _r(p["1"]), "px": _r(p["X"]), "p2": _r(p["2"]), "ue25": _r(p["Ü2.5"]), "bts": _r(p["BTS"]),
           "xg": [round(float(xh), 2), round(float(xa), 2)], "tipp": p.get("tipp")}
    if ht is not None:
        hh, ha = kern.erwartete_tore(ht)
        out["ht"] = [_r(x) for x in p["HT"]]
        out["ht2"] = [_r(x) for x in p["HT2"]] if p.get("HT2") is not None else None
        out["h1o05"] = _r(1 - ht[0, 0] / ht.sum())
        out["h2o05"] = _r(1 - math.exp(-(max(xh - hh, 0.05) + max(xa - ha, 0.05))))
    else:
        out["ht"] = None
        out["ht2"] = None
        out["h1o05"] = _r(1 - math.exp(-(xh + xa) * share))
        out["h2o05"] = _r(1 - math.exp(-(xh + xa) * (1 - share)))
    return out


def ergaenzen(p, share):
    """Fehlende Felder (z. B. bei RF/XGBoost) aus der eigenen Tor-Erwartung desselben Modells."""
    if not p or not p.get("xg"):
        return p
    l = sum(p["xg"])
    p.setdefault("h1o05", _r(1 - math.exp(-l * share)))
    p.setdefault("h2o05", _r(1 - math.exp(-l * (1 - share))))
    return p


# ---------------- 1) Basis-Modelle (trainiere / vorhersage) ----------------
def gruppen(spiele, HALL):
    G = {}
    for land, codes in LAENDER.items():
        for c in codes:
            G[c] = tuple(codes)
    for s in spiele:
        G.setdefault(s["liga"], (s["liga"],))
    return G


def basis(HALL, spiele, grenze, share, sicht, h2h_ok):
    G = gruppen(spiele, HALL)
    je = defaultdict(list)
    for i, s in enumerate(spiele):
        je[G[s["liga"]]].append(i)
    mods = {k: importlib.import_module(k) for k in BASIS}
    fehler = defaultdict(int)
    for codes, idx in je.items():
        L = HALL[HALL["Div"].isin(codes) & (HALL["Date"] < grenze)]
        if len(L) < 300:
            continue
        teams = set(L["Home"]) | set(L["Away"])
        La = t_ha(L)
        for k, mod in mods.items():
            try:
                m = mod.trainiere(La, grenze)
                for i in idx:
                    s = spiele[i]
                    if s["heim"] in teams and s["gast"] in teams:
                        o = mod.vorhersage(m, s["heim"] + "@H", s["gast"] + "@A")
                        sicht[i]["ha"][k] = aus_matrix(o["ft"], o.get("ht"), share.get(s["liga"], 0.44))
            except Exception as ex:
                log(f"  {k} Heim/Ausw {codes[0]}: {str(ex)[:70]}")
            # Nur H2H: je Spiel wird NUR dieses Paar auf seine Duelle reduziert, der Rest der Liga bleibt
            for i in idx:
                if i not in h2h_ok:
                    continue
                h, a = spiele[i]["heim"], spiele[i]["gast"]
                try:
                    m = mod.trainiere(t_h2h(L, [(h, a)], grenze), grenze)
                    o = mod.vorhersage(m, h, a)
                    sicht[i]["h2h"][k] = aus_matrix(o["ft"], o.get("ht"), share.get(spiele[i]["liga"], 0.44))
                except Exception as ex:
                    fehler[k] += 1
    log("Basis-Modelle fertig", dict(fehler) if fehler else "")


# ---------------- 2) Profi: LightGBM, Residual, Waage ----------------
def profi(HP, spiele, grenze, share, sicht, laeufe):
    """Heim/Ausw: Modell lernt und rechnet auf der umbenannten Historie.
    Nur H2H: dasselbe Modell wie Gesamt (gelernt auf allen Spielen), aber die Eingaben (Merkmale) der zwei Teams
    stammen nur aus ihren Duellen - sonst wuerde das Training fast aller Teams eines Spieltags wegfallen."""
    import profi_modelle as PM
    import waage as WG
    import neue_modelle as NM
    ligen = {l: i for i, l in enumerate(sorted(HP["Div"].unique()))}

    def lerne(X, Y, D, NW):
        bw = PM.trainfenster(D, grenze)
        m = PM.Modelle(X, Y, None, bw, D)
        W = WG.Waage(X, NW, WG.ergebnis(Y), bw)
        st = PM._stat_lam(X[bw])
        return m, W, kern.lerne_rho(Y[bw, 0], Y[bw, 1], st[:, 0], st[:, 1])
    basis_m = None
    for name, Hv, ziele, ren in laeufe(HP):
        try:
            X, Y, D, M, zf, _ = PM.durchlauf(Hv, ziele, ligen)
            NW, ZN = np.array(PM.WAAGE_N, dtype=float), dict(PM.WAAGE_ZN)
            if name == "ha":
                m, W, rho_w = lerne(X, Y, D, NW)
            else:
                if basis_m is None:
                    Xb, Yb, Db, _, _, _ = PM.durchlauf(HP[HP["Date"] < pd.Timestamp(grenze)], [], ligen)
                    basis_m = lerne(Xb, Yb, Db, np.array(PM.WAAGE_N, dtype=float))
                m, W, rho_w = basis_m
            keys = sorted(zf)
            if not keys:
                continue
            XL = np.array([zf[k] for k in keys], dtype=float)
            lh, la = m.lgbm(XL); rl = m.rho_fuer("lgbm", XL)
            rh, ra = m.resid(XL); rr = m.rho_fuer("resid", XL)
            kw = [k for k in keys if k in ZN]
            PW = dict(zip(kw, W.vorhersage(np.array([zf[k] for k in kw], dtype=float), np.array([ZN[k] for k in kw], dtype=float)))) if kw else {}
            for j, k in enumerate(keys):
                sh = share.get(spiele[k]["liga"], 0.44)
                sicht[k][name]["lgbm"] = NM.maerkte(float(lh[j]), float(la[j]), sh, rho=float(rl[j]))
                sicht[k][name]["resid"] = NM.maerkte(float(rh[j]), float(ra[j]), sh, rho=float(rr[j]))
                if k in PW:
                    a, b = PM._stat_lam(XL[j:j + 1])[0]
                    q = NM.maerkte(float(a), float(b), sh, rho=rho_w)
                    q["p1"], q["px"], q["p2"] = (round(float(v) * 100, 1) for v in PW[k])
                    sicht[k][name]["waage"] = q
            log(f"  Profi {name}: {len(keys)} Spiele")
        except Exception as ex:
            log(f"  Profi {name} nicht moeglich: {str(ex)[:120]}")


# ---------------- 3) SVR + K2 (neue_modelle) ----------------
def svr(HN, spiele, grenze, share, sicht, laeufe):
    import neue_modelle as NM
    basis_m = None
    for name, Hv, ziele, ren in laeufe(HN):
        try:
            out = NM.durchlauf(Hv, ziele)
            zf = out[5]
            if name == "ha":
                mo = NM.svr_modell(out[0], out[3], out[4], grenze)
            else:
                if basis_m is None:
                    b = NM.durchlauf(HN[HN["Date"] < pd.Timestamp(grenze)], [])
                    basis_m = NM.svr_modell(b[0], b[3], b[4], grenze)
                mo = basis_m
            for k, (f, *_x) in zf.items():
                s = spiele[k]
                sicht[k][name]["svr"] = NM.maerkte(*mo(f), share.get(s["liga"], 0.44), rho=NM.rho_von(mo, s["liga"]))
            log(f"  SVR {name}: {len(zf)} Spiele")
        except Exception as ex:
            log(f"  SVR {name} nicht moeglich: {str(ex)[:120]}")


def _eintrag(r, h=None, a=None):
    ht = [int(r.HTHG), int(r.HTAG)] if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)) else None
    return {"datum": r.Date.strftime("%Y-%m-%d"), "heim": h or r.Home, "gast": a or r.Away, "ft": [int(r.HG), int(r.AG)], "ht": ht}


def k2(HN, HALL, spiele, grenze, sicht, h2h_ok):
    import neue_modelle as NM
    try:
        NM.k2_rho_lernen(HN, grenze)
    except Exception as ex:
        log("  K2-Remis-Korrektur:", ex)
    sai = HN[(HN["Date"] >= pd.Timestamp(NM.SAISON_START)) & (HN["Date"] < grenze)]
    saison, saison_ha = defaultdict(list), defaultdict(list)
    for r in sai.itertuples(index=False):
        saison[r.Div].append(_eintrag(r))
        saison_ha[r.Div].append(_eintrag(r, r.Home + "@H", r.Away + "@A"))
    xg = NM.lade_xg()
    n = 0
    for i, s in enumerate(spiele):
        liga = s["liga"]
        # Heim/Auswaerts: Heimteam nur Heimspiele, Gastteam nur Auswaertsspiele (auch beim xG)
        s2 = dict(s, heim=s["heim"] + "@H", gast=s["gast"] + "@A")
        try:
            o = NM.konsens2(s2, saison_ha.get(liga, []), [])
            xl = xg.get(liga, [])
            if o and xl:
                Ax = xg_ort(xl, s["heim"], s["datum"], 0)
                Bx = xg_ort(xl, s["gast"], s["datum"], 1)
                if Ax and Bx:
                    L = NM.liga_kennz([e for e in saison.get(liga, []) if e["datum"] < s["datum"]])
                    o["k2_xg"] = NM.maerkte(*NM.lam_xg(Ax, Bx), L["share"], rho=NM.k2_rho("korr", liga))
            sicht[i]["ha"].update(o or {})
        except Exception:
            pass
        # Nur H2H: Saison der Liga ohne die zwei Teams + alle Duelle seit 2000 (aktuelle mehrfach)
        if i in h2h_ok:
            h, a = s["heim"], s["gast"]
            try:
                rest = [e for e in saison.get(liga, []) if not ({e["heim"], e["gast"]} & {h, a})]
                du = []
                for r in duelle(HALL, h, a, grenze).itertuples(index=False):
                    du += [_eintrag(r)] * _kopien(r.Date, grenze)
                hist = sorted(rest + du, key=lambda e: e["datum"])
                o = NM.konsens2(s, hist, [])
                sicht[i]["h2h"].update(o or {})
                n += 1
            except Exception:
                pass
    log(f"  K2: Heim/Ausw + H2H ({n})")


def xg_ort(xl, name, vor, seite):
    """xG eines Teams nur aus Heimspielen (seite 0) bzw. nur Auswaertsspielen (seite 1)."""
    import neue_modelle as NM, difflib
    namen = {}
    for d, h, g, *_ in xl:
        namen.setdefault(NM._ascii(h), h); namen.setdefault(NM._ascii(g), g)
    z = NM._ascii(name)
    if z not in namen:
        b = difflib.get_close_matches(z, list(namen), n=1, cutoff=0.6)
        if not b:
            return None
        z = b[0]
    echt = namen[z]
    f, g = [], []
    for d, h, a, xh, xa in xl:
        if d >= vor:
            continue
        if seite == 0 and h == echt:
            f.append(xh); g.append(xa)
        elif seite == 1 and a == echt:
            f.append(xa); g.append(xh)
    return (sum(f) / len(f), sum(g) / len(g)) if len(f) >= 2 else None


# ---------------- 4) Random Forest + XGBoost (ohne Quoten) ----------------
def baeume(HALL, spiele, grenze, share, sicht, laeufe):
    """Random Forest und XGBoost - beide OHNE Buchmacher-Quote (gleiche Merkmale wie RF).
    Heim/Ausw: lernen + rechnen auf der umbenannten Historie. Nur H2H: Gesamt-Modell, Merkmale nur aus Duellen."""
    import rf_modell as RF, xgb_modell as XG
    ligen = {d: i for i, d in enumerate(sorted(HALL["Div"].unique()))}
    ZIELE = ("y1", "yo", "yb", "yh", "yh2")

    def lerne(Hs):
        vor, ende = XG._features(Hs)
        rows = []
        for r in Hs.itertuples(index=False):
            f = vor.get(XG._key(r.Div, r.Date, r.Home, r.Away))
            if f:
                rows.append(dict(f, liga=ligen.get(r.Div, -1), **RF._ziele(r)))
        T = pd.DataFrame(rows)
        M = {}
        for art in ("rf", "xgb"):
            M[art] = {}
            for z in ZIELE:
                tt = T.dropna(subset=[z]) if z in T else None
                if tt is None or len(tt) < 1000 or tt[z].nunique() < 2:
                    continue
                mod = RF._modell() if art == "rf" else XG._modell(tt[z].nunique())
                mod.fit(tt[RF.FEAT], tt[z].astype(int))
                M[art][z] = mod
        log(f"    Baeume gelernt: {len(T)} Spiele")
        return M, ende

    def rechne(M, ende, ziele, name):
        n = 0
        for z in ziele:
            if z["liga"] not in ende:
                continue
            f = XG._zukunft(ende[z["liga"]], z["liga"], z["datum"], z["heim"], z["gast"])
            X = pd.DataFrame([dict(f, liga=ligen.get(z["liga"], -1))])[RF.FEAT]
            for art, ms in M.items():
                pr = lambda q: ms[q].predict_proba(X)[0] if q in ms else None
                p1, po, pb, ph, ph2 = (pr(q) for q in ZIELE)
                if p1 is None or po is None:
                    continue
                r0 = lambda x: round(float(x) * 100, 2)
                p = {"p1": r0(p1[0]), "px": r0(p1[1]), "p2": r0(p1[2]), "ue25": r0(po[1]),
                     "bts": r0(pb[1]) if pb is not None else r0(f["sb"]),
                     "ht": [r0(x) for x in ph] if ph is not None else None,
                     "ht2": [r0(x) for x in ph2] if ph2 is not None else None,
                     "xg": [round(f["lh"], 2), round(f["la"], 2)]}
                sicht[z["key"]][name][art] = ergaenzen(p, share.get(z["liga"], 0.44))
                n += 1
        log(f"  RF/XGBoost {name}: {n} Prognosen")
    H0 = HALL[HALL["Date"] >= pd.Timestamp("2019-07-01")]
    basis_m = None
    for name, Hv, ziele, ren in laeufe(H0):
        try:
            if name == "ha":
                M, ende = lerne(Hv)
            else:
                if basis_m is None:
                    basis_m = lerne(H0[H0["Date"] < pd.Timestamp(grenze)])[0]
                M = basis_m
                ende = XG._features(Hv)[1]
            rechne(M, ende, ziele, name)
        except Exception as ex:
            log(f"  RF/XGBoost {name} nicht moeglich: {str(ex)[:120]}")


# ---------------- 5) Stacking (Gewichte aus engine, ohne Markt) ----------------
def stacking(d, sicht):
    import stacking as STK
    w = {k: v for k, v in (d.get("stacking_gew") or {}).items() if k != "markt"}
    if not w:
        return
    for i, sv in sicht.items():
        for name in ("ha", "h2h"):
            ids = [m for m in w if m in sv[name]]
            if len(ids) < 2:
                continue
            g = np.array([w[m] for m in ids], float)
            if g.sum() <= 0:
                continue
            try:
                ps = [{k: v for k, v in sv[name][m].items() if k not in ("tipp",)} for m in ids]
                sv[name]["st"] = STK.mischen(ps, g / g.sum())
            except Exception:
                pass


# ---------------- X-Werte, letzte Spiele ----------------
def _zeile(r, team):
    heim = r.Home == team
    t, g = (r.HG, r.AG) if heim else (r.AG, r.HG)
    ht = None
    if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)):
        ht = [int(r.HTHG), int(r.HTAG)] if heim else [int(r.HTAG), int(r.HTHG)]
    return [r.Date.strftime("%Y-%m-%d"), r.Away if heim else r.Home, "H" if heim else "A", int(t), int(g), ht]


def xwerte(HALL, s, N=10):
    vor = pd.Timestamp(s["datum"])
    h, a = s["heim"], s["gast"]
    E = HALL[HALL["Date"] < vor]
    out = {}
    for seite, t in (("h", h), ("a", a)):
        T = E[(E["Home"] == t) | (E["Away"] == t)]
        out[seite] = {"ges": [_zeile(r, t) for r in T.tail(N).itertuples(index=False)][::-1],
                      "heim": [_zeile(r, t) for r in T[T["Home"] == t].tail(N).itertuples(index=False)][::-1],
                      "ausw": [_zeile(r, t) for r in T[T["Away"] == t].tail(N).itertuples(index=False)][::-1]}
    du = duelle(HALL, h, a, vor)
    out["h2h"] = [_zeile(r, h) for r in du.tail(N).itertuples(index=False)][::-1]
    out["h2h_n"] = int(len(du))
    return out


# ---------------- Liga-Normalwerte je Markt (wie die Seite gewinnt) ----------------
def _sg(x, y):
    return "1" if x > y else "X" if x == y else "2"


def gewinne(hg, ag, hh, ha):
    """{markt: [Ausgang gewonnen? (True/False/None)]} - gleiche Regeln wie msWin/torWin auf der Seite."""
    e = _sg(hg, ag)
    W = {"x1": [e == "1", e == "X", e == "2"], "x": [e != "2", e != "1", e != "X"], "ou": [hg + ag > 2, hg + ag <= 2],
         "bts": [hg > 0 and ag > 0, not (hg > 0 and ag > 0)]}
    fg = None
    if hg + ag == 0:
        fg = [False, False]
    elif ag == 0:
        fg = [True, False]
    elif hg == 0:
        fg = [False, True]
    if hh is not None:
        h2, a2 = hg - hh, ag - ha
        e1, e2 = _sg(hh, ha), _sg(h2, a2)
        W.update({"h2": [h2 + a2 > 0, h2 + a2 == 0], "bts2": [h2 > 0 and a2 > 0, not (h2 > 0 and a2 > 0)],
                  "w1": [e1 == "1", e1 == "X", e1 == "2"], "w2": [e2 == "1", e2 == "X", e2 == "2"],
                  "ht": [e1 != "2", e1 != "1", e1 != "X"], "ht2a": [e2 != "2", e2 != "1", e2 != "X"]})
        if fg is None:
            if hh + ha > 0:
                fg = [True, False] if ha == 0 else [False, True] if hh == 0 else None
            else:
                fg = [True, False] if a2 == 0 else [False, True] if h2 == 0 else None
        W["ng2"] = [False, False] if h2 + a2 == 0 else [True, False] if a2 == 0 else [False, True] if h2 == 0 else None
    W["fg"] = fg
    return W


def normalwerte(HALL, grenze):
    E = HALL[(HALL["Date"] < grenze) & (HALL["Date"] >= pd.Timestamp(grenze) - pd.DateOffset(years=3))]
    acc = defaultdict(lambda: defaultdict(lambda: None))
    for r in E.itertuples(index=False):
        hh = None if pd.isna(r.HTHG) else int(r.HTHG)
        ha = None if pd.isna(r.HTAG) else int(r.HTAG)
        for mk, v in gewinne(int(r.HG), int(r.AG), hh, ha if hh is not None else None).items():
            if v is None:
                continue
            a = acc[r.Div][mk]
            if a is None:
                a = acc[r.Div][mk] = [0, [0] * len(v)]
            a[0] += 1
            for j, x in enumerate(v):
                a[1][j] += bool(x)
    return {liga: {mk: [round(h / n * 100, 1) for h in hs] for mk, (n, hs) in M.items() if n >= 50} for liga, M in acc.items()}


# ---------------- Modell-Erwartung vs. echtes xG ----------------
def xg_vergleich(spiele_alle):
    import neue_modelle as NM
    xl = NM.lade_xg()
    idx = {}
    for liga, L in xl.items():
        for dd, h, g, xh, xa in L:
            idx[(liga, dd, NM._ascii(h)[:6], NM._ascii(g)[:6])] = (xh, xa)
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for s in spiele_alle:
        x = idx.get((s.get("liga"), s.get("datum"), NM._ascii(s.get("heim", ""))[:6], NM._ascii(s.get("gast", ""))[:6]))
        if not x:
            continue
        quellen = [("ges", s.get("prog") or {})] + [(n, (s.get("sicht") or {}).get(n) or {}) for n in ("ha", "h2h")]
        for name, P in quellen:
            for mid, p in P.items():
                if not isinstance(p, dict) or not p.get("xg"):
                    continue
                a = acc[(mid, name)]
                a[0] += 1
                a[1] += p["xg"][0] - x[0]
                a[2] += p["xg"][1] - x[1]
                a[3] += abs(p["xg"][0] - x[0]) + abs(p["xg"][1] - x[1])
    out = defaultdict(dict)
    for (mid, name), (n, bh, ba, mae) in acc.items():
        if n >= 5:
            out[mid][name] = {"n": n, "bias_h": round(bh / n, 2), "bias_a": round(ba / n, 2), "mae": round(mae / (2 * n), 2)}
    return out


def rund(x):
    if isinstance(x, dict):
        return {k: rund(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rund(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 2)
    return x


# ---------------- Hauptprogramm ----------------
def main():
    d = json.load(open(PFAD, encoding="utf-8"))
    spiele = d.get("spiele", [])
    if not spiele:
        log("keine Spiele")
        return
    grenze = pd.Timestamp(min(s["datum"] for s in spiele))
    HALL = daten.alle_spiele()
    HALL["Date"] = pd.to_datetime(HALL["Date"]).dt.normalize()
    for c in ("HTHG", "HTAG", "HST", "AST"):
        HALL[c] = pd.to_numeric(HALL[c], errors="coerce") if c in HALL else np.nan
    HALL = HALL.sort_values("Date", kind="stable").reset_index(drop=True)
    log(f"Historie {len(HALL)} Spiele, Grenze {grenze.date()}, {len(spiele)} angezeigte Spiele")
    share = {}
    for div, g in HALL.dropna(subset=["HTHG", "HTAG"]).groupby("Div"):
        tot = (g["HG"] + g["AG"]).sum()
        share[div] = float((g["HTHG"] + g["HTAG"]).sum() / tot) if tot else 0.44

    sicht = {i: {"ha": {}, "h2h": {}} for i in range(len(spiele))}
    h2h_ok = {i for i, s in enumerate(spiele) if len(duelle(HALL, s["heim"], s["gast"], grenze)) >= MIN_DUELLE}
    log(f"H2H moeglich (>= {MIN_DUELLE} Duelle): {len(h2h_ok)} von {len(spiele)}")
    B = chargen([(i, (spiele[i]["heim"], spiele[i]["gast"])) for i in sorted(h2h_ok)])

    def laeufe(H):
        """Heim/Ausw (umbenannt) und je H2H-Gruppe ein Lauf: (name, Historie, Ziele, umbenannt)."""
        H = H[H["Date"] < grenze]
        z = [{"key": i, "liga": s["liga"], "datum": s["datum"], "heim": s["heim"] + "@H", "gast": s["gast"] + "@A", "quoten": None}
             for i, s in enumerate(spiele)]
        yield "ha", t_ha(H), z, True
        for b in B:
            z = [{"key": i, "liga": spiele[i]["liga"], "datum": spiele[i]["datum"], "heim": h, "gast": a, "quoten": None} for i, h, a in b]
            yield "h2h", t_h2h(H, [(h, a) for _, h, a in b], grenze), z, False

    gs = grenze.strftime("%Y-%m-%d")
    schritte = [("Basis", lambda: basis(HALL, spiele, grenze, share, sicht, h2h_ok))]
    try:
        import profi_modelle as PM
        HP = PM.historie()
        schritte.append(("Profi", lambda: profi(HP, spiele, gs, share, sicht, laeufe)))
    except Exception as ex:
        log("Profi-Historie:", ex)
    try:
        import neue_modelle as NM
        HN = daten.alle_spiele()
        HN = HN[HN["Date"] >= pd.Timestamp(NM.AB)].copy()
        HN["HTHG"] = pd.to_numeric(HN["HTHG"], errors="coerce"); HN["HTAG"] = pd.to_numeric(HN["HTAG"], errors="coerce")
        HN = NM.mit_quoten(HN)
        schritte.append(("SVR", lambda: svr(HN, spiele, gs, share, sicht, laeufe)))
        schritte.append(("K2", lambda: k2(HN, HALL, spiele, grenze, sicht, h2h_ok)))
    except Exception as ex:
        log("Neue-Modelle-Historie:", ex)
    schritte.append(("Baeume", lambda: baeume(HALL, spiele, grenze, share, sicht, laeufe)))
    nur = [x for x in os.environ.get("SICHT_NUR", "").split(",") if x]     # nur zum Testen einzelner Schritte
    for nm_, f in schritte:
        if nur and nm_ not in nur:
            continue
        log("==", nm_)
        try:
            f()
        except Exception as ex:
            import traceback
            log(nm_, "Fehler:", ex, traceback.format_exc()[-500:])
    stacking(d, sicht)

    # Eigene Datei docs/sichten.json (nur dieser Lauf schreibt sie -> kein Konflikt mit daten.json beim Speichern)
    n = defaultdict(int)
    je = {}
    for i, s in enumerate(spiele):
        e = {"sicht": rund(sicht[i])}
        for name, P in sicht[i].items():
            n[name] += bool(P)
        try:
            e["x"] = xwerte(HALL, s)
        except Exception:
            pass
        je[f'{s["datum"]}|{s["heim"]}|{s["gast"]}'] = e
        s["sicht"] = e["sicht"]          # nur fuer den xG-Vergleich unten
    try:
        arch = json.load(open(ARCHIV, encoding="utf-8"))
    except Exception:
        arch = []
    info = {"stand": pd.Timestamp.today().strftime("%Y-%m-%d"), "grenze": str(grenze.date()),
            "spiele": len(spiele), "ha": n["ha"], "h2h": n["h2h"], "h2h_moeglich": len(h2h_ok),
            "gruppen": len(B), "sekunden": round(time.time() - T0)}
    out = {"info": info, "normal": normalwerte(HALL, grenze), "xgvgl": xg_vergleich(arch + spiele), "spiele": je}
    with open(AUS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    d = {"sichten_info": info}
    log("fertig:", d["sichten_info"])


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        import traceback
        log("Fehler (sichten.json unveraendert):", ex, traceback.format_exc()[-800:])
    sys.exit(0)
