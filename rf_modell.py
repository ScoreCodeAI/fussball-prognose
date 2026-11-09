# ===== MODELL RF · RANDOM FOREST (reine Statistik, ohne Quoten) =====
# Viele unabhaengige Entscheidungsbaeume, jeder auf einer Zufalls-Stichprobe der Spiele und Merkmale;
# die Prognose ist der Durchschnitt aller Baeume (robust, kaum Ueberanpassung).
# Merkmale (gleiche Aufbereitung wie XGBoost, aber OHNE Buchmacher-Quoten):
#   Schuss-Staerken, Tor-Staerken (Angriff/Abwehr), Elo, Form (letzte 5), Ruhetage, Liga.
# Dadurch Prognose fuer ALLE Spiele, auch ohne Quoten, und unabhaengig vom Markt.
# Ehrlich: Training nur mit Spielen VOR der angezeigten Spielwoche.
import numpy as np
import pandas as pd

import kern
import xgb_modell as X

INFO = {"id": "rf", "name": "Random Forest · Statistik", "kurz": "RandomForest", "icon": "🌲",
        "farbe1": "#2bf5a0", "farbe2": "#7ff0ff",
        "beschreibung": "Random Forest aus Schuss- und Tor-Stärken, Elo, Form und Ruhetagen – ohne Buchmacher-Quoten"}
MARKT = {"m1", "mx", "m2", "mo"}
FEAT = [f for f in X.FEAT if f not in MARKT]
BAEUME, BLATT = 400, 60

# ===== RF+ · Vergleichs-Fassung mit neuen SportsAPI-Kennzahlen (nur Daten VOR dem Spiel) =====
#  xpts  = Punkte minus erwartete Punkte (xPts aus xG beider Teams, Poisson)  -> Glueck / Pech
#  fin   = Tore minus xG                                                        -> Abschluss-Form
#  gkp   = goalsPrevented (Torwart haelt mehr/weniger als erwartet)             -> Torwart-Form
#  xgd   = xG minus xG des Gegners                                              -> Chancen-Bilanz
#  bcm   = vergebene Grosschancen
# Je Team Mittel der letzten N Spiele MIT Statistik; ohne Statistik 0 (+ Anzahl als Merkmal).
# Angezeigt wird RF+ erst, wenn es im ehrlichen Vergleich auf genug neuen Spielen besser ist.
EXTRA = ["xpts_h", "xpts_a", "fin_h", "fin_a", "gkp_h", "gkp_a", "xgd_h", "xgd_a", "bcm_h", "bcm_a", "nst_h", "nst_a"]
N_LETZTE = 6
MIN_VERGLEICH = 500


def _xpts(xh, xa):
    from scipy.stats import poisson
    k = np.arange(11)
    M = np.outer(poisson.pmf(k, max(xh, 0.05)), poisson.pmf(k, max(xa, 0.05)))
    ph, px = np.tril(M, -1).sum(), np.trace(M)
    pa = 1 - ph - px
    return 3 * ph + px, 3 * pa + px


def _extra(spiele):
    """-> vor {key: merkmale vor dem Spiel}, team {(liga, team): aktuelle merkmale}"""
    hist, vor = {}, {}
    if "HXG" not in spiele:
        return vor, {}
    d = spiele.dropna(subset=["HG", "AG"]).sort_values("Date")

    def mm(lst):
        if not lst:
            return dict(xpts=0.0, fin=0.0, gkp=0.0, xgd=0.0, bcm=0.0, nst=0)
        a = np.array(lst, dtype=float)
        return dict(xpts=float(np.nanmean(a[:, 0])), fin=float(np.nanmean(a[:, 1])), gkp=float(np.nan_to_num(np.nanmean(a[:, 2]))),
                    xgd=float(np.nanmean(a[:, 3])), bcm=float(np.nan_to_num(np.nanmean(a[:, 4]))), nst=len(lst))
    for r in d.itertuples(index=False):
        a, b = (r.Div, r.Home), (r.Div, r.Away)
        ma, mb = mm(hist.get(a, [])[-N_LETZTE:]), mm(hist.get(b, [])[-N_LETZTE:])
        vor[X._key(r.Div, r.Date, r.Home, r.Away)] = {f"{k}_h": v for k, v in ma.items()} | {f"{k}_a": v for k, v in mb.items()}
        xh, xa = getattr(r, "HXG", np.nan), getattr(r, "AXG", np.nan)
        if pd.isna(xh) or pd.isna(xa):
            continue
        eh, ea = _xpts(float(xh), float(xa))
        hg, ag = float(r.HG), float(r.AG)
        ph = 3.0 if hg > ag else 1.0 if hg == ag else 0.0
        pa = 3.0 if ag > hg else 1.0 if hg == ag else 0.0
        g = lambda c: getattr(r, c, np.nan)
        hist.setdefault(a, []).append([ph - eh, hg - xh, g("HGP"), xh - xa, g("HBCM")])
        hist.setdefault(b, []).append([pa - ea, ag - xa, g("AGP"), xa - xh, g("ABCM")])
    team = {t: mm(v[-N_LETZTE:]) for t, v in hist.items()}
    return vor, team


def _ex_live(team, liga, h, a):
    ma = team.get((liga, h)) or dict(xpts=0.0, fin=0.0, gkp=0.0, xgd=0.0, bcm=0.0, nst=0)
    mb = team.get((liga, a)) or dict(xpts=0.0, fin=0.0, gkp=0.0, xgd=0.0, bcm=0.0, nst=0)
    return {f"{k}_h": v for k, v in ma.items()} | {f"{k}_a": v for k, v in mb.items()}


def _modell():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=BAEUME, min_samples_leaf=BLATT, max_features="sqrt",
                                  n_jobs=-1, random_state=7)


def _ziele(r):
    hg, ag = int(r.HG), int(r.AG)
    z = dict(y1=0 if hg > ag else 1 if hg == ag else 2, yo=int(hg + ag > 2), yb=int(hg > 0 and ag > 0))
    if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)):
        h1, a1 = int(r.HTHG), int(r.HTAG)
        z["yh"] = 0 if h1 > a1 else 1 if h1 == a1 else 2
        z["yh2"] = 0 if hg - h1 > ag - a1 else 1 if hg - h1 == ag - a1 else 2
    return z


def berechne(spiele, liste, log=print):
    vor, ende = X._features(spiele)
    ex_vor, ex_team = _extra(spiele)
    ligen = {d: i for i, d in enumerate(sorted(spiele["Div"].unique()))}
    stichtag = min(pd.Timestamp(e["datum"]) for e in liste) if liste else pd.Timestamp.today()
    rows = []
    for r in spiele.dropna(subset=["HG", "AG"]).itertuples(index=False):
        if r.Date >= stichtag:
            continue
        k = X._key(r.Div, r.Date, r.Home, r.Away)
        f = vor.get(k)
        if f:
            rows.append(dict(f, liga=ligen.get(r.Div, -1), **_ziele(r), **(ex_vor.get(k) or _ex_live({}, 0, 0, 0))))
    T = pd.DataFrame(rows)
    if len(T) < 1000:
        log(f"RandomForest: zu wenig Trainingsspiele ({len(T)})")
        return {}
    FPLUS = FEAT + EXTRA
    mit_stat = int((T["nst_h"] > 0).sum()) if "nst_h" in T else 0

    def lerne(F):
        ms = {}
        for ziel in ("y1", "yo", "yb", "yh", "yh2"):
            t = T.dropna(subset=[ziel]) if ziel in T else None
            if t is None or len(t) < 1000 or t[ziel].nunique() < 2:
                continue
            m = _modell(); m.fit(t[F], t[ziel].astype(int)); ms[ziel] = m
        return ms
    basis = lerne(FEAT)
    plus = lerne(FPLUS) if mit_stat >= 50 else {}
    status = lade_status()
    aktiv = bool(plus) and status.get("n", 0) >= MIN_VERGLEICH and status.get("rps_neu", 9) < status.get("rps_alt", 0)
    global VERGLEICH
    VERGLEICH = []
    out = {}
    for i, e in enumerate(liste):
        k = (e["liga"], e["datum"], e["heim"], e["gast"])
        f = vor.get(k) if e.get("ergebnis") else None
        if f is None:
            if e["liga"] not in ende:
                continue
            f = X._zukunft(ende[e["liga"]], e["liga"], e["datum"], e["heim"], e["gast"])
        ex = ex_vor.get(k) if e.get("ergebnis") else None
        ex = ex or _ex_live(ex_team, e["liga"], e["heim"], e["gast"])
        Xb = pd.DataFrame([dict(f, liga=ligen.get(e["liga"], -1))])[FEAT]
        Xp = pd.DataFrame([dict(f, liga=ligen.get(e["liga"], -1), **ex)])[FPLUS]
        if plus and "y1" in plus and "y1" in basis and (ex["nst_h"] > 0 or ex["nst_a"] > 0):
            VERGLEICH.append((e, basis["y1"].predict_proba(Xb)[0], plus["y1"].predict_proba(Xp)[0]))
        modelle, Xe = (plus, Xp) if aktiv else (basis, Xb)
        pr = lambda z: modelle[z].predict_proba(Xe)[0] if z in modelle else None
        p1, po, pb, ph, ph2 = pr("y1"), pr("yo"), pr("yb"), pr("yh"), pr("yh2")
        if p1 is None or po is None:
            continue
        M = kern.matrix_unabhaengig(f["lh"], f["la"])
        a, b = np.unravel_index(M.argmax(), M.shape)
        r0 = lambda x: round(float(x) * 100, 2)
        out[i] = {"p1": r0(p1[0]), "px": r0(p1[1]), "p2": r0(p1[2]), "ue25": r0(po[1]),
                  "bts": r0(pb[1]) if pb is not None else r0(f["sb"]),
                  "ht": [r0(x) for x in ph] if ph is not None else None,
                  "ht2": [r0(x) for x in ph2] if ph2 is not None else None,
                  "tipp": f"{a}:{b}", "xg": [round(f["lh"], 2), round(f["la"], 2)], "value": [],
                  # Halbzeit OVER 0,5 aus den erwarteten Toren (Anteil 1. HZ 45 %, 2. HZ 55 %)
                  "h1o05": r0(1 - np.exp(-0.45 * (f["lh"] + f["la"]))), "h2o05": r0(1 - np.exp(-0.55 * (f["lh"] + f["la"])))}
    log(f"RandomForest: {len(T)} Trainingsspiele ({mit_stat} mit SportsAPI-Kennzahlen), {len(out)} Prognosen | "
        f"angezeigt: {'RF+ (mit neuen Kennzahlen)' if aktiv else 'RF (bisher)'}")
    return out


# ===== Vergleich RF gegen RF+ (dauerhaft in docs/rf_plus.json) =====
PFAD = None


def lade_status():
    import json, os
    if not PFAD or not os.path.exists(PFAD):
        return {}
    try:
        return json.load(open(PFAD, encoding="utf-8")).get("status", {})
    except Exception:
        return {}


def vergleich_speichern(log=print):
    """Fertige Spiele (aus der angezeigten Woche, nicht im Training) mit Kennzahlen: RPS alt gegen neu protokollieren."""
    import json, os
    if not PFAD:
        return None
    alt = json.load(open(PFAD, encoding="utf-8")) if os.path.exists(PFAD) else {"spiele": {}}
    sp = alt.get("spiele", {})
    rps = lambda p, o: ((p[0] - (o == 0)) ** 2 + (p[0] + p[1] - (o <= 1)) ** 2) / 2
    for e, pb, pp in VERGLEICH:
        if not e.get("ergebnis"):
            continue
        k = f'{e["liga"]}|{e["datum"]}|{e["heim"]}|{e["gast"]}'
        if k in sp:
            continue
        h, a = map(int, e["ergebnis"].split(":"))
        o = 0 if h > a else 1 if h == a else 2
        sp[k] = [e["datum"], round(float(rps(pb, o)), 5), round(float(rps(pp, o)), 5)]
    werte = sorted(sp.values())
    n = len(werte)
    st = {"n": n, "ziel": MIN_VERGLEICH}
    if n:
        st["rps_alt"] = round(float(np.mean([w[1] for w in werte])), 5)
        st["rps_neu"] = round(float(np.mean([w[2] for w in werte])), 5)
        st["besser_neu"] = int(sum(w[2] < w[1] for w in werte))
        tage, cum, verlauf = {}, 0.0, []
        for i, w in enumerate(werte, 1):
            cum += w[1] - w[2]
            tage[w[0]] = (i, round(cum / i * 1000, 3))
        st["verlauf"] = [[d, v[0], v[1]] for d, v in sorted(tage.items())]
    st["aktiv"] = bool(n >= MIN_VERGLEICH and st.get("rps_neu", 9) < st.get("rps_alt", 0))
    json.dump({"status": st, "spiele": sp}, open(PFAD, "w", encoding="utf-8"), ensure_ascii=False)
    log(f"RF+ Vergleich: {n}/{MIN_VERGLEICH} Spiele | RPS RF {st.get('rps_alt')} vs RF+ {st.get('rps_neu')}"
        f"{' -> RF+ aktiv' if st['aktiv'] else ''}")
    return st


def rueckblick(spiele, ab, log=print):
    """Ehrlicher Test: Training vor 'ab', Prognose ab 'ab'."""
    vor, _ = X._features(spiele)
    ligen = {d: i for i, d in enumerate(sorted(spiele["Div"].unique()))}
    ab = pd.Timestamp(ab); tr, te = [], []
    for r in spiele.dropna(subset=["HG", "AG"]).itertuples(index=False):
        f = vor.get(X._key(r.Div, r.Date, r.Home, r.Away))
        if f:
            z = dict(f, liga=ligen.get(r.Div, -1), **_ziele(r), HG=int(r.HG), AG=int(r.AG))
            (tr if r.Date < ab else te).append(z)
    T, E = pd.DataFrame(tr), pd.DataFrame(te)
    m1, mo = _modell(), _modell()
    m1.fit(T[FEAT], T["y1"]); mo.fit(T[FEAT], T["yo"])
    return E, m1.predict_proba(E[FEAT]), mo.predict_proba(E[FEAT])
