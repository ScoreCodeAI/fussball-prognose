# ===== MODELL XG · EXPECTED GOALS (SportsAPI) =====
# Wie das Schuss-Modell, aber mit echtem xG: Lernziel = Mischung aus Toren und xG (falls vorhanden),
# sonst Schuesse aufs Tor x Liga-Verwertung, sonst nur Tore. Je mehr Spiele mit xG im Archiv, desto mehr wirkt es.
# Anteil xG wird je Land im Backtest automatisch gewaehlt (0 % bis 80 %) - ehrlich, nur aus frueheren Spielen.
# ----- urspruengliche Beschreibung (Schuss-Modell) -----
# Tore sind zufaellig, Schuesse aufs Tor zeigen die echte Staerke frueher.
# Jedes Team hat Angriff/Abwehr (log-Skala), laufend nachgefuehrt wie bei Elo - aber das Ziel ist
# eine Mischung aus Toren und "Schuss-Toren" (Schuesse aufs Tor x Liga-Verwertung).
# Im Test 2024-26 (16 Ligen) in jeder Liga genauer als das gleiche Modell nur mit Toren.
# Automatische Auswahl: Lernrate und Schuss-Anteil werden je Land im Backtest gewaehlt.
# Halbzeit: eigene Halbzeit-Staerke aus Halbzeit-Toren und Schuss-Anteil.
import numpy as np
import pandas as pd

import kern

NAME = "xG · Expected Goals"
KURZ = "xG"
ICON = "🎯"
FARBE1 = "#ff7a59"
FARBE2 = "#ffe04d"
BESCHREIBUNG = "Teamstärke nur aus Statistik: xG, xGOT, Großchancen, Schüsse im Strafraum, Schüsse aufs Tor – keine Tore"

JAHRE = 5
VALID_TAGE = 150
GITTER_K = [0.02, 0.03, 0.045]
GITTER_W = [1.0]      # nur Statistik (keine Tore)
STANDARD = (0.03, 0.6)
_TUNING = {}


def _ziel(r, conv):
    """Lernziel je Team NUR aus Statistik (keine Tore): Mittel der verfuegbaren Werte, je in Tor-Einheiten umgerechnet.
    xG zaehlt direkt; xGOT, Grosschancen, Schuesse im Strafraum, Schuesse aufs Tor ueber laufende Liga-Umrechnung.
    Gibt None zurueck, wenn das Spiel keine Statistik hat (dann lernt das Modell nichts daraus)."""
    out = []
    for h, a in (("HXG", "AXG"), ("AXG", "HXG")):
        pass
    th, ta, n = 0.0, 0.0, 0
    for kh, ka, k in (("HXG", "AXG", None), ("HXGOT", "AXGOT", "xgot"), ("HBC", "ABC", "bc"), ("HSIB", "ASIB", "sib"), ("HST", "AST", "sot")):
        vh, va = getattr(r, kh, np.nan), getattr(r, ka, np.nan)
        if pd.isna(vh) or pd.isna(va):
            continue
        f = 1.0 if k is None else conv[k]
        th += vh * f; ta += va * f; n += 1
    return (th / n, ta / n) if n else None


def _lauf(d, K, w=None, bis=None):
    """Laeuft chronologisch durch die Spiele; Teamstaerke nur aus Statistik. ll ab 'bis' (gegen echte Tore gemessen)."""
    A, D = {}, {}
    mu, h, anteil_ht = np.log(1.35), 0.12, 0.45
    conv = {"xgot": 0.75, "bc": 0.4, "sib": 0.12, "sot": 0.32}
    ll, n, n_stat = 0.0, 0, 0
    for r in d.itertuples(index=False):
        a, b = r.Home, r.Away
        lh = np.exp(mu + h + A.get(a, 0) - D.get(b, 0))
        la = np.exp(mu - h + A.get(b, 0) - D.get(a, 0))
        hg, ag = float(r.HG), float(r.AG)
        z = _ziel(r, conv)
        if z is None:
            continue
        if bis is not None and r.Date >= bis:
            ll += hg * np.log(lh) - lh + ag * np.log(la) - la
            n += 1
        n_stat += 1
        th, ta = z
        # Umrechnungen laufend nachfuehren (Verhaeltnis xG zu Kennzahl, nur wenn xG vorhanden)
        xh, xa = getattr(r, "HXG", np.nan), getattr(r, "AXG", np.nan)
        for k, (kh, ka) in {"xgot": ("HXGOT", "AXGOT"), "bc": ("HBC", "ABC"), "sib": ("HSIB", "ASIB"), "sot": ("HST", "AST")}.items():
            vh, va = getattr(r, kh, np.nan), getattr(r, ka, np.nan)
            if not (pd.isna(vh) or pd.isna(va) or pd.isna(xh) or pd.isna(xa)) and vh + va > 0:
                conv[k] = 0.99 * conv[k] + 0.01 * ((xh + xa) / (vh + va))
        eh, ea = th - lh, ta - la
        A[a] = A.get(a, 0) + K * eh; D[b] = D.get(b, 0) - K * eh
        A[b] = A.get(b, 0) + K * ea; D[a] = D.get(a, 0) - K * ea
        mu += 0.002 * (eh + ea)
        h += 0.002 * (eh - ea)
        if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)) and hg + ag > 0:
            anteil_ht = 0.99 * anteil_ht + 0.01 * ((r.HTHG + r.HTAG) / (hg + ag))
    return {"A": A, "D": D, "mu": mu, "h": h, "ht": float(np.clip(anteil_ht, 0.35, 0.5)), "n_stat": n_stat}, (ll / n if n else None)


def _tune(d, heute):
    key = (tuple(sorted(d["Div"].unique())), heute.year, (heute.month - 1) // 3)
    if key in _TUNING:
        return _TUNING[key]
    grenze = heute - pd.Timedelta(days=VALID_TAGE)
    best, best_ll = STANDARD, -np.inf
    if (d["Date"] >= grenze).sum() >= 80:
        for K in GITTER_K:
            for w in GITTER_W:
                _, ll = _lauf(d, K, w, bis=grenze)
                if ll is not None and ll > best_ll:
                    best, best_ll = (K, w), ll
    _TUNING[key] = best
    return best


def trainiere(df, heute):
    heute = pd.Timestamp(heute)
    d = df[(df["Date"] < heute) & (df["Date"] >= heute - pd.DateOffset(years=JAHRE))]
    d = d.dropna(subset=["HG", "AG"]).sort_values("Date")
    if len(d) < 200:
        raise ValueError("zu wenig Daten")
    K, w = _tune(d, heute)
    m, _ = _lauf(d, K, w)
    m.update(K=K, w=w, n=len(d), schuss=int(d["HST"].notna().sum()), xg=int(d["HXG"].notna().sum()) if "HXG" in d else 0)
    return m


def vorhersage(m, heim, gast):
    lh = float(np.exp(m["mu"] + m["h"] + m["A"].get(heim, 0) - m["D"].get(gast, 0)))
    la = float(np.exp(m["mu"] - m["h"] + m["A"].get(gast, 0) - m["D"].get(heim, 0)))
    ft = kern.matrix_dixon_coles(lh, la, kern.REMIS_RHO)
    ht = kern.matrix_unabhaengig(lh * m["ht"], la * m["ht"])
    return {"ft": ft, "ht": ht, "xg": [lh, la]}


def diagnose(m):
    return f"Lernrate {m['K']}, nur Statistik | {m['xg']} Spiele mit echtem xG, {m['n_stat']} mit Statistik von {m['n']}"
