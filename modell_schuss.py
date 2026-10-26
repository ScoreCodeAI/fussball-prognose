# ===== MODELL SCHUSS · TORSCHUESSE STATT TORE =====
# Tore sind zufaellig, Schuesse aufs Tor zeigen die echte Staerke frueher.
# Jedes Team hat Angriff/Abwehr (log-Skala), laufend nachgefuehrt wie bei Elo - aber das Ziel ist
# eine Mischung aus Toren und "Schuss-Toren" (Schuesse aufs Tor x Liga-Verwertung).
# Im Test 2024-26 (16 Ligen) in jeder Liga genauer als das gleiche Modell nur mit Toren.
# Automatische Auswahl: Lernrate und Schuss-Anteil werden je Land im Backtest gewaehlt.
# Halbzeit: eigene Halbzeit-Staerke aus Halbzeit-Toren und Schuss-Anteil.
import numpy as np
import pandas as pd

import kern

NAME = "Schuss · Torschüsse"
KURZ = "Schuss"
ICON = "⚡"
FARBE1 = "#2bf5a0"
FARBE2 = "#22e4ff"
BESCHREIBUNG = "Teamstärke aus Schüssen aufs Tor statt nur aus Toren, Lernrate und Schuss-Anteil automatisch gewählt"

JAHRE = 5
VALID_TAGE = 150
GITTER_K = [0.02, 0.03, 0.045]
GITTER_W = [0.4, 0.6, 0.8]      # Anteil Schuss-Tore am Lernziel
STANDARD = (0.03, 0.6)
_TUNING = {}


def _lauf(d, K, w, bis=None):
    """Laeuft chronologisch durch die Spiele; liefert Zustand und (optional) Log-Likelihood ab 'bis'."""
    A, D = {}, {}
    mu, h, conv, anteil_ht = np.log(1.35), 0.12, 0.32, 0.45
    ll, n = 0.0, 0
    for r in d.itertuples(index=False):
        a, b = r.Home, r.Away
        lh = np.exp(mu + h + A.get(a, 0) - D.get(b, 0))
        la = np.exp(mu - h + A.get(b, 0) - D.get(a, 0))
        hg, ag = float(r.HG), float(r.AG)
        if bis is not None and r.Date >= bis:
            ll += hg * np.log(lh) - lh + ag * np.log(la) - la
            n += 1
        sh, sa = r.HST, r.AST
        if pd.isna(sh) or pd.isna(sa):
            th, ta = hg, ag
        else:
            th, ta = w * sh * conv + (1 - w) * hg, w * sa * conv + (1 - w) * ag
            s = sh + sa
            if s > 0:
                conv = 0.995 * conv + 0.005 * ((hg + ag) / s)
        eh, ea = th - lh, ta - la
        A[a] = A.get(a, 0) + K * eh; D[b] = D.get(b, 0) - K * eh
        A[b] = A.get(b, 0) + K * ea; D[a] = D.get(a, 0) - K * ea
        mu += 0.002 * (eh + ea)
        h += 0.002 * (eh - ea)
        if not (pd.isna(r.HTHG) or pd.isna(r.HTAG)) and hg + ag > 0:
            anteil_ht = 0.99 * anteil_ht + 0.01 * ((r.HTHG + r.HTAG) / (hg + ag))
    return {"A": A, "D": D, "mu": mu, "h": h, "ht": float(np.clip(anteil_ht, 0.35, 0.5))}, (ll / n if n else None)


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
    m.update(K=K, w=w, n=len(d), schuss=int(d["HST"].notna().sum()))
    return m


def vorhersage(m, heim, gast):
    lh = float(np.exp(m["mu"] + m["h"] + m["A"].get(heim, 0) - m["D"].get(gast, 0)))
    la = float(np.exp(m["mu"] - m["h"] + m["A"].get(gast, 0) - m["D"].get(heim, 0)))
    ft = kern.matrix_dixon_coles(lh, la, -0.05)
    ht = kern.matrix_unabhaengig(lh * m["ht"], la * m["ht"])
    return {"ft": ft, "ht": ht, "xg": [lh, la]}


def diagnose(m):
    return f"Lernrate {m['K']}, Schuss-Anteil {m['w']:.0%}, Halbzeit-Anteil {m['ht']:.0%} | {m['schuss']} von {m['n']} Spielen mit Schussdaten"
