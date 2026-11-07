# ===== MODELL ELO · CLUBELO + EIGENES ELO =====
# Teamstaerke als Elo-Zahl (wie im Schach):
#   - Wo vorhanden, wird der echte ClubElo-Wert vor dem Spiel uebernommen (Spalten HomeElo/AwayElo der Historie).
#     ClubElo kennt auch Europapokal-Spiele und Staerke zwischen Ligen - Information, die die anderen Modelle nicht haben.
#   - Fehlt ClubElo (aktuelle Wochen, MLS), wird die Zahl nach jedem Spiel selbst fortgeschrieben:
#     Erwartung = 1 / (1 + 10^(-(Elo_heim + Heimvorteil - Elo_gast) / 400)), Anpassung K x Tordifferenz-Faktor.
#   - Aus der Elo-Differenz werden per Poisson-Regression die erwarteten Tore beider Teams gelernt (je Land, ehrlich:
#     nur Spiele VOR dem Stichtag). Keine Quoten, keine festen Favoriten-Boni.
import numpy as np
import pandas as pd

import kern

NAME = "Elo · ClubElo"
KURZ = "Elo"
ICON = "♟"
FARBE1 = "#7ff0ff"
FARBE2 = "#2bf5a0"
BESCHREIBUNG = "Teamstärke als Elo-Zahl: echte ClubElo-Werte (inkl. Europapokal), danach selbst nach jedem Spiel fortgeschrieben"

JAHRE = 5
K = 20.0
HEIM = 65.0
START = 1500.0


def _faktor(gd):
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


def lauf(d):
    """Chronologisch: Elo vor jedem Spiel (fuer das Lernen) und Endstand. -> (vorher-Liste, R, verlauf)"""
    R, verlauf, vor = {}, {}, []
    for r in d.itertuples(index=False):
        a, b = r.Home, r.Away
        he, ae = getattr(r, "HE", np.nan), getattr(r, "AE", np.nan)
        if not pd.isna(he):
            R[a] = float(he)
        if not pd.isna(ae):
            R[b] = float(ae)
        ra, rb = R.get(a, START), R.get(b, START)
        vor.append((ra - rb, float(r.HG), float(r.AG)))
        e = 1 / (1 + 10 ** (-(ra + HEIM - rb) / 400))
        w = 1.0 if r.HG > r.AG else 0.5 if r.HG == r.AG else 0.0
        dlt = K * _faktor(r.HG - r.AG) * (w - e)
        R[a], R[b] = ra + dlt, rb - dlt
        for t, v in ((a, R[a]), (b, R[b])):
            verlauf.setdefault(t, []).append(round(v, 1))
    return vor, R, verlauf


def _poisson(x, y):
    w = np.array([np.log(max(y.mean(), 0.1)), 0.0])
    X = np.c_[np.ones_like(x), x]
    for _ in range(25):
        mu = np.exp(np.clip(X @ w, -5, 3))
        w = w + np.linalg.solve(X.T @ (X * mu[:, None]) + 1e-6 * np.eye(2), X.T @ (y - mu))
    return w


def trainiere(df, heute):
    heute = pd.Timestamp(heute)
    d = df[(df["Date"] < heute) & (df["Date"] >= heute - pd.DateOffset(years=JAHRE))]
    d = d.dropna(subset=["HG", "AG"]).sort_values("Date")
    if len(d) < 200:
        raise ValueError("zu wenig Daten")
    vor, R, _ = lauf(d)
    v = np.array(vor[min(len(vor) // 4, 300):])
    x = v[:, 0] / 100
    wh, wa = _poisson(x, v[:, 1]), _poisson(-x, v[:, 2])
    ht = d.dropna(subset=["HTHG", "HTAG"])
    anteil = float((ht["HTHG"] + ht["HTAG"]).sum() / max((ht["HG"] + ht["AG"]).sum(), 1)) if len(ht) > 100 else 0.45
    club = int(d["HE"].notna().sum()) if "HE" in d else 0
    return {"R": R, "wh": wh, "wa": wa, "ht": anteil, "n": len(d), "club": club}


def vorhersage(m, heim, gast):
    x = (m["R"].get(heim, START) - m["R"].get(gast, START)) / 100
    lh = float(np.clip(np.exp(m["wh"][0] + m["wh"][1] * x), 0.15, 5))
    la = float(np.clip(np.exp(m["wa"][0] + m["wa"][1] * -x), 0.15, 5))
    ft = kern.matrix_dixon_coles(lh, la, kern.REMIS_RHO)
    ht = kern.matrix_unabhaengig(lh * m["ht"], la * m["ht"])
    return {"ft": ft, "ht": ht, "xg": [lh, la]}


def diagnose(m):
    return f"{m['n']} Spiele, davon {m['club']} mit ClubElo-Wert | +100 Elo = Heimtore x{np.exp(m['wh'][1]):.2f}"


def anzeige(df, liste, log=print):
    """Webseite: Elo beider Teams vor dem Spiel, Rang in der Liga und Trend (letzte 5 Spiele)."""
    n = 0
    for liga, d in df.dropna(subset=["HG", "AG"]).sort_values("Date").groupby("Div"):
        d = d[d["Date"] >= d["Date"].max() - pd.DateOffset(years=JAHRE)]
        spiele = [e for e in liste if e["liga"] == liga]
        if not spiele:
            continue
        cache = {}
        for e in spiele:
            if e["datum"] not in cache:
                vor = d[d["Date"] < pd.Timestamp(e["datum"])]
                if len(vor) < 50:
                    cache[e["datum"]] = None
                else:
                    _, R, verl = lauf(vor)
                    akt = set(vor[vor["Date"] >= vor["Date"].max() - pd.Timedelta(days=200)][["Home", "Away"]].values.ravel())
                    cache[e["datum"]] = (R, verl, sorted([t for t in akt if t in R], key=lambda t: -R[t]))
            if not cache[e["datum"]]:
                continue
            R, verl, rang = cache[e["datum"]]
            out = {}
            for k, t in (("h", e["heim"]), ("a", e["gast"])):
                if t in R:
                    v = verl.get(t, [])
                    out[k] = {"elo": round(R[t]), "rang": (rang.index(t) + 1) if t in rang else None, "von": len(rang),
                              "trend": round(v[-1] - v[-6], 1) if len(v) > 5 else None}
            if out:
                ex = 1 / (1 + 10 ** (-((out.get("h", {}).get("elo", START) + HEIM) - out.get("a", {}).get("elo", START)) / 400))
                out["erw"] = round(ex * 100, 1)
                e["elo"] = out
                n += 1
    log(f"Elo-Anzeige: {n} Spiele")
