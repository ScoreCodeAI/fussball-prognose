# ===== MODELL: Copula · Torverteilung (Engine 3) =====
# Vollzeit und Halbzeit sind zwei GETRENNTE Modelle mit eigenen Parametern,
# trainiert auf echten Vollzeit- bzw. Halbzeitstaenden.
#
# Je Ziel (Vollzeit / Halbzeit):
#  1) Poisson-Grundmodell mit getrennter Heim- und Auswaertsstaerke je Team
#     (Partial Pooling zur Team-Grundstaerke, siehe kern.poisson_fit)
#  2) Zeitgewichtung: neuere Spiele zaehlen mehr. Wie schnell alte Spiele an Gewicht verlieren
#     und wie stark Heim/Auswaerts zusammengezogen werden, wird AUTOMATISCH gewaehlt:
#     Validierung auf den letzten 120 Tagen VOR dem Prognosezeitpunkt (kein Blick in die Zukunft),
#     neu bestimmt einmal pro Quartal.
#  3) Abhaengigkeit zwischen Heim- und Gasttoren wird automatisch gewaehlt:
#     unabhaengig, Dixon-Coles (mehr/weniger knappe Ergebnisse) oder Frank-Copula.
#  4) Aus der gemeinsamen Ergebnis-Matrix werden alle Maerkte abgeleitet.
#
# Gelernt aus Daten: alle Teamstaerken, Heimvorteil, Zeitgewichtung, Pooling-Staerke, Abhaengigkeit.
# Feste Annahmen: Poisson-Raender, max. 4 Jahre Daten (Rechenzeit; alte Spiele wiegen ohnehin kaum).
# Unbekannte Teams (z. B. Aufsteiger ohne Historie) bekommen Liga-Durchschnitt.
import numpy as np
import pandas as pd

import kern

NAME = "Copula · Torverteilung"
KURZ = "Copula"
ICON = "C"
FARBE1 = "#22e4ff"
FARBE2 = "#a855f7"
BESCHREIBUNG = "Gemeinsame Verteilung von Heim- und Gasttoren, Heim/Auswärts getrennt, Halbzeit eigenständig"

DATEN_JAHRE = 4
VALID_TAGE = 120
GITTER_XI = [0.0015, 0.003, 0.005, 0.008]     # Halbwertszeit ca. 460 / 230 / 140 / 87 Tage
GITTER_SPLIT = [1.0, 4.0, 16.0]               # schwach / mittel / stark zusammengezogen
LAM_BASIS = 0.5
STANDARD = (0.003, 4.0)
_TUNING = {}


def _spalten(ziel):
    return ("HG", "AG") if ziel == "ft" else ("HTHG", "HTAG")


def _auswahl(df, von, bis, ziel):
    d = df[(df["Date"] < bis) & (df["Date"] >= von)]
    sh, sa = _spalten(ziel)
    return d.dropna(subset=[sh, sa])


def _fit(d, heute, xi, split, ziel):
    sh, sa = _spalten(ziel)
    teams = sorted(set(d["Home"]) | set(d["Away"]))
    idx = {t: k for k, t in enumerate(teams)}
    i, j = d["Home"].map(idx).to_numpy(), d["Away"].map(idx).to_numpy()
    w = np.exp(-xi * (heute - d["Date"]).dt.days.to_numpy())
    p = kern.poisson_fit(i, j, d[sh].to_numpy(), d[sa].to_numpy(), w, len(teams), LAM_BASIS, split)
    return {"p": p, "n": len(teams), "idx": idx, "xi": xi, "split": split}


def _lambdas(f, heim, gast):
    i = np.array([f["idx"].get(t, -1) for t in np.atleast_1d(heim)])
    j = np.array([f["idx"].get(t, -1) for t in np.atleast_1d(gast)])
    return kern.poisson_lambdas(f["p"], f["n"], i, j)


def _tune(df, heute, ziel):
    key = (tuple(sorted(df["Div"].unique())), ziel, heute.year, (heute.month - 1) // 3)
    if key in _TUNING:
        return _TUNING[key]
    grenze = heute - pd.Timedelta(days=VALID_TAGE)
    tr = _auswahl(df, grenze - pd.DateOffset(years=DATEN_JAHRE), grenze, ziel)
    va = _auswahl(df, grenze, heute, ziel)
    best = STANDARD
    if len(tr) >= 300 and len(va) >= 80:
        sh, sa = _spalten(ziel)
        best_ll = -np.inf
        for xi in GITTER_XI:
            for split in GITTER_SPLIT:
                f = _fit(tr, grenze, xi, split, ziel)
                lh, la = _lambdas(f, va["Home"].to_numpy(), va["Away"].to_numpy())
                ll = np.sum(kern.poisson.logpmf(va[sh].to_numpy(), lh) + kern.poisson.logpmf(va[sa].to_numpy(), la))
                if ll > best_ll:
                    best, best_ll = (xi, split), ll
    _TUNING[key] = best
    return best


def trainiere(df, heute):
    m = {}
    for ziel in ("ft", "ht"):
        d = _auswahl(df, heute - pd.DateOffset(years=DATEN_JAHRE), heute, ziel)
        if len(d) < 200:
            m[ziel] = None
            continue
        xi, split = _tune(df, heute, ziel)
        f = _fit(d, heute, xi, split, ziel)
        sh, sa = _spalten(ziel)
        lh, la = _lambdas(f, d["Home"].to_numpy(), d["Away"].to_numpy())
        w = np.exp(-xi * (heute - d["Date"]).dt.days.to_numpy())
        f["abh"] = kern.waehle_abhaengigkeit(d[sh].to_numpy(), d[sa].to_numpy(), lh, la, w)
        m[ziel] = f
    if m["ft"] is None:
        raise ValueError("zu wenig Daten")
    return m


def vorhersage(m, heim, gast):
    lh, la = _lambdas(m["ft"], heim, gast)
    lh, la = float(lh[0]), float(la[0])
    out = {"ft": kern.matrix(lh, la, m["ft"]["abh"]), "ht": None, "xg": [lh, la]}
    if m["ht"] is not None:
        hh, ha = _lambdas(m["ht"], heim, gast)
        out["ht"] = kern.matrix(float(hh[0]), float(ha[0]), m["ht"]["abh"])
    return out


def diagnose(m):
    t = []
    for ziel, name in (("ft", "Vollzeit"), ("ht", "Halbzeit")):
        f = m.get(ziel)
        if f is None:
            t.append(f"{name}: keine Daten")
        else:
            typ, par = f["abh"]
            t.append(f"{name}: Halbwertszeit {np.log(2) / f['xi']:.0f} Tage, Pooling {f['split']:g}, "
                     f"Abhängigkeit {typ} ({par:+.3f}), Heimvorteil {np.exp(f['p'][1]):.2f}x")
    return " | ".join(t)
