# ===== MODELL: Tore-Durchschnitt (einfaches Vergleichsmodell) =====
import pandas as pd

NAME = "Tore-Durchschnitt (12 Monate)"


def trainiere(df, heute):
    d = df[df["Date"] >= heute - pd.Timedelta(days=365)]
    heim = d.groupby("Home").agg(geschossen=("HG", "mean"), kassiert=("AG", "mean"))
    gast = d.groupby("Away").agg(geschossen=("AG", "mean"), kassiert=("HG", "mean"))
    return {"hs": d["HG"].mean(), "gs": d["AG"].mean(), "heim": heim, "gast": gast}


def tore_erwartet(m, heim, gast):
    h = m["heim"].loc[heim]
    g = m["gast"].loc[gast]
    lh = h["geschossen"] * g["kassiert"] / m["hs"]
    la = g["geschossen"] * h["kassiert"] / m["gs"]
    return max(float(lh), 0.1), max(float(la), 0.1)
