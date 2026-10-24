# ===== MODELL: Poisson (zeitgewichtet) =====
# Jedes Modell braucht: NAME, trainiere(df, heute), tore_erwartet(modell, heim, gast)
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

NAME = "Poisson (zeitgewichtet)"
JAHRE = 3            # wie viele Jahre zurueck gelernt wird
HALBWERTSZEIT = 240  # nach so vielen Tagen zaehlt ein Spiel nur noch halb


def trainiere(df, heute):
    df = df[df["Date"] >= heute - pd.DateOffset(years=JAHRE)]
    w = 0.5 ** ((heute - df["Date"]).dt.days / HALBWERTSZEIT)
    a = pd.DataFrame({"team": df["Home"], "gegner": df["Away"], "heim": 1, "tore": df["HG"], "w": w})
    b = pd.DataFrame({"team": df["Away"], "gegner": df["Home"], "heim": 0, "tore": df["AG"], "w": w})
    d = pd.concat([a, b], ignore_index=True)
    d = d[np.isfinite(d["w"]) & (d["w"] > 0) & d["tore"].notna()]   # ungueltige Zeilen entfernen
    if len(d) < 50:
        raise ValueError("zu wenig Daten")
    return smf.glm("tore ~ heim + team + gegner", data=d,
                   family=sm.families.Poisson(), freq_weights=d["w"]).fit()


def tore_erwartet(modell, heim, gast):
    lh = modell.predict(pd.DataFrame({"team": [heim], "gegner": [gast], "heim": [1]})).iloc[0]
    la = modell.predict(pd.DataFrame({"team": [gast], "gegner": [heim], "heim": [0]})).iloc[0]
    return float(lh), float(la)
