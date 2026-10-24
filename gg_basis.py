# ===== BASIS: Modell "Gemeinsame Gegner" (Common Opponents) =====
# Idee: Fuer Heim gegen Gast werden beide Teams an DENSELBEN Gegnern gemessen.
#   1) Wie hat das Heimteam gegen die Teams gespielt, gegen die auch das Gastteam gespielt hat?
#   2) Wie hat das Gastteam gegen die Teams gespielt, gegen die auch das Heimteam gespielt hat?
#   3) Direkter Vergleich der beiden Teams (falls vorhanden)
# Gibt es zu wenige gemeinsame Gegner, wird automatisch mehr Gewicht auf die
# allgemeine Form im Zeitfenster gelegt (und notfalls auf den Liga-Schnitt).
# Heimvorteil wird herausgerechnet und bei der Prognose wieder eingerechnet.
import numpy as np
import pandas as pd

H2H_JAHRE = 2        # direkter Vergleich: so weit zurueck
H2H_ANTEIL = 0.2     # Gewicht des direkten Vergleichs, wenn vorhanden
SICHERHEIT = 3.0     # ab so vielen gemeinsamen Spielen vertraut das Modell ihnen zu ~50 %


def _wmittel(werte, gew):
    g = np.sum(gew)
    return float(np.dot(werte, gew) / g) if g > 0 else None


def trainiere(df, heute, wochen):
    liga = df[(df["Date"] < heute) & (df["Date"] >= heute - pd.Timedelta(days=365))]
    if len(liga) < 50:
        raise ValueError("zu wenig Daten")
    hg, ag = liga["HG"].mean(), liga["AG"].mean()
    mitte = (hg + ag) / 2
    fH, fA = hg / mitte, ag / mitte                     # Heim- und Auswaertsfaktor
    halbwert = max(wochen * 7 / 2, 7)

    def tabelle(d):
        t = {}
        alter = (heute - d["Date"]).dt.days.to_numpy()
        w = 0.5 ** (alter / halbwert)
        for (h, a, x, y, ww, dt) in zip(d["Home"], d["Away"], d["HG"], d["AG"], w, d["Date"]):
            t.setdefault(h, []).append((a, x / fH, y / fA, ww, dt))   # neutralisierte Tore
            t.setdefault(a, []).append((h, y / fA, x / fH, ww, dt))
        return t

    fenster = df[(df["Date"] < heute) & (df["Date"] >= heute - pd.Timedelta(weeks=wochen))]
    h2h = df[(df["Date"] < heute) & (df["Date"] >= heute - pd.DateOffset(years=H2H_JAHRE))]
    return {"mitte": mitte, "fH": fH, "fA": fA, "fenster": tabelle(fenster), "h2h": tabelle(h2h)}


def tore_erwartet(m, heim, gast):
    F, mitte = m["fenster"], m["mitte"]
    if heim not in F and gast not in F and heim not in m["h2h"] and gast not in m["h2h"]:
        raise KeyError("Teams unbekannt")
    A, B = F.get(heim, []), F.get(gast, [])

    def form(spiele, nur=None):
        s = [x for x in spiele if nur is None or x[0] in nur]
        if not s:
            return None, None, 0.0
        w = np.array([x[3] for x in s])
        return _wmittel([x[1] for x in s], w), _wmittel([x[2] for x in s], w), float(len(s))

    gemeinsam = ({x[0] for x in A} & {x[0] for x in B}) - {heim, gast}
    att_a, def_a, n_a = form(A, gemeinsam)
    att_b, def_b, n_b = form(B, gemeinsam)
    att_a0, def_a0, n_a0 = form(A)
    att_b0, def_b0, n_b0 = form(B)

    def mix(wert, n, basis):
        basis = mitte if basis is None else basis
        if wert is None:
            return basis
        a = n / (n + SICHERHEIT)
        return a * wert + (1 - a) * basis

    # gemeinsame Gegner -> allgemeine Form im Fenster -> Liga-Schnitt
    att_a = mix(att_a, n_a, mix(att_a0, n_a0, None))
    def_a = mix(def_a, n_a, mix(def_a0, n_a0, None))
    att_b = mix(att_b, n_b, mix(att_b0, n_b0, None))
    def_b = mix(def_b, n_b, mix(def_b0, n_b0, None))

    lh = (att_a + def_b) / 2
    la = (att_b + def_a) / 2

    # direkter Vergleich
    d = [x for x in m["h2h"].get(heim, []) if x[0] == gast]
    if d:
        w = np.array([x[3] for x in d])
        anteil = H2H_ANTEIL * min(1.0, len(d) / 3)
        lh = (1 - anteil) * lh + anteil * _wmittel([x[1] for x in d], w)
        la = (1 - anteil) * la + anteil * _wmittel([x[2] for x in d], w)

    return max(lh * m["fH"], 0.15), max(la * m["fA"], 0.15)
