# ===== MODELL: Gemeinsame Gegner · 4 Wochen =====
# Rechenweg siehe gg_basis.py - hier wird nur das Zeitfenster festgelegt.
import gg_basis

NAME = "Gemeinsame Gegner · 4 Wochen"
KURZ = "GG 4W"
ICON = "G4"
FARBE1 = "#2bf5a0"
FARBE2 = "#22e4ff"
BESCHREIBUNG = "Beide Teams an denselben Gegnern gemessen, letzte 4 Wochen, plus direkter Vergleich"
WOCHEN = 4


def trainiere(df, heute):
    return gg_basis.trainiere(df, heute, WOCHEN)


def tore_erwartet(modell, heim, gast):
    return gg_basis.tore_erwartet(modell, heim, gast)
