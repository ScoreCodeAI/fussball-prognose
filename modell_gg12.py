# ===== MODELL: Gemeinsame Gegner · 12 Wochen =====
# Rechenweg siehe gg_basis.py - hier wird nur das Zeitfenster festgelegt.
import gg_basis

NAME = "Gemeinsame Gegner · 12 Wochen"
KURZ = "GG 12W"
ICON = "G12"
FARBE1 = "#4fa8ff"
FARBE2 = "#7b5cff"
BESCHREIBUNG = "Beide Teams an denselben Gegnern gemessen, letzte 12 Wochen, plus direkter Vergleich"
WOCHEN = 12


def trainiere(df, heute):
    return gg_basis.trainiere(df, heute, WOCHEN)


def tore_erwartet(modell, heim, gast):
    return gg_basis.tore_erwartet(modell, heim, gast)
