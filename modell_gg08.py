# ===== MODELL: Gemeinsame Gegner · 8 Wochen =====
# Rechenweg siehe gg_basis.py - hier wird nur das Zeitfenster festgelegt.
import gg_basis

NAME = "Gemeinsame Gegner · 8 Wochen"
KURZ = "GG 8W"
ICON = "G8"
FARBE1 = "#9dff3c"
FARBE2 = "#2bf5a0"
BESCHREIBUNG = "Beide Teams an denselben Gegnern gemessen, letzte 8 Wochen, plus direkter Vergleich"
WOCHEN = 8


def trainiere(df, heute):
    return gg_basis.trainiere(df, heute, WOCHEN)


def tore_erwartet(modell, heim, gast):
    return gg_basis.tore_erwartet(modell, heim, gast)
