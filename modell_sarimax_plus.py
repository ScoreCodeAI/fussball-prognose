# ===== MODELL: SARIMAX · Plus (dein Ansatz + Gegnerstaerke) =====
# Wie das Original, zusaetzlich die Gegentor-Neigung des Gegners als aeusserer Einfluss (X).
# Rechenweg siehe sarimax_basis.py
import sarimax_basis

NAME = "SARIMAX · Plus"
KURZ = "SARIMAX+"
ICON = "S+"
FARBE1 = "#e449ff"
FARBE2 = "#7b5cff"
BESCHREIBUNG = "Wie SARIMAX Original, zusätzlich mit der Stärke des Gegners als Einflussgröße"
BACKTEST_WOCHEN = 5


def trainiere(df, heute):
    return sarimax_basis.trainiere(df, heute, plus=True)


def vorhersage(modell, heim, gast):
    return sarimax_basis.vorhersage(modell, heim, gast)
