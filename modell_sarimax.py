# ===== MODELL: SARIMAX · Original (dein Ansatz) =====
# Rechenweg siehe sarimax_basis.py
import sarimax_basis

NAME = "SARIMAX · Original"
KURZ = "SARIMAX"
ICON = "S"
FARBE1 = "#ffb020"
FARBE2 = "#ff6b3d"
BESCHREIBUNG = "Auto-ARIMA: Heimteam aus seinen Heimspielen, Gastteam aus seinen Auswärtsspielen, Halbzeit eigenständig"
BACKTEST_WOCHEN = 5     # ARIMA ist rechenintensiv: rueckwirkender Test nur ueber die letzten Wochen


def trainiere(df, heute):
    return sarimax_basis.trainiere(df, heute, plus=False)


def vorhersage(modell, heim, gast):
    return sarimax_basis.vorhersage(modell, heim, gast)
