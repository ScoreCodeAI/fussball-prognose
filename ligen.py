# ===== LIGEN-EINSTELLUNGEN =====
# Neue Liga hinzufuegen: einfach eine Zeile beim passenden Land ergaenzen.
# Codes von football-data.co.uk. Ligen eines Landes werden gemeinsam gerechnet (Auf-/Absteiger).

LAENDER = {
    "Deutschland": {"D1": "Bundesliga", "D2": "2. Bundesliga"},
    "England": {"E0": "Premier League", "E1": "Championship", "E2": "League One", "E3": "League Two", "EC": "National League"},
    "Spanien": {"SP1": "La Liga", "SP2": "Segunda División"},
    "Italien": {"I1": "Serie A", "I2": "Serie B"},
    "Frankreich": {"F1": "Ligue 1", "F2": "Ligue 2"},
    "Niederlande": {"N1": "Eredivisie"},
    "Portugal": {"P1": "Primeira Liga"},
    "Belgien": {"B1": "Jupiler Pro League"},
    "Griechenland": {"G1": "Super League"},
    "Schottland": {"SC0": "Premiership", "SC1": "Championship", "SC2": "League One", "SC3": "League Two"},
    "Dänemark": {"DNK": "Superliga"},
    "Norwegen": {"NOR": "Eliteserien"},
    "Schweden": {"SWE": "Allsvenskan"},
    "Finnland": {"FIN": "Veikkausliiga"},
    "Österreich": {"AUT": "Bundesliga"},
    "Schweiz": {"SWZ": "Super League"},
    "Polen": {"POL": "Ekstraklasa"},
    "Rumänien": {"ROU": "Liga I"},
    "Irland": {"IRL": "Premier Division"},
    "Russland": {"RUS": "Premier Liga"},
    "USA": {"USA": "MLS"},
    "Mexiko": {"MEX": "Liga MX"},
    "Brasilien": {"BRA": "Série A"},
    "Argentinien": {"ARG": "Liga Profesional"},
    "Japan": {"JPN": "J1 League"},
    "China": {"CHN": "Super League"},
}

# Diese Ligen kommen aus den "Extra-Ligen"-Dateien (nur 1. Liga, Quoten nur 1X2, keine Halbzeit)
EXTRA_LIGEN = {"DNK": "Denmark", "NOR": "Norway", "SWE": "Sweden", "FIN": "Finland", "AUT": "Austria",
               "SWZ": "Switzerland", "POL": "Poland", "ROU": "Romania", "IRL": "Ireland", "RUS": "Russia",
               "USA": "USA", "MEX": "Mexico", "BRA": "Brazil", "ARG": "Argentina", "JPN": "Japan", "CHN": "China"}

# Aktuelle Saison im Format von football-data.co.uk (2026/27 -> "2627")
SAISON = "2627"
SAISON_START = "2026-07-01"
