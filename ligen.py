# ===== LIGEN-EINSTELLUNGEN =====
# Neue Liga hinzufuegen: einfach eine Zeile beim passenden Land ergaenzen.
# Codes von football-data.co.uk. Ligen eines Landes werden gemeinsam gerechnet (Auf-/Absteiger).

LAENDER = {
    "Deutschland": {"D1": "Bundesliga", "D2": "2. Bundesliga"},
    "England": {"E0": "Premier League", "E1": "Championship", "E2": "League One", "E3": "League Two"},
    "Spanien": {"SP1": "La Liga", "SP2": "Segunda División"},
    "Italien": {"I1": "Serie A", "I2": "Serie B"},
    "Frankreich": {"F1": "Ligue 1", "F2": "Ligue 2"},
    "Niederlande": {"N1": "Eredivisie"},
    "Belgien": {"B1": "Jupiler Pro League"},
    "Griechenland": {"G1": "Super League"},
    "Schottland": {"SC0": "Premiership"},
    "USA": {"USA": "MLS"},
}

# Diese Ligen kommen aus den "Extra-Ligen"-Dateien (nur 1. Liga, Quoten nur 1X2, keine Halbzeit)
EXTRA_LIGEN = {"USA": "USA"}

# Aktuelle Saison im Format von football-data.co.uk (2026/27 -> "2627")
SAISON = "2627"
SAISON_START = "2026-07-01"
