# ===== LIGEN-EINSTELLUNGEN =====
# Neue Liga hinzufuegen: einfach eine Zeile beim passenden Land ergaenzen.
# Die Codes sind die von football-data.co.uk (z. B. "N1" = Niederlande).
# Ligen eines Landes werden gemeinsam gerechnet (wichtig fuer Auf-/Absteiger).

LAENDER = {
    "Deutschland": {"D1": "Bundesliga", "D2": "2. Bundesliga"},
    "England": {"E0": "Premier League", "E1": "Championship"},
    "Spanien": {"SP1": "La Liga", "SP2": "Segunda División"},
    "Italien": {"I1": "Serie A", "I2": "Serie B"},
    "Frankreich": {"F1": "Ligue 1", "F2": "Ligue 2"},
}

# Aktuelle Saison im Format von football-data.co.uk (2026/27 -> "2627")
SAISON = "2627"
SAISON_START = "2026-07-01"
