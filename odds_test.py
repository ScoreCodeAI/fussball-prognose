# Test The Odds API: welche Ligen gibt es, kommen Quoten an, wie viele Credits sind uebrig?
# Verbraucht nur 2 Credits (ein Quoten-Abruf Bundesliga: 1X2 + OVER/UNDER).
import os, requests
KEY = os.environ.get("ODDS_API_KEY", "")
if not KEY:
    raise SystemExit("FEHLER: Secret ODDS_API_KEY fehlt")
B = "https://api.the-odds-api.com/v4/"
UNSERE = {"D1": "soccer_germany_bundesliga", "D2": "soccer_germany_bundesliga2", "E0": "soccer_epl",
          "E1": "soccer_efl_champ", "E2": "soccer_england_league1", "E3": "soccer_england_league2",
          "SP1": "soccer_spain_la_liga", "SP2": "soccer_spain_segunda_division", "I1": "soccer_italy_serie_a",
          "I2": "soccer_italy_serie_b", "F1": "soccer_france_ligue_one", "F2": "soccer_france_ligue_two",
          "N1": "soccer_netherlands_eredivisie", "P1": "soccer_portugal_primeira_liga", "B1": "soccer_belgium_first_div",
          "G1": "soccer_greece_super_league", "SC0": "soccer_spl", "DNK": "soccer_denmark_superliga",
          "NOR": "soccer_norway_eliteserien", "SWE": "soccer_sweden_allsvenskan", "FIN": "soccer_finland_veikkausliiga",
          "AUT": "soccer_austria_bundesliga", "SWZ": "soccer_switzerland_superleague", "POL": "soccer_poland_ekstraklasa",
          "IRL": "soccer_league_of_ireland", "RUS": "soccer_russia_premier_league", "USA": "soccer_usa_mls",
          "MEX": "soccer_mexico_ligamx", "BRA": "soccer_brazil_campeonato", "ARG": "soccer_argentina_primera_division",
          "JPN": "soccer_japan_j_league", "CHN": "soccer_china_superleague"}

r = requests.get(B + "sports", params={"apiKey": KEY, "all": "true"}, timeout=30)
print("1) KONTO: Credits uebrig", r.headers.get("x-requests-remaining"), "| verbraucht", r.headers.get("x-requests-used"))
alle = {s["key"]: s for s in r.json() if s.get("group") == "Soccer"}
print(f"2) FUSSBALL-LIGEN bei The Odds API: {len(alle)}")
for code, key in UNSERE.items():
    s = alle.get(key)
    print(f"   {code:4} {key:40} {'OK' if s else 'FEHLT'}{'' if not s else (' (aktiv)' if s.get('active') else ' (gerade keine Spiele)')}")
fehlt = [k for k in alle if k not in UNSERE.values()]
print("   weitere vorhandene Ligen:", ", ".join(sorted(fehlt))[:1500])

o = requests.get(B + "sports/soccer_germany_bundesliga/odds",
                 params={"apiKey": KEY, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"}, timeout=30)
print("3) BUNDESLIGA-QUOTEN: Status", o.status_code, "| Credits uebrig", o.headers.get("x-requests-remaining"),
      "| letzter Abruf kostete", o.headers.get("x-requests-last"))
ev = o.json() if o.status_code == 200 else []
print(f"   Spiele: {len(ev)}")
for e in ev[:3]:
    bm = e.get("bookmakers", [])
    print(f"   {e['commence_time'][:16]} {e['home_team']} - {e['away_team']} | Buchmacher: {len(bm)}")
    if bm:
        print("     z.B.:", ", ".join(b["title"] for b in bm[:10]))
        for b in bm[:1]:
            for m in b["markets"]:
                print("     ", m["key"], [(x["name"], x.get("point"), x["price"]) for x in m["outcomes"]])
print("\nERGEBNIS:", "QUOTEN KOMMEN AN ✓" if ev else "KEINE QUOTEN ✕")
