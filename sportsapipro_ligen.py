# ===== TEST 2 · SportsAPI Pro: welche Ligen? (genau 1 Abruf) =====
# Holt den Spielplan vom 31.10. (352 Spiele) und listet ALLE Wettbewerbe mit Land und Anzahl Spiele,
# dazu ob Halbzeitstaende vorhanden sind. Markiert unsere Ligen.
import os, sys, requests
KEY = os.environ.get("SPORTSAPIPRO_KEY", "") or sys.exit("SPORTSAPIPRO_KEY fehlt")
r = requests.get("https://v2.football.sportsapipro.com/api/schedule/2026-10-31", headers={"x-api-key": KEY}, timeout=30)
print("Status", r.status_code, "| uebrig:", r.headers.get("x-ratelimit-remaining"))
j = r.json()
ev = next((j[k] for k in ("events", "data", "matches") if isinstance(j, dict) and isinstance(j.get(k), list)), j if isinstance(j, list) else [])
UNSERE = ["Bundesliga", "2. Bundesliga", "Premier League", "Championship", "League One", "League Two", "National League", "LaLiga",
          "LaLiga 2", "Serie A", "Serie B", "Ligue 1", "Ligue 2", "Eredivisie", "Liga Portugal", "Pro League", "Super League",
          "Premiership", "Superliga", "Eliteserien", "Allsvenskan", "Veikkausliiga", "Ekstraklasa", "Liga I", "Premier Division",
          "MLS", "Liga MX", "Brasileirão", "Liga Profesional", "J1 League", "Chinese Super League"]
L = {}
for e in ev:
    t = e.get("tournament") or {}
    land = (t.get("category") or {}).get("name", "?") if isinstance(t, dict) else "?"
    name = t.get("name", "?") if isinstance(t, dict) else str(t)
    hs = e.get("homeScore") or {}
    z = L.setdefault((land, name), [0, 0])
    z[0] += 1
    z[1] += hs.get("period1") is not None
print(f"\n{len(L)} Wettbewerbe an diesem Tag ({len(ev)} Spiele)\n")
for (land, name), (n, ht) in sorted(L.items()):
    m = "  <== UNSERE" if any(u.lower() in name.lower() for u in UNSERE) else ""
    print(f"{land:22} | {name:40} | {n:3} Spiele | HZ {ht}/{n}{m}")
