# Test API-Football: 3 Abrufe (Status, naechste Bundesliga-Spiele, Quoten eines Spiels)
import os, json, requests
KEY = os.environ.get("API_FOOTBALL_KEY", "")
H = {"x-apisports-key": KEY}
B = "https://v3.football.api-sports.io/"
if not KEY:
    raise SystemExit("FEHLER: Secret API_FOOTBALL_KEY fehlt")
def get(p, **q):
    r = requests.get(B + p, headers=H, params=q, timeout=30).json()
    if r.get("errors"):
        print("   API-Meldung:", r["errors"])
    return r
s = get("status")["response"]
print("1) KONTO:", s.get("subscription", {}).get("plan"), "| Abrufe heute:", s.get("requests"))
f = get("fixtures", league=78, season=2026, next=3)
print(f"2) BUNDESLIGA 2026/27 naechste Spiele: {f.get('results')}")
for x in f.get("response", []):
    print("   ", x["fixture"]["date"][:16], x["teams"]["home"]["name"], "-", x["teams"]["away"]["name"])
if f.get("response"):
    o = get("odds", fixture=f["response"][0]["fixture"]["id"])
    print(f"3) QUOTEN fuer das erste Spiel: {o.get('results')} Eintraege")
    for e in o.get("response", []):
        bms = e["bookmakers"]
        print("   Buchmacher:", len(bms), "->", ", ".join(b["name"] for b in bms[:12]))
        if bms:
            print("   Wettarten (1. Buchmacher):", len(bms[0]["bets"]))
            for b in bms[0]["bets"][:40]:
                print("     -", b["id"], b["name"], "|", len(b["values"]), "Werte")
print("\nERGEBNIS:", "AKTUELLE SAISON FREI ✓" if f.get("results") else "AKTUELLE SAISON GESPERRT ✕")
