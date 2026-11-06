# ===== GESAMTTEST · SportsAPI Pro (max. 20 Abrufe) =====
# Prueft fuer unsere Ligen: Spiele pro Samstag (Budget), xG/Statistik je Liga, alle Quoten-Maerkte,
# zweite Seite im Spielplan (Begrenzung?), Rueckblick in die letzte Saison.
import json, os, sys
import requests

KEY = os.environ.get("SPORTSAPIPRO_KEY", "") or sys.exit("SPORTSAPIPRO_KEY fehlt")
B = "https://v2.football.sportsapipro.com"
H = {"x-api-key": KEY}
n = [0]


def hole(pfad, params=None):
    if n[0] >= 20:
        print("STOPP 20 Abrufe"); return None
    n[0] += 1
    r = requests.get(B + pfad, headers=H, params=params, timeout=30)
    print(f"[Abruf {n[0]}] {pfad} {params or ''} -> {r.status_code} | uebrig {r.headers.get('x-ratelimit-remaining')}")
    try:
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def ev_liste(j):
    if isinstance(j, dict):
        for k in ("events", "data", "matches"):
            if isinstance(j.get(k), list):
                return j[k]
        if isinstance(j.get("data"), dict):
            for k in ("events", "matches"):
                if isinstance(j["data"].get(k), list):
                    return j["data"][k]
    return j if isinstance(j, list) else []


UNSERE = {("Germany", "Bundesliga"): "D1", ("Germany", "2. Bundesliga"): "D2", ("England", "Premier League"): "E0",
          ("England", "Championship"): "E1", ("England", "League One"): "E2", ("England", "League Two"): "E3",
          ("Spain", "LaLiga"): "SP1", ("Spain", "LaLiga 2"): "SP2", ("Italy", "Serie A"): "I1", ("Italy", "Serie B"): "I2",
          ("France", "Ligue 1"): "F1", ("France", "Ligue 2"): "F2", ("Netherlands", "VriendenLoterij Eredivisie"): "N1",
          ("Belgium", "Pro League"): "B1", ("Greece", "Stoiximan Super League"): "G1",
          ("Scotland", "Scottish Premiership"): "SC0", ("USA", "MLS"): "USA"}


def code(e):
    t = e.get("tournament") or {}
    land = (t.get("category") or {}).get("name", "")
    return UNSERE.get((land, t.get("name", "")))


# 1) Samstag 31.10. + Seite 2 pruefen
j = hole("/api/schedule/2026-10-31")
ev = ev_liste(j)
print("Antwort-Schluessel:", list(j.keys()) if isinstance(j, dict) else type(j), "| Spiele:", len(ev))
if isinstance(j, dict):
    print("Seiten-Infos:", {k: j[k] for k in j if k not in ("events", "data")})
j2 = hole("/api/schedule/2026-10-31", {"page": 2})
ev2 = ev_liste(j2)
neu = {e.get("id") for e in ev2} - {e.get("id") for e in ev}
print(f"Seite 2: {len(ev2)} Spiele, davon {len(neu)} NEUE (0 = keine Begrenzung)")
pro = {}
for e in ev + [x for x in ev2 if x.get("id") in neu]:
    c = code(e)
    if c:
        pro.setdefault(c, []).append(e)
print("\n=== UNSERE LIGEN am Samstag 31.10. ===")
for c in sorted(pro):
    print(f"  {c:4} {len(pro[c])} Spiele")
print(f"  SUMME: {sum(len(v) for v in pro.values())} Spiele (= Abrufe fuer xG, wenn alle)")
fehlt = sorted(set(UNSERE.values()) - set(pro))
print("  nicht gefunden:", fehlt)

# 2) xG/Statistik in den Ligen, die wir noch nicht getestet haben
print("\n=== xG je Liga (1 Spiel je Liga) ===")
erstes = None
for c in ["D2", "SP2", "I2", "F2", "B1", "G1", "SC0", "N1", "USA"]:
    e = next((x for x in pro.get(c, []) if (x.get("homeScore") or {}).get("current") is not None), None)
    if not e:
        print(f"  {c}: kein beendetes Spiel"); continue
    erstes = erstes or e
    s = hole(f"/api/match/{e['id']}/statistics")
    t = json.dumps(s).lower() if s else ""
    print(f"  {c}: xG {'expectedgoals' in t} | xGOT {'expectedgoalsontarget' in t} | Ecken {'corner' in t} | "
          f"Grosschancen {'bigchance' in t} | Karten {'yellowcard' in t} | HZ-Statistik {'1st' in t}")

# 3) ALLE Quoten-Maerkte eines Spiels
e = (pro.get("E0") or pro.get("D1") or [erstes])[0]
if e:
    o = hole(f"/api/match/{e['id']}/odds", {"scope": "all"})
    d = o.get("data", o) if isinstance(o, dict) else {}
    mk = d.get("markets", []) if isinstance(d, dict) else []
    print(f"\n=== QUOTEN-MAERKTE ({len(mk)}) fuer {e.get('homeTeam', {}).get('name')} - {e.get('awayTeam', {}).get('name')} ===")
    for m in mk:
        ch = ", ".join(f"{c.get('name')} {c.get('fractionalValue')}" for c in m.get("choices", []))
        print(f"  {m.get('marketName')} | {m.get('marketPeriod', '')} | {m.get('choiceGroup', '')} | {ch}")

# 4) Rueckblick: letzte Saison
print("\n=== Rueckblick 01.11.2025 ===")
ev3 = ev_liste(hole("/api/schedule/2025-11-01"))
alt = [x for x in ev3 if code(x)]
print(f"  {len(ev3)} Spiele, davon {len(alt)} in unseren Ligen")
if alt:
    s = hole(f"/api/match/{alt[0]['id']}/statistics")
    t = json.dumps(s).lower() if s else ""
    print(f"  altes Spiel {alt[0].get('homeTeam', {}).get('name')}: xG {'expectedgoals' in t} | Ecken {'corner' in t}")
print(f"\nVerbraucht: {n[0]} Abrufe")
