# ===== TEST · SportsAPI Pro (Gratis-Plan) · Fokus England: Premier League, Championship, League One, League Two =====
# Listet je Liga: Ergebnis, Halbzeit, alle Statistik-Werte (Ecken, Schuesse, xG ...), alle Quoten-Maerkte.
# Hoechstens 14 Abrufe. Zeigt den Abruf-Zaehler aus den Antwort-Kopfzeilen.
import json, os, sys
from datetime import date, timedelta
import requests

KEY = os.environ.get("SPORTSAPIPRO_KEY", "")
if not KEY:
    sys.exit("Kein Schluessel: GitHub-Secret SPORTSAPIPRO_KEY fehlt")
BASEN = ["https://v2.football.sportsapipro.com", "https://api.sportsapipro.com/v2/football"]
H = {"x-api-key": KEY, "Accept": "application/json"}
n, MAX = [0], 14


def hole(pfad, params=None):
    for b in list(BASEN):
        if n[0] >= MAX:
            print("STOPP: Test-Grenze erreicht"); return None
        n[0] += 1
        try:
            r = requests.get(b + pfad, headers=H, params=params, timeout=30)
        except Exception as ex:
            print(f"[{n[0]}] {pfad} -> Fehler {ex}"); continue
        kopf = {k: v for k, v in r.headers.items() if any(w in k.lower() for w in ("limit", "remaining", "quota", "plan", "usage"))}
        print(f"[Abruf {n[0]}] {pfad} -> Status {r.status_code} | Kopfzeilen: {kopf}")
        if r.status_code == 200:
            BASEN[:] = [b]
            try:
                return r.json()
            except Exception:
                print("   keine JSON-Antwort:", r.text[:200]); return None
        print("   Antwort:", r.text[:300])
    return None


def liste(j):
    if isinstance(j, dict):
        for k in ("events", "data", "matches", "response"):
            if isinstance(j.get(k), list):
                return j[k]
    return j if isinstance(j, list) else []


def schluessel(o, pfad="", out=None):
    """alle Feldnamen (flach) sammeln"""
    out = set() if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("name", "key") and isinstance(v, str):
                out.add(v)
            schluessel(v, k, out)
    elif isinstance(o, list):
        for v in o:
            schluessel(v, pfad, out)
    return out


ENGLAND = {"Premier League": None, "Championship": None, "League One": None, "League Two": None}
heute = date.today()
sams = heute - timedelta(days=(heute.weekday() - 5) % 7 or 7)
for d in (sams.isoformat(), (sams + timedelta(days=1)).isoformat()):
    if all(ENGLAND.values()):
        break
    ev = liste(hole(f"/api/schedule/{d}"))
    print(f"\n=== Spielplan {d}: {len(ev)} Spiele ===")
    for e in ev:
        t = e.get("tournament") or {}
        land = ((t.get("category") or {}).get("name", "") if isinstance(t, dict) else "")
        name = t.get("name", "") if isinstance(t, dict) else str(t)
        if "england" not in land.lower():
            continue
        for liga in ENGLAND:
            if name.strip().lower() == liga.lower() and ENGLAND[liga] is None and (e.get("homeScore") or {}).get("current") is not None:
                ENGLAND[liga] = e
for liga, e in ENGLAND.items():
    print(f"\n################ {liga} ################")
    if not e:
        print("  kein beendetes Spiel im Spielplan gefunden"); continue
    hs, as_ = e.get("homeScore") or {}, e.get("awayScore") or {}
    print(f"  Spiel: {(e.get('homeTeam') or {}).get('name')} - {(e.get('awayTeam') or {}).get('name')} | "
          f"Endstand {hs.get('current')}:{as_.get('current')} | Halbzeit {hs.get('period1')}:{as_.get('period1')}")
    s = hole(f"/api/match/{e.get('id')}/statistics")
    k = sorted(schluessel(s)) if s else []
    print(f"  STATISTIK-WERTE ({len(k)}): {', '.join(k)}")
    txt = json.dumps(s, ensure_ascii=False).lower() if s else ""
    print("  -> Ecken:", "corner" in txt, "| xG:", ("expectedgoals" in txt or '"xg"' in txt), "| Schuesse:", "shot" in txt)
    if liga in ("Premier League", "League Two"):      # Quoten nur 2x testen (spart Abrufe)
        o = hole(f"/api/match/{e.get('id')}/odds", {"scope": "all"})
        m = []
        if isinstance(o, dict):
            for mk in o.get("markets") or []:
                if isinstance(mk, dict):
                    m.append(f'{mk.get("marketName") or mk.get("name")}' + (f' ({mk.get("choiceGroup")})' if mk.get("choiceGroup") else ""))
        print(f"  QUOTEN-MAERKTE ({len(m)}): {', '.join(m) if m else json.dumps(o, ensure_ascii=False)[:800]}")
print(f"\nVerbrauchte Abrufe in diesem Test: {n[0]} (Gratis-Plan: 100 pro Tag)")
