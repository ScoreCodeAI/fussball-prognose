"""archiv.py – sichert vergangene Spiele (schlank) fuer die Auto-Wahl in docs/archiv_spiele.json.
Laeuft VOR engine.py mit der alten daten.json (dort stehen schon die Endstaende).
Aendert keine Modelle, nur Archiv fuer die Messung. Haelt 120 Tage."""
import json, os
from datetime import date, timedelta
SRC, DST, TAGE = "docs/daten.json", "docs/archiv_spiele.json", 120
FELDER = ("p1", "px", "p2", "ue25", "bts")

def schlank(s):
    prog = {}
    for k, v in (s.get("prog") or {}).items():
        if isinstance(v, dict) and v.get("p1") is not None:
            prog[k] = {f: v[f] for f in FELDER if v.get(f) is not None}
    return {k: s.get(k) for k in ("id", "liga", "datum", "zeit", "heim", "gast", "ergebnis", "ergebnis_ht")} | {"prog": prog}

def main():
    try:
        alt = {s["id"]: s for s in json.load(open(DST, encoding="utf-8"))}
    except Exception:
        alt = {}
    try:
        d = json.load(open(SRC, encoding="utf-8"))
    except Exception as e:
        print("archiv: keine daten.json", e); return
    heute = date.today().isoformat()
    n = 0
    for s in d.get("spiele", []):
        if s.get("id") and s.get("datum", "9") < heute and s.get("ergebnis"):
            alt[s["id"]] = schlank(s); n += 1
    grenze = (date.today() - timedelta(days=TAGE)).isoformat()
    out = sorted((s for s in alt.values() if s.get("datum", "") >= grenze), key=lambda s: (s["datum"], s.get("zeit") or ""))
    os.makedirs("docs", exist_ok=True)
    json.dump(out, open(DST, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"archiv: {n} Spiele uebernommen, gesamt {len(out)}")

if __name__ == "__main__":
    main()
