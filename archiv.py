"""archiv.py – sichert vergangene Spiele (schlank) in docs/archiv_spiele.json.
Laeuft VOR engine.py mit der alten daten.json (dort stehen schon die Endstaende).
Aendert keine Modelle, nur Archiv fuer Messung, Auto-Wahl und Mystery Box. Haelt 120 Tage.
Neu: Schluessel = Datum + Teams (die IDs in daten.json wechseln taeglich),
     zusaetzlich xG-Form, Elo und Markt-% (1X2, O/U 2,5) – keine API-Credits noetig."""
import json, os
from datetime import date, timedelta
SRC, DST, TAGE = "docs/daten.json", "docs/archiv_spiele.json", 120
FELDER = ("p1", "px", "p2", "ue25", "bts", "ht", "ht2", "h2o05", "xg")   # ht/ht2/h2o05/xg fuer die Halbzeit-Maerkte des MASTER

def key(s):
    return f'{s.get("datum")}|{s.get("heim")}|{s.get("gast")}'

def markt(s):
    sv = (s.get("prog") or {}).get("sv") or {}
    m = {}
    for mk in ("x", "ou"):
        a = sv.get(mk) or {}
        v = a.get("shin") or a.get("fair")
        if v:
            m[mk] = v
    return m

def schlank(s):
    prog = {}
    for k, v in (s.get("prog") or {}).items():
        if isinstance(v, dict) and v.get("p1") is not None:
            prog[k] = {f: v[f] for f in FELDER if v.get(f) is not None}
    out = {k: s.get(k) for k in ("id", "liga", "datum", "zeit", "heim", "gast", "ergebnis", "ergebnis_ht")}
    out["prog"] = prog
    sicht = {}                                    # 🧩 Modelle: Heim/Ausw und H2H je Modell (fuer die Schwelle aus der Historie)
    for name, P in (s.get("sicht") or {}).items():
        q = {k: {f: v[f] for f in FELDER if v.get(f) is not None} for k, v in (P or {}).items() if isinstance(v, dict)}
        if q:
            sicht[name] = q
    if sicht:
        out["sicht"] = sicht
    if s.get("xgform"):
        out["xgform"] = s["xgform"]
    if s.get("elo"):
        out["elo"] = s["elo"]
    mk = markt(s)
    if mk:
        out["mkt"] = mk
    return out

def main():
    alt = {}
    try:
        for s in json.load(open(DST, encoding="utf-8")):
            alt[key(s)] = s
    except Exception:
        pass
    try:
        d = json.load(open(SRC, encoding="utf-8"))
    except Exception as e:
        print("archiv: keine daten.json", e); return
    try:                                          # 🧩 Sichten je Modell stehen in eigener Datei
        sj = json.load(open("docs/sichten.json", encoding="utf-8")).get("spiele", {})
        for s in d.get("spiele", []):
            e = sj.get(key(s))
            if e and e.get("sicht"):
                s["sicht"] = e["sicht"]
    except Exception:
        pass
    heute = date.today().isoformat()
    n = 0
    for s in d.get("spiele", []):
        if s.get("datum", "9") < heute and s.get("ergebnis"):
            alt[key(s)] = schlank(s); n += 1
    grenze = (date.today() - timedelta(days=TAGE)).isoformat()
    out = sorted((s for s in alt.values() if s.get("datum", "") >= grenze), key=lambda s: (s["datum"], s.get("zeit") or ""))
    os.makedirs("docs", exist_ok=True)
    json.dump(out, open(DST, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"archiv: {n} Spiele uebernommen, gesamt {len(out)}")

if __name__ == "__main__":
    main()
