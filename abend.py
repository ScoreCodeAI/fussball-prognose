# ===== ABEND-ERGEBNISSE · nur Endstaende fuer die Webseite (eigenstaendig) =====
# Holt bei SportsAPI den Spielplan von heute/gestern (2-3 Abrufe, keine Quoten, keine Odds-API-Credits)
# und traegt fertige Ergebnisse + Halbzeit in docs/daten.json ein - nur bei Spielen OHNE Ergebnis.
# Veraendert KEINE Modelle, KEINE Prognosen, KEIN Archiv. Der Morgenlauf baut daten.json ohnehin komplett neu.
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import sportsapi as sa

HIER = os.path.dirname(os.path.abspath(__file__))
PFAD = os.path.join(HIER, "docs", "daten.json")
MAX_ABRUFE = 4


def log(*a):
    print("Abend:", *a, flush=True)


def main():
    key = os.environ.get("SPORTSAPIPRO_KEY", "")
    if not key:
        log("kein Schluessel - nichts zu tun")
        return
    try:
        d = json.load(open(PFAD, encoding="utf-8"))
    except Exception as e:
        log("daten.json nicht lesbar", e)
        return
    jetzt = datetime.now(timezone.utc)
    heute = jetzt.astimezone(sa.DE).strftime("%Y-%m-%d")
    gestern = (jetzt.astimezone(sa.DE) - timedelta(days=1)).strftime("%Y-%m-%d")
    offen = [e for e in d.get("spiele", []) if e.get("datum") in (heute, gestern) and not e.get("ergebnis")]
    if not offen:
        log("keine offenen Spiele von heute/gestern")
        return
    hole = sa.Abruf(key, log)
    a = {"spiele": {}}
    tage = sorted({(jetzt + timedelta(days=k)).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d") for k in (-1, 0)})
    for tag in tage[:MAX_ABRUFE]:
        for ev in sa._ev(hole(f"/api/schedule/{tag}")):
            c = sa._code(ev)
            if c and ev.get("id"):
                a["spiele"][str(ev["id"])] = sa._spiel(ev, c)
    idx = sa._index(a)
    n = 0
    for e in offen:
        s = sa._finde(idx, e["liga"], e["datum"], e["heim"], e["gast"])
        if not s or not s.get("fertig") or not s.get("ft"):
            continue
        e["ergebnis"] = f'{s["ft"][0]}:{s["ft"][1]}'
        e["equelle"] = "SportsAPI"
        if s.get("ht") and not e.get("ergebnis_ht"):
            e["ergebnis_ht"] = f'{s["ht"][0]}:{s["ht"][1]}'
        n += 1
    if n:
        with open(PFAD, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
    log(f"{hole.n} Abrufe · {n} von {len(offen)} offenen Spielen mit Ergebnis ergaenzt")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log("Fehler (nichts veraendert):", ex)
    sys.exit(0)
