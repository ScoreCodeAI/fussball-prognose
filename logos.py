# ===== LOGOS · Vereinswappen fuer die Spielkarten (eigenstaendig, aendert keine Berechnung) =====
# Quelle: TheSportsDB (kostenlos, Test-Schluessel "123"), KEINE Odds-API- oder SportsAPI-Abrufe.
# Ablauf: Teams aus docs/daten.json lesen -> welche haben noch kein Wappen? -> nur fuer diese Ligen
#         die Teamliste bei TheSportsDB holen (1 Abruf je Liga) -> Namen zuordnen -> Wappen laden,
#         auf 96 px verkleinern, in docs/logos/ speichern, docs/logos/logos.json aktualisieren.
# Jedes Wappen wird nur EINMAL geladen. Fehlt eins, versucht es der naechste Lauf wieder.
# MLS bewusst ausgelassen. Fehler brechen nie den Workflow ab.
import difflib
import io
import json
import os
import re
import sys
import time
import unicodedata

import requests

BASE = os.environ.get("TSDB_BASE", "https://www.thesportsdb.com/api/v1/json/123")
HIER = os.path.dirname(os.path.abspath(__file__))
DATEN = os.path.join(HIER, "docs", "daten.json")
ORDNER = os.path.join(HIER, "docs", "logos")
KARTE = os.path.join(ORDNER, "logos.json")
GROESSE = 96
NAMENSSUCHE_MAX = 25          # Einzelsuche je Lauf fuer Teams, die in der Ligaliste nicht gefunden wurden
OHNE = {"USA"}                # MLS nicht gewuenscht

LIGEN = {"E0": 4328, "E1": 4329, "E2": 4396, "E3": 4397, "D1": 4331, "D2": 4399, "SP1": 4335, "SP2": 4400,
         "I1": 4332, "I2": 4394, "F1": 4334, "F2": 4401, "N1": 4337, "B1": 4338, "G1": 4336, "SC0": 4330}

ALIAS = {"man united": "manchester united", "man city": "manchester city", "nott'm forest": "nottingham forest",
         "sheffield weds": "sheffield wednesday", "qpr": "queens park rangers", "west brom": "west bromwich albion",
         "wolves": "wolverhampton wanderers", "spurs": "tottenham hotspur", "brighton": "brighton and hove albion",
         "ath madrid": "atletico madrid", "ath bilbao": "athletic bilbao", "sp gijon": "sporting gijon",
         "espanol": "espanyol", "la coruna": "deportivo la coruna", "sociedad": "real sociedad",
         "sociedad b": "real sociedad b", "vallecano": "rayo vallecano", "betis": "real betis", "celta": "celta vigo",
         "celta b": "celta vigo b", "santander": "racing santander", "ein frankfurt": "eintracht frankfurt",
         "m'gladbach": "borussia monchengladbach", "fc koln": "fc koln", "leverkusen": "bayer leverkusen",
         "bayern munich": "bayern munich", "hertha": "hertha berlin", "greuther furth": "greuther furth",
         "st pauli": "fc st pauli", "paris sg": "paris saint germain", "inter": "inter milan", "milan": "ac milan",
         "for sittard": "fortuna sittard", "nijmegen": "nec nijmegen", "st. gilloise": "union saint gilloise",
         "st truiden": "sint truiden", "waregem": "zulte waregem", "oud-heverlee leuven": "oh leuven",
         "olympiakos": "olympiacos", "dundee united": "dundee united", "st johnstone": "st johnstone",
         "st mirren": "st mirren", "juve stabia": "juve stabia", "virtus entella": "entella"}


def log(*a):
    print("Logos:", *a, flush=True)


def norm(x):
    x = ALIAS.get(x.lower().strip(), x)
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower()
    x = re.sub(r"\b(fc|cf|sc|ac|afc|as|ss|ssc|us|rc|sv|vfb|vfl|tsg|fk|cd|ud|sd|rcd|ogc|club|de|fsv|bv|calcio|kv|krc|kaa|rsc|sk)\b", " ", x)
    return re.sub(r"[^a-z0-9]", "", x)


def slug(t):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()) + ".png"


def holen(url, **kw):
    for v in range(3):
        try:
            r = requests.get(url, timeout=15, **kw)
            if r.status_code == 429:
                time.sleep(20)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            if v == 2:
                log("Abruf fehlgeschlagen", url.split("?")[0], e)
            time.sleep(3)
    return None


def kandidaten(teams):
    out = []
    for t in teams or []:
        b = t.get("strBadge") or t.get("strTeamBadge")
        if not b:
            continue
        namen = [t.get("strTeam") or "", t.get("strTeamShort") or ""] + (t.get("strTeamAlternate") or "").split(",")
        out.append((b, {norm(n) for n in namen if n and n.strip()}))
    return out


def zuordnen(name, kand):
    k = norm(name)
    if not k:
        return None
    for b, ns in kand:
        if k in ns:
            return b
    for b, ns in kand:
        if any((len(k) > 4 and len(n) > 4 and (k in n or n in k)) or (len(k) >= 4 and n.startswith(k)) for n in ns):
            return b
    alle = {n: b for b, ns in kand for n in ns}
    m = difflib.get_close_matches(k, list(alle), n=1, cutoff=0.8)
    return alle[m[0]] if m else None


def speichern(team, url, karte):
    from PIL import Image
    r = holen(url + "/small") if "thesportsdb.com" in url and not url.endswith(("/small", "/tiny", "/medium")) else None
    r = r or holen(url)
    if not r:
        return False
    try:
        im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        im.thumbnail((GROESSE, GROESSE))
        f = slug(team)
        im.save(os.path.join(ORDNER, f), optimize=True)
        karte[team] = f
        return True
    except Exception as e:
        log("Bild unlesbar", team, e)
        return False


START_URL = "https://raw.githubusercontent.com/luukhopman/football-logos/master/"
START = {"AZ Alkmaar": "logos/Netherlands - Eredivisie/AZ Alkmaar.png", "Aberdeen": "logos/Scotland - Scottish Premiership/Aberdeen FC.png", "Ajax": "history/2021-22/Netherlands - Eredivisie/Ajax.png", "Alaves": "history/2021-22/Spain - LaLiga/Alavés.png", "Almeria": "history/2023-24/Spain - LaLiga/UD Almería.png", "Anderlecht": "logos/Belgium - Jupiler Pro League/RSC Anderlecht.png", "Antwerp": "logos/Belgium - Jupiler Pro League/Royal Antwerp FC.png", "Aris": "logos/Greece - Super League 1/Aris Thessaloniki.png", "Arsenal": "logos/England - Premier League/Arsenal FC.png", "Asteras Tripolis": "history/2023-24/Greece - Super League 1/Asteras Tripolis.png", "Aston Villa": "logos/England - Premier League/Aston Villa.png", "Atalanta": "logos/Italy - Serie A/Atalanta BC.png", "Atromitos": "logos/Greece - Super League 1/Atromitos Athens.png", "Augsburg": "logos/Germany - Bundesliga/FC Augsburg.png", "Auxerre": "logos/France - Ligue 1/AJ Auxerre.png", "Barcelona": "logos/Spain - LaLiga/FC Barcelona.png", "Bayern Munich": "logos/Germany - Bundesliga/Bayern Munich.png", "Betis": "logos/Spain - LaLiga/Real Betis Balompié.png", "Beveren": "logos/Belgium - Jupiler Pro League/SK Beveren.png", "Bielefeld": "history/2021-22/Germany - Bundesliga/Arminia Bielefeld.png", "Bochum": "history/2024-25/Germany - Bundesliga/VfL Bochum.png", "Bologna": "logos/Italy - Serie A/Bologna FC 1909.png", "Bournemouth": "logos/England - Premier League/AFC Bournemouth.png", "Brentford": "logos/England - Premier League/Brentford FC.png", "Brest": "logos/France - Ligue 1/Stade Brestois 29.png", "Brighton": "history/2021-22/England - Premier League/Brighton.png", "Burnley": "history/2025-26/England - Premier League/Burnley FC.png", "Cadiz": "history/2023-24/Spain - LaLiga/Cádiz CF.png", "Cagliari": "logos/Italy - Serie A/Cagliari Calcio.png", "Cambuur": "history/2021-22/Netherlands - Eredivisie/SC Cambuur.png", "Celta": "logos/Spain - LaLiga/Celta de Vigo.png", "Celtic": "logos/Scotland - Scottish Premiership/Celtic FC.png", "Cercle Brugge": "logos/Belgium - Jupiler Pro League/Cercle Brugge.png", "Charleroi": "logos/Belgium - Jupiler Pro League/Royal Charleroi SC.png", "Chelsea": "logos/England - Premier League/Chelsea FC.png", "Clermont": "history/2023-24/France - Ligue 1/Clermont Foot 63.png", "Club Brugge": "history/2021-22/Belgium - Jupiler Pro League/Club Brugge.png", "Como": "logos/Italy - Serie A/Como 1907.png", "Coventry": "logos/England - Premier League/Coventry City.png", "Cremonese": "history/2025-26/Italy - Serie A/US Cremonese.png", "Crystal Palace": "logos/England - Premier League/Crystal Palace.png", "Darmstadt": "history/2023-24/Germany - Bundesliga/SV Darmstadt 98.png", "Den Haag": "logos/Netherlands - Eredivisie/ADO Den Haag.png", "Dortmund": "logos/Germany - Bundesliga/Borussia Dortmund.png", "Dundee United": "logos/Scotland - Scottish Premiership/Dundee United FC.png", "Elche": "logos/Spain - LaLiga/Elche CF.png", "Elversberg": "logos/Germany - Bundesliga/SV 07 Elversberg.png", "Empoli": "history/2024-25/Italy - Serie A/FC Empoli.png", "Everton": "logos/England - Premier League/Everton FC.png", "Excelsior": "logos/Netherlands - Eredivisie/Excelsior Rotterdam.png", "FC Koln": "logos/Germany - Bundesliga/1.FC Köln.png", "Falkirk": "logos/Scotland - Scottish Premiership/Falkirk FC.png", "Feyenoord": "history/2021-22/Netherlands - Eredivisie/Feyenoord.png", "Fiorentina": "logos/Italy - Serie A/ACF Fiorentina.png", "Freiburg": "logos/Germany - Bundesliga/SC Freiburg.png", "Frosinone": "logos/Italy - Serie A/Frosinone Calcio.png", "Fulham": "logos/England - Premier League/Fulham FC.png", "Genk": "logos/Belgium - Jupiler Pro League/KRC Genk.png", "Genoa": "logos/Italy - Serie A/Genoa CFC.png", "Gent": "logos/Belgium - Jupiler Pro League/KAA Gent.png", "Getafe": "logos/Spain - LaLiga/Getafe CF.png", "Girona": "history/2025-26/Spain - LaLiga/Girona FC.png", "Go Ahead Eagles": "logos/Netherlands - Eredivisie/Go Ahead Eagles.png", "Granada": "history/2023-24/Spain - LaLiga/Granada CF.png", "Greuther Furth": "history/2021-22/Germany - Bundesliga/Greuther Fürth.png", "Groningen": "logos/Netherlands - Eredivisie/FC Groningen.png", "Hamburg": "logos/Germany - Bundesliga/Hamburger SV.png", "Heerenveen": "logos/Netherlands - Eredivisie/SC Heerenveen.png", "Heidenheim": "history/2025-26/Germany - Bundesliga/1.FC Heidenheim 1846.png", "Hertha": "history/2022-23/Germany - Bundesliga/Hertha BSC.png", "Hibernian": "logos/Scotland - Scottish Premiership/Hibernian FC.png", "Hoffenheim": "logos/Germany - Bundesliga/TSG 1899 Hoffenheim.png", "Holstein Kiel": "history/2024-25/Germany - Bundesliga/Holstein Kiel.png", "Hull": "logos/England - Premier League/Hull City.png", "Ipswich": "logos/England - Premier League/Ipswich Town.png", "Iraklis": "logos/Greece - Super League 1/Iraklis Thessaloniki.png", "Juventus": "logos/Italy - Serie A/Juventus FC.png", "Kalamata": "logos/Greece - Super League 1/Kalamata FC.png", "Kifisia": "logos/Greece - Super League 1/AE Kifisia.png", "Kilmarnock": "logos/Scotland - Scottish Premiership/Kilmarnock FC.png", "Kortrijk": "logos/Belgium - Jupiler Pro League/KV Kortrijk.png", "Las Palmas": "history/2024-25/Spain - LaLiga/UD Las Palmas.png", "Lazio": "logos/Italy - Serie A/SS Lazio.png", "Le Havre": "logos/France - Ligue 1/Le Havre AC.png", "Le Mans": "logos/France - Ligue 1/Le Mans FC.png", "Lecce": "logos/Italy - Serie A/US Lecce.png", "Leeds": "logos/England - Premier League/Leeds United.png", "Leganes": "history/2024-25/Spain - LaLiga/CD Leganés.png", "Lens": "logos/France - Ligue 1/RC Lens.png", "Levante": "logos/Spain - LaLiga/Levante UD.png", "Leverkusen": "logos/Germany - Bundesliga/Bayer 04 Leverkusen.png", "Lille": "logos/France - Ligue 1/LOSC Lille.png", "Liverpool": "logos/England - Premier League/Liverpool FC.png", "Lommel SK": "logos/Belgium - Jupiler Pro League/Lommel SK.png", "Lorient": "logos/France - Ligue 1/FC Lorient.png", "Luton": "history/2023-24/England - Premier League/Luton Town.png", "Lyon": "logos/France - Ligue 1/Olympique Lyon.png", "M'gladbach": "history/2021-22/Germany - Bundesliga/Bor. M'gladbach.png", "Mainz": "logos/Germany - Bundesliga/1.FSV Mainz 05.png", "Malaga": "logos/Spain - LaLiga/Málaga CF.png", "Mallorca": "history/2025-26/Spain - LaLiga/RCD Mallorca.png", "Man City": "history/2021-22/England - Premier League/Man City.png", "Marseille": "history/2021-22/France - Ligue 1/Marseille.png", "Mechelen": "logos/Belgium - Jupiler Pro League/KV Mechelen.png", "Metz": "history/2025-26/France - Ligue 1/FC Metz.png", "Milan": "logos/Italy - Serie A/AC Milan.png", "Monaco": "logos/France - Ligue 1/AS Monaco.png", "Montpellier": "history/2021-22/France - Ligue 1/Montpellier.png", "Monza": "logos/Italy - Serie A/AC Monza.png", "Motherwell": "logos/Scotland - Scottish Premiership/Motherwell FC.png", "Nantes": "history/2025-26/France - Ligue 1/FC Nantes.png", "Napoli": "logos/Italy - Serie A/SSC Napoli.png", "Newcastle": "logos/England - Premier League/Newcastle United.png", "Nice": "logos/France - Ligue 1/OGC Nice.png", "Nijmegen": "logos/Netherlands - Eredivisie/NEC Nijmegen.png", "Norwich": "history/2021-22/England - Premier League/Norwich.png", "OFI Crete": "logos/Greece - Super League 1/OFI Crete.png", "Osasuna": "logos/Spain - LaLiga/CA Osasuna.png", "Oud-Heverlee Leuven": "logos/Belgium - Jupiler Pro League/Oud-Heverlee Leuven.png", "Oviedo": "history/2025-26/Spain - LaLiga/Real Oviedo.png", "PAOK": "logos/Greece - Super League 1/PAOK Thessaloniki.png", "PSV Eindhoven": "logos/Netherlands - Eredivisie/PSV Eindhoven.png", "Paderborn": "logos/Germany - Bundesliga/SC Paderborn 07.png", "Panathinaikos": "logos/Greece - Super League 1/Panathinaikos.png", "Panetolikos": "logos/Greece - Super League 1/Panetolikos.png", "Paris FC": "logos/France - Ligue 1/Paris FC.png", "Paris SG": "history/2021-22/France - Ligue 1/Paris SG.png", "Parma": "logos/Italy - Serie A/Parma Calcio 1913.png", "Pisa": "history/2025-26/Italy - Serie A/Pisa Sporting Club.png", "RAAL La Louviere": "logos/Belgium - Jupiler Pro League/RAAL La Louvière.png", "RB Leipzig": "logos/Germany - Bundesliga/RB Leipzig.png", "Rangers": "logos/Scotland - Scottish Premiership/Rangers FC.png", "Real Madrid": "logos/Spain - LaLiga/Real Madrid.png", "Red Star": "history/2021-22/Serbia - Super liga Srbije/Red Star.png", "Reims": "history/2024-25/France - Ligue 1/Stade Reims.png", "Roma": "logos/Italy - Serie A/AS Roma.png", "Sampdoria": "history/2021-22/Italy - Serie A/Sampdoria.png", "Santander": "logos/Spain - LaLiga/Racing Santander.png", "Sassuolo": "logos/Italy - Serie A/US Sassuolo.png", "Schalke 04": "logos/Germany - Bundesliga/FC Schalke 04.png", "Sevilla": "logos/Spain - LaLiga/Sevilla FC.png", "Sheffield United": "history/2023-24/England - Premier League/Sheffield United.png", "Sociedad": "logos/Spain - LaLiga/Real Sociedad.png", "Southampton": "history/2024-25/England - Premier League/Southampton FC.png", "Sparta Rotterdam": "logos/Netherlands - Eredivisie/Sparta Rotterdam.png", "St Johnstone": "logos/Scotland - Scottish Premiership/St. Johnstone FC.png", "St Mirren": "logos/Scotland - Scottish Premiership/St. Mirren FC.png", "St Pauli": "history/2025-26/Germany - Bundesliga/FC St. Pauli.png", "Standard": "logos/Belgium - Jupiler Pro League/Standard Liège.png", "Strasbourg": "logos/France - Ligue 1/RC Strasbourg Alsace.png", "Stuttgart": "logos/Germany - Bundesliga/VfB Stuttgart.png", "Sunderland": "logos/England - Premier League/Sunderland AFC.png", "Telstar": "logos/Netherlands - Eredivisie/SC Telstar.png", "Torino": "logos/Italy - Serie A/Torino FC.png", "Tottenham": "logos/England - Premier League/Tottenham Hotspur.png", "Toulouse": "logos/France - Ligue 1/FC Toulouse.png", "Troyes": "logos/France - Ligue 1/ESTAC Troyes.png", "Twente": "logos/Netherlands - Eredivisie/FC Twente Enschede.png", "Udinese": "logos/Italy - Serie A/Udinese Calcio.png", "Union Berlin": "logos/Germany - Bundesliga/1.FC Union Berlin.png", "Utrecht": "logos/Netherlands - Eredivisie/FC Utrecht.png", "Valencia": "logos/Spain - LaLiga/Valencia CF.png", "Valladolid": "history/2024-25/Spain - LaLiga/Real Valladolid CF.png", "Vallecano": "logos/Spain - LaLiga/Rayo Vallecano.png", "Venezia": "logos/Italy - Serie A/Venezia FC.png", "Verona": "history/2025-26/Italy - Serie A/Hellas Verona.png", "Villarreal": "logos/Spain - LaLiga/Villarreal CF.png", "Volos NFC": "logos/Greece - Super League 1/Volos NFC.png", "Waregem": "logos/Belgium - Jupiler Pro League/Zulte Waregem.png", "Watford": "history/2021-22/England - Premier League/Watford FC.png", "Werder Bremen": "logos/Germany - Bundesliga/SV Werder Bremen.png", "West Ham": "history/2025-26/England - Premier League/West Ham United.png", "Westerlo": "logos/Belgium - Jupiler Pro League/KVC Westerlo.png", "Willem II": "history/2021-22/Netherlands - Eredivisie/Willem II.png", "Wolfsburg": "history/2025-26/Germany - Bundesliga/VfL Wolfsburg.png", "Zwolle": "logos/Netherlands - Eredivisie/PEC Zwolle.png"}


def startpaket(karte):
    """Gepruefte Startzuordnung (luukhopman/football-logos) - nur fehlende Teams, nie ueberschreiben."""
    n = 0
    for team, pfad in START.items():
        if team in karte and os.path.exists(os.path.join(ORDNER, karte[team])):
            continue
        if speichern(team, START_URL + requests.utils.quote(pfad), karte):
            n += 1
    return n


def main():
    os.makedirs(ORDNER, exist_ok=True)
    try:
        karte = json.load(open(KARTE, encoding="utf-8"))
    except Exception:
        karte = {}
    s = startpaket(karte)
    if s:
        log(f"{s} Wappen aus dem Startpaket uebernommen")
        json.dump(dict(sorted(karte.items())), open(KARTE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    try:
        spiele = json.load(open(DATEN, encoding="utf-8")).get("spiele", [])
    except Exception as e:
        log("daten.json nicht lesbar", e)
        return
    je_liga = {}
    for s in spiele:
        if s.get("liga") in OHNE:
            continue
        for t in (s.get("heim"), s.get("gast")):
            if t and (t not in karte or not os.path.exists(os.path.join(ORDNER, karte[t]))):
                je_liga.setdefault(s.get("liga"), set()).add(t)
    offen = sum(len(v) for v in je_liga.values())
    log(f"{len(karte)} Wappen vorhanden, {offen} fehlen")
    if not offen:
        return
    neu, rest, aus = 0, [], 0
    for liga, teams in sorted(je_liga.items()):
        lid = LIGEN.get(liga)
        kand = []
        if lid:
            r = holen(f"{BASE}/lookup_all_teams.php", params={"id": lid})
            aus = aus + 1 if r is None else 0
            if aus >= 2:
                log("TheSportsDB nicht erreichbar – Abbruch, naechster Lauf versucht es wieder")
                rest = []
                break
            try:
                kand = kandidaten((r.json() or {}).get("teams")) if r else []
            except Exception:
                kand = []
            time.sleep(2.5)
        treffer = 0
        for t in sorted(teams):
            b = zuordnen(t, kand)
            if b and speichern(t, b, karte):
                neu += 1
                treffer += 1
            else:
                rest.append(t)
        log(f"{liga}: {treffer}/{len(teams)} neu ({len(kand)} Teams bei TheSportsDB)")
    for t in rest[:NAMENSSUCHE_MAX]:
        r = holen(f"{BASE}/searchteams.php", params={"t": ALIAS.get(t.lower(), t)})
        time.sleep(2.5)
        try:
            kand = kandidaten([x for x in ((r.json() or {}).get("teams") or []) if x.get("strSport") == "Soccer"]) if r else []
        except Exception:
            kand = []
        b = zuordnen(t, kand) or (kand[0][0] if len(kand) == 1 else None)
        if b and speichern(t, b, karte):
            neu += 1
    json.dump(dict(sorted(karte.items())), open(KARTE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    fehlt = [t for v in je_liga.values() for t in v if t not in karte]
    log(f"{neu} neue Wappen gespeichert, jetzt {len(karte)}" + (f" · noch ohne: {', '.join(sorted(fehlt)[:40])}" if fehlt else ""))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("Fehler (Workflow laeuft trotzdem weiter):", e)
    sys.exit(0)
