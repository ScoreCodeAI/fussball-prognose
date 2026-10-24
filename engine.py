# ===== SCORE CODE AGENT - BACKTEST- UND PROGNOSE-MOTOR =====
# Rechnet fuer jede Liga und jedes Modell:
#   - Prognosen fuer die kommenden Spiele (inkl. Value gegen die Quote)
#   - Backtest: was das Modell VOR jedem Spiel getippt haette -> ROI, Trefferquote
#   - Letzter Spieltag: Tipp vs. Ergebnis
# Ergebnis: docs/daten.json (wird von der Webseite angezeigt)
import glob, importlib, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")
ORDNER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ORDNER)
import daten
from ligen import LAENDER, SAISON_START

# ---------- Einstellungen ----------
BACKTEST_START = "2025-07-01"   # ab hier wird rueckwirkend getestet (Saison 2025/26 + 2026/27)
SAISONS_QUOTEN = ["2526", "2627"]
NEU_TRAINIEREN_TAGE = 7          # Modell wird jede Woche neu trainiert (wie in echt)
MIN_VALUE = 0.05                 # ab +5 % Vorteil wird gewettet
MAX_VALUE = 0.40                 # ueber +40 % ist meist ein Datenfehler -> ignorieren
FLAGGEN = {"Deutschland": "🇩🇪", "England": "🏴", "Spanien": "🇪🇸", "Italien": "🇮🇹", "Frankreich": "🇫🇷",
           "Niederlande": "🇳🇱", "Portugal": "🇵🇹", "Belgien": "🇧🇪", "Türkei": "🇹🇷", "Griechenland": "🇬🇷", "Schottland": "🏴"}
MODELL_FARBEN = {"modell_poisson": ("λ", "#22e4ff", "#a855f7", "Angriffs- und Abwehrstärke, neuere Spiele zählen mehr"),
                 "modell_elo": ("E", "#a855f7", "#e449ff", "Teamstärke aus Elo-Werten"),
                 "modell_durchschnitt": ("Ø", "#ffb020", "#ff6b3d", "Einfaches Vergleichsmodell aus 12 Monaten")}
MAERKTE = {"1": ["AvgH", "B365H"], "X": ["AvgD", "B365D"], "2": ["AvgA", "B365A"],
           "Ü2.5": ["Avg>2.5", "B365>2.5"], "U2.5": ["Avg<2.5", "B365<2.5"]}
MARKT_NAME = {"1": "Heimsieg", "X": "Remis", "2": "Auswärtssieg", "Ü2.5": "Über 2,5", "U2.5": "Unter 2,5"}
heute = pd.Timestamp.today().normalize()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------- Modelle finden ----------
def modelle_laden():
    m = {}
    for pfad in sorted(glob.glob(os.path.join(ORDNER, "modell_*.py"))):
        key = os.path.basename(pfad)[:-3]
        m[key] = importlib.import_module(key)
    return m


# ---------- Daten ----------
def quoten_spiele():
    """Gespielte Spiele mit Quoten von football-data (fuer den Backtest)."""
    teile = []
    spalten = sorted({c for v in MAERKTE.values() for c in v})
    for s in SAISONS_QUOTEN:
        for liga in daten.ALLE_LIGEN:
            try:
                d = daten._csv(daten.FD_SAISON.format(saison=s, liga=liga))
            except Exception:
                continue
            d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
            out = pd.DataFrame({"Div": liga, "Home": d["HomeTeam"], "Away": d["AwayTeam"],
                                "HG": pd.to_numeric(d["FTHG"], errors="coerce"),
                                "AG": pd.to_numeric(d["FTAG"], errors="coerce"),
                                "Date": pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")})
            for c in spalten:
                out[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
            teile.append(out)
    q = pd.concat(teile).dropna(subset=["Date", "HG", "AG"])
    return q.drop_duplicates(subset=["Div", "Date", "Home", "Away"], keep="last").sort_values("Date")


# ---------- Rechnen ----------
def maerkte(lh, la):
    M = np.outer(poisson.pmf(range(11), lh), poisson.pmf(range(11), la))
    M /= M.sum()
    unter = M[np.add.outer(range(11), range(11)) <= 2].sum()
    i, j = np.unravel_index(M.argmax(), M.shape)
    return {"1": np.tril(M, -1).sum(), "X": np.trace(M), "2": np.triu(M, 1).sum(),
            "Ü2.5": 1 - unter, "U2.5": unter, "BTS": 1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0],
            "tipp": f"{i}:{j}"}


def quote(row, namen):
    for n in namen:
        v = row.get(n)
        if v is not None and pd.notna(v) and v > 1:
            return float(v)
    return None


def value_wetten(p, row):
    w = []
    for mk, sp in MAERKTE.items():
        q = quote(row, sp)
        if q:
            edge = p[mk] * q - 1
            if MIN_VALUE < edge < MAX_VALUE:
                w.append((mk, q, edge))
    return sorted(w, key=lambda x: -x[2])


def gewonnen(mk, hg, ag):
    return {"1": hg > ag, "X": hg == ag, "2": hg < ag, "Ü2.5": hg + ag > 2, "U2.5": hg + ag <= 2}[mk]


def r(x, n=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def backtest(mod, spiele_land, quoten_land):
    """Walk-forward: jede Woche nur mit Daten VOR dieser Woche trainieren."""
    zeilen = []
    for w in pd.date_range(BACKTEST_START, heute, freq=f"{NEU_TRAINIEREN_TAGE}D"):
        woche = quoten_land[(quoten_land["Date"] >= w) & (quoten_land["Date"] < w + pd.Timedelta(days=NEU_TRAINIEREN_TAGE))]
        if woche.empty:
            continue
        try:
            modell = mod.trainiere(spiele_land[spiele_land["Date"] < w], w)
        except Exception as e:
            log("   Training fehlgeschlagen", w.date(), e)
            continue
        for _, s in woche.iterrows():
            try:
                lh, la = mod.tore_erwartet(modell, s["Home"], s["Away"])
            except Exception:
                continue
            p = maerkte(lh, la)
            wetten = value_wetten(p, s)
            zeilen.append({"Div": s["Div"], "Date": s["Date"], "Home": s["Home"], "Away": s["Away"],
                           "HG": int(s["HG"]), "AG": int(s["AG"]), "p": p, "row": s, "wetten": wetten})
    return zeilen


def kennzahlen(wetten):
    n = len(wetten)
    if n == 0:
        return {"wetten": 0, "roi": None, "treffer": None, "gewinn": 0.0}
    g = sum(x["profit"] for x in wetten)
    return {"wetten": n, "roi": r(g / n * 100), "treffer": r(sum(x["profit"] > 0 for x in wetten) / n * 100, 0), "gewinn": r(g, 2)}


def auswerten(zeilen, liga):
    z = [x for x in zeilen if x["Div"] == liga]
    alle_wetten = []
    for x in z:
        for mk, q, edge in x["wetten"]:
            win = gewonnen(mk, x["HG"], x["AG"])
            alle_wetten.append({"Date": x["Date"], "markt": mk, "profit": (q - 1) if win else -1.0})
    saison = [w for w in alle_wetten if w["Date"] >= pd.Timestamp(SAISON_START)]
    letzte30 = [w for w in alle_wetten if w["Date"] >= heute - pd.Timedelta(days=30)]

    # Gewinnkurve (pro Woche kumuliert)
    kurve = []
    if alle_wetten:
        df = pd.DataFrame(alle_wetten).set_index("Date").sort_index()
        kurve = [r(v, 2) for v in df["profit"].resample("W").sum().cumsum().tolist()]

    maerkte_roi = {}
    for mk in MAERKTE:
        k = kennzahlen([w for w in alle_wetten if w["markt"] == mk])
        maerkte_roi[MARKT_NAME[mk]] = k

    # Letzter Spieltag: alle Spiele der letzten 5 Tage vor dem juengsten Spiel
    letzter = []
    if z:
        ende = max(x["Date"] for x in z)
        for x in [x for x in z if x["Date"] > ende - pd.Timedelta(days=5)]:
            p = x["p"]
            if x["wetten"]:
                mk, q, _ = x["wetten"][0]
                value = True
            else:
                mk = max(["1", "X", "2"], key=lambda k: p[k])
                q, value = quote(x["row"], MAERKTE[mk]), False
            win = gewonnen(mk, x["HG"], x["AG"])
            letzter.append({"datum": x["Date"].strftime("%d.%m."), "heim": x["Home"], "gast": x["Away"],
                            "ergebnis": f'{x["HG"]}:{x["AG"]}', "tipp": mk, "prog": p["tipp"], "quote": r(q, 2),
                            "value": value, "richtig": bool(win),
                            "profit": r(((q - 1) if win else -1.0) if (value and q) else 0.0, 2)})
    return {"saison": kennzahlen(saison), "letzte30": kennzahlen(letzte30), "gesamt": kennzahlen(alle_wetten),
            "kurve": kurve, "maerkte": maerkte_roi, "letzter_spieltag": letzter}


def prognosen(mod, modell, fx_liga):
    out = []
    for _, s in fx_liga.iterrows():
        try:
            lh, la = mod.tore_erwartet(modell, s["HomeTeam"], s["AwayTeam"])
        except Exception:
            continue
        p = maerkte(lh, la)
        w = value_wetten(p, s)
        out.append({"datum": s["Date"].strftime("%Y-%m-%d") if pd.notna(s["Date"]) else None,
                    "zeit": str(s.get("Time", "") or "")[:5], "heim": s["HomeTeam"], "gast": s["AwayTeam"],
                    "p1": r(p["1"] * 100, 0), "px": r(p["X"] * 100, 0), "p2": r(p["2"] * 100, 0),
                    "ue25": r(p["Ü2.5"] * 100, 0), "bts": r(p["BTS"] * 100, 0), "tipp": p["tipp"],
                    "xg": [r(lh), r(la)],
                    "quoten": {mk: r(quote(s, sp), 2) for mk, sp in MAERKTE.items()},
                    "value": [{"markt": MARKT_NAME[mk], "quote": r(q, 2), "edge": r(e * 100, 0)} for mk, q, e in w]})
    return out


# ---------- Hauptprogramm ----------
def main():
    t0 = time.time()
    modelle = modelle_laden()
    log("Modelle:", ", ".join(m.NAME for m in modelle.values()))
    log("Lade Historie, Quoten und kommende Spiele ...")
    hist = daten.historie()
    q = quoten_spiele()
    spiele = pd.concat([hist, q[daten.SPALTEN]]).dropna(subset=["Date", "HG", "AG"])
    spiele = spiele.drop_duplicates(subset=["Div", "Date", "Home", "Away"], keep="last").sort_values("Date")
    fx = daten.kommende_spiele()
    log(f"Spiele: {len(spiele):,} | mit Quoten: {len(q):,} | kommend: {len(fx)}")

    ergebnis = {"stand": heute.strftime("%Y-%m-%d"), "daten_bis": spiele["Date"].max().strftime("%Y-%m-%d"),
                "anzahl_spiele": int(len(spiele)), "einstellungen": {"min_value": MIN_VALUE, "backtest_start": BACKTEST_START},
                "laender": [], "modelle": [], "ligen": {}}
    for key, mod in modelle.items():
        ic, c1, c2, txt = MODELL_FARBEN.get(key, ("•", "#22e4ff", "#a855f7", ""))
        ergebnis["modelle"].append({"id": key, "name": mod.NAME, "icon": ic, "farbe1": c1, "farbe2": c2,
                                    "beschreibung": getattr(mod, "BESCHREIBUNG", txt)})

    gesamt_wetten = {k: [] for k in modelle}
    for land, ligen in LAENDER.items():
        ergebnis["laender"].append({"name": land, "flagge": FLAGGEN.get(land, "🏳"),
                                    "ligen": [{"code": c, "name": n} for c, n in ligen.items()]})
        codes = list(ligen)
        sp_land = spiele[spiele["Div"].isin(codes)]
        q_land = q[(q["Div"].isin(codes)) & (q["Date"] >= pd.Timestamp(BACKTEST_START))]
        for key, mod in modelle.items():
            log(f"{land} · {mod.NAME}: Backtest ...")
            zeilen = backtest(mod, sp_land, q_land)
            modell_jetzt = mod.trainiere(sp_land, heute)
            for c in codes:
                a = auswerten(zeilen, c)
                a["prognosen"] = prognosen(mod, modell_jetzt, fx[fx["Div"] == c])
                ergebnis["ligen"].setdefault(c, {"name": ligen[c], "land": land, "modelle": {}})["modelle"][key] = a
                s = a["saison"]
                log(f"   {ligen[c]:<18} Wetten {s['wetten']:>4} | ROI {s['roi']} % | Treffer {s['treffer']} % | "
                    f"Prognosen {len(a['prognosen'])}")
            for x in zeilen:
                for mk, qq, _ in x["wetten"]:
                    if x["Date"] >= pd.Timestamp(SAISON_START):
                        gesamt_wetten[key].append({"profit": (qq - 1) if gewonnen(mk, x["HG"], x["AG"]) else -1.0})
    for m in ergebnis["modelle"]:
        m["saison"] = kennzahlen(gesamt_wetten[m["id"]])

    os.makedirs(os.path.join(ORDNER, "docs"), exist_ok=True)
    pfad = os.path.join(ORDNER, "docs", "daten.json")
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, ensure_ascii=False, separators=(",", ":"))
    log(f"Fertig in {time.time() - t0:.0f} s -> {pfad} ({os.path.getsize(pfad) / 1024:.0f} KB)")
    print("\n===== MODELL-RANKING (Saison, alle Ligen) =====")
    for m in sorted(ergebnis["modelle"], key=lambda m: -(m["saison"]["roi"] or -999)):
        s = m["saison"]
        print(f"{m['name']:<32} Wetten {s['wetten']:>4} | ROI {s['roi']} % | Treffer {s['treffer']} % | Gewinn {s['gewinn']} Einheiten")


if __name__ == "__main__":
    main()
