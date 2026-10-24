# ===== SCORE CODE AGENT - BACKTEST- UND PROGNOSE-MOTOR (v2, auto-kalibriert) =====
# 1) Walk-forward-Backtest: jede Woche nur mit Daten VOR dieser Woche trainieren
# 2) Auto-Kalibrierung: sucht pro Modell die besten Einstellungen
#    (Markt-Mischung, Mindest-Value, Quotenbereich innerhalb 1,5-5,0)
#    - gelernt an der Vorsaison, ehrlich geprueft an der laufenden Saison
# 3) Prognosen fuer kommende Spiele mit den auf ALLEN Daten kalibrierten Werten
# Ergebnis: docs/daten.json
import glob, importlib, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")
ORDNER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ORDNER)
import daten
from ligen import LAENDER, SAISON_START

# ---------- Grundeinstellungen (nur Grenzen - die Werte sucht der Motor selbst) ----------
BACKTEST_START = "2025-07-01"
SAISONS_QUOTEN = ["2526", "2627"]
NEU_TRAINIEREN_TAGE = 7
QUOTE_GRENZEN = (1.5, 5.0)       # aeusserer Rahmen fuer erlaubte Quoten
MAX_VALUE = 0.50                 # ueber +50 % ist praktisch immer ein Datenfehler
MIN_WETTEN = 60                  # eine Einstellung braucht genug Wetten, sonst zaehlt sie nicht

# Suchraum der Auto-Kalibrierung
GITTER_MISCHUNG = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]       # Anteil Modell vs. Markt (1.0 = nur Modell)
GITTER_MIN_VALUE = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
GITTER_QUOTE_MIN = [1.5, 1.7, 1.9, 2.1, 2.5]
GITTER_QUOTE_MAX = [2.5, 3.0, 3.5, 4.0, 5.0]

FLAGGEN = {"Deutschland": "🇩🇪", "England": "🏴", "Spanien": "🇪🇸", "Italien": "🇮🇹", "Frankreich": "🇫🇷",
           "Niederlande": "🇳🇱", "Portugal": "🇵🇹", "Belgien": "🇧🇪", "Türkei": "🇹🇷", "Griechenland": "🇬🇷", "Schottland": "🏴"}
MODELL_FARBEN = {"modell_poisson": ("λ", "#22e4ff", "#a855f7", "Angriffs- und Abwehrstärke, neuere Spiele zählen mehr"),
                 "modell_elo": ("E", "#a855f7", "#e449ff", "Teamstärke aus Elo-Werten"),
                 "modell_durchschnitt": ("Ø", "#ffb020", "#ff6b3d", "Einfaches Vergleichsmodell aus 12 Monaten")}
MAERKTE = {"1": ["AvgH", "B365H"], "X": ["AvgD", "B365D"], "2": ["AvgA", "B365A"],
           "Ü2.5": ["Avg>2.5", "B365>2.5"], "U2.5": ["Avg<2.5", "B365<2.5"]}
GRUPPE = {"1": "1X2", "X": "1X2", "2": "1X2", "Ü2.5": "OU", "U2.5": "OU"}
MARKT_NAME = {"1": "Heimsieg", "X": "Remis", "2": "Auswärtssieg", "Ü2.5": "Über 2,5", "U2.5": "Unter 2,5"}
heute = pd.Timestamp.today().normalize()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def r(x, n=1):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) or np.isinf(x) else round(x, n)


def modelle_laden():
    m = {}
    for pfad in sorted(glob.glob(os.path.join(ORDNER, "modell_*.py"))):
        key = os.path.basename(pfad)[:-3]
        m[key] = importlib.import_module(key)
    return m


# ---------- Daten ----------
def quoten_spiele():
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


# ---------- Wahrscheinlichkeiten ----------
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
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isnan(v) and v > 1:
            return v
    return None


def markt_wkeiten(row):
    """Faire Markt-Wahrscheinlichkeiten aus den Quoten (Buchmacher-Marge herausgerechnet)."""
    q = {mk: quote(row, sp) for mk, sp in MAERKTE.items()}
    out = {}
    for grp in [["1", "X", "2"], ["Ü2.5", "U2.5"]]:
        if all(q[m] for m in grp):
            inv = {m: 1 / q[m] for m in grp}
            s = sum(inv.values())
            out.update({m: inv[m] / s for m in grp})
    return q, out


def gewonnen(mk, hg, ag):
    return {"1": hg > ag, "X": hg == ag, "2": hg < ag, "Ü2.5": hg + ag > 2, "U2.5": hg + ag <= 2}[mk]


# ---------- Backtest ----------
def backtest(mod, spiele_land, quoten_land):
    spiele_out, kandidaten = [], []
    for w in pd.date_range(BACKTEST_START, heute, freq=f"{NEU_TRAINIEREN_TAGE}D"):
        woche = quoten_land[(quoten_land["Date"] >= w) & (quoten_land["Date"] < w + pd.Timedelta(days=NEU_TRAINIEREN_TAGE))]
        if woche.empty:
            continue
        try:
            modell = mod.trainiere(spiele_land[spiele_land["Date"] < w], w)
        except Exception as e:
            log("   Training übersprungen", w.date(), str(e)[:60])
            continue
        for _, s in woche.iterrows():
            try:
                lh, la = mod.tore_erwartet(modell, s["Home"], s["Away"])
            except Exception:
                continue
            p = maerkte(lh, la)
            q, pk = markt_wkeiten(s)
            sid = len(spiele_out)
            spiele_out.append({"id": sid, "Div": s["Div"], "Date": s["Date"], "Home": s["Home"], "Away": s["Away"],
                               "HG": int(s["HG"]), "AG": int(s["AG"]), "p": p, "q": q})
            for mk in MAERKTE:
                if q[mk] and mk in pk:
                    kandidaten.append({"id": sid, "Div": s["Div"], "Date": s["Date"], "mk": mk, "grp": GRUPPE[mk],
                                       "pm": p[mk], "pk": pk[mk], "q": q[mk], "win": gewonnen(mk, s["HG"], s["AG"])})
    return spiele_out, kandidaten


def wetten_waehlen(K, par):
    """Waehlt die Wetten fuer eine Einstellung: hoechstens 1 Wette je Spiel und Marktgruppe."""
    if K.empty:
        return K.assign(edge=[], profit=[])
    pf = par["mischung"] * K["pm"] + (1 - par["mischung"]) * K["pk"]
    edge = pf * K["q"] - 1
    m = (edge > par["min_value"]) & (edge < MAX_VALUE) & (K["q"] >= par["quote_min"]) & (K["q"] <= par["quote_max"])
    W = K[m].assign(edge=edge[m])
    W = W.sort_values("edge", ascending=False).drop_duplicates(["id", "grp"])
    return W.assign(profit=np.where(W["win"], W["q"] - 1, -1.0))


def kalibrieren(K):
    """Probiert alle Kombinationen und nimmt die mit dem besten 'vorsichtigen' ROI
    (Durchschnittsgewinn minus Unsicherheit) - so gewinnen keine Zufallstreffer."""
    best, best_score = None, -1e9
    for mi in GITTER_MISCHUNG:
        for mv in GITTER_MIN_VALUE:
            for qmin in GITTER_QUOTE_MIN:
                for qmax in GITTER_QUOTE_MAX:
                    if qmax <= qmin or qmin < QUOTE_GRENZEN[0] or qmax > QUOTE_GRENZEN[1]:
                        continue
                    par = {"mischung": mi, "min_value": mv, "quote_min": qmin, "quote_max": qmax}
                    W = wetten_waehlen(K, par)
                    n = len(W)
                    if n < MIN_WETTEN:
                        continue
                    score = W["profit"].mean() - W["profit"].std() / np.sqrt(n)
                    if score > best_score:
                        best, best_score = par, score
    return best or {"mischung": 0.5, "min_value": 0.10, "quote_min": 1.5, "quote_max": 5.0}


def kennzahlen(W):
    n = len(W)
    if n == 0:
        return {"wetten": 0, "roi": None, "treffer": None, "gewinn": 0.0}
    g = W["profit"].sum()
    return {"wetten": int(n), "roi": r(g / n * 100), "treffer": r((W["profit"] > 0).mean() * 100, 0), "gewinn": r(g, 2)}


# ---------- Auswertung je Liga ----------
def auswerten(spiele, W, liga):
    z = [s for s in spiele if s["Div"] == liga]
    Wl = W[W["Div"] == liga]
    saison = Wl[Wl["Date"] >= pd.Timestamp(SAISON_START)]
    kurve = []
    if not saison.empty:
        kurve = [r(v, 2) for v in saison.set_index("Date")["profit"].sort_index().resample("W").sum().cumsum()]
    maerkte_roi = {MARKT_NAME[mk]: kennzahlen(saison[saison["mk"] == mk]) for mk in MAERKTE}

    letzter = []
    if z:
        ende = max(s["Date"] for s in z)
        wetten_id = {}
        for _, w in Wl.iterrows():
            wetten_id.setdefault(w["id"], []).append(w)
        for s in [s for s in z if s["Date"] > ende - pd.Timedelta(days=5)]:
            p = s["p"]
            if s["id"] in wetten_id:
                w = max(wetten_id[s["id"]], key=lambda x: x["edge"])
                mk, q, value = w["mk"], w["q"], True
            else:
                mk = max(["1", "X", "2"], key=lambda k: p[k])
                q, value = s["q"].get(mk), False
            win = gewonnen(mk, s["HG"], s["AG"])
            letzter.append({"datum": s["Date"].strftime("%d.%m."), "heim": s["Home"], "gast": s["Away"],
                            "ergebnis": f'{s["HG"]}:{s["AG"]}', "tipp": mk, "prog": p["tipp"], "quote": r(q, 2),
                            "value": value, "richtig": bool(win),
                            "profit": r(((q - 1) if win else -1.0) if value else 0.0, 2)})
    return {"saison": kennzahlen(saison), "gesamt": kennzahlen(Wl), "kurve": kurve,
            "maerkte": maerkte_roi, "letzter_spieltag": letzter}


def prognosen(mod, modell, fx_liga, par):
    out = []
    for _, s in fx_liga.iterrows():
        try:
            lh, la = mod.tore_erwartet(modell, s["HomeTeam"], s["AwayTeam"])
        except Exception:
            continue
        p = maerkte(lh, la)
        q, pk = markt_wkeiten(s)
        value = []
        for grp in [["1", "X", "2"], ["Ü2.5", "U2.5"]]:
            best = None
            for mk in grp:
                if not (q[mk] and mk in pk) or not (par["quote_min"] <= q[mk] <= par["quote_max"]):
                    continue
                e = (par["mischung"] * p[mk] + (1 - par["mischung"]) * pk[mk]) * q[mk] - 1
                if par["min_value"] < e < MAX_VALUE and (best is None or e > best[2]):
                    best = (mk, q[mk], e)
            if best:
                value.append({"markt": MARKT_NAME[best[0]], "quote": r(best[1], 2), "edge": r(best[2] * 100, 0)})
        out.append({"datum": s["Date"].strftime("%Y-%m-%d") if pd.notna(s["Date"]) else None,
                    "zeit": str(s.get("Time", "") or "")[:5], "heim": s["HomeTeam"], "gast": s["AwayTeam"],
                    "p1": r(p["1"] * 100, 0), "px": r(p["X"] * 100, 0), "p2": r(p["2"] * 100, 0),
                    "ue25": r(p["Ü2.5"] * 100, 0), "bts": r(p["BTS"] * 100, 0), "tipp": p["tipp"],
                    "xg": [r(lh), r(la)], "quoten": {mk: r(v, 2) for mk, v in q.items()}, "value": value})
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
                "anzahl_spiele": int(len(spiele)), "laender": [], "modelle": [], "ligen": {}}
    for land, ligen in LAENDER.items():
        ergebnis["laender"].append({"name": land, "flagge": FLAGGEN.get(land, "🏳"),
                                    "ligen": [{"code": c, "name": n} for c, n in ligen.items()]})

    for key, mod in modelle.items():
        log(f"===== {mod.NAME} =====")
        alle_spiele, alle_kand, live_modelle = [], [], {}
        for land, ligen in LAENDER.items():
            codes = list(ligen)
            sp_land = spiele[spiele["Div"].isin(codes)]
            q_land = q[(q["Div"].isin(codes)) & (q["Date"] >= pd.Timestamp(BACKTEST_START))]
            log(f"{land}: Backtest ...")
            s_out, k_out = backtest(mod, sp_land, q_land)
            off = len(alle_spiele)
            for s in s_out:
                s["id"] += off
            for k in k_out:
                k["id"] += off
            alle_spiele += s_out
            alle_kand += k_out
            live_modelle[land] = mod.trainiere(sp_land, heute)
        K = pd.DataFrame(alle_kand)

        # Kalibrierung 1 (ehrlich): nur an der Vorsaison lernen -> laufende Saison ist unbekannt
        vor = K[K["Date"] < pd.Timestamp(SAISON_START)]
        par_test = kalibrieren(vor)
        W_test = wetten_waehlen(K, par_test)
        # Kalibrierung 2 (live): an allen Daten lernen -> fuer die kommenden Prognosen
        par_live = kalibrieren(K)
        log(f"Einstellung (an Vorsaison gelernt): {par_test}")
        log(f"Einstellung (live, alle Daten):    {par_live}")

        for land, ligen in LAENDER.items():
            for c in ligen:
                a = auswerten(alle_spiele, W_test, c)
                a["prognosen"] = prognosen(mod, live_modelle[land], fx[fx["Div"] == c], par_live)
                ergebnis["ligen"].setdefault(c, {"name": ligen[c], "land": land, "modelle": {}})["modelle"][key] = a
                s = a["saison"]
                log(f"   {ligen[c]:<18} Wetten {s['wetten']:>4} | ROI {s['roi']} % | Treffer {s['treffer']} %")

        ic, c1, c2, txt = MODELL_FARBEN.get(key, ("•", "#22e4ff", "#a855f7", ""))
        saison_w = W_test[W_test["Date"] >= pd.Timestamp(SAISON_START)]
        ergebnis["modelle"].append({
            "id": key, "name": mod.NAME, "icon": ic, "farbe1": c1, "farbe2": c2,
            "beschreibung": getattr(mod, "BESCHREIBUNG", txt),
            "saison": kennzahlen(saison_w),
            "vorsaison": kennzahlen(W_test[W_test["Date"] < pd.Timestamp(SAISON_START)]),
            "einstellung_test": par_test, "einstellung_live": par_live})

    os.makedirs(os.path.join(ORDNER, "docs"), exist_ok=True)
    pfad = os.path.join(ORDNER, "docs", "daten.json")
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, ensure_ascii=False, separators=(",", ":"))
    log(f"Fertig in {time.time() - t0:.0f} s -> {pfad} ({os.path.getsize(pfad) / 1024:.0f} KB)")

    print("\n===== MODELL-RANKING =====")
    print("Vorsaison = daran wurde gelernt | Saison = ehrlicher Test mit unbekannten Spielen\n")
    for m in sorted(ergebnis["modelle"], key=lambda m: -(m["saison"]["roi"] if m["saison"]["roi"] is not None else -999)):
        v, s, p = m["vorsaison"], m["saison"], m["einstellung_test"]
        print(f"{m['name']}")
        print(f"   Einstellung: Modell-Anteil {p['mischung']:.0%} | Value ab {p['min_value']:.0%} | Quoten {p['quote_min']}-{p['quote_max']}")
        print(f"   Vorsaison: {v['wetten']:>4} Wetten | ROI {v['roi']} % | Treffer {v['treffer']} %")
        print(f"   SAISON:    {s['wetten']:>4} Wetten | ROI {s['roi']} % | Treffer {s['treffer']} % | Gewinn {s['gewinn']} Einheiten\n")


if __name__ == "__main__":
    main()
