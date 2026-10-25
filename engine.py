# ===== SCORE CODE AGENT - BACKTEST- UND PROGNOSE-MOTOR (v3) =====
# Jedes Modell ist eine eigene Datei modell_*.py und wird automatisch erkannt.
# Schnittstelle eines Modells:
#   NAME, KURZ, ICON, FARBE1, FARBE2, BESCHREIBUNG   (Anzeige im Dashboard)
#   BACKTEST_WOCHEN (optional)                        (kuerzerer Rueckwaerts-Test fuer langsame Modelle)
#   trainiere(df, heute) -> modell                    (nur Spiele VOR "heute")
#   vorhersage(modell, heim, gast) -> {"ft": Matrix, "ht": Matrix oder None, "xg": [h, a]}
#   diagnose(modell) -> Text (optional)
# Ablauf:
#  1) Walk-forward-Backtest: jede Woche nur mit Daten VOR dieser Woche trainieren
#  2) Qualitaetsmasse (RPS, Log-Loss, ...) und Trefferquoten je Markt
#  3) Value-Auto-Kalibrierung (Quotenbereich 1,5-5,0)
#  4) Prognosen fuer kommende Spiele -> docs/daten.json
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # Prozesse laufen parallel - jeder nur mit einem Rechenkern

import glob, importlib, json, multiprocessing, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ORDNER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ORDNER)
import daten
import kern
import markt
import quoten_modell
import values
from ligen import LAENDER, SAISON_START, EXTRA_LIGEN

# ---------- Grundeinstellungen ----------
BACKTEST_START = "2025-07-01"
SAISONS_QUOTEN = ["2122", "2223", "2324", "2425", "2526", "2627"]   # aeltere Saisons fuellen Luecken der Historie (z. B. Griechenland)
NEU_TRAINIEREN_TAGE = 7
QUOTE_GRENZEN = (1.5, 5.0)
MAX_VALUE = 0.50
MIN_WETTEN = 60
GITTER_MISCHUNG = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
GITTER_MIN_VALUE = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
GITTER_QUOTE_MIN = [1.5, 1.7, 1.9, 2.1, 2.5]
GITTER_QUOTE_MAX = [2.5, 3.0, 3.5, 4.0, 5.0]

FLAGGEN = {"Deutschland": "🇩🇪", "England": "🇬🇧", "Spanien": "🇪🇸", "Italien": "🇮🇹", "Frankreich": "🇫🇷",
           "Niederlande": "🇳🇱", "Portugal": "🇵🇹", "Belgien": "🇧🇪", "Griechenland": "🇬🇷", "Schottland": "🇬🇧",
           "Dänemark": "🇩🇰", "Norwegen": "🇳🇴", "Schweden": "🇸🇪", "Finnland": "🇫🇮", "Türkei": "🇹🇷"}
MAERKTE = {"1": ["AvgH", "B365H"], "X": ["AvgD", "B365D"], "2": ["AvgA", "B365A"],
           "Ü2.5": ["Avg>2.5", "B365>2.5"], "U2.5": ["Avg<2.5", "B365<2.5"]}
GRUPPE = {"1": "1X2", "X": "1X2", "2": "1X2", "Ü2.5": "OU", "U2.5": "OU"}
MARKT_NAME = {"1": "Heimsieg", "X": "Remis", "2": "Auswärtssieg", "Ü2.5": "Über 2,5", "U2.5": "Unter 2,5"}
EXTRA_QUOTEN = ["PSH", "PSD", "PSA", "P>2.5", "P<2.5", "MaxH", "MaxD", "MaxA", "Max>2.5", "Max<2.5"]  # nur fuer Markt-Vergleich
K_SPALTEN = ["id", "Div", "Date", "mk", "grp", "pm", "pk", "q", "win"]
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
    spalten = sorted({c for v in MAERKTE.values() for c in v} | set(EXTRA_QUOTEN))
    quellen = [(s, liga) for s in SAISONS_QUOTEN for liga in daten.ALLE_LIGEN if liga not in EXTRA_LIGEN]
    quellen += [("neu", liga) for liga in EXTRA_LIGEN if liga in daten.ALLE_LIGEN]
    for s, liga in quellen:
        try:
            d = daten.extra_liga(liga) if s == "neu" else daten._csv(daten.FD_SAISON.format(saison=s, liga=liga))
        except Exception:
            continue
        d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        out = pd.DataFrame({"Div": liga, "Home": d["HomeTeam"], "Away": d["AwayTeam"],
                            "HG": pd.to_numeric(d["FTHG"], errors="coerce"),
                            "AG": pd.to_numeric(d["FTAG"], errors="coerce"),
                            "Date": pd.to_datetime(d["Date"], dayfirst=True, errors="coerce"),
                            "Time": d["Time"].fillna("").astype(str).str[:5] if "Time" in d else ""})
        for c in spalten:
            out[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        for c in daten.STAT_SPALTEN:
            out[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        teile.append(out)
    q = pd.concat(teile).dropna(subset=["Date", "HG", "AG"])
    q = q[q["Date"] >= pd.Timestamp(BACKTEST_START) - pd.DateOffset(years=4)]
    return q.drop_duplicates(subset=["Div", "Date", "Home", "Away"], keep="last").sort_values("Date")


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


def _int(v):
    try:
        return None if pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


# ---------- Modell aufrufen ----------
def berechne(mod, modell, heim, gast):
    if hasattr(mod, "vorhersage"):
        o = mod.vorhersage(modell, heim, gast)
        ft, ht = o["ft"], o.get("ht")
        xg = o.get("xg") or kern.erwartete_tore(ft)
    else:   # alte Schnittstelle: nur erwartete Tore, keine eigene Halbzeit
        lh, la = mod.tore_erwartet(modell, heim, gast)
        ft, ht, xg = kern.matrix_unabhaengig(lh, la), None, [lh, la]
    return ft, ht, [r(xg[0]), r(xg[1])]


# ---------- Backtest ----------
def backtest(mod, spiele_land, quoten_land, start):
    spiele_out, kandidaten = [], []
    for w in pd.date_range(start, heute, freq=f"{NEU_TRAINIEREN_TAGE}D"):
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
                ft, ht, xg = berechne(mod, modell, s["Home"], s["Away"])
            except Exception:
                continue
            p = kern.maerkte(ft, ht)
            q, pk = markt_wkeiten(s)
            hg, ag, hth, hta = int(s["HG"]), int(s["AG"]), _int(s.get("HTHG")), _int(s.get("HTAG"))
            sid = len(spiele_out)
            spiele_out.append({"id": sid, "Div": s["Div"], "Date": s["Date"], "Home": s["Home"], "Away": s["Away"],
                               "HG": hg, "AG": ag, "p": p, "q": q, "Time": str(s.get("Time", "") or "")[:5], "xg": xg,
                               "HT": f"{hth}:{hta}" if hth is not None and hta is not None else None,
                               "qual": kern.qualitaet(p, ft, ht, hg, ag, hth, hta)})
            for mk in MAERKTE:
                if q[mk] and mk in pk:
                    kandidaten.append({"id": sid, "Div": s["Div"], "Date": s["Date"], "mk": mk, "grp": GRUPPE[mk],
                                       "pm": p[mk], "pk": pk[mk], "q": q[mk], "win": gewonnen(mk, hg, ag)})
    return spiele_out, kandidaten


# ---------- Parallele Berechnung je Land ----------
_G = {}


def _land_arbeit(job):
    key, land = job
    t0 = time.time()
    try:
        mod = sys.modules.get(key) or importlib.import_module(key)
        codes = list(LAENDER[land])
        sp_land = _G["spiele"][_G["spiele"]["Div"].isin(codes)]
        start = _G["start"][key]
        q_land = _G["q"][(_G["q"]["Div"].isin(codes)) & (_G["q"]["Date"] >= start)]
        s_out, k_out = backtest(mod, sp_land, q_land, start)
        live, diag = [], ""
        try:
            lm = mod.trainiere(sp_land, heute)
            diag = mod.diagnose(lm) if hasattr(mod, "diagnose") else ""
            for _, f in _G["fx"][_G["fx"]["Div"].isin(codes)].iterrows():
                if pd.isna(f["Date"]):
                    continue
                try:
                    ft, ht, xg = berechne(mod, lm, f["HomeTeam"], f["AwayTeam"])
                except Exception:
                    continue
                qq, _ = markt_wkeiten(f)
                live.append({"Div": f["Div"], "Date": f["Date"], "Time": str(f.get("Time", "") or "")[:5],
                             "Home": f["HomeTeam"], "Away": f["AwayTeam"], "p": kern.maerkte(ft, ht), "xg": xg, "q": qq})
        except Exception as e:
            diag = f"Live-Training fehlgeschlagen: {str(e)[:80]}"
        return land, s_out, k_out, live, diag, time.time() - t0
    except Exception as e:
        return land, [], [], [], f"FEHLER: {str(e)[:120]}", time.time() - t0


def _parallel(jobs):
    n = min(len(jobs), os.cpu_count() or 1)
    if n > 1:
        try:
            with multiprocessing.get_context("fork").Pool(n) as pool:
                return list(pool.imap_unordered(_land_arbeit, jobs))
        except Exception as e:
            log("Parallel nicht möglich, rechne nacheinander:", str(e)[:80])
    return [_land_arbeit(j) for j in jobs]


# ---------- Value-Kalibrierung ----------
def wetten_waehlen(K, par):
    if K.empty:
        return K.assign(edge=[], profit=[])
    pf = par["mischung"] * K["pm"] + (1 - par["mischung"]) * K["pk"]
    edge = pf * K["q"] - 1
    m = (edge > par["min_value"]) & (edge < MAX_VALUE) & (K["q"] >= par["quote_min"]) & (K["q"] <= par["quote_max"])
    W = K[m].assign(edge=edge[m])
    W = W.sort_values("edge", ascending=False).drop_duplicates(["id", "grp"])
    return W.assign(profit=np.where(W["win"], W["q"] - 1, -1.0))


def kalibrieren(K):
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


def value_liste(p, q, par):
    pk = {}
    for grp in [["1", "X", "2"], ["Ü2.5", "U2.5"]]:
        if all(q.get(m) for m in grp):
            inv = {m: 1 / q[m] for m in grp}
            t = sum(inv.values())
            pk.update({m: inv[m] / t for m in grp})
    out = []
    for grp in [["1", "X", "2"], ["Ü2.5", "U2.5"]]:
        best = None
        for mk in grp:
            if not (q.get(mk) and mk in pk) or not (par["quote_min"] <= q[mk] <= par["quote_max"]):
                continue
            e = (par["mischung"] * p[mk] + (1 - par["mischung"]) * pk[mk]) * q[mk] - 1
            if par["min_value"] < e < MAX_VALUE and (best is None or e > best[2]):
                best = (mk, q[mk], e)
        if best:
            out.append({"markt": MARKT_NAME[best[0]], "quote": r(best[1], 2), "edge": r(best[2] * 100, 0)})
    return out


def prog_eintrag(p, xg, q, par):
    return {"p1": r(p["1"] * 100, 0), "px": r(p["X"] * 100, 0), "p2": r(p["2"] * 100, 0),
            "ue25": r(p["Ü2.5"] * 100, 0), "bts": r(p["BTS"] * 100, 0), "tipp": p["tipp"], "xg": xg,
            "ht": None if p["HT"] is None else [r(x * 100, 0) for x in p["HT"]],
            "htft": [r(x * 100, 1) for x in p["HTFT"]],
            "_csg": [[r(x * 100, 2) for x in z] for z in p["CSG"]],
            "cs": [[k, r(v * 100, 1)] for k, v in p["CS"]], "value": value_liste(p, q, par)}


# ---------- Hauptprogramm ----------
def main():
    t0 = time.time()
    modelle = modelle_laden()
    if not modelle:
        raise SystemExit("Keine Modell-Dateien (modell_*.py) gefunden.")
    log("Modelle:", ", ".join(m.NAME for m in modelle.values()), f"| Rechenkerne: {os.cpu_count()}")
    log("Lade Historie, Quoten und kommende Spiele ...")
    hist = daten.historie()
    q = quoten_spiele()
    spiele = pd.concat([hist, q[daten.SPALTEN]]).dropna(subset=["Date", "HG", "AG"])
    spiele = spiele.drop_duplicates(subset=["Div", "Date", "Home", "Away"], keep="last").sort_values("Date")
    fx = daten.kommende_spiele()
    log(f"Spiele: {len(spiele):,} | mit Halbzeit: {spiele['HTHG'].notna().sum():,} | mit Quoten: {len(q):,} | kommend: {len(fx)}")

    ergebnis = {"stand": heute.strftime("%Y-%m-%d"), "daten_bis": spiele["Date"].max().strftime("%Y-%m-%d"),
                "anzahl_spiele": int(len(spiele)), "laender": [], "modelle": [], "spiele": []}
    woche_ende = q["Date"].max()
    woche_start = woche_ende - pd.Timedelta(days=6)
    log(f"Letzte Spielwoche: {woche_start.date()} bis {woche_ende.date()}")
    for land, ligen in LAENDER.items():
        ergebnis["laender"].append({"name": land, "flagge": FLAGGEN.get(land, "🏳"),
                                    "ligen": [{"code": c, "name": n} for c, n in ligen.items()]})
    spiel_map = {}

    def eintrag(liga, datum, zeit, heim, gast, ergebnis_txt, quoten, ht=None):
        key = (liga, datum.strftime("%Y-%m-%d"), heim, gast)
        if key not in spiel_map:
            spiel_map[key] = {"liga": liga, "datum": key[1], "zeit": zeit or "", "heim": heim, "gast": gast,
                              "ergebnis": ergebnis_txt, "ergebnis_ht": ht,
                              "quoten": {k: r(v, 2) for k, v in quoten.items()}, "prog": {}}
        return spiel_map[key]

    _G.update(spiele=spiele, q=q, fx=fx, start={})
    groesse = {l: int(spiele["Div"].isin(list(c)).sum()) for l, c in LAENDER.items()}
    reihenfolge = sorted(LAENDER, key=lambda l: -groesse[l])     # grosse Laender zuerst
    qual_roh = {}

    for key, mod in modelle.items():
        tm = time.time()
        wochen = getattr(mod, "BACKTEST_WOCHEN", None)
        start = pd.Timestamp(BACKTEST_START)
        if wochen:
            start = max(start, heute - pd.Timedelta(weeks=wochen))
        _G["start"][key] = start
        log(f"===== {mod.NAME} ===== (Backtest ab {start.date()})")
        res = {x[0]: x for x in _parallel([(key, land) for land in reihenfolge])}
        alle_spiele, alle_kand, live = [], [], []
        for land in LAENDER:
            _, s_out, k_out, lv, diag, dauer = res[land]
            off = len(alle_spiele)
            for s in s_out:
                s["id"] += off
            for k in k_out:
                k["id"] += off
            alle_spiele += s_out
            alle_kand += k_out
            live += lv
            log(f"   {land:<13} {len(s_out):>4} Backtest-Spiele, {len(lv):>3} kommend, {dauer:5.0f} s  {diag}")
        K = pd.DataFrame(alle_kand, columns=K_SPALTEN)

        par_test = kalibrieren(K[K["Date"] < pd.Timestamp(SAISON_START)])
        W_test = wetten_waehlen(K, par_test)
        par_live = kalibrieren(K)

        n_alt = 0
        for sp in alle_spiele:
            if woche_start <= sp["Date"] <= woche_ende:
                e = eintrag(sp["Div"], sp["Date"], sp["Time"], sp["Home"], sp["Away"], f'{sp["HG"]}:{sp["AG"]}', sp["q"], sp["HT"])
                e["prog"][key] = prog_eintrag(sp["p"], sp["xg"], sp["q"], par_live)
                n_alt += 1
        for lv in live:
            e = eintrag(lv["Div"], lv["Date"], lv["Time"], lv["Home"], lv["Away"], None, lv["q"])
            e["prog"][key] = prog_eintrag(lv["p"], lv["xg"], lv["q"], par_live)
        qual_roh[key] = [(s["Date"], s["qual"]) for s in alle_spiele]
        log(f"   Prognosen: {n_alt} letzte Spielwoche, {len(live)} kommend | {time.time() - tm:.0f} s")

        saison_w = W_test[W_test["Date"] >= pd.Timestamp(SAISON_START)]
        ergebnis["modelle"].append({
            "id": key, "name": mod.NAME, "kurz": getattr(mod, "KURZ", mod.NAME.split(" ")[0]),
            "icon": getattr(mod, "ICON", "•"), "farbe1": getattr(mod, "FARBE1", "#2bf5a0"),
            "farbe2": getattr(mod, "FARBE2", "#22e4ff"), "beschreibung": getattr(mod, "BESCHREIBUNG", ""),
            "backtest_ab": start.strftime("%Y-%m-%d"), "saison": kennzahlen(saison_w),
            "vorsaison": kennzahlen(W_test[W_test["Date"] < pd.Timestamp(SAISON_START)]),
            "einstellung_test": par_test, "einstellung_live": par_live})

    # Qualitaet: alle Modelle auf demselben Zeitraum vergleichen
    vergleich_ab = max(_G["start"].values())
    for m in ergebnis["modelle"]:
        rows = [qq for d, qq in qual_roh[m["id"]] if d >= vergleich_ab]
        ks = sorted({k for qq in rows for k in qq})
        m["qualitaet"] = {"ab": vergleich_ab.strftime("%Y-%m-%d"), "spiele": len(rows),
                          **{k: r(np.mean([qq[k] for qq in rows if k in qq]), 4) for k in ks}}

    ids = [m["id"] for m in ergebnis["modelle"]]
    for i in ids:
        fehlt = sum(1 for e in spiel_map.values() if i not in e["prog"])
        if fehlt:
            log(f"ACHTUNG: {fehlt} Spiele ohne Prognose von {i} - diese Spiele werden nicht angezeigt")
    liste = [e for e in spiel_map.values() if all(i in e["prog"] for i in ids)]
    liste.sort(key=lambda e: (e["datum"], e["zeit"], e["liga"]))
    for i, e in enumerate(liste):
        e["id"] = i
    ergebnis["spiele"] = liste

    # Markt-Vergleich: was erwarten die Buchmacher? -> Vorteile je Modell und Markt
    zeilen = {}
    for _, z in q.iterrows():
        zeilen[(z["Div"], z["Date"].strftime("%Y-%m-%d"), z["Home"], z["Away"])] = z
    for _, z in fx.iterrows():
        if not pd.isna(z["Date"]):
            zeilen[(z["Div"], z["Date"].strftime("%Y-%m-%d"), z["HomeTeam"], z["AwayTeam"])] = z
    n_mk = 0
    for e in liste:
        try:
            mk = markt.block(zeilen.get((e["liga"], e["datum"], e["heim"], e["gast"])))
        except Exception:
            mk = None
        if mk:
            n_mk += 1
            e["markt"] = {k: v for k, v in mk.items() if k != "csg"}
        for pr in e["prog"].values():
            pr["vt"] = markt.vorteile(pr, mk)
    log(f"Markt-Vergleich: {n_mk} von {len(liste)} Spielen mit Quoten")
    try:   # Zusatz: Wettquoten-Values (veraendert keine Modelle)
        ergebnis["values"] = values.berechne(liste, fx, ergebnis["modelle"])
        log(f"Values: {len(ergebnis['values'])} gefunden")
    except Exception as ex:
        ergebnis["values"] = []
        log("Values konnten nicht berechnet werden:", ex)
    for e in liste:
        for pr in e["prog"].values():
            pr.pop("_csg", None)

    # Modell Q · Quoten-Analyse (Georgis Berechnung auf den Hoechstquoten)
    ergebnis["modelle"].append(dict(quoten_modell.MODELL_INFO))
    n_q = 0
    for e in liste:
        try:
            qe = quoten_modell.eintrag(zeilen.get((e["liga"], e["datum"], e["heim"], e["gast"])))
        except Exception:
            qe = None
        e["prog"]["quoten"] = qe
        n_q += qe is not None
    log(f"Quoten-Analyse: {n_q} von {len(liste)} Spielen")
    os.makedirs(os.path.join(ORDNER, "docs"), exist_ok=True)
    pfad = os.path.join(ORDNER, "docs", "daten.json")
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, ensure_ascii=False, separators=(",", ":"))
    log(f"Fertig in {time.time() - t0:.0f} s -> {pfad} ({os.path.getsize(pfad) / 1024:.0f} KB, {len(liste)} Spiele)")

    print(f"\n===== MODELL-RANKING (Qualität ab {vergleich_ab.date()}, kleiner = besser) =====")
    for m in sorted([m for m in ergebnis["modelle"] if "qualitaet" in m], key=lambda m: m["qualitaet"].get("rps") or 9):
        qq, s = m["qualitaet"], m["saison"]
        print(f"{m['name']}")
        print(f"   RPS {qq.get('rps')} | LogLoss 1X2 {qq.get('logloss_1x2')} | Ü/U {qq.get('logloss_ou')} | "
              f"BTS {qq.get('logloss_bts')} | HZ {qq.get('logloss_ht')} | Correct Score {qq.get('nll_cs')} ({qq['spiele']} Spiele)")
        print(f"   Value-Wetten Saison: {s['wetten']} | ROI {s['roi']} % | Treffer {s['treffer']} %\n")


if __name__ == "__main__":
    main()
