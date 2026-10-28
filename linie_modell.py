# ===== MODELL L · LINIE (Quotenbewegung) – fuer ALLE Modelle =====
# Frage je Modell: Erkennt es frueh, in welche Richtung sich die Pinnacle-Quote bis zum Anpfiff bewegt?
#   Vorteil = Modell-Chance / faire Pinnacle-Chance (Erfassungs-Quote) - 1
#   Rueckblick = die Backtest-Prognosen des Modells (nur Daten VOR dem Spiel):
#     CLV = Erfassungs-Quote x faire Schluss-Chance - 1   (> 0 = man hatte frueh die bessere Quote)
# Selbstpruefung bei jedem Update: Ein Modell bekommt Tipps nur in den Maerkten (1X2, OVER/UNDER),
#   in denen sein Rueckblick ein positives CLV mit genug Faellen hat. Sonst: kein Tipp.
# XGBoost hat keinen eigenen Rueckblick (es wird einmal trainiert) -> dafuer zaehlt sein Ergebnis
#   auf der letzten Spielwoche nicht; es wird nur geprueft, wenn genug Faelle vorliegen.
import numpy as np

import markt

INFO = {"id": "linie", "typ": "linie", "name": "Linie · Quotenbewegung", "kurz": "Linie", "icon": "📉",
        "farbe1": "#7cf5ff", "farbe2": "#a855f7",
        "beschreibung": "Tipps nur von Modellen, die im Rückblick die Quotenbewegung vorhergesehen haben"}

SCHWELLE = 0.10
MAX_QUOTE = 6.0
MIN_FAELLE = 120
MAERKTE = {"x": (["H", "D", "A"], "PS", "PSC", ["1", "X", "2"]),
           "ou": ([">2.5", "<2.5"], "P", "PC", ["OVER", "UNDER"])}


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if not np.isnan(v) and v > 1.0 else None


def _quoten(row, pre, suf):
    q = [_f(row.get(pre + s)) for s in suf]
    return q if all(q) else None


def _wk_bt(p, m):
    """Backtest-Prognose (Anteile) -> Liste je Ausgang"""
    return [p["1"], p["X"], p["2"]] if m == "x" else [p["Ü2.5"], p["U2.5"]]


def _wk_prog(pr, m):
    """Website-Prognose (Prozent) -> Anteile je Ausgang"""
    if m == "x":
        return [pr["p1"] / 100, pr["px"] / 100, pr["p2"] / 100]
    return [pr["ue25"] / 100, 1 - pr["ue25"] / 100]


def berechne(bt, liste, zeilen, modelle, log=print):
    """bt: {modell_id: [Backtest-Spiele mit Div, Date, Home, Away, HG, AG, p]}, modelle: [{id, kurz}]"""
    stand = {}
    for mod in modelle:
        mid = mod["id"]
        stand[mid] = {"kurz": mod["kurz"]}
        for m, (suf, po, pc, lab) in MAERKTE.items():
            faelle = []
            for sp in bt.get(mid, []):
                row = zeilen.get((sp["Div"], sp["Date"].strftime("%Y-%m-%d"), sp["Home"], sp["Away"]))
                if row is None:
                    continue
                qo, qc = _quoten(row, po, suf), _quoten(row, pc, suf)
                if not qo or not qc:
                    continue
                fo, fc = markt.marge_entfernen(qo), markt.marge_entfernen(qc)
                hg, ag = sp["HG"], sp["AG"]
                res = (0 if hg > ag else 1 if hg == ag else 2) if m == "x" else (0 if hg + ag > 2 else 1)
                for i, p in enumerate(_wk_bt(sp["p"], m)):
                    if p / fo[i] - 1 > SCHWELLE and qo[i] < MAX_QUOTE:
                        faelle.append((qo[i] * fc[i] - 1, fc[i] > fo[i], qo[i] - 1 if res == i else -1.0))
            n = len(faelle)
            a = np.array(faelle) if n else np.zeros((0, 3))
            clv = float(a[:, 0].mean()) if n else None
            stand[mid][m] = {"n": n, "clv": None if clv is None else round(clv * 100, 2),
                             "faellt": round(float(a[:, 1].mean()) * 100, 1) if n else None,
                             "roi": round(float(a[:, 2].mean()) * 100, 1) if n else None,
                             "aktiv": bool(n >= MIN_FAELLE and clv > 0)}
        log(f"Linie {mod['kurz']:8}: 1X2 {stand[mid]['x']['n']} Fälle CLV {stand[mid]['x']['clv']}% "
            f"{'AKTIV' if stand[mid]['x']['aktiv'] else '-'} | O/U {stand[mid]['ou']['n']} Fälle CLV {stand[mid]['ou']['clv']}% "
            f"{'AKTIV' if stand[mid]['ou']['aktiv'] else '-'}")

    n_sp = 0
    for e in liste:
        row = zeilen.get((e["liga"], e["datum"], e["heim"], e["gast"]))
        tipps = []
        if row is not None:
            for mod in modelle:
                mid, pr = mod["id"], e["prog"].get(mod["id"])
                if not pr:
                    continue
                for m, (suf, po, pc, lab) in MAERKTE.items():
                    if not stand[mid][m]["aktiv"]:
                        continue
                    qo = _quoten(row, po, suf)
                    if not qo:
                        continue
                    fo = markt.marge_entfernen(qo)
                    for i, p in enumerate(_wk_prog(pr, m)):
                        v = p / fo[i] - 1
                        if v > SCHWELLE and qo[i] < MAX_QUOTE:
                            tipps.append({"modell": mod["kurz"], "m": m, "tipp": lab[i], "v": round(v * 100, 1)})
        tipps.sort(key=lambda t: -t["v"])
        e["prog"]["linie"] = {"tipps": tipps}
        n_sp += bool(tipps)
    log(f"Linie: {n_sp} Spiele mit Tipp")
    return stand
