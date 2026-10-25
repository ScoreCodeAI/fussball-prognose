# ===== ★ BESTE + P · PINNACLE =====
# 1) Backtest-Vergleich: fuer jeden Markt wird gemessen, ob ein Modell besser vorhersagt als
#    Pinnacle (Log-Loss auf denselben Spielen; kleiner = besser).
# 2) Modell "P · Pinnacle": Prognosen nur aus den Pinnacle-Quoten (ohne Marge).
# 3) Modell "★ Beste": je Markt die Prognose des Modells, das dort im Backtest am besten war -
#    oder Pinnacle selbst, wenn kein Modell den Markt schlaegt.
# Veraendert KEIN bestehendes Modell.
import numpy as np

import markt

MAERKTE = [("1x2", "Spielausgang"), ("ou", "OVER / UNDER 2,5"), ("bts", "Beide treffen"),
           ("ht", "1. Halbzeit"), ("ht2", "2. Halbzeit")]
MIN_SPIELE = 80
EPS = 1e-9

INFO_P = {"id": "pinnacle", "typ": "pinnacle", "name": "Pinnacle · Markt", "kurz": "Pinnacle", "icon": "P",
          "farbe1": "#ff4d7d", "farbe2": "#ffb02b",
          "beschreibung": "Faire Wahrscheinlichkeiten aus den Pinnacle-Quoten (ohne Marge)"}
INFO_B = {"id": "beste", "typ": "beste", "name": "Beste · je Markt", "kurz": "Beste", "icon": "★",
          "farbe1": "#ffd84d", "farbe2": "#2bf5a0",
          "beschreibung": "Je Markt das Modell, das im Backtest am besten war (oder Pinnacle)"}


def _sg(a, b):
    return 0 if a > b else 1 if a == b else 2


def _ergebnisse(hg, ag, ht):
    out = {"1x2": _sg(hg, ag), "ou": 0 if hg + ag > 2 else 1, "bts": 0 if (hg > 0 and ag > 0) else 1}
    if ht:
        try:
            h1, a1 = (int(x) for x in ht.split(":"))
            out["ht"] = _sg(h1, a1)
            out["ht2"] = _sg(hg - h1, ag - a1)
        except ValueError:
            pass
    return out


def _wk_modell(p):
    """kern.maerkte-Dict (Anteile) -> {markt: [Wahrscheinlichkeiten]}"""
    w = {"1x2": [p["1"], p["X"], p["2"]], "ou": [p["Ü2.5"], p["U2.5"]], "bts": [p["BTS"], 1 - p["BTS"]]}
    if p.get("HT"):
        w["ht"] = p["HT"]
    if p.get("HT2"):
        w["ht2"] = p["HT2"]
    return w


def _wk_markt(mk):
    return {k: [x / 100 for x in mk[k]] for k, _ in MAERKTE if mk.get(k)}


def vergleich(bt, zeilen, kurz):
    """bt: {modell_id: [Backtest-Spiele]}, zeilen: {(Div, Datum, Heim, Gast): Quoten-Zeile}.
    -> {"maerkte": {markt: {"wahl": id, "modelle": {id: {"diff": %, "n": n}}}}}"""
    cache = {}

    def mk_von(key):
        if key not in cache:
            try:
                cache[key] = markt.block(zeilen.get(key))
            except Exception:
                cache[key] = None
        return cache[key]

    summen = {k: {} for k, _ in MAERKTE}
    for mid, spiele in bt.items():
        for s in spiele:
            key = (s["Div"], s["Date"].strftime("%Y-%m-%d"), s["Home"], s["Away"])
            mk = mk_von(key)
            if not mk:
                continue
            erg = _ergebnisse(s["HG"], s["AG"], s.get("HT"))
            wm, wp = _wk_modell(s["p"]), _wk_markt(mk)
            for k, _ in MAERKTE:
                if k in erg and k in wm and k in wp:
                    o = erg[k]
                    lm = -np.log(max(wm[k][o], EPS))
                    lp = -np.log(max(wp[k][o], EPS))
                    a = summen[k].setdefault(mid, [0.0, 0.0, 0])
                    a[0] += lm; a[1] += lp; a[2] += 1
    out = {}
    for k, name in MAERKTE:
        mods = {}
        for mid, (lm, lp, n) in summen[k].items():
            if n >= MIN_SPIELE:
                mods[mid] = {"diff": round((lm / lp - 1) * 100, 2), "n": n, "kurz": kurz.get(mid, mid)}
        wahl = "pinnacle"
        if mods:
            bester = min(mods, key=lambda m: mods[m]["diff"])
            if mods[bester]["diff"] < 0:
                wahl = bester
        out[k] = {"name": name, "wahl": wahl, "modelle": mods}
    return {"maerkte": out, "spiele_markt": sum(1 for v in cache.values() if v)}


FELDER = {"1x2": ["p1", "px", "p2"], "ou": ["ue25"], "bts": ["bts"], "ht": ["ht"], "ht2": ["ht2"]}
VALUE_MARKT = {"Heimsieg": "1x2", "Remis": "1x2", "Auswärtssieg": "1x2", "OVER 2,5": "ou", "UNDER 2,5": "ou"}


def beste_prog(e, wahl, kurz):
    """Setzt die ★-Prognose eines Spiels aus den gewaehlten Quellen zusammen."""
    out, quelle = {}, {}
    for k, felder in FELDER.items():
        src = wahl.get(k, "pinnacle")
        pr = e["prog"].get(src) or e["prog"].get("pinnacle")
        if not pr:
            return None
        for f in felder:
            out[f] = pr.get(f)
        quelle[k] = kurz.get(src, "Pinnacle") if src in e["prog"] else "Pinnacle"
    basis = e["prog"].get(wahl.get("1x2", "pinnacle")) or e["prog"].get("pinnacle")
    out["tipp"], out["xg"] = basis.get("tipp"), basis.get("xg")
    out["value"] = []
    for k in ("1x2", "ou"):
        pr = e["prog"].get(wahl.get(k, "pinnacle")) or {}
        out["value"] += [v for v in pr.get("value", []) if VALUE_MARKT.get(v["markt"]) == k]
    out["quelle"] = quelle
    return out
