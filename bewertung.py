# ===== BEWERTUNG · Wie zuverlaessig waren die Kombi-Tipps wirklich? =====
# Spielt die Kombi-Regeln der Webseite auf den Backtest-Prognosen nach (nur Daten VOR dem Spiel):
#   Quellen:   MAX (hoechste % je Ausgang ueber alle Modelle) und Elo (nur Elo)
#   Maerkte:   1X2, OVER/UNDER 2,5, 1. Halbzeit 1X2 (bewusst nur 3 Maerkte -> wenig Zufallstreffer)
#   Varianten: A = Uebereinstimmung ✓ · B = ✓ + Kontra 0 · C = ✓ + Konsens / hoechste %
# Zeitraum wird geteilt: aelterer Teil = "Lern", juengerer Teil = "Pruef" (nur Pruef zaehlt in der Tabelle).
# "Gepruefte Trefferquote" je %-Stufe: aus allen Backtest-Tipps (fuer die heutigen Tipps sind das alles fruehere Spiele).
import math
from collections import defaultdict

import pandas as pd

MAERKTE = {"x": ("1", "X", "2"), "ou25": ("O", "U"), "ht": ("1", "X", "2")}
STUFEN = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 101)]
MIN_N = 100
GROESSEN = (3, 5, 10)


def _r2(x):                      # wie Math.round(x*100)/100 in JavaScript
    return math.floor(x * 100 + 0.5) / 100


def _ue(q):
    s0 = sum(1 / x for x in q)
    pay = _r2(100 / s0)
    fp = [_r2(100 / x) for x in q]
    rp = [_r2(1 / x * 100 / s0) for x in q]
    ue = [_r2(100 - abs(a - b) / a * 100) for a, b in zip(fp, rp)]
    return pay, ue


def _vals(p, mk):
    if not p:
        return None
    if mk == "x":
        v = [p.get("1"), p.get("X"), p.get("2")]
    elif mk == "ou25":
        v = [p.get("Ü2.5"), p.get("U2.5")]
    else:
        v = p.get("HT")
    if not v or any(x is None or x <= 0 for x in v):
        return None
    return [_r2(float(x) * 100) for x in v]


def _sg(h, a):
    return "1" if h > a else "X" if h == a else "2"


def _echt(sp, mk):
    hg, ag = int(sp["HG"]), int(sp["AG"])
    if mk == "x":
        return _sg(hg, ag)
    if mk == "ou25":
        return "O" if hg + ag > 2 else "U"
    if not sp.get("HT"):
        return None
    h1, a1 = map(int, str(sp["HT"]).split(":"))
    return _sg(h1, a1)


def _max_tipp(V, mk, px_avg):
    L = MAERKTE[mk]
    mx = [max(v[i] for v in V) for i in range(len(L))]
    s = sum(mx)
    norm = [x / s * 100 for x in mx]
    q = [_r2(100 / x) for x in mx]
    pay, ue = _ue(q)
    tk = [i for i in range(len(L)) if ue[i] == pay]
    if not tk or len(tk) == len(L):
        return None
    avg = [sum(v[i] for v in V) / len(V) for i in range(len(L))]
    kons = avg.index(max(avg))
    i = sorted(tk, key=lambda j: -norm[j])[0]
    kg = len(V) > 1 and max(v[i] for v in V) - min(v[i] for v in V) > 15
    if mk != "ou25" and len(tk) == 1 and L[tk[0]] != "X" and px_avg is not None and px_avg >= 28:
        kg = True
    return {"tk": [L[i] for i in tk], "pr": sum(norm[i] for i in tk), "kontra": kg, "kons": kons in tk}


def _einzel_tipp(v, mk):
    L = MAERKTE[mk]
    q = [_r2(100 / x) for x in v]
    pay, ue = _ue(q)
    tk = [i for i in range(len(L)) if ue[i] == pay]
    if not tk or len(tk) == len(L):
        return None
    ti = v.index(max(v))
    kg = len(L) == 3 and len(tk) == 1 and L[tk[0]] != "X" and v[1] >= 28
    return {"tk": [L[i] for i in tk], "pr": sum(v[i] for i in tk), "kontra": kg, "kons": ti in tk}


def _stufe(pr):
    for a, b in STUFEN:
        if a <= pr < b:
            return f"{a}"
    return "80"


def berechne(bt, log=print):
    ids = [m for m in bt if bt[m]]
    if not ids:
        return None
    spiele = defaultdict(dict)
    for m in ids:
        for sp in bt[m]:
            k = (sp["Div"], pd.Timestamp(sp["Date"]).strftime("%Y-%m-%d"), sp["Home"], sp["Away"])
            spiele[k][m] = sp
    tipps = []          # (datum, quelle, variante, markt, pr, treffer)
    for k, ms in spiele.items():
        sp0 = next(iter(ms.values()))
        if sp0.get("HG") is None:
            continue
        for mk in MAERKTE:
            echt = _echt(sp0, mk)
            if echt is None:
                continue
            V = [v for v in (_vals(sp["p"], mk) for sp in ms.values()) if v]
            if len(V) >= 3:
                pxs = [x for x in (_vals(sp["p"], "x") for sp in ms.values()) if x]
                t = _max_tipp(V, mk, sum(x[1] for x in pxs) / len(pxs) if pxs else None)
                if t:
                    tipps.append((k[1], "MAX", t, mk, echt in t["tk"]))
            if mk != "ht" and "modell_elo" in ms:
                v = _vals(ms["modell_elo"]["p"], mk)
                t = _einzel_tipp(v, mk) if v else None
                if t:
                    tipps.append((k[1], "Elo", t, mk, echt in t["tk"]))
    if not tipps:
        return None
    daten = sorted({t[0] for t in tipps})
    grenze = daten[int(len(daten) * 0.6)]
    out = {"lern_bis": grenze, "von": daten[0], "bis": daten[-1], "gruppen": {}, "stufen": {}}
    for q in ("MAX", "Elo"):
        for mk in MAERKTE:
            if q == "Elo" and mk == "ht":
                continue
            for var in ("a", "b", "c"):
                sel = [t for t in tipps if t[1] == q and t[3] == mk and
                       (var == "a" or (var == "b" and not t[2]["kontra"]) or (var == "c" and t[2]["kons"]))]
                pruef = [t for t in sel if t[0] >= grenze]
                lern = [t for t in sel if t[0] < grenze]
                g = {"n": len(pruef), "n_lern": len(lern)}
                if len(pruef) >= MIN_N:
                    g["treffer"] = round(sum(t[4] for t in pruef) / len(pruef) * 100, 1)
                    g["treffer_lern"] = round(sum(t[4] for t in lern) / len(lern) * 100, 1) if len(lern) >= MIN_N else None
                    tage = defaultdict(list)
                    for t in pruef:
                        tage[t[0]].append(t)
                    for n in GROESSEN:
                        sch = [sorted(v, key=lambda t: -t[2]["pr"])[:n] for v in tage.values() if len(v) >= n]
                        g[f"k{n}"] = round(sum(all(t[4] for t in s) for s in sch) / len(sch) * 100, 1) if len(sch) >= 10 else None
                        g[f"k{n}_n"] = len(sch)
                    kal = []
                    for a, b in STUFEN:
                        zl = [t for t in lern if a <= t[2]["pr"] < b]
                        zp = [t for t in pruef if a <= t[2]["pr"] < b]
                        kal.append({"stufe": f"{a}–{b - 1 if b < 101 else 100} %",
                                    "lern": round(sum(t[4] for t in zl) / len(zl) * 100, 1) if len(zl) >= 30 else None,
                                    "pruef": round(sum(t[4] for t in zp) / len(zp) * 100, 1) if len(zp) >= 30 else None,
                                    "n": len(zp)})
                    g["kal"] = kal
                out["gruppen"][f"{q}|{mk}|{var}"] = g
                st = {}
                for a, b in STUFEN:
                    z = [t for t in sel if a <= t[2]["pr"] < b]
                    if len(z) >= 30:
                        st[str(a)] = round(sum(t[4] for t in z) / len(z) * 100, 1)
                out["stufen"][f"{q}|{mk}|{var}"] = st
    log(f"Bewertung: {len(tipps)} Backtest-Tipps ({daten[0]} bis {daten[-1]}), Prüfzeitraum ab {grenze}")
    return out
