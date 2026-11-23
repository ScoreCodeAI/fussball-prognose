"""master_gewichte.py – Rueckblick-Bilanz fuer den MASTER auf der Seite.

Fuer jede Quelle (Modell) und jeden Markt wird gezaehlt: wie oft hat sie getippt (n) und wie oft lag sie richtig (h).
Die Regeln (welcher Tipp, wann richtig) sind GENAU dieselben wie im MASTER auf der Seite (index.html: msVec/msWin).
Grundlage sind nur Vorhersagen, die das Modell OHNE Kenntnis des Spiels gemacht hat (Rueckblick-Test).
Ergebnis landet in daten.json unter "master_gew": {quelle_id: {markt: [n, h]}}.
Aendert keine Modelle und keine Prognosen.
"""
import math

MAERKTE = ["x1", "x", "ou", "h2", "bts", "bts2", "w1", "w2", "ht", "ht2a", "h1x", "h2x", "cs"]
CSL = [f"{i}:{j}" for i in range(6) for j in range(6)]


def _sg(h, a):
    return 0 if h > a else (1 if h == a else 2)


def _ok3(v):
    return v is not None and len(v) == 3 and all(x is not None and x == x for x in v)


def vec(mk, p):
    """Wahrscheinlichkeiten (in %) der Optionen eines Marktes, wie msVec auf der Seite."""
    g = p.get
    x3 = [g("p1"), g("px"), g("p2")]
    dc = lambda v: [v[0] + v[1], v[1] + v[2], v[0] + v[2]]
    if mk == "x1":
        return x3 if _ok3(x3) else None
    if mk == "x":
        return dc(x3) if _ok3(x3) else None
    if mk == "ou":
        return None if g("ue25") is None else [g("ue25"), 100 - g("ue25")]
    if mk == "h2":
        return None if g("h2o05") is None else [g("h2o05"), 100 - g("h2o05")]
    if mk == "bts":
        return None if g("bts") is None else [g("bts"), 100 - g("bts")]
    if mk == "bts2":
        xg = g("xg")
        if g("h2o05") is None or not xg:
            return None
        t = -math.log(max(1e-6, 1 - g("h2o05") / 100))
        sh = (xg[0] + xg[1]) or 1
        v = (1 - math.exp(-t * xg[0] / sh)) * (1 - math.exp(-t * xg[1] / sh)) * 100
        return [v, 100 - v]
    if mk == "cs":
        xg = g("xg")
        if not xg or xg[0] is None:
            return None
        P = lambda l, k: math.exp(-l) * l ** k / math.factorial(k)
        return [P(xg[0], int(c[0])) * P(xg[1], int(c[2])) * 100 for c in CSL]
    ht, ht2 = g("ht"), g("ht2")
    if mk == "w1":
        return [ht[0], ht[2]] if _ok3(ht) else None
    if mk == "w2":
        return [ht2[0], ht2[2]] if _ok3(ht2) else None
    if mk == "ht":
        return dc(ht) if _ok3(ht) else None
    if mk == "ht2a":
        return dc(ht2) if _ok3(ht2) else None
    if mk == "h1x":
        return ht if _ok3(ht) else None
    if mk == "h2x":
        return ht2 if _ok3(ht2) else None
    return None


def win(mk, k, r):
    """Treffer ja/nein (None = nicht entscheidbar), wie msWin auf der Seite. r = (h, a, hh, ha)."""
    h, a, hh, ha = r
    e = _sg(h, a)
    if mk == "cs":
        return CSL[k] == f"{h}:{a}"
    if mk == "x1":
        return e == k
    if mk == "x":
        return e != (2 if k == 0 else 0 if k == 1 else 1)
    if mk == "ou":
        return (h + a > 2) == (k == 0)
    if mk == "bts":
        return (h > 0 and a > 0) == (k == 0)
    if hh is None or ha is None:
        return None
    h1, h2, a2 = _sg(hh, ha), h - hh, a - ha
    e2 = _sg(h2, a2)
    if mk == "h2":
        return (h2 + a2 > 0) == (k == 0)
    if mk == "bts2":
        return (h2 > 0 and a2 > 0) == (k == 0)
    if mk == "w1":
        return h1 == (2 if k else 0)
    if mk == "w2":
        return e2 == (2 if k else 0)
    if mk == "ht":
        return h1 != (2 if k == 0 else 0 if k == 1 else 1)
    if mk == "ht2a":
        return e2 != (2 if k == 0 else 0 if k == 1 else 1)
    if mk == "h1x":
        return h1 == k
    if mk == "h2x":
        return e2 == k
    return None


def zaehlen(stats, quelle, p, r):
    for mk in MAERKTE:
        v = vec(mk, p)
        if not v:
            continue
        if mk in ("bts", "bts2"):          # Beide treffen: nur JA-Tipps zaehlen (wie auf der Seite)
            if v[0] < 50:
                continue
            k = 0
        else:
            k = v.index(max(v))
        w = win(mk, k, r)
        if w is None:
            continue
        e = stats.setdefault(quelle, {}).setdefault(mk, [0, 0])
        e[0] += 1
        e[1] += int(bool(w))


def _bt_zu_seite(sp):
    """Backtest-Zeile der Engine (kern.maerkte) in das Format der Seite umrechnen."""
    q = sp.get("p") or {}
    f = lambda k: None if q.get(k) is None else float(q[k]) * 100
    p = {"p1": f("1"), "px": f("X"), "p2": f("2"), "ue25": f("Ü2.5"), "bts": f("BTS"), "xg": sp.get("xg")}
    if q.get("HT"):
        p["ht"] = [float(x) * 100 for x in q["HT"]]
    if q.get("HT2"):
        p["ht2"] = [float(x) * 100 for x in q["HT2"]]
    cs2 = q.get("CS2")
    if cs2 and "0:0" in cs2:
        p["h2o05"] = (1 - cs2["0:0"]) * 100          # wie auf der Seite (engine._tt_kb)
    return p


def _ht(sp):
    try:
        a, b = str(sp.get("HT") or "").split(":")
        return int(a), int(b)
    except Exception:
        return None, None


def aus_bt(bt, log=print, ab="2025-07-01"):
    """Rueckblick der Engine-Modelle (Woche fuer Woche ohne Zukunftswissen trainiert)."""
    import pandas as pd
    stats = {}
    for mid, rows in (bt or {}).items():
        for sp in rows or []:
            if sp.get("HG") is None or sp.get("AG") is None:
                continue
            try:
                if pd.Timestamp(sp["Date"]).strftime("%Y-%m-%d") < ab:
                    continue
                hh, ha = _ht(sp)
                zaehlen(stats, mid, _bt_zu_seite(sp), (int(sp["HG"]), int(sp["AG"]), hh, ha))
            except Exception:
                continue
    log("Master-Gewichte (Engine-Rueckblick):", ", ".join(f"{m} {s.get('x1', [0])[0]}" for m, s in stats.items()))
    return stats


def eintragen(d, stats):
    g = d.setdefault("master_gew", {})
    for q, s in stats.items():
        g[q] = s
