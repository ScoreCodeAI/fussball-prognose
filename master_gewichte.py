"""master_gewichte.py – Rueckblick-Bilanz fuer den MASTER auf der Seite.

Fuer jede Quelle (Modell) und jeden Markt wird gezaehlt: wie oft hat sie getippt (n) und wie oft lag sie richtig (h).
Die Regeln (welcher Tipp, wann richtig) sind GENAU dieselben wie im MASTER auf der Seite (index.html: msVec/msWin).
Grundlage sind nur Vorhersagen, die das Modell OHNE Kenntnis des Spiels gemacht hat (Rueckblick-Test).
Ergebnis landet in daten.json unter "master_gew": {quelle_id: {markt: [n, h]}}.
Aendert keine Modelle und keine Prognosen.
"""
import math

KAL_MAERKTE = ("x1", "x", "ou")   # Maerkte mit echten Buchmacher-Quoten
MAERKTE = ["x1", "x", "ou", "h2", "bts", "bts2", "w1", "w2", "ht", "ht2a", "h1x", "h2x", "fg", "ng2"]


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
    if mk in ("fg", "ng2"):              # wer trifft zuerst (Vollzeit / 2. HZ), ohne Option "kein Tor"
        xg = g("xg")
        if not xg or xg[0] is None:
            return None
        h, a = float(xg[0]), float(xg[1])
        if mk == "ng2":
            if g("h2o05") is None:
                return None
            t = -math.log(max(1e-6, 1 - g("h2o05") / 100))
            sh = (h + a) or 1
            h, a = t * h / sh, t * a / sh
        lam = h + a
        if lam <= 0:
            return None
        gg = 1 - math.exp(-lam)
        return [h / lam * gg * 100, a / lam * gg * 100]
    ht, ht2 = g("ht"), g("ht2")
    if mk == "w1":                      # 1. HZ 1 X 2
        return ht if _ok3(ht) else None
    if mk == "w2":                      # 2. HZ 1 X 2
        return ht2 if _ok3(ht2) else None
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
    if mk == "fg":
        want = 2 if k else 0
        if h + a == 0:
            return False
        if a == 0:
            return want == 0
        if h == 0:
            return want == 2
        if hh is None or ha is None:
            return None
        if hh + ha > 0:
            return (want == 0) if ha == 0 else (want == 2) if hh == 0 else None
        h2, a2 = h - hh, a - ha
        return (want == 0) if a2 == 0 else (want == 2) if h2 == 0 else None
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
        return h1 == k
    if mk == "w2":
        return e2 == k
    if mk == "ht":
        return h1 != (2 if k == 0 else 0 if k == 1 else 1)
    if mk == "ht2a":
        return e2 != (2 if k == 0 else 0 if k == 1 else 1)
    if mk == "ng2":
        want = 2 if k else 0
        if h2 + a2 == 0:
            return False
        return (want == 0) if a2 == 0 else (want == 2) if h2 == 0 else None
    if mk == "h1x":
        return h1 == k
    if mk == "h2x":
        return e2 == k
    return None


# ---------------- SELBST-EICHUNG (immer aktiv auf der Seite) ----------------
# Fuer jede Quelle und jedes Grundfeld, aus dem die 12 Maerkte gerechnet werden, wird gezaehlt:
#   _eich:   "feld|ausgang|stufe" -> [n, treffer]     wie oft trat ein Ausgang ein, wenn das Modell ~X % sagte (10er-Stufen)
#   _muster: "feld|ausgang|liga"  -> [n, summe %, treffer]   systematische Abweichung je Liga (Fehler-Muster)
EICH_FELDER = ("x", "ue25", "bts", "ht", "ht2", "h1o05", "h2o05")


def _eich_ausgaenge(feld, p, r):
    """[(ausgang, wahrscheinlichkeit %, eingetreten?)] fuer ein Grundfeld; None wenn nicht pruefbar."""
    h, a, hh, ha = r
    g = p.get
    if feld == "x":
        v = [g("p1"), g("px"), g("p2")]
        if not _ok3(v):
            return None
        e = _sg(h, a)
        return [(k, v[k], e == k) for k in range(3)]
    if feld in ("ue25", "bts"):
        v = g(feld)
        if v is None:
            return None
        hit = (h + a > 2) if feld == "ue25" else (h > 0 and a > 0)
        return [(0, v, hit)]
    if hh is None or ha is None:
        return None
    if feld in ("ht", "ht2"):
        v = g(feld)
        if not _ok3(v):
            return None
        e = _sg(hh, ha) if feld == "ht" else _sg(h - hh, a - ha)
        return [(k, v[k], e == k) for k in range(3)]
    if feld in ("h1o05", "h2o05"):
        v = g(feld)
        if v is None:
            return None
        hit = (hh + ha > 0) if feld == "h1o05" else (h - hh + a - ha > 0)
        return [(0, v, hit)]
    return None


def eichen_zaehlen(stats, quelle, p, r, liga=None):
    E = stats.setdefault("_eich", {}).setdefault(quelle, {})
    Mu = stats.setdefault("_muster", {}).setdefault(quelle, {})
    for feld in EICH_FELDER:
        try:
            aus = _eich_ausgaenge(feld, p, r)
        except Exception:
            aus = None
        for k, v, hit in aus or []:
            if v is None or v != v:
                continue
            st = min(9, max(0, int(float(v) // 10)))
            e = E.setdefault(f"{feld}|{k}|{st}", [0, 0])
            e[0] += 1
            e[1] += int(bool(hit))
            if liga:
                m = Mu.setdefault(f"{feld}|{k}|{liga}", [0, 0.0, 0])
                m[0] += 1
                m[1] = round(m[1] + float(v), 2)
                m[2] += int(bool(hit))


def zaehlen(stats, quelle, p, r, liga=None):
    eichen_zaehlen(stats, quelle, p, r, liga)
    for mk in MAERKTE:
        v = vec(mk, p)
        if not v:
            continue
        k = v.index(max(v))                # auch Beide treffen: JA oder NEIN (wie auf der Seite)
        w = win(mk, k, r)
        if w is None:
            continue
        e = stats.setdefault(quelle, {}).setdefault(mk, [0, 0])
        e[0] += 1
        e[1] += int(bool(w))
        if mk in KAL_MAERKTE:              # Kalibrierung: "Modell sagt ~X %" -> wie oft traf es wirklich
            kb = stats.setdefault("_kal", {}).setdefault(f"{mk}|{int(v[k] // 5) * 5}", [0, 0])
            kb[0] += 1
            kb[1] += int(bool(w))


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
                zaehlen(stats, mid, _bt_zu_seite(sp), (int(sp["HG"]), int(sp["AG"]), hh, ha), sp.get("Div"))
            except Exception:
                continue
    log("Master-Gewichte (Engine-Rueckblick):", ", ".join(f"{m} {s.get('x1', [0])[0]}" for m, s in stats.items()))
    return stats


def eintragen(d, stats):
    g = d.setdefault("master_gew", {})
    for q, s in stats.items():
        if q in ("_eich", "_muster"):      # je Quelle uebernehmen (jede Pipeline hat ihre eigenen Quellen)
            g.setdefault(q, {}).update(s)
            continue
        if q == "_kal":                    # Kalibrierung aufsummieren statt ueberschreiben
            k = g.setdefault("_kal", {})
            for b, (n, h) in s.items():
                e = k.setdefault(b, [0, 0])
                e[0] += n
                e[1] += h
            continue
        g[q] = s
