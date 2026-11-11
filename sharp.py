# ===== SHARP-SIGNAL · Wo bewegt sich das Profi-Geld? (eigenstaendig, fliesst in nichts anderes ein) =====
# Zwei Signale, jeweils fuer 1X2 und OVER/UNDER 2,5, Marge immer nach Shin entfernt:
#  S1 · Pinnacle gegen Markt: faire Chance bei Pinnacle (Profi-Buchmacher) minus faire Chance im Markt-Durchschnitt.
#       Pinnacle hoeher = Profis sehen den Ausgang wahrscheinlicher als die "weichen" Buchmacher.
#  S2 · Bewegung: faire Chance bei Pinnacle frueh (Tage vor dem Spiel, football-data) -> jetzt/Schluss.
#       Steigt die Chance deutlich (Quote faellt), floss Geld auf diesen Ausgang.
# Ehrlicher Test auf allen historischen Spielen (football-data, feste Schwellen, nichts angepasst):
#  Trefferquote des Signals gegen die Erwartung und Rendite (ROI) zur Hoechstquote.
import numpy as np
import pandas as pd

S1_MIN = 1.5      # Prozentpunkte Abstand Pinnacle zu Markt
S2_MIN = 2.0      # Prozentpunkte Bewegung frueh -> spaet

M = {"x": (("H", "D", "A"), ("1", "X", "2")), "ou": ((">2.5", "<2.5"), ("O", "U"))}


def shin(qs):
    from scipy.optimize import brentq
    q = np.array(qs, dtype=float)
    if not np.all(np.isfinite(q)) or np.any(q <= 1):
        return None
    pi = 1 / q
    S = pi.sum()
    if S < 1:
        return pi / S
    f = lambda z: (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / S) - z).sum() / (2 * (1 - z)) - 1
    try:
        z = brentq(f, 0, 0.4)
        p = (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / S) - z) / (2 * (1 - z))
    except Exception:
        p = pi / S
    return p / p.sum()


def _get(r, cols):
    try:
        v = [float(r.get(c)) for c in cols]
        return v if all(np.isfinite(x) and x > 1 for x in v) else None
    except Exception:
        return None


def _pre(mk, art):
    s = M[mk][0]
    if mk == "x":
        return {"pin": ["PS" + x for x in s], "pinc": ["PSC" + x for x in s], "avg": ["Avg" + x for x in s],
                "max": ["Max" + x for x in s], "maxc": ["MaxC" + x for x in s]}[art]
    return {"pin": ["P" + x for x in s], "pinc": ["PC" + x for x in s], "avg": ["Avg" + x for x in s],
            "max": ["Max" + x for x in s], "maxc": ["MaxC" + x for x in s]}[art]


def _ausgang(r, mk):
    hg, ag = int(r["HG"]), int(r["AG"])
    if mk == "x":
        return 0 if hg > ag else 1 if hg == ag else 2
    return 0 if hg + ag > 2 else 1


def test(q, log=print):
    out = {}
    for mk in M:
        z = {"s1": [], "s2": [], "s2f": []}
        for _, r in q.iterrows():
            if pd.isna(r.get("HG")):
                continue
            o = _ausgang(r, mk)
            pin, avg, mx = _get(r, _pre(mk, "pin")), _get(r, _pre(mk, "avg")), _get(r, _pre(mk, "max"))
            pinc, mxc = _get(r, _pre(mk, "pinc")), _get(r, _pre(mk, "maxc"))
            mxc = mxc or pinc                     # keine Schluss-Hoechstquote -> Pinnacle-Schlussquote (realistisch erreichbar)
            pp, pa = (shin(pin) if pin else None), (shin(avg) if avg else None)
            if pp is not None and pa is not None and mx:
                d = (pp - pa) * 100
                i = int(np.argmax(d))
                if d[i] >= S1_MIN:
                    z["s1"].append((o == i, pp[i], mx[i]))
            if pp is not None and pinc:
                pc = shin(pinc)
                if pc is not None:
                    d = (pc - pp) * 100
                    i = int(np.argmax(d))
                    if d[i] >= S2_MIN:
                        if mxc:
                            z["s2"].append((o == i, pc[i], mxc[i]))
                        if mx:
                            z["s2f"].append((o == i, pc[i], mx[i]))
        res = {}
        for k, v in z.items():
            if len(v) < 50:
                res[k] = {"n": len(v)}
                continue
            a = np.array(v, dtype=float)
            res[k] = {"n": len(v), "treffer": round(float(a[:, 0].mean()) * 100, 1), "erwartung": round(float(a[:, 1].mean()) * 100, 1),
                      "roi": round(float((a[:, 0] * a[:, 2] - 1).mean() * 100), 2), "quote": round(float(a[:, 2].mean()), 2)}
        out[mk] = res
        log(f"Sharp-Test {mk}: " + " | ".join(f"{k} n={v['n']} Treffer {v.get('treffer')} % (erw. {v.get('erwartung')}) ROI {v.get('roi')} %" for k, v in res.items()))
    von = q["Date"].min().strftime("%Y-%m-%d") if len(q) else None
    return {"von": von, "bis": q["Date"].max().strftime("%Y-%m-%d") if len(q) else None, "s1_min": S1_MIN, "s2_min": S2_MIN, **out}


def spiel(frueh, jetzt):
    """frueh: football-data-Zeile (Pinnacle vor Tagen) oder None; jetzt: Odds-API-Zeile (Pinnacle + Durchschnitt) oder None."""
    res = {}
    for mk, (suf, lab) in M.items():
        e = {"lab": list(lab)}
        pin = _get(jetzt, _pre(mk, "pin")) if jetzt is not None else None
        avg = _get(jetzt, _pre(mk, "avg")) if jetzt is not None else None
        mx = _get(jetzt, _pre(mk, "max")) if jetzt is not None else None
        if pin is None and jetzt is not None:                       # nach dem Spiel: Pinnacle-Schluss als "jetzt"
            pin = _get(jetzt, _pre(mk, "pinc"))
        pp, pa = (shin(pin) if pin else None), (shin(avg) if avg else None)
        if pp is not None:
            e["pin_q"] = pin
            e["pin_p"] = [round(float(x) * 100, 1) for x in pp]
        if pa is not None:
            e["markt_p"] = [round(float(x) * 100, 1) for x in pa]
        if mx:
            e["max_q"] = mx
        if pp is not None and pa is not None:
            d = [round(float(a - b) * 100, 2) for a, b in zip(pp, pa)]
            e["s1"] = d
            i = int(np.argmax(d))
            if d[i] >= S1_MIN:
                e["s1_tipp"] = i
        fr = _get(frueh, _pre(mk, "pin")) if frueh is not None else None
        pf = shin(fr) if fr else None
        if pf is not None and pp is not None:
            e["frueh_q"] = fr
            e["frueh_p"] = [round(float(x) * 100, 1) for x in pf]
            d = [round(float(a - b) * 100, 2) for a, b in zip(pp, pf)]
            e["s2"] = d
            i = int(np.argmax(d))
            if d[i] >= S2_MIN:
                e["s2_tipp"] = i
        if len(e) > 1:
            res[mk] = e
    return res or None
