# ===== SCORE CODE AGENT - Fussball-Prognose-Portal =====
import glob
import importlib
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

import daten
from ligen import LAENDER, SAISON_START

st.set_page_config(page_title="Score Code Agent", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.sca-titel {font-size: 1.6rem; font-weight: 700; color: #1f9d55; margin-bottom: 0;}
.sca-sub {color: #888; margin-top: 0;}
.sca-form span {display:inline-block; width:24px; text-align:center; border-radius:4px;
                margin-right:3px; font-weight:600; color:white;}
.S {background:#1f9d55;} .U {background:#9e9e9e;} .N {background:#d64545;}
</style>
""", unsafe_allow_html=True)

VALUE_SCHWELLE = 0.05
QUOTEN = {"1": ["AvgH", "B365H"], "X": ["AvgD", "B365D"], "2": ["AvgA", "B365A"],
          "Ü2.5": ["Avg>2.5", "B365>2.5"], "U2.5": ["Avg<2.5", "B365<2.5"]}


# ---------- Modelle automatisch finden (alle Dateien modell_*.py) ----------
def modelle_laden():
    ordner = os.path.dirname(os.path.abspath(__file__))
    if ordner not in sys.path:
        sys.path.insert(0, ordner)
    gefunden = {}
    for pfad in sorted(glob.glob(os.path.join(ordner, "modell_*.py"))):
        mod = importlib.import_module(os.path.basename(pfad)[:-3])
        gefunden[mod.NAME] = mod
    return gefunden


MODELLE = modelle_laden()


@st.cache_data(ttl=6 * 3600, show_spinner="Lade Spieldaten ...")
def lade_daten():
    return daten.alle_spiele(), daten.kommende_spiele()


@st.cache_resource(ttl=6 * 3600, show_spinner="Berechne Modell ...")
def modell_fuer(modell_name, land, stand):
    spiele, _ = lade_daten()
    df = spiele[spiele["Div"].isin(list(LAENDER[land]))]
    return MODELLE[modell_name].trainiere(df, pd.Timestamp(stand))


# ---------- Berechnungen ----------
def maerkte(lh, la):
    M = np.outer(poisson.pmf(range(11), lh), poisson.pmf(range(11), la))
    M = M / M.sum()
    unter = sum(M[i, j] for i in range(11) for j in range(11) if i + j <= 2)
    return {"M": M, "1": np.tril(M, -1).sum(), "X": np.trace(M), "2": np.triu(M, 1).sum(),
            "Ü2.5": 1 - unter, "U2.5": unter,
            "BTS": 1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0]}


def quote(row, namen):
    for n in namen:
        if n in row and pd.notna(row[n]):
            try:
                return float(row[n])
            except ValueError:
                pass
    return None


def value_tipps(m, row):
    tipps = []
    for markt, spalten in QUOTEN.items():
        q = quote(row, spalten)
        if q and m[markt] * q - 1 > VALUE_SCHWELLE:
            tipps.append(f"{markt} @{q:.2f} (+{(m[markt] * q - 1) * 100:.0f}%)")
    return tipps


def prognose(mod, modell, row):
    try:
        lh, la = mod.tore_erwartet(modell, row["HomeTeam"], row["AwayTeam"])
    except Exception:
        return None
    return lh, la, maerkte(lh, la)


def form(spiele, team, n=5):
    t = spiele[(spiele["Home"] == team) | (spiele["Away"] == team)].tail(n)
    out = []
    for _, r in t.iterrows():
        tf, tg = (r["HG"], r["AG"]) if r["Home"] == team else (r["AG"], r["HG"])
        out.append("S" if tf > tg else "U" if tf == tg else "N")
    return out


def form_html(liste):
    return '<div class="sca-form">' + "".join(f'<span class="{x}">{x}</span>' for x in liste) + "</div>"


def tabelle(df):
    t = {}
    for _, r in df.iterrows():
        for team, tf, tg in [(r["Home"], r["HG"], r["AG"]), (r["Away"], r["AG"], r["HG"])]:
            d = t.setdefault(team, {"Sp": 0, "S": 0, "U": 0, "N": 0, "Tore": 0, "Gegen": 0, "Pkt": 0})
            d["Sp"] += 1
            d["Tore"] += int(tf)
            d["Gegen"] += int(tg)
            if tf > tg:
                d["S"] += 1; d["Pkt"] += 3
            elif tf == tg:
                d["U"] += 1; d["Pkt"] += 1
            else:
                d["N"] += 1
    tab = pd.DataFrame.from_dict(t, orient="index")
    if tab.empty:
        return tab
    tab["Diff"] = tab["Tore"] - tab["Gegen"]
    tab = tab.sort_values(["Pkt", "Diff", "Tore"], ascending=False)
    tab.index.name = "Team"
    tab = tab.reset_index()
    tab.index = range(1, len(tab) + 1)
    return tab[["Team", "Sp", "S", "U", "N", "Tore", "Gegen", "Diff", "Pkt"]]


# ---------- Seitenmenue ----------
st.sidebar.markdown('<p class="sca-titel">⚽ Score Code Agent</p>', unsafe_allow_html=True)
seite = st.sidebar.radio("Bereich", ["Spiele & Prognosen", "Spielanalyse", "Tabelle"])
land = st.sidebar.selectbox("Land", list(LAENDER))
liga = st.sidebar.selectbox("Liga", list(LAENDER[land]), format_func=lambda c: LAENDER[land][c])
modell_name = st.sidebar.selectbox("Modell", list(MODELLE))

try:
    spiele, fx = lade_daten()
except Exception as e:
    st.error(f"Die Daten konnten nicht geladen werden. Versuch es später noch einmal. ({e})")
    st.stop()

heute = pd.Timestamp.today().normalize()
mod = MODELLE[modell_name]
modell = modell_fuer(modell_name, land, str(heute.date()))
land_spiele = spiele[spiele["Div"].isin(list(LAENDER[land]))]
liga_fx = fx[fx["Div"] == liga]

st.sidebar.caption(f"Daten bis {spiele['Date'].max():%d.%m.%Y} · {len(spiele):,} Spiele".replace(",", "."))
st.sidebar.caption("Prognosen sind Wahrscheinlichkeiten, keine Garantien.")

st.markdown(f'<p class="sca-titel">{LAENDER[land][liga]}</p>'
            f'<p class="sca-sub">{land} · Modell: {modell_name}</p>', unsafe_allow_html=True)

# ---------- Seite 1: Spiele & Prognosen ----------
if seite == "Spiele & Prognosen":
    if liga_fx.empty:
        st.info("Für diese Liga sind gerade keine kommenden Spiele eingetragen.")
    else:
        zeilen = []
        for _, s in liga_fx.iterrows():
            p = prognose(mod, modell, s)
            if not p:
                continue
            lh, la, m = p
            i, j = np.unravel_index(m["M"].argmax(), m["M"].shape)
            v = value_tipps(m, s)
            zeilen.append({"Datum": s["Date"].strftime("%a %d.%m.") if pd.notna(s["Date"]) else "",
                           "Spiel": f"{s['HomeTeam']} – {s['AwayTeam']}",
                           "1": m["1"] * 100, "X": m["X"] * 100, "2": m["2"] * 100,
                           "Ü2.5": m["Ü2.5"] * 100, "BTS": m["BTS"] * 100,
                           "Tipp": f"{i}:{j}", "xTore": f"{lh:.1f} : {la:.1f}",
                           "Value": ", ".join(v) if v else "–"})
        tab = pd.DataFrame(zeilen)
        if st.toggle("Nur Spiele mit Value-Tipp"):
            tab = tab[tab["Value"] != "–"]
        pct = st.column_config.ProgressColumn
        st.dataframe(tab, hide_index=True, use_container_width=True, column_config={
            "1": pct("1", format="%.0f%%", min_value=0, max_value=100),
            "X": pct("X", format="%.0f%%", min_value=0, max_value=100),
            "2": pct("2", format="%.0f%%", min_value=0, max_value=100),
            "Ü2.5": pct("Ü2.5", format="%.0f%%", min_value=0, max_value=100),
            "BTS": pct("Beide treffen", format="%.0f%%", min_value=0, max_value=100)})
        st.caption("Value = Modell sieht eine höhere Wahrscheinlichkeit als die Quote. "
                   "Sehr hohe Werte (über +30 %) sind meist ein Modellfehler, kein echter Vorteil.")

# ---------- Seite 2: Spielanalyse ----------
elif seite == "Spielanalyse":
    if liga_fx.empty:
        st.info("Für diese Liga sind gerade keine kommenden Spiele eingetragen.")
    else:
        labels = [f"{s['HomeTeam']} – {s['AwayTeam']}" for _, s in liga_fx.iterrows()]
        wahl = st.selectbox("Spielpaarung", range(len(labels)), format_func=lambda k: labels[k])
        s = liga_fx.iloc[wahl]
        heim, gast = s["HomeTeam"], s["AwayTeam"]
        p = prognose(mod, modell, s)
        if not p:
            st.warning("Für eines der Teams gibt es zu wenig Daten.")
        else:
            lh, la, m = p
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Sieg {heim}", f"{m['1']:.0%}", help=f"Quote: {quote(s, QUOTEN['1']) or '–'}")
            c2.metric("Remis", f"{m['X']:.0%}", help=f"Quote: {quote(s, QUOTEN['X']) or '–'}")
            c3.metric(f"Sieg {gast}", f"{m['2']:.0%}", help=f"Quote: {quote(s, QUOTEN['2']) or '–'}")
            c4, c5, c6 = st.columns(3)
            c4.metric("Erwartete Tore", f"{lh:.1f} : {la:.1f}")
            c5.metric("Über 2,5 Tore", f"{m['Ü2.5']:.0%}")
            c6.metric("Beide treffen", f"{m['BTS']:.0%}")

            v = value_tipps(m, s)
            if v:
                st.success("Value-Tipps: " + ", ".join(v))

            st.subheader("Wahrscheinlichste Ergebnisse")
            M = m["M"]
            top = sorted(((M[i, j], f"{i}:{j}") for i in range(6) for j in range(6)), reverse=True)[:6]
            st.dataframe(pd.DataFrame({"Ergebnis": [t[1] for t in top],
                                       "Wahrscheinlichkeit": [t[0] * 100 for t in top]}),
                         hide_index=True, use_container_width=True, column_config={
                             "Wahrscheinlichkeit": st.column_config.ProgressColumn(
                                 format="%.1f%%", min_value=0, max_value=max(top)[0] * 100)})

            st.subheader("Form (letzte 5)")
            f1, f2 = st.columns(2)
            f1.markdown(f"**{heim}**" + form_html(form(land_spiele, heim)), unsafe_allow_html=True)
            f2.markdown(f"**{gast}**" + form_html(form(land_spiele, gast)), unsafe_allow_html=True)

            st.subheader("Direkter Vergleich")
            h2h = spiele[((spiele["Home"] == heim) & (spiele["Away"] == gast)) |
                         ((spiele["Home"] == gast) & (spiele["Away"] == heim))].tail(10).iloc[::-1]
            if h2h.empty:
                st.caption("Keine direkten Duelle in den Daten.")
            else:
                sh = ((h2h["Home"] == heim) & (h2h["HG"] > h2h["AG"])) | ((h2h["Away"] == heim) & (h2h["AG"] > h2h["HG"]))
                sg = ((h2h["Home"] == gast) & (h2h["HG"] > h2h["AG"])) | ((h2h["Away"] == gast) & (h2h["AG"] > h2h["HG"]))
                st.write(f"{heim} {sh.sum()} Siege · {(h2h['HG'] == h2h['AG']).sum()} Remis · {gast} {sg.sum()} Siege "
                         f"· Ø {(h2h['HG'] + h2h['AG']).mean():.1f} Tore")
                st.dataframe(pd.DataFrame({"Datum": h2h["Date"].dt.strftime("%d.%m.%Y"),
                                           "Spiel": h2h["Home"] + " – " + h2h["Away"],
                                           "Ergebnis": h2h["HG"].astype(int).astype(str) + ":" + h2h["AG"].astype(int).astype(str)}),
                             hide_index=True, use_container_width=True)

# ---------- Seite 3: Tabelle ----------
else:
    saison_df = spiele[(spiele["Div"] == liga) & (spiele["Date"] >= pd.Timestamp(SAISON_START))]
    tab = tabelle(saison_df)
    if tab.empty:
        st.info("Noch keine Spiele in dieser Saison.")
    else:
        tab["Form"] = [" ".join(form(land_spiele, t)) for t in tab["Team"]]
        st.dataframe(tab, use_container_width=True)
