# ===== DATEN LADEN =====
# Historie: xgabora (GitHub) | aktuelle Saison + kommende Spiele: football-data.co.uk
import pandas as pd
import requests

from ligen import LAENDER, SAISON, EXTRA_LIGEN

ALLE_LIGEN = [code for ligen in LAENDER.values() for code in ligen]
SPALTEN = ["Div", "Date", "Home", "Away", "HG", "AG"]

XG_RAW = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data/main/data/"
XG_API = "https://api.github.com/repos/xgabora/Club-Football-Match-Data/contents/data"
FD_SAISON = "https://www.football-data.co.uk/mmz4281/{saison}/{liga}.csv"
FD_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
FD_NEU = "https://www.football-data.co.uk/new/{liga}.csv"
FD_NEU_FIXTURES = "https://www.football-data.co.uk/new_league_fixtures.csv"
EXTRA_QUOTEN = {"AvgCH": "AvgH", "AvgCD": "AvgD", "AvgCA": "AvgA", "PSCH": "B365H", "PSCD": "B365D", "PSCA": "B365A"}


def extra_liga(liga):
    """Extra-Liga (z. B. Daenemark) im selben Format wie die Hauptligen."""
    d = _csv(FD_NEU.format(liga=liga))
    d = d.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
    for alt, neu in EXTRA_QUOTEN.items():
        if alt in d and neu not in d:
            d[neu] = d[alt]
    d["Div"] = liga
    return d


def _csv(url):
    d = pd.read_csv(url, encoding="latin-1")
    d.columns = [str(c).replace("ï»¿", "").replace("\ufeff", "").strip() for c in d.columns]
    return d


def _xgabora_url():
    for name in ["Matches.csv", "MATCHES.csv", "matches.csv"]:
        try:
            if requests.head(XG_RAW + name, timeout=15).status_code == 200:
                return XG_RAW + name
        except Exception:
            pass
    files = requests.get(XG_API, timeout=30).json()
    return [f["download_url"] for f in files
            if "match" in f["name"].lower() and f["name"].endswith(".csv")][0]


def historie():
    cols = ["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"]
    h = pd.read_csv(_xgabora_url(), usecols=cols, low_memory=False)[cols]
    h = h[h["Division"].isin(ALLE_LIGEN)].copy()
    h.columns = SPALTEN
    h["Date"] = pd.to_datetime(h["Date"], errors="coerce")
    return h


def saison():
    teile = []
    for liga in ALLE_LIGEN:
        try:
            d = extra_liga(liga) if liga in EXTRA_LIGEN else _csv(FD_SAISON.format(saison=SAISON, liga=liga))
            d = d[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"]].copy()
            d.columns = ["Home", "Away", "HG", "AG", "Date"]
            d["Div"] = liga
            d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
            teile.append(d[SPALTEN])
        except Exception:
            pass
    return pd.concat(teile) if teile else pd.DataFrame(columns=SPALTEN)


def alle_spiele():
    df = pd.concat([historie(), saison()])
    df["HG"] = pd.to_numeric(df["HG"], errors="coerce")
    df["AG"] = pd.to_numeric(df["AG"], errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away", "HG", "AG"])
    df = df.drop_duplicates(subset=["Div", "Date", "Home", "Away"], keep="last")
    return df.sort_values("Date").reset_index(drop=True)


def kommende_spiele():
    fx = _csv(FD_FIXTURES)
    try:
        nf = _csv(FD_NEU_FIXTURES).rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
        land_code = {v: k for k, v in EXTRA_LIGEN.items()}
        nf["Div"] = nf["Country"].map(land_code)
        for alt, neu in EXTRA_QUOTEN.items():
            if alt in nf and neu not in nf:
                nf[neu] = nf[alt]
        fx = pd.concat([fx, nf.dropna(subset=["Div"])], ignore_index=True)
    except Exception:
        pass
    fx = fx[fx["Div"].isin(ALLE_LIGEN)].copy()
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    return fx.sort_values(["Date", "Div"]).reset_index(drop=True)
