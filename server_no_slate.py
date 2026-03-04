from __future__ import annotations

import os
import io
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Config
# -------------------------

# Local folder (optional). On Render this is typically the app working dir.
DATA_DIR = os.getenv("SHARP_DATA_DIR", os.getcwd())

# GitHub raw base (recommended primary source)
# Example: https://raw.githubusercontent.com/Vision919/cbb-sharp-data/main
GITHUB_RAW_BASE = os.getenv("SHARP_GITHUB_RAW_BASE", "").rstrip("/")

# Optional simple auth for Actions
SHARP_API_KEY = os.getenv("SHARP_API_KEY", "")

# CSV filenames your pipeline produces
FILES = {
    "kenpom": os.getenv("SHARP_KENPOM_FILE", "kenpom_live.csv"),
    "vegas": os.getenv("SHARP_VEGAS_FILE", "vegas_odds.csv"),
    "players": os.getenv("SHARP_PLAYERS_FILE", "player_stats.csv"),
    # "slate": os.getenv("SHARP_SLATE_FILE", "active_slate.csv"),  # if you removed slate
    "teamrankings": os.getenv("SHARP_TEAMRANKINGS_FILE", "teamrankings_live.csv"),
}

# Cache to avoid re-reading every request (0 = disable cache)
CACHE_TTL_SECONDS = int(os.getenv("SHARP_CACHE_TTL", "0"))


@dataclass
class CacheItem:
    ts: float
    df: pd.DataFrame


_cache: Dict[str, CacheItem] = {}


def _require_api_key(x_api_key: Optional[str]) -> None:
    if not SHARP_API_KEY:
        return  # auth disabled
    if not x_api_key or x_api_key != SHARP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _local_path(key: str) -> str:
    return os.path.join(DATA_DIR, FILES[key])


def _read_csv_local(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Robust read: avoid a single malformed row crashing the service
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")


def _read_csv_github(filename: str) -> pd.DataFrame:
    if not GITHUB_RAW_BASE:
        raise FileNotFoundError("GitHub raw base not configured.")
    # Cache-bust to avoid CDN staleness
    url = f"{GITHUB_RAW_BASE}/{filename}?v={int(time.time())}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), on_bad_lines="skip")


def _get_df_strict(key: str) -> pd.DataFrame:
    """Strict bridge fetch.

    - Prefer GitHub raw first (freshest)
    - Fall back to local only if GitHub fetch fails

    Raises if neither exists.
    """
    now = time.time()
    cached = _cache.get(key)
    if cached and (now - cached.ts) < CACHE_TTL_SECONDS:
        return cached.df

    try:
        df = _read_csv_github(FILES[key])
    except Exception:
        df = _read_csv_local(_local_path(key))

    df.columns = [str(c).strip() for c in df.columns]

    _cache[key] = CacheItem(ts=now, df=df)
    return df


def _get_df_soft(key: str) -> tuple[pd.DataFrame, bool, Optional[str]]:
    """Soft bridge fetch.

    Returns (df, missing, error_message). If missing, df is an empty DataFrame.

    This prevents 500s during preview/ledger when a non-critical file (like slate)
    is missing or stale.
    """
    try:
        df = _get_df_strict(key)
        return df, False, None
    except Exception as e:
        return pd.DataFrame(), True, str(e)


def _df_to_records(df: pd.DataFrame, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records:
      - NaN -> None
      - +inf/-inf -> None

    Prevents: ValueError: Out of range float values are not JSON compliant
    """
    if limit is not None:
        df = df.head(limit)

    safe = df.copy()
    safe = safe.replace([np.inf, -np.inf], np.nan)
    safe = safe.where(pd.notnull(safe), None)

    return safe.to_dict(orient="records")


app = FastAPI(title="Sharp-CBB Data API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/sharp/health")
def health(x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)

    status: Dict[str, Any] = {
        "ok": True,
        "data_dir": DATA_DIR,
        "github_raw_base": GITHUB_RAW_BASE or None,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }
    for k in FILES:
        status[f"{k}_local_exists"] = os.path.exists(_local_path(k))
    return status


@app.get("/sharp/data")
def sharp_data(
    x_api_key: Optional[str] = Header(default=None),
    include_rows: bool = Query(default=True),
    preview_rows: int = Query(default=0, ge=0, le=200),
) -> Dict[str, Any]:
    """Returns bridge metadata (and optionally rows).

    KEY CHANGE:
    - This endpoint will NOT 500 just because slate is missing.
    - It reports missing=true and rows=0 for missing tables.

    Your GPT can then enforce "stop if <5 rows" cleanly without tool errors.
    """
    _require_api_key(x_api_key)

    payload: Dict[str, Any] = {"ok": True, "generated_at_unix": int(time.time()), "tables": {}}

    for key in ["kenpom", "vegas", "players", "teamrankings"]:
        df, missing, err = _get_df_soft(key)

        table_info: Dict[str, Any] = {
            "rows": int(df.shape[0]) if not missing else 0,
            "cols": [str(c) for c in df.columns.tolist()] if not missing else [],
            "missing": bool(missing),
        }
        if missing and err:
            table_info["error"] = err

        if include_rows:
            limit = preview_rows if preview_rows > 0 else None
            table_info["data"] = _df_to_records(df, limit=limit) if not missing else []

        payload["tables"][key] = table_info

    return payload


@app.get("/sharp/game")
def sharp_game(
    team: str = Query(..., description="Team name fragment; matches Home or Away in vegas table"),
    x_api_key: Optional[str] = Header(default=None),
    max_results: int = Query(default=10, ge=1, le=50),
) -> Dict[str, Any]:
    _require_api_key(x_api_key)

    vegas = _get_df_strict("vegas").copy()

    cols = {c.lower(): c for c in vegas.columns}
    if "home" not in cols or "away" not in cols:
        raise HTTPException(status_code=500, detail="Vegas CSV must contain Home and Away columns.")

    home_col = cols["home"]
    away_col = cols["away"]

    team_l = team.strip().lower()
    mask = (
        vegas[home_col].astype(str).str.lower().str.contains(team_l)
        | vegas[away_col].astype(str).str.lower().str.contains(team_l)
    )
    matches = vegas.loc[mask].head(max_results)

    return {
        "ok": True,
        "query": team,
        "matches": _df_to_records(matches),
        "match_count": int(matches.shape[0]),
    }


@app.get("/sharp/table/{table_name}/preview")
def table_preview(
    table_name: str,
    n: int = Query(default=25, ge=1, le=200),
    x_api_key: Optional[str] = Header(default=None),
):
    _require_api_key(x_api_key)

    if table_name not in FILES:
        raise HTTPException(status_code=404, detail="Unknown table")

    df, missing, err = _get_df_soft(table_name)
    if missing:
        return {
            "ok": True,
            "table": table_name,
            "rows": 0,
            "cols": [],
            "missing": True,
            "error": err,
            "data": [],
        }

    return {
        "ok": True,
        "table": table_name,
        "rows": int(df.shape[0]),
        "cols": [str(c) for c in df.columns.tolist()],
        "missing": False,
        "data": _df_to_records(df, limit=n),
    }


@app.get("/sharp/team")
def team_lookup(
    team: str = Query(..., description="Team name fragment"),
    x_api_key: Optional[str] = Header(default=None),
    max_results: int = Query(default=50, ge=1, le=200),
):
    _require_api_key(x_api_key)

    out: Dict[str, Any] = {"ok": True, "query": team, "kenpom": [], "players": []}
    team_l = team.strip().lower()

    # KenPom
    kp = _get_df_strict("kenpom").copy()
    kp_cols = {c.lower(): c for c in kp.columns}
    kp_team_col = kp_cols.get("team") or kp_cols.get("school") or kp_cols.get("team_name")
    if kp_team_col:
        m = kp[kp[kp_team_col].astype(str).str.lower().str.contains(team_l)].head(max_results)
        out["kenpom"] = _df_to_records(m)

    # Players
    pl = _get_df_strict("players").copy()
    pl_cols = {c.lower(): c for c in pl.columns}
    pl_team_col = pl_cols.get("team") or pl_cols.get("school")
    if pl_team_col:
        m = pl[pl[pl_team_col].astype(str).str.lower().str.contains(team_l)].head(max_results)
        out["players"] = _df_to_records(m)

    return out

# --- TABLE PATHS ---
TABLE_PATHS = {
    "kenpom": os.path.join(DATA_DIR, "kenpom_live.csv"),
    "vegas": os.path.join(DATA_DIR, "vegas_odds.csv"),
    "players": os.path.join(DATA_DIR, "players_live.csv"),
    "slate": os.path.join(DATA_DIR, "active_slate.csv"),
    "teamrankings": os.path.join(DATA_DIR, "teamrankings_live.csv"),
}

def _normalize_team(name: str) -> str:
    s = (name or "").lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[\.\'’]", "", s)
    s = re.sub(r"\s+", " ", s)
    # mirror the normalizer used in the teamrankings scraper
    s = s.replace("st ", "state ")
    s = s.replace("st-", "state ")
    s = s.replace("st.", "state")
    return s

def load_table(table_name: str) -> pd.DataFrame:
    path = TABLE_PATHS.get(table_name)
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# --- TABLE PREVIEW ---
@app.get("/sharp/table/{table_name}/preview")
def tablePreview(table_name: str, n: int = 25):
    if table_name not in {"kenpom", "vegas", "players", "slate", "teamrankings"}:
        return {"ok": False, "error": "Unknown table name"}

    df = load_table(table_name)
    cols = df.columns.tolist() if not df.empty else []
    data = df.head(n).to_dict(orient="records") if not df.empty else []
    return {
        "ok": True,
        "table": table_name,
        "rows": int(len(df)),
        "cols": cols,
        "missing": df.empty,
        "data": data,
    }


# --- LEDGER ---
@app.get("/sharp/data")
def getSharpData(include_rows: bool = True, preview_rows: int = 0):
    tables = {}
    for tname in ["kenpom", "vegas", "players", "slate", "teamrankings"]:
        df = load_table(tname)
        tables[tname] = {
            "rows": int(len(df)),
            "cols": df.columns.tolist() if not df.empty else [],
            "missing": df.empty,
        }

    out = {"ok": True, "generated_at_unix": int(time.time()), "tables": tables}
    return out


# --- TEAM LOOKUP ---
@app.get("/sharp/team")
def teamLookup(team: str, max_results: int = 50):
    kenpom = load_table("kenpom")
    players = load_table("players")
    teamrankings = load_table("teamrankings")

    q = (team or "").strip().lower()

    # KenPom match
    kp = []
    if not kenpom.empty and "Team" in kenpom.columns:
        mask = kenpom["Team"].astype(str).str.lower().str.contains(q, na=False)
        kp = kenpom.loc[mask].head(max_results).to_dict(orient="records")

    # Players match
    pr = []
    if not players.empty and "Team" in players.columns:
        mask = players["Team"].astype(str).str.lower().str.contains(q, na=False)
        pr = players.loc[mask].head(max_results).to_dict(orient="records")

    # TeamRankings match (prefer Team_norm)
    tr_rows = []
    if not teamrankings.empty:
        if "Team_norm" in teamrankings.columns:
            qn = _normalize_team(team)
            mask = teamrankings["Team_norm"].astype(str).str.contains(qn, na=False)
            tr_rows = teamrankings.loc[mask].head(max_results).to_dict(orient="records")
        elif "Team" in teamrankings.columns:
            mask = teamrankings["Team"].astype(str).str.lower().str.contains(q, na=False)
            tr_rows = teamrankings.loc[mask].head(max_results).to_dict(orient="records")

    return {
        "ok": True,
        "query": team,
        "kenpom": kp,
        "players": pr,
        "teamrankings": tr_rows,
    }

# Run:
#   pip install fastapi uvicorn pandas requests numpy
#   uvicorn server_no_slate:app --host 0.0.0.0 --port 8000