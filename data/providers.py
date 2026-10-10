"""Market-data providers. Tiingo is the supported clean EOD source."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "prices"
USER_AGENT = "quant-portfolio/1.0 (research backtest)"


def cached_tiingo_tickers(start: str, end: str | None = None, field: str = "adjClose") -> list[str]:
    """Tickers already on disk (or marked as no-data) for this start/end window."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    end_key = end or "latest"
    prefix = "tiingo" if field == "adjClose" else f"tiingo{field.lower()}"
    out: list[str] = []
    for path in [*CACHE_DIR.glob(f"{prefix}_*_{start}_{end_key}.csv"),
                 *CACHE_DIR.glob(f"{prefix}_*_{start}_{end_key}.empty")]:
        # tiingo_AAPL_2018-01-01_latest.csv
        parts = path.name.split("_")
        if len(parts) >= 4:
            out.append(parts[1].upper())
    return sorted(set(out))


def load_cached_frame(tickers: list[str], start: str, end: str | None = None, field: str = "adjClose") -> pd.DataFrame:
    """Cache-only read (never hits the API); names not on disk are left out."""
    end_key = end or "latest"
    prefix = "tiingo" if field == "adjClose" else f"tiingo{field.lower()}"
    cols = {}
    for t in tickers:
        path = CACHE_DIR / f"{prefix}_{normalize_ticker(t)}_{start}_{end_key}.csv"
        if path.exists():
            cols[normalize_ticker(t)] = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return pd.DataFrame(cols).sort_index()


def _prefix(field: str) -> str:
    return "tiingo" if field == "adjClose" else f"tiingo{field.lower()}"


def normalize_ticker(ticker: str) -> str:
    """BRK.B -> BRK-B style used by Tiingo US equities."""
    return ticker.replace(".", "-").upper().strip()


def env_value(key: str) -> str:
    """Environment variable, falling back to the repo-local .env (no python-dotenv needed)."""
    if not os.environ.get(key):
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key and v.strip():
                    os.environ[key] = v.strip().strip('"').strip("'")
                    break
    return os.environ.get(key, "").strip()


def _tiingo_token() -> str:
    return env_value("TIINGO_API_KEY")


def fetch_tiingo_adj_close(
    ticker: str,
    start: str,
    end: str | None = None,
    *,
    max_retries: int = 8,
    field: str = "adjClose",
) -> pd.Series:
    """Tiingo adjusted close (or another daily field such as the as-traded "close")
    — requires TIINGO_API_KEY (free starter OK)."""
    df = fetch_tiingo_daily(ticker, start, end, max_retries=max_retries)
    t = normalize_ticker(ticker)
    if df.empty:
        return pd.Series(dtype=float, name=t)
    s = df[field if field in df.columns else "close"].astype(float)
    s.name = t
    return s


def fetch_tiingo_daily(
    ticker: str,
    start: str,
    end: str | None = None,
    *,
    max_retries: int = 8,
) -> pd.DataFrame:
    """All Tiingo daily fields (close, adjClose, ...) in one request."""
    key = _tiingo_token()
    if not key:
        raise RuntimeError(
            "TIINGO_API_KEY is not set. Create a free key at https://www.tiingo.com/ "
            "and put it in .env as TIINGO_API_KEY=..."
        )

    t = normalize_ticker(ticker)
    params = {"startDate": start, "format": "json", "token": key}
    if end:
        params["endDate"] = end
    url = f"https://api.tiingo.com/tiingo/daily/{t.lower()}/prices"
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}

    for attempt in range(max_retries):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 404:
            return pd.DataFrame()
        if resp.status_code == 429:
            wait = min(90 * (attempt + 1), 600)
            print(f"  rate-limited on {t}; sleeping {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
        return df.set_index("date").sort_index()

    raise RuntimeError(f"Tiingo rate limit persisted for {t}")


def load_price_frame(
    tickers: list[str],
    start: str,
    end: str | None = None,
    *,
    provider: str = "tiingo",
    use_cache: bool = True,
    pause_sec: float = 1.0,  # only applies on live fetches; cached reads are free
    field: str = "adjClose",
) -> pd.DataFrame:
    """Download (or cache-load) adjusted closes; columns = normalized tickers.
    field="close" loads as-traded (split/dividend-unadjusted) closes instead."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    provider = provider.lower().strip()
    if provider != "tiingo":
        raise ValueError(
            "Only provider='tiingo' is supported for clean EOD data. "
            "Yahoo/Stooq were removed due to quality / bot-block issues."
        )

    series_list: list[pd.Series] = []
    seen: set[str] = set()
    ordered = [normalize_ticker(t) for t in tickers if not (normalize_ticker(t) in seen or seen.add(normalize_ticker(t)))]

    for i, t in enumerate(ordered, 1):
        end_key = end or "latest"
        prefix = "tiingo" if field == "adjClose" else f"tiingo{field.lower()}"
        cache_path = CACHE_DIR / f"{prefix}_{t}_{start}_{end_key}.csv"
        # Tiingo had no data for this symbol (e.g. renamed ticker): don't spend a request again
        empty_marker = cache_path.with_suffix(".empty")
        if use_cache and empty_marker.exists():
            s = pd.Series(dtype=float, name=t)
        elif use_cache and cache_path.exists():
            s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.name = t
        else:
            try:
                df = fetch_tiingo_daily(t, start, end)
                s = df[field].astype(float).rename(t) if field in df.columns else pd.Series(dtype=float, name=t)
                if use_cache and not s.empty:
                    s.to_csv(cache_path, header=True)
                    # the same response carries the other close; cache it to save a request later
                    for other in {"adjClose", "close"} - {field}:
                        other_path = CACHE_DIR / f"{_prefix(other)}_{t}_{start}_{end_key}.csv"
                        if other in df.columns and not other_path.exists():
                            df[other].astype(float).rename(t).to_csv(other_path, header=True)
                elif use_cache:
                    empty_marker.touch()
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                # never echo URLs that embed the API token
                if "tiingo.com" in msg.lower() or "token=" in msg.lower():
                    msg = msg.split(" for url:")[0].strip()
                print(f"  skip {t}: {msg}")
                s = pd.Series(dtype=float, name=t)
            if pause_sec > 0:
                time.sleep(pause_sec)
        if not s.empty:
            series_list.append(s)
        if i % 25 == 0 or i == len(ordered):
            print(f"  prices {i}/{len(ordered)} (tiingo)")

    if not series_list:
        raise RuntimeError("No prices downloaded via Tiingo")
    return pd.concat(series_list, axis=1).sort_index().ffill(limit=5)
