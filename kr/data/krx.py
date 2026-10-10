"""
KRX prices and point-in-time KOSPI200 membership via pykrx.

Adjusted prices work without a login; unadjusted prices and index
constituents need a free data.krx.co.kr account (KRX_ID / KRX_PW in .env).
Results are cached under data/cache/kr/.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from data.providers import env_value

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "kr"


def _ensure_login() -> None:
    for key in ("KRX_ID", "KRX_PW"):
        value = env_value(key)
        if value:
            os.environ[key] = value


def _stock():
    _ensure_login()
    from pykrx import stock

    return stock


def kospi200_asof(date: str | pd.Timestamp, index_code: str = "1028") -> list[str]:
    """Index constituents (6-digit tickers) on `date`; requires a KRX login."""
    d = pd.Timestamp(date).strftime("%Y%m%d")
    path = CACHE_DIR / "universe" / f"{index_code}_{d}.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str)["ticker"].tolist()
    tickers = list(_stock().get_index_portfolio_deposit_file(index_code, d))
    if not tickers:
        raise RuntimeError(f"No constituents for {index_code} on {d}; is KRX_ID/KRX_PW set in .env?")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
    return tickers


def membership_matrix(dates: pd.DatetimeIndex, snapshot_dates: list[pd.Timestamp],
                      index_code: str = "1028") -> pd.DataFrame:
    """Boolean (dates x tickers): member as of the latest snapshot on or before each date."""
    snaps = {pd.Timestamp(d): set(kospi200_asof(d, index_code)) for d in snapshot_dates}
    tickers = sorted(set().union(*snaps.values()))
    keys = sorted(snaps)
    rows, k = [], -1
    for d in dates:
        while k + 1 < len(keys) and keys[k + 1] <= d:
            k += 1
        members = snaps[keys[k]] if k >= 0 else set()
        rows.append([t in members for t in tickers])
    return pd.DataFrame(rows, index=dates, columns=tickers)


def close_series(ticker: str, start: str, end: str | None = None, adjusted: bool = True) -> pd.Series:
    end_d = pd.Timestamp(end or pd.Timestamp.today()).strftime("%Y%m%d")
    start_d = pd.Timestamp(start).strftime("%Y%m%d")
    kind = "adj" if adjusted else "raw"
    path = CACHE_DIR / "prices" / f"{kind}_{ticker}_{start_d}_{'latest' if end is None else end_d}.csv"
    if path.exists():
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = ticker
        return s
    df = _stock().get_market_ohlcv(start_d, end_d, ticker, adjusted=adjusted)
    s = df.iloc[:, 3].astype(float) if not df.empty else pd.Series(dtype=float)  # close column
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(path, header=True)
    return s


def price_frame(tickers: list[str], start: str, end: str | None = None, adjusted: bool = True) -> pd.DataFrame:
    cols = {t: close_series(t, start, end, adjusted) for t in tickers}
    return pd.DataFrame({t: s for t, s in cols.items() if not s.empty}).sort_index()
