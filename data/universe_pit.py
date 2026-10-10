"""Point-in-time S&P universe via pitindex (no Yahoo membership scraping)."""

from __future__ import annotations

import warnings
from functools import lru_cache

import pandas as pd

try:
    import pitindex
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install pitindex") from exc

from data.pit_sector_overrides import SECTOR_OVERRIDES, SUB_INDUSTRY_OVERRIDES


def normalize_ticker(t: str) -> str:
    return str(t).replace(".", "-").upper()


def _silence_stale_warning() -> None:
    warnings.filterwarnings("ignore", category=getattr(pitindex, "StaleDataWarning", Warning))


@lru_cache(maxsize=8)
def _history(start: str, end: str, index: str) -> pd.DataFrame:
    _silence_stale_warning()
    return pitindex.get_constituents_history(start, end, index=index)


def constituents_asof(as_of: str | pd.Timestamp, index: str = "sp500") -> pd.DataFrame:
    _silence_stale_warning()
    df = pitindex.get_constituents(pd.Timestamp(as_of).date().isoformat(), index=index).copy()
    fill = df["ticker"].map(lambda t: SECTOR_OVERRIDES.get(normalize_ticker(t)))
    df["gics_sector"] = df["gics_sector"].fillna(fill)
    sub_fill = df["ticker"].map(lambda t: SUB_INDUSTRY_OVERRIDES.get(normalize_ticker(t)))
    df["gics_sub_industry"] = df["gics_sub_industry"].fillna(sub_fill)
    return df


def pit_price_tickers(
    start: str,
    *,
    index: str = "sp500",
    max_names: int = 50,
    seed: list[str] | None = None,
    include_unknown_sector: bool = False,
) -> list[str]:
    """
    Names to price: members as of `start` only (no look-ahead), `seed` first,
    then round-robin across GICS sectors (alphabetical within a sector) so the
    universe widens evenly instead of by ticker spelling. pitindex leaves most
    S&P 400 sectors blank; include_unknown_sector adds them as one more queue.
    """
    roster = constituents_asof(start, index=index)
    roster["ticker"] = roster["ticker"].map(normalize_ticker)
    members = set(roster["ticker"])
    picked = [t for t in dict.fromkeys(normalize_ticker(s) for s in (seed or [])) if t in members]

    if include_unknown_sector:
        roster["gics_sector"] = roster["gics_sector"].fillna("Unknown")
    queues = {
        sector: sorted(set(part["ticker"]) - set(picked))
        for sector, part in roster.dropna(subset=["gics_sector"]).groupby("gics_sector")
    }
    while len(picked) < max_names and any(queues.values()):
        for sector in sorted(queues):
            if queues[sector] and len(picked) < max_names:
                picked.append(queues[sector].pop(0))
    return picked[:max_names]


def historical_tickers(start: str, end: str | None = None, index: str = "sp500") -> list[str]:
    """Unique tickers that were index members at any point in [start, end]."""
    end = end or pd.Timestamp.today().date().isoformat()
    hist = _history(start, end, index)
    return sorted({normalize_ticker(t) for t in hist["ticker"].dropna()})


def membership_matrix(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    *,
    index: str = "sp500",
) -> pd.DataFrame:
    """
    Boolean DataFrame (dates x tickers): True if ticker was an index member
    as of that date (forward-filled from PIT snapshots).
    """
    tickers = [normalize_ticker(t) for t in tickers]
    if len(dates) == 0:
        return pd.DataFrame(columns=tickers)

    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    hist = _history(start, end, index).copy()
    hist["as_of"] = pd.to_datetime(hist["as_of"])
    hist["ticker"] = hist["ticker"].map(normalize_ticker)

    wide = (
        hist.assign(v=True)
        .pivot_table(index="as_of", columns="ticker", values="v", aggfunc="max", fill_value=False)
        .sort_index()
    )
    for t in tickers:
        if t not in wide.columns:
            wide[t] = False
    wide = wide.reindex(columns=tickers).fillna(False)

    # Seed days before first snapshot
    if dates.min() < wide.index.min():
        seed = constituents_asof(dates.min(), index=index)
        seed_set = {normalize_ticker(t) for t in seed["ticker"]}
        seed_row = pd.DataFrame(
            [[t in seed_set for t in tickers]],
            index=[dates.min()],
            columns=tickers,
        )
        wide = pd.concat([seed_row, wide]).sort_index()
        wide = wide[~wide.index.duplicated(keep="last")]

    aligned = wide.reindex(wide.index.union(dates)).sort_index().ffill()
    return aligned.reindex(dates).fillna(False).astype(bool)


def sector_groups_asof(
    as_of: str | pd.Timestamp,
    tickers: list[str] | None = None,
    *,
    index: str = "sp500",
    min_names: int = 2,
    level: str = "sector",
) -> dict[str, list[str]]:
    """
    GICS group -> tickers present as of date (and optionally in `tickers`).
    level="sector" groups by GICS sector; level="sub_industry" by GICS
    sub-industry (the finest GICS level), keyed "Sector / Sub-industry".
    """
    df = constituents_asof(as_of, index=index).copy()
    df["ticker"] = df["ticker"].map(normalize_ticker)
    if tickers is not None:
        allowed = {normalize_ticker(t) for t in tickers}
        df = df[df["ticker"].isin(allowed)]
    df["gics_sector"] = df["gics_sector"].fillna("Unknown")
    if level == "sub_industry":
        df["group"] = df["gics_sector"] + " / " + df["gics_sub_industry"].fillna("Unknown")
    elif level == "sector":
        df["group"] = df["gics_sector"]
    else:
        raise ValueError(f"unknown group level: {level}")
    groups: dict[str, list[str]] = {}
    for sector, part in df.groupby("group"):
        names = sorted(part["ticker"].unique().tolist())
        if len(names) >= min_names:
            groups[str(sector)] = names
    return groups


def coverage_report(membership: pd.DataFrame, prices: pd.DataFrame) -> dict[str, float]:
    """Share of PIT members that have a price that day (delist gap proxy)."""
    common = membership.columns.intersection(prices.columns)
    if len(common) == 0:
        return {"avg_member_coverage": 0.0, "priced_tickers": 0.0}
    mem = membership[common]
    priced = prices[common].notna()
    denom = mem.sum(axis=1).replace(0, pd.NA)
    cov = (mem & priced).sum(axis=1) / denom
    return {
        "avg_member_coverage": float(cov.mean()),
        "priced_tickers": float(len(common)),
        "membership_days": float(len(membership)),
    }
