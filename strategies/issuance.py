"""
Net share issuance long-short: firms shrinking their share count (buybacks)
tend to outperform firms growing it (equity issuance, heavy stock comp).
Monthly, dollar-neutral, equal-weight within each side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pick_sides(
    signal: pd.Series,
    *,
    frac: float = 0.3,
    sectors: dict[str, str] | None = None,
    min_group: int = 4,
) -> tuple[list[str], list[str]]:
    """Long the lowest `frac` of share growth, short the highest; ranked across all
    names, or within each sector (sector-neutral) when `sectors` is given."""
    signal = signal.dropna()
    if sectors is None:
        groups = [signal]
    else:
        key = signal.index.map(lambda t: sectors.get(t, "Unknown"))
        groups = [g for _, g in signal.groupby(key) if len(g) >= min_group]
    longs, shorts = [], []
    for g in groups:
        n = max(1, int(round(len(g) * frac)))
        ordered = g.sort_values(kind="stable")
        longs += list(ordered.index[:n])
        shorts += list(ordered.index[-n:])
    return longs, shorts


def event_overlay_weights(
    index: pd.DatetimeIndex,
    columns: list[str],
    events: pd.DataFrame,
    *,
    threshold: float = 0.02,
    hold_days: int = 63,
    long_weight: float = 0.02,
    short_weight: float = 0.03,
    long_cap: float = 0.5,
    short_cap: float = 0.75,
    membership: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    events: ticker, filed, share_growth (one row per filing). The latest filing of
    a name decides its state for hold_days trading days, starting the first
    trading day after the filing date: share_growth <= -threshold -> long
    (buyback), >= +threshold -> short (issuance), otherwise flat. Each side is
    scaled down pro rata when it exceeds its cap.
    """
    col = {c: k for k, c in enumerate(columns)}
    sign = np.zeros((len(index), len(columns)))
    for t, g in events.sort_values("filed").groupby("ticker"):
        c = col.get(t)
        if c is None:
            continue
        starts = index.searchsorted(pd.DatetimeIndex(g["filed"]), side="right")
        vals = g["share_growth"].to_numpy()
        for k, (s, v) in enumerate(zip(starts, vals)):
            nxt = starts[k + 1] if k + 1 < len(starts) else len(index)
            e = min(s + hold_days, nxt, len(index))
            if v <= -threshold:
                sign[s:e, c] = 1.0
            elif v >= threshold:
                sign[s:e, c] = -1.0
    w = pd.DataFrame(np.where(sign > 0, long_weight, np.where(sign < 0, -short_weight, 0.0)),
                     index=index, columns=columns)
    if membership is not None:
        w = w.where(membership.reindex(index=index, columns=columns).fillna(False), 0.0)
    if prices is not None:
        w = w.where(prices.notna(), 0.0)
    longs = w.clip(lower=0.0).sum(axis=1)
    shorts = -w.clip(upper=0.0).sum(axis=1)
    long_scale = (long_cap / longs).where(longs > long_cap, 1.0)
    short_scale = (short_cap / shorts).where(shorts > short_cap, 1.0)
    return w.clip(lower=0.0).mul(long_scale, axis=0) + w.clip(upper=0.0).mul(short_scale, axis=0)


def long_short_weights(
    index: pd.DatetimeIndex,
    columns: list[str],
    signals: dict[pd.Timestamp, pd.Series],
    *,
    frac: float = 0.3,
    sectors_by_date: dict[pd.Timestamp, dict[str, str]] | None = None,
    gross: float = 1.0,
    membership: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rebalance on each signal date (gross/2 per side), hold until the next; a name
    that leaves the index or stops trading is closed to cash on that day."""
    w = pd.DataFrame(0.0, index=index, columns=columns)
    dates = sorted(signals)
    for k, d in enumerate(dates):
        nxt = dates[k + 1] if k + 1 < len(dates) else index[-1] + pd.Timedelta(days=1)
        span = (index >= d) & (index < nxt)
        if not span.any():
            continue
        sig = signals[d].reindex(columns)
        if membership is not None:
            sig = sig[membership.loc[d].reindex(sig.index).fillna(False).to_numpy()]
        if prices is not None:
            sig = sig[np.isfinite(prices.loc[d].reindex(sig.index)).to_numpy()]
        longs, shorts = pick_sides(
            sig, frac=frac, sectors=sectors_by_date.get(d) if sectors_by_date else None
        )
        row = pd.Series(0.0, index=columns)
        if longs and shorts:
            row[longs] = gross / 2 / len(longs)
            row[shorts] = -gross / 2 / len(shorts)
        w.loc[span] = row.to_numpy()
    if membership is not None:
        w = w.where(membership.reindex(index=index, columns=columns).fillna(False), 0.0)
    if prices is not None:
        w = w.where(prices.notna(), 0.0)
    return w
