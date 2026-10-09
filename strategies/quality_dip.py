"""
Quality dip-buying (long-only): buy a financially strong name after a sharp
short-term drop and sell when it recovers to its moving average.

Quality is a point-in-time cross-sectional score from SEC filings: the mean
percentile rank of ROE (high), debt ratio (low) and earnings yield (high).
Losses, negative equity and missing filings rank last.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quality_scores(table: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Adds `score` (0 = best) per date among index members on that date."""
    out = []
    for d, g in table.groupby("date"):
        g = g[g["ticker"].map(lambda t: t in membership.columns and bool(membership.at[d, t]))].copy()
        ranks = [
            g["roe"].rank(pct=True, ascending=False),
            g["debt_ratio"].rank(pct=True, ascending=True),
            g["ey"].rank(pct=True, ascending=False),
        ]
        g["score"] = sum(r.fillna(1.0) for r in ranks) / len(ranks)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def quality_sets(scored: pd.DataFrame, top_frac: float) -> dict[pd.Timestamp, set[str]]:
    sets = {}
    for d, g in scored.groupby("date"):
        n = int(round(len(g) * top_frac))
        sets[d] = set(g.sort_values("score", kind="stable")["ticker"].iloc[:n])
    return sets


def _asof_sets(index: pd.DatetimeIndex, sets: dict[pd.Timestamp, set[str]]) -> list[set[str]]:
    keys = sorted(sets)
    out, k = [], -1
    for d in index:
        while k + 1 < len(keys) and keys[k + 1] <= d:
            k += 1
        out.append(sets[keys[k]] if k >= 0 else set())
    return out


def build_quality_dip(
    prices: pd.DataFrame,
    sets: dict[pd.Timestamp, set[str]],
    *,
    membership: pd.DataFrame | None = None,
    ret_days: int = 5,
    vol_window: int = 60,
    entry_sigma: float = 2.0,
    exit_ma: int = 20,
    max_hold: int = 20,
    slot_weight: float = 0.10,
    max_slots: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Signals use closes through day t and the position is held from t's close
    (weights at t earn t+1 returns). Entry: name is in the latest quality set,
    is an index member, and its ret_days log return is <= -entry_sigma times
    its trailing daily vol * sqrt(ret_days). Most negative moves fill free slots
    first. Exit: close >= exit_ma-day average, max_hold days, or index exit.
    """
    idx, cols = prices.index, list(prices.columns)
    logp = np.log(prices)
    daily = logp.diff()
    vol = daily.rolling(vol_window, min_periods=vol_window).std().shift(1)
    move = (logp - logp.shift(ret_days)) / (vol * np.sqrt(ret_days))
    ma = prices.rolling(exit_ma, min_periods=exit_ma).mean()

    move_a, px_a, ma_a = move.to_numpy(), prices.to_numpy(), ma.to_numpy()
    member = (
        membership.reindex(index=idx, columns=cols).fillna(False).to_numpy()
        if membership is not None else np.ones(prices.shape, dtype=bool)
    )
    col = {c: k for k, c in enumerate(cols)}
    day_sets = _asof_sets(idx, sets)

    weights = np.zeros(prices.shape)
    held: dict[int, int] = {}  # column -> entry day
    trades = []
    for t in range(len(idx)):
        for c, t0 in list(held.items()):
            reason = None
            if not member[t, c] or not np.isfinite(px_a[t, c]):
                reason = "index_exit"
            elif np.isfinite(ma_a[t, c]) and px_a[t, c] >= ma_a[t, c]:
                reason = "recovered"
            elif t - t0 >= max_hold:
                reason = "timeout"
            if reason:
                trades.append({"ticker": cols[c], "entry": idx[t0], "exit": idx[t],
                               "days": t - t0, "reason": reason, "entry_move": move_a[t0, c]})
                del held[c]
        free = max_slots - len(held)
        if free > 0:
            cands = []
            for name in day_sets[t]:
                c = col.get(name)
                if c is None or c in held or not member[t, c]:
                    continue
                m = move_a[t, c]
                if np.isfinite(m) and m <= -entry_sigma:
                    cands.append((m, c))
            for _, c in sorted(cands)[:free]:
                held[c] = t
        for c in held:
            weights[t, c] = slot_weight

    for c, t0 in held.items():
        trades.append({"ticker": cols[c], "entry": idx[t0], "exit": idx[-1],
                       "days": len(idx) - 1 - t0, "reason": "open", "entry_move": move_a[t0, c]})
    return pd.DataFrame(weights, index=idx, columns=cols), pd.DataFrame(trades)


def basket_weights(
    index: pd.DatetimeIndex,
    columns: list[str],
    sets: dict[pd.Timestamp, set[str]],
    rebalance_dates: list[pd.Timestamp],
    membership: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Equal weight in the set as of each rebalance date, held (drifting) until the next.
    Names that leave the index or stop trading are sold to cash."""
    w = pd.DataFrame(0.0, index=index, columns=columns)
    rets = prices.pct_change().fillna(0.0) if prices is not None else None
    reb = sorted(rebalance_dates)
    keys = sorted(sets)
    current = pd.Series(0.0, index=columns)
    k = 0
    for t, d in enumerate(index):
        if k < len(reb) and d >= reb[k]:
            s = sets[max(x for x in keys if x <= d)]
            names = [n for n in s if n in w.columns
                     and (prices is None or np.isfinite(prices.at[d, n]))
                     and (membership is None or bool(membership.at[d, n]))]
            current = pd.Series(0.0, index=columns)
            if names:
                current[names] = 1.0 / len(names)
            while k < len(reb) and d >= reb[k]:
                k += 1
        elif rets is not None and t > 0:
            r = rets.iloc[t]
            current = current * (1.0 + r) / (1.0 + float((current * r).sum()))
            if membership is not None:
                current[~membership.iloc[t].reindex(columns).fillna(False).to_numpy()] = 0.0
        w.iloc[t] = current.to_numpy()
    return w
