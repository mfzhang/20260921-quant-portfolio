"""Composite value score: percentile ranks of several yields, combined per name."""

from __future__ import annotations

import numpy as np
import pandas as pd


def composite_scores(
    metrics: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    method: str = "geometric",
    min_metrics: int = 4,
) -> pd.Series:
    """
    metrics: one row per name (index = ticker) for a single date. Each column is
    ranked ascending (higher yield = cheaper = higher percentile) as rank / (n + 1),
    so percentiles lie strictly in (0, 1). Names with fewer than `min_metrics`
    available are dropped. geometric = exp(mean log pct): a name must be cheap on
    every measure to score high; arithmetic = plain mean.
    """
    m = metrics[list(columns)]
    m = m[m.notna().sum(axis=1) >= min_metrics]
    if m.empty:
        return pd.Series(dtype=float)
    pct = m.rank(axis=0) / (m.notna().sum(axis=0) + 1)
    if method == "geometric":
        return np.exp(np.log(pct).mean(axis=1))
    if method == "arithmetic":
        return pct.mean(axis=1)
    raise ValueError(f"unknown method: {method}")


def quantile_sets(scores: pd.Series, n: int = 5) -> list[set[str]]:
    """Names split into n equal-count buckets by score, cheapest bucket last."""
    if scores.empty:
        return [set() for _ in range(n)]
    order = scores.rank(method="first")
    bucket = np.ceil(order / len(order) * n).astype(int).clip(1, n)
    return [set(scores.index[bucket == k]) for k in range(1, n + 1)]
