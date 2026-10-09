"""
Spread-trend entry filter for the pair book, production pair schedule, raw
book (no guards). The threshold is fixed in advance (R^2 >= 0.5 of a linear
trend over 2 x lookback, pointing the same way as the divergence); nothing is
swept. Baseline trades are also bucketed by trend score for diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.providers import normalize_ticker
from data.universe_pit import membership_matrix
from run_pair_book_experiment import SPLIT, evaluate, trade_pnl
from run_pair_cap_sweep import _load_prices
from run_sub_industry_experiment import trade_stats
from strategies.pair_book import build_pair_book

TREND_MAX_R2 = 0.5
BUCKETS = [-1.0, 0.0, 0.25, 0.5, 1.0001]


def main() -> None:
    out_dir = ROOT / "output" / "trend_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    membership = membership_matrix(prices.index, list(prices.columns), index=config.PIT_INDEX)
    log = pd.read_csv(ROOT / "output" / "rolling_pairs_log.csv", parse_dates=["refresh_date"])
    common = dict(
        lookback=config.STATARB_LOOKBACK,
        entry_z=config.STATARB_ENTRY_Z,
        exit_z=config.STATARB_EXIT_Z,
        stop_z=config.STATARB_STOP_Z,
        timeout_mult=config.STATARB_HL_TIMEOUT_MULT,
        membership=membership,
        formation_days=config.COINT_FORMATION_DAYS,
        refresh_days=config.COINT_REFRESH_DAYS,
        pair_budget=1.0 / config.COINT_MAX_PAIRS,
        spread=config.STATARB_SPREAD,
        fallback_pairs=[(normalize_ticker(a), normalize_ticker(b)) for a, b in config.PAIRS],
    )

    rows, stat_rows, trades = [], [], {}
    for label, r2 in (("baseline", None), (f"trend filter R2>={TREND_MAX_R2}", TREND_MAX_R2)):
        w, t = build_pair_book(prices, log, trend_max_r2=r2, **common)
        t = trade_pnl(prices, t)
        t.to_csv(out_dir / f"trades_{'baseline' if r2 is None else 'filtered'}.csv", index=False)
        trades[label] = t
        rows.append(evaluate(prices, w, label))
        for period, part in (("train", t[t["entry"] <= SPLIT]), ("test", t[t["entry"] > SPLIT])):
            stat_rows.append({"book": label, "period": period, **trade_stats(part)})

    base = trades["baseline"].copy()
    base["bucket"] = pd.cut(base["trend_score"], BUCKETS, right=False)
    base["period"] = (base["entry"] > SPLIT).map({False: "train", True: "test"})
    bucket_rows = [
        {"trend_score": str(b), "period": per, **trade_stats(part)}
        for (b, per), part in base.groupby(["bucket", "period"], observed=True)
    ]

    table, stats, buckets = pd.DataFrame(rows), pd.DataFrame(stat_rows), pd.DataFrame(bucket_rows)
    table.to_csv(out_dir / "comparison.csv", index=False)
    stats.to_csv(out_dir / "trade_stats.csv", index=False)
    buckets.to_csv(out_dir / "buckets.csv", index=False)
    fmt = lambda v: f"{v:.4f}"
    print(f"\n=== Book comparison (train <= {SPLIT.date()} < test, raw book) ===")
    print(table.to_string(index=False, float_format=fmt))
    print("\n=== Per-trade stats (bps of pair capital) ===")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== Baseline trades by trend score (signed R^2; + = drifting with divergence) ===")
    print(buckets.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
