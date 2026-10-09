"""
Price-only entry filters on top of the z signal, production pair schedule,
raw pair book (no guards). Thresholds are fixed in advance, nothing is swept:
  jump      skip if either leg had an idiosyncratic daily move >= 4 sigma
            (vs equal-weight universe, trailing 60d vol) in the last 5 days
  reversion skip if the AR(1) half-life of the last 120d spread is > 2x the
            screening half-life, or the spread is not reverting at all
Baseline trades are bucketed by each score for diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

JUMP_MAX_Z = 4.0
HL_MAX_RATIO = 2.0
VARIANTS = {
    "baseline": {},
    "jump filter": {"jump_max_z": JUMP_MAX_Z},
    "reversion filter": {"hl_max_ratio": HL_MAX_RATIO},
    "jump + reversion": {"jump_max_z": JUMP_MAX_Z, "hl_max_ratio": HL_MAX_RATIO},
}
BUCKETS = {
    "jump_score": [0.0, 2.0, 3.0, 4.0, np.inf],
    "hl_ratio": [0.0, 0.5, 1.0, 2.0, np.inf],
}


def main() -> None:
    out_dir = ROOT / "output" / "signal_filter"
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

    rows, stat_rows, base = [], [], None
    for label, kwargs in VARIANTS.items():
        w, t = build_pair_book(prices, log, **kwargs, **common)
        t = trade_pnl(prices, t)
        t.to_csv(out_dir / f"trades_{label.replace(' ', '_').replace('+', 'and')}.csv", index=False)
        if base is None:
            base = t
        rows.append(evaluate(prices, w, label))
        for period, part in (("train", t[t["entry"] <= SPLIT]), ("test", t[t["entry"] > SPLIT])):
            stat_rows.append({"book": label, "period": period, **trade_stats(part)})

    base = base.copy()
    base["period"] = (base["entry"] > SPLIT).map({False: "train", True: "test"})
    bucket_rows = []
    for score, edges in BUCKETS.items():
        cut = pd.cut(base[score].replace(np.inf, 1e9), edges, right=False)
        for (b, per), part in base.groupby([cut, "period"], observed=True):
            bucket_rows.append({"score": score, "bucket": str(b), "period": per, **trade_stats(part)})

    table, stats, buckets = pd.DataFrame(rows), pd.DataFrame(stat_rows), pd.DataFrame(bucket_rows)
    table.to_csv(out_dir / "comparison.csv", index=False)
    stats.to_csv(out_dir / "trade_stats.csv", index=False)
    buckets.to_csv(out_dir / "buckets.csv", index=False)
    print(f"\n=== Book comparison (train <= {SPLIT.date()} < test, raw book) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Per-trade stats (bps of pair capital) ===")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== Baseline trades by score ===")
    print(buckets.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
