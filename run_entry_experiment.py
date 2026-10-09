"""
Entry timing and deviation strength on the pair book (raw, no guards).

  1. entry_mode: "cross" (first touch of entry_z) vs "turn" (first day |z| falls)
     vs "recross" (|z| back below entry_z)
  2. trades bucketed by the peak |z| reached before entry, train vs held-out
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
from strategies.pair_book import build_pair_book

MODES = ("cross", "turn", "recross")
BUCKETS = [2.5, 3.0, 3.5, 4.0]


def bucket_table(trades: pd.DataFrame, budget: float) -> pd.DataFrame:
    t = trades.copy()
    t["peak_bucket"] = pd.cut(t["peak_abs_z"], BUCKETS, right=False)
    t["period"] = t["entry"].map(lambda d: "train" if d <= SPLIT else "test")
    t["gross_bps"] = t["gross_pnl"] / budget * 1e4
    t["net_bps"] = t["net_pnl"] / budget * 1e4
    t["is_stop"] = t["reason"] == "stop"
    return t.groupby(["period", "peak_bucket"], observed=True).agg(
        n=("net_bps", "size"),
        gross_bps=("gross_bps", "mean"),
        net_bps=("net_bps", "mean"),
        hit=("gross_pnl", lambda s: (s > 0).mean()),
        stop_rate=("is_stop", "mean"),
        days=("days", "mean"),
    ).reindex(["train", "test"], level=0)


def main() -> None:
    out_dir = ROOT / "output" / "entry_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    membership = membership_matrix(prices.index, list(prices.columns), index=config.PIT_INDEX)
    log = pd.read_csv(ROOT / "output" / "rolling_pairs_log.csv", parse_dates=["refresh_date"])
    budget = 1.0 / config.COINT_MAX_PAIRS
    common = dict(
        lookback=config.STATARB_LOOKBACK,
        entry_z=config.STATARB_ENTRY_Z,
        exit_z=config.STATARB_EXIT_Z,
        stop_z=config.STATARB_STOP_Z,
        timeout_mult=config.STATARB_HL_TIMEOUT_MULT,
        membership=membership,
        formation_days=config.COINT_FORMATION_DAYS,
        refresh_days=config.COINT_REFRESH_DAYS,
        pair_budget=budget,
        spread=config.STATARB_SPREAD,
        fallback_pairs=[(normalize_ticker(a), normalize_ticker(b)) for a, b in config.PAIRS],
    )

    rows = []
    for mode in MODES:
        w, trades = build_pair_book(prices, log, entry_mode=mode, **common)
        trades = trade_pnl(prices, trades)
        trades.to_csv(out_dir / f"trades_{mode}.csv", index=False)
        row = evaluate(prices, w, mode)
        row["trades"] = len(trades)
        row["stop_rate"] = float((trades["reason"] == "stop").mean())
        row["hit"] = float((trades["gross_pnl"] > 0).mean())
        row["gross_bps_per_trade"] = float(trades["gross_pnl"].mean() / budget * 1e4)
        rows.append(row)

        print(f"\n=== {mode}: by peak |z| before entry (per trade, bps of pair capital) ===")
        print(bucket_table(trades, budget).to_string(float_format=lambda v: f"{v:.2f}"))
        print(trades.groupby("reason").agg(
            n=("net_pnl", "size"), gross=("gross_pnl", "sum"),
            hit=("gross_pnl", lambda s: (s > 0).mean()), days=("days", "mean"),
        ).to_string(float_format=lambda v: f"{v:.4f}"))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "comparison.csv", index=False)
    cols = ["book", "trades", "hit", "stop_rate", "gross_bps_per_trade", "gross_full", "net_full",
            "net_train", "net_test", "mdd_full", "turnover", "avg_gross", "net_per_gross_full"]
    print(f"\n=== Entry timing (train <= {SPLIT.date()} < test) ===")
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
