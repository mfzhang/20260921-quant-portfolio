"""
Compare the legacy pair weights with the event-driven pair book
(strategies/pair_book.py) on the production pair schedule, raw book (no guards).

Sector exclusion is decided on the train window only (gross P&L < 0 with at
least MIN_SECTOR_TRADES trades), then scored on the held-out window.
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
from run_pair_cap_sweep import _load_prices
from strategies.pair_book import build_pair_book
from strategies.rolling_pairs import weights_from_pair_log

SPLIT = pd.Timestamp("2023-05-10")
MIN_SECTOR_TRADES = 5
ANN = config.ANNUALIZATION
TRADE_BPS = (config.COMMISSION_BPS + config.SLIPPAGE_BPS) / 10_000.0
BORROW_DAILY = config.BORROW_FEE_ANNUAL / ANN


def _cagr(r: pd.Series) -> float:
    return float((1.0 + r.fillna(0.0)).prod() ** (ANN / max(len(r), 1)) - 1.0)


def _mdd(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def evaluate(prices: pd.DataFrame, w: pd.DataFrame, label: str) -> dict:
    rets = prices.pct_change().fillna(0.0)
    lag = w.shift(1).fillna(0.0)
    gross = (lag * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * TRADE_BPS - lag.clip(upper=0.0).abs().sum(axis=1) * BORROW_DAILY
    start = prices.index[config.COINT_FORMATION_DAYS]
    live = gross.index >= start
    train = live & (gross.index <= SPLIT)
    test = gross.index > SPLIT
    expo = w.abs().sum(axis=1)
    return {
        "book": label,
        "gross_full": _cagr(gross[live]),
        "net_full": _cagr(net[live]),
        "net_train": _cagr(net[train]),
        "net_test": _cagr(net[test]),
        "gross_test": _cagr(gross[test]),
        "mdd_full": _mdd(net[live]),
        "turnover": float(turnover[live].mean() * ANN),
        "avg_gross": float(expo[live].mean()),
        "net_expo": float(w.sum(axis=1)[live].abs().mean()),
        "net_per_gross_full": _cagr(net[live]) / max(float(expo[live].mean()), 1e-9),
    }


def trade_pnl(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().fillna(0.0)
    out = trades.copy()
    gross = []
    for tr in trades.itertuples(index=False):
        span = rets.loc[(rets.index > tr.entry) & (rets.index <= tr.exit)]
        gross.append(float((tr.wy * span[tr.y] + tr.wx * span[tr.x]).sum()))
    out["gross_pnl"] = gross
    out["cost"] = 2.0 * (out["wy"].abs() + out["wx"].abs()) * TRADE_BPS
    out["net_pnl"] = out["gross_pnl"] - out["cost"]
    return out


def sector_table(trades: pd.DataFrame) -> pd.DataFrame:
    return trades.groupby("sector").agg(
        n=("net_pnl", "size"),
        gross=("gross_pnl", "sum"),
        net=("net_pnl", "sum"),
        hit=("gross_pnl", lambda s: (s > 0).mean()),
        days=("days", "mean"),
    ).sort_values("gross")


def main() -> None:
    out_dir = ROOT / "output" / "pair_book"
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    membership = membership_matrix(prices.index, list(prices.columns), index=config.PIT_INDEX)
    log = pd.read_csv(ROOT / "output" / "rolling_pairs_log.csv", parse_dates=["refresh_date"])
    fallback = [(normalize_ticker(a), normalize_ticker(b)) for a, b in config.PAIRS]
    common = dict(
        lookback=config.STATARB_LOOKBACK,
        entry_z=config.STATARB_ENTRY_Z,
        exit_z=config.STATARB_EXIT_Z,
        stop_z=config.STATARB_STOP_Z,
        membership=membership,
        formation_days=config.COINT_FORMATION_DAYS,
        refresh_days=config.COINT_REFRESH_DAYS,
        fallback_pairs=fallback,
    )
    budget = 1.0 / config.COINT_MAX_PAIRS

    rows = []
    legacy = weights_from_pair_log(prices, log, timeout_mult=config.STATARB_HL_TIMEOUT_MULT, **common)
    rows.append(evaluate(prices, legacy, "legacy (production)"))

    books = {}
    for label, spread, tmult in [
        ("pair book, price spread, timeout 2xHL", "price", config.STATARB_HL_TIMEOUT_MULT),
        ("pair book, log spread, timeout 2xHL", "log", config.STATARB_HL_TIMEOUT_MULT),
        ("pair book, log spread, no timeout", "log", 0.0),
    ]:
        w, trades = build_pair_book(
            prices, log, timeout_mult=tmult, spread=spread, pair_budget=budget, **common
        )
        books[label] = (spread, tmult, trade_pnl(prices, trades))
        rows.append(evaluate(prices, w, label))

    print("\n=== Train-window sector P&L (decides exclusions) ===")
    excluded: dict[str, tuple[str, ...]] = {}
    for label in ("pair book, log spread, timeout 2xHL", "pair book, log spread, no timeout"):
        spread, tmult, t = books[label]
        t.to_csv(out_dir / f"trades_{'timeout' if tmult > 0 else 'no_timeout'}.csv", index=False)
        st = sector_table(t[t["exit"] <= SPLIT])
        drop = tuple(st[(st["gross"] < 0) & (st["n"] >= MIN_SECTOR_TRADES)].index)
        excluded[label] = drop
        print(f"\n{label}:")
        print(st.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"-> exclude: {list(drop)}")
        w, _ = build_pair_book(
            prices, log, timeout_mult=tmult, spread=spread, pair_budget=budget,
            excluded_sectors=drop, **common,
        )
        rows.append(evaluate(prices, w, f"{label} + sector exclusion"))

        print("Held-out sector P&L (for reference, not used to choose):")
        print(sector_table(t[t["entry"] > SPLIT]).to_string(float_format=lambda v: f"{v:.4f}"))

    for label in ("pair book, log spread, timeout 2xHL", "pair book, log spread, no timeout"):
        t = books[label][2]
        print(f"\n{label} exit reasons (all trades):")
        print(t.groupby("reason").agg(
            n=("net_pnl", "size"), gross=("gross_pnl", "sum"), net=("net_pnl", "sum"),
            hit=("gross_pnl", lambda s: (s > 0).mean()), days=("days", "mean"),
        ).to_string(float_format=lambda v: f"{v:.4f}"))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print(f"\n=== Comparison (train <= {SPLIT.date()} < test) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
