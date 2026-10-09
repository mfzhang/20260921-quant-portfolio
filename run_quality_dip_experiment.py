"""
Quality dip-buying vs simple benchmarks, point-in-time universe, costs included.
Rules are fixed in advance (strategies/quality_dip.py defaults); nothing swept.

  quality dip      long-only, 10 slots x 10%: buy top-third quality names after a
                   5-day drop <= -2 sigma, sell at the 20-day average or after 20 days
  quality basket   equal-weight top-third quality names, rebalanced quarterly
  equal weight     equal-weight all priced index members, rebalanced monthly
  SPY              S&P 500 ETF buy and hold
Quality = mean percentile rank of ROE, debt ratio and earnings yield from SEC
filings dated on or before each month end.
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
from data.fundamentals import ticker_ciks, value_screen_table
from data.providers import load_cached_frame, load_price_frame
from data.universe_pit import _history, membership_matrix
from run_pair_cap_sweep import _load_prices
from strategies.quality_dip import basket_weights, build_quality_dip, quality_scores, quality_sets

OUT = ROOT / "output" / "quality_dip"
SPLIT = pd.Timestamp("2023-05-10")
TOP_FRAC = 1 / 3
ANN = config.ANNUALIZATION
TRADE_BPS = (config.COMMISSION_BPS + config.SLIPPAGE_BPS) / 10_000.0


def _cagr(r: pd.Series) -> float:
    return float((1.0 + r).prod() ** (ANN / max(len(r), 1)) - 1.0)


def _mdd(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def book_returns(prices: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    rets = prices.pct_change().fillna(0.0)
    lag = w.shift(1).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = (lag * rets).sum(axis=1) - turnover * TRADE_BPS
    return net, turnover, w.sum(axis=1)


def metrics(label: str, r: pd.Series, spy: pd.Series, turnover=None, invested=None) -> dict:
    train, test = r[r.index <= SPLIT], r[r.index > SPLIT]
    beta = float(np.cov(r, spy)[0, 1] / np.var(spy, ddof=1))
    return {
        "book": label,
        "cagr": _cagr(r),
        "cagr_train": _cagr(train),
        "cagr_test": _cagr(test),
        "excess_vs_spy": _cagr(r) - _cagr(spy),
        "vol": float(r.std() * np.sqrt(ANN)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else np.nan,
        "sharpe_train": float(train.mean() / train.std() * np.sqrt(ANN)),
        "sharpe_test": float(test.mean() / test.std() * np.sqrt(ANN)),
        "mdd": _mdd(r),
        "beta": beta,
        "turnover": float(turnover.mean() * ANN) if turnover is not None else 0.0,
        "avg_invested": float(invested.mean()) if invested is not None else 1.0,
    }


def trade_table(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().fillna(0.0)
    out = trades.copy()
    out["ret"] = [
        float((1.0 + rets.loc[(rets.index > tr.entry) & (rets.index <= tr.exit), tr.ticker]).prod() - 1.0)
        for tr in trades.itertuples(index=False)
    ]
    out["net_ret"] = out["ret"] - 2 * TRADE_BPS
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    tickers = list(prices.columns)
    raw = load_cached_frame(tickers, config.START, config.END, field="close")
    membership = membership_matrix(prices.index, tickers, index=config.PIT_INDEX)
    spy_px = load_price_frame(["SPY"], config.START, config.END)["SPY"].reindex(prices.index).ffill()

    month_ends = list(prices.index.to_series().groupby(prices.index.to_period("M")).max())
    table_path = OUT / "fundamentals_monthly.csv"
    if table_path.exists():
        table = pd.read_csv(table_path, parse_dates=["date"])
    else:
        ciks = ticker_ciks(_history(config.START, prices.index[-1].date().isoformat(), config.PIT_INDEX))
        table = value_screen_table(month_ends, tickers, prices, raw, ciks)
        table.to_csv(table_path, index=False)
    scored = quality_scores(table, membership)
    scored.to_csv(OUT / "quality_scores.csv", index=False)
    sets = quality_sets(scored, TOP_FRAC)
    all_members = {d: set(g["ticker"]) for d, g in scored.groupby("date")}
    cover = scored.groupby("date").agg(
        members=("ticker", "size"),
        no_roe=("roe", lambda s: int(s.isna().sum())),
        no_debt=("debt_ratio", lambda s: int(s.isna().sum())),
        no_ey=("ey", lambda s: int(s.isna().sum())),
    )
    print("Fundamental coverage (members per month end; missing counts):")
    print(cover.describe().loc[["mean", "min", "max"]].round(1).to_string())

    start = prices.index[60]
    live = prices.index >= start

    w_dip, trades = build_quality_dip(prices, sets, membership=membership)
    w_basket = basket_weights(prices.index, tickers, sets, month_ends[::3], membership, prices)
    w_ew = basket_weights(prices.index, tickers, all_members, month_ends, membership, prices)
    for w in (w_dip, w_basket, w_ew):
        w.loc[~live] = 0.0

    spy = spy_px.pct_change().fillna(0.0)[live]
    rows = []
    for label, w in (("quality dip (B)", w_dip), ("quality basket (quarterly)", w_basket),
                     ("equal weight members", w_ew)):
        net, turnover, invested = book_returns(prices, w)
        rows.append(metrics(label, net[live], spy, turnover[live], invested[live]))
        (1.0 + net[live]).cumprod().to_csv(OUT / f"equity_{label.split(' (')[0].replace(' ', '_')}.csv")
    rows.append(metrics("SPY buy & hold", spy, spy))
    (1.0 + spy).cumprod().to_csv(OUT / "equity_SPY.csv")

    table_out = pd.DataFrame(rows)
    table_out.to_csv(OUT / "comparison.csv", index=False)
    t = trade_table(prices, trades)
    t.to_csv(OUT / "trades.csv", index=False)
    t["period"] = np.where(t["entry"] <= SPLIT, "train", "test")
    tstats = t.groupby("period").agg(
        trades=("net_ret", "size"), hit=("net_ret", lambda s: (s > 0).mean()),
        avg_net_ret=("net_ret", "mean"), se=("net_ret", lambda s: s.std() / np.sqrt(len(s))),
        days=("days", "mean"),
    )
    reasons = t.groupby("reason").agg(n=("net_ret", "size"), avg_net_ret=("net_ret", "mean"))

    print(f"\n=== {start.date()} ~ {prices.index[-1].date()} (train <= {SPLIT.date()} < test), costs included ===")
    print(table_out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Quality dip trades (net of 30bps round trip) ===")
    print(tstats.to_string(float_format=lambda v: f"{v:.4f}"))
    print(reasons.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
