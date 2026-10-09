"""
F-score (10-point, net share issuance weighted 2) buy-and-hold vs the same
book plus a leveraged buyback/issuance event overlay. Rules fixed in advance.

  FS hold         equal-weight index members with F-score >= 8 (latest 10-K
                  filed by the rebalance date), rebalanced quarterly
  FS + events     FS hold (100%) plus an event book: a filing showing weighted
                  diluted shares down >= 2% YoY -> +2% long (cap +50%), up >= 2%
                  -> -3% short (cap -75%), for 63 trading days from the day after
                  the filing (public filing date stands in for news)
  events only     the event book alone
Costs: 15bps per unit turnover, 1.5%/yr borrow on shorts, 3%/yr financing on
long exposure above 100% of capital.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.fundamentals import FSCORE_ITEMS, fscore_asof, load_facts, share_events, ticker_ciks
from data.providers import load_price_frame
from data.universe_pit import _history, membership_matrix
from run_pair_cap_sweep import _load_prices
from strategies.issuance import event_overlay_weights
from strategies.quality_dip import basket_weights

OUT = ROOT / "output" / "fscore"
SPLIT = pd.Timestamp("2023-05-10")
MIN_SCORE = 8
ANN = config.ANNUALIZATION
TRADE_BPS = (config.COMMISSION_BPS + config.SLIPPAGE_BPS) / 10_000.0
BORROW_DAILY = config.BORROW_FEE_ANNUAL / ANN
FINANCING_DAILY = 0.03 / ANN

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def _cagr(r: pd.Series) -> float:
    return float((1.0 + r).prod() ** (ANN / max(len(r), 1)) - 1.0)


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else np.nan


def _mdd(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def book_returns(prices: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    rets = prices.pct_change().fillna(0.0)
    lag = w.shift(1).fillna(0.0)
    longs, shorts = lag.clip(lower=0.0).sum(axis=1), -lag.clip(upper=0.0).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = ((lag * rets).sum(axis=1) - turnover * TRADE_BPS - shorts * BORROW_DAILY
           - (longs - 1.0).clip(lower=0.0) * FINANCING_DAILY)
    return net, turnover, longs, shorts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    tickers = list(prices.columns)
    membership = membership_matrix(prices.index, tickers, index=config.PIT_INDEX)
    spy_px = load_price_frame(["SPY"], config.START, config.END)["SPY"].reindex(prices.index).ffill()
    month_ends = list(prices.index.to_series().groupby(prices.index.to_period("M")).max())[:-1]

    fs_path, ev_path = OUT / "fscore_by_filing.csv", OUT / "share_events.csv"
    if fs_path.exists() and ev_path.exists():
        fs = pd.read_csv(fs_path, parse_dates=["filed"])
        events = pd.read_csv(ev_path, parse_dates=["filed"])
    else:
        ciks = ticker_ciks(_history(config.START, prices.index[-1].date().isoformat(), config.PIT_INDEX))
        fs_rows, ev_rows = [], []
        for t in tickers:
            facts = load_facts(ciks[t]) if t in ciks else pd.DataFrame()
            if facts.empty:
                continue
            k_dates = sorted(facts.loc[facts["form"].isin(["10-K", "10-K/A"]), "filed"].unique())
            for d in k_dates:
                if pd.Timestamp(d) >= pd.Timestamp(config.START) - pd.Timedelta(days=470):
                    fs_rows.append({"ticker": t, "filed": pd.Timestamp(d), **fscore_asof(facts, pd.Timestamp(d))})
            ev = share_events(facts)
            ev["ticker"] = t
            ev_rows.append(ev)
        fs = pd.DataFrame(fs_rows)
        events = pd.concat(ev_rows, ignore_index=True)
        fs.to_csv(fs_path, index=False)
        events.to_csv(ev_path, index=False)

    fs = fs.dropna(subset=["score"]).sort_values("filed")
    sets, monthly = {}, []
    for d in month_ends:
        last = fs[(fs["filed"] <= d) & (fs["filed"] > d - pd.Timedelta(days=470))]
        last = last.drop_duplicates("ticker", keep="last").set_index("ticker")
        members = [t for t in tickers if bool(membership.at[d, t]) and np.isfinite(prices.at[d, t])]
        scores = last["score"].reindex(members)
        sets[d] = set(scores[scores >= MIN_SCORE].index)
        monthly.append({"date": d, "members": len(members), "scored": int(scores.notna().sum()),
                        "fs>=8": len(sets[d]), "mean_score": float(scores.mean())})
    monthly = pd.DataFrame(monthly)
    monthly.to_csv(OUT / "monthly_counts.csv", index=False)
    print("Names per month end:")
    print(monthly[["members", "scored", "fs>=8", "mean_score"]].describe().loc[["mean", "min", "max"]]
          .round(1).to_string())
    print("Score distribution (all 10-K filings):")
    print(fs["score"].value_counts().sort_index().to_string())

    start = month_ends[0]
    live = prices.index > start
    fs_hold = basket_weights(prices.index, tickers, sets, month_ends[::3], membership, prices)
    overlay = event_overlay_weights(prices.index, tickers, events, membership=membership, prices=prices)
    ew = basket_weights(prices.index, tickers,
                        {d: {t for t in tickers if bool(membership.at[d, t])} for d in month_ends},
                        month_ends, membership, prices)
    ev_live = events[(events["filed"] > start)]
    print(f"Events after {start.date()}: buyback (<= -2%) {int((ev_live.share_growth <= -0.02).sum())}, "
          f"issuance (>= +2%) {int((ev_live.share_growth >= 0.02).sum())}, filings {len(ev_live)}")

    spy = spy_px.pct_change().fillna(0.0)[live]
    rows, series = [], {}
    for label, w in (("FS >= 8 hold", fs_hold), ("FS >= 8 hold + event leverage", fs_hold + overlay),
                     ("event book only", overlay), ("equal weight members", ew)):
        w = w.copy()
        w.loc[~live] = 0.0
        net, turnover, longs, shorts = book_returns(prices, w)
        net, turnover, longs, shorts = net[live], turnover[live], longs[live], shorts[live]
        series[label] = net
        train, test = net[net.index <= SPLIT], net[net.index > SPLIT]
        rows.append({
            "book": label, "cagr": _cagr(net), "cagr_train": _cagr(train), "cagr_test": _cagr(test),
            "excess_vs_spy": _cagr(net) - _cagr(spy), "vol": float(net.std() * np.sqrt(ANN)),
            "sharpe": _sharpe(net), "sharpe_train": _sharpe(train), "sharpe_test": _sharpe(test),
            "mdd": _mdd(net), "beta": float(np.cov(net, spy)[0, 1] / np.var(spy, ddof=1)),
            "avg_long": float(longs.mean()), "avg_short": float(shorts.mean()),
            "max_long": float(longs.max()), "max_short": float(shorts.max()),
            "turnover": float(turnover.mean() * ANN),
        })
    series["SPY"] = spy
    train, test = spy[spy.index <= SPLIT], spy[spy.index > SPLIT]
    rows.append({"book": "SPY", "cagr": _cagr(spy), "cagr_train": _cagr(train), "cagr_test": _cagr(test),
                 "excess_vs_spy": 0.0, "vol": float(spy.std() * np.sqrt(ANN)), "sharpe": _sharpe(spy),
                 "sharpe_train": _sharpe(train), "sharpe_test": _sharpe(test), "mdd": _mdd(spy), "beta": 1.0,
                 "avg_long": 1.0, "avg_short": 0.0, "max_long": 1.0, "max_short": 0.0, "turnover": 0.0})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(series).to_csv(OUT / "daily_returns.csv")
    diff = series["FS >= 8 hold"] - series["equal weight members"]
    print(f"FS hold minus equal weight: {diff.mean() * ANN:.2%}/yr, "
          f"tracking error {diff.std() * np.sqrt(ANN):.2%}, t = {diff.mean() / diff.std() * np.sqrt(len(diff)):.2f}")
    yearly = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in series.items()})
    yearly.to_csv(OUT / "yearly.csv")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    for k, v in series.items():
        ax.plot(v.index, ((1 + v).cumprod() - 1) * 100, label=k, lw=2 if k.startswith("FS") else 1.2)
    ax.axvline(SPLIT, color="black", ls="--", lw=1, label=f"학습 | 보류 ({SPLIT.date()})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("F-스코어 보유 vs 이벤트 레버리지 (누적 수익률 %, 비용 포함)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    (yearly * 100).plot.bar(ax=axes[1], width=0.85)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title("연도별 수익률 (%)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fscore.png", dpi=110)

    fmt = lambda v: f"{v:.4f}"
    print(f"\n=== {prices.index[live][0].date()} ~ {prices.index[-1].date()} (train <= {SPLIT.date()} < test) ===")
    print(table.to_string(index=False, float_format=fmt))
    print("\n=== Calendar years ===")
    print(yearly.to_string(float_format=fmt))
    print("\nF-score item pass rates (all 10-K filings):")
    print(fs[list(FSCORE_ITEMS)].mean().round(3).to_string())


if __name__ == "__main__":
    main()
