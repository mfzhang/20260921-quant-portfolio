"""
Piotroski as designed: F-score separates winners from losers *inside* the cheap
(high book-to-market) group. Rules fixed in advance:

  value group     index members in the top third by book-to-market at each
                  quarter end (book = latest filed total equity, per weighted
                  diluted share, split-corrected; market = as-traded close);
                  negative book excluded. The paper uses the top fifth; a third
                  keeps enough names in a ~185-name universe.
  value high FS   value group with F-score >= 8 (10-point, net issuance x2)
  value low FS    value group with F-score <= 4
  value L/S       long value high FS (100%) / short value low FS (100%)
Equal weight, quarterly rebalance, same costs as run_fscore_experiment.py.
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
from data.fundamentals import book_to_market_table, ticker_ciks
from data.providers import load_cached_frame, load_price_frame
from data.universe_pit import _history, membership_matrix
from run_fscore_experiment import ANN, SPLIT, _cagr, _mdd, _sharpe, book_returns
from run_pair_cap_sweep import _load_prices
from strategies.quality_dip import basket_weights

OUT = ROOT / "output" / "piotroski"
FS_PATH = ROOT / "output" / "fscore" / "fscore_by_filing.csv"
VALUE_FRACTION = 1 / 3
HIGH, LOW = 8, 4

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def _tstat(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    tickers = list(prices.columns)
    raw = load_cached_frame(tickers, config.START, config.END, field="close")
    membership = membership_matrix(prices.index, tickers, index=config.PIT_INDEX)
    spy_px = load_price_frame(["SPY"], config.START, config.END)["SPY"].reindex(prices.index).ffill()
    month_ends = list(prices.index.to_series().groupby(prices.index.to_period("M")).max())[:-1]
    rebal = month_ends[::3]

    bm_path = OUT / "bm_quarterly.csv"
    if bm_path.exists():
        bm = pd.read_csv(bm_path, parse_dates=["date"])
    else:
        ciks = ticker_ciks(_history(config.START, prices.index[-1].date().isoformat(), config.PIT_INDEX))
        bm = book_to_market_table(rebal, tickers, prices, raw, ciks)
        bm.to_csv(bm_path, index=False)
    bm = bm.pivot(index="date", columns="ticker", values="bm")

    fs = pd.read_csv(FS_PATH, parse_dates=["filed"]).dropna(subset=["score"]).sort_values("filed")
    sets = {k: {} for k in ("all members", "value", "value high FS", "value low FS", "all high FS")}
    counts = []
    for d in rebal:
        members = [t for t in tickers if bool(membership.at[d, t]) and np.isfinite(prices.at[d, t])]
        last = fs[(fs["filed"] <= d) & (fs["filed"] > d - pd.Timedelta(days=470))]
        score = last.drop_duplicates("ticker", keep="last").set_index("ticker")["score"].reindex(members)
        b = bm.loc[d].reindex(members) if d in bm.index else pd.Series(np.nan, index=members)
        b = b[b > 0]
        value = set(b[b >= b.quantile(1 - VALUE_FRACTION)].index) if len(b) else set()
        sets["all members"][d] = set(members)
        sets["value"][d] = value
        sets["value high FS"][d] = {t for t in value if score.get(t, np.nan) >= HIGH}
        sets["value low FS"][d] = {t for t in value if score.get(t, np.nan) <= LOW}
        sets["all high FS"][d] = set(score[score >= HIGH].index)
        counts.append({"date": d, "members": len(members), "bm_known": len(b), "value": len(value),
                       "value_high": len(sets["value high FS"][d]), "value_low": len(sets["value low FS"][d]),
                       "value_mean_fs": float(score.reindex(list(value)).mean())})
    counts = pd.DataFrame(counts)
    counts.to_csv(OUT / "quarterly_counts.csv", index=False)
    print("Names per quarter end:")
    print(counts.drop(columns="date").describe().loc[["mean", "min", "max"]].round(1).to_string())

    start = rebal[0]
    live = prices.index > start
    weights = {k: basket_weights(prices.index, tickers, s, rebal, membership, prices) for k, s in sets.items()}
    weights["value L/S"] = weights["value high FS"] - weights["value low FS"]

    spy = spy_px.pct_change().fillna(0.0)[live]
    rows, series = [], {}
    order = ("value high FS", "value L/S", "value", "value low FS", "all high FS", "all members")
    for label in order:
        w = weights[label].copy()
        w.loc[~live] = 0.0
        net, turnover, longs, shorts = book_returns(prices, w)
        net, turnover, longs, shorts = net[live], turnover[live], longs[live], shorts[live]
        series[label] = net
        train, test = net[net.index <= SPLIT], net[net.index > SPLIT]
        rows.append({
            "book": label, "cagr": _cagr(net), "cagr_train": _cagr(train), "cagr_test": _cagr(test),
            "vol": float(net.std() * np.sqrt(ANN)), "sharpe": _sharpe(net),
            "sharpe_train": _sharpe(train), "sharpe_test": _sharpe(test), "mdd": _mdd(net),
            "beta": float(np.cov(net, spy)[0, 1] / np.var(spy, ddof=1)),
            "avg_long": float(longs.mean()), "avg_short": float(shorts.mean()),
            "turnover": float(turnover.mean() * ANN),
        })
    series["SPY"] = spy
    train, test = spy[spy.index <= SPLIT], spy[spy.index > SPLIT]
    rows.append({"book": "SPY", "cagr": _cagr(spy), "cagr_train": _cagr(train), "cagr_test": _cagr(test),
                 "vol": float(spy.std() * np.sqrt(ANN)), "sharpe": _sharpe(spy), "sharpe_train": _sharpe(train),
                 "sharpe_test": _sharpe(test), "mdd": _mdd(spy), "beta": 1.0, "avg_long": 1.0,
                 "avg_short": 0.0, "turnover": 0.0})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(series).to_csv(OUT / "daily_returns.csv")

    print(f"\n=== {prices.index[live][0].date()} ~ {prices.index[-1].date()} (train <= {SPLIT.date()} < test) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSpreads (annualized mean, t-stat; full / train / test):")
    for a, b in (("value high FS", "value"), ("value high FS", "value low FS"),
                 ("value", "all members"), ("value high FS", "all high FS")):
        diff = series[a] - series[b]
        parts = [f"{diff[m].mean() * ANN:+.2%} (t {_tstat(diff[m]):+.2f})"
                 for m in (diff.index == diff.index, diff.index <= SPLIT, diff.index > SPLIT)]
        print(f"  {a} - {b}: " + " / ".join(parts))

    yearly = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in series.items()})
    yearly.to_csv(OUT / "yearly.csv")
    print("\n=== Calendar years ===")
    print(yearly.to_string(float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    for k, v in series.items():
        ax.plot(v.index, ((1 + v).cumprod() - 1) * 100, label=k, lw=2 if k.startswith("value high") else 1.2)
    ax.axvline(SPLIT, color="black", ls="--", lw=1, label=f"학습 | 보류 ({SPLIT.date()})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("저평가 집단 안의 F-스코어 (누적 수익률 %, 비용 포함)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    (yearly * 100).plot.bar(ax=axes[1], width=0.85)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title("연도별 수익률 (%)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=8, ncol=4)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "piotroski.png", dpi=110)


if __name__ == "__main__":
    main()
