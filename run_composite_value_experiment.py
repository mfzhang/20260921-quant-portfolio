"""
Composite value on S&P 400 mid caps. Rules fixed in advance:

  universe   S&P 400 members (point-in-time), priced sample of 200 of the
             2018-01 roster; banks/lenders/brokers/insurers (SEC SIC 6000-6499)
             excluded, REITs kept; at least 4 of the 5 yields available
  yields     E/P (TTM net income), B/M (latest equity), CF/P (TTM operating cash
             flow), S/P (TTM revenue) over market cap; EBIT/EV (TTM operating
             income over market cap + debt - cash). Filed by the rebalance date.
  score      percentile rank per yield, combined by geometric mean (primary);
             arithmetic mean reported for comparison only
  book       top 20% by score, equal weight, quarterly rebalance; the five score
             quintiles are reported to check the ordering
Costs as in run_fscore_experiment.py. Benchmarks: equal weight of the eligible
names, equal weight of all priced members, IJH (S&P 400 ETF).
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
from data.cik_lookup import fill_ciks, sic_codes
from data.fundamentals import VALUE_METRICS, ticker_ciks, value_metrics_table
from data.providers import load_cached_frame
from data.universe_pit import _history, membership_matrix, normalize_ticker, pit_price_tickers
from run_fscore_experiment import ANN, SPLIT, _cagr, _mdd, _sharpe, book_returns
from strategies.composite_value import composite_scores, quantile_sets
from strategies.quality_dip import basket_weights

OUT = ROOT / "output" / "composite_value"
INDEX = "sp400"
SAMPLE = 200
BENCH = "IJH"
FIN_SIC = (6000, 6499)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def _tstat(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else np.nan


def _stats(label: str, net: pd.Series, bench: pd.Series, extra: dict | None = None) -> dict:
    train, test = net[net.index <= SPLIT], net[net.index > SPLIT]
    return {"book": label, "cagr": _cagr(net), "cagr_train": _cagr(train), "cagr_test": _cagr(test),
            "vol": float(net.std() * np.sqrt(ANN)), "sharpe": _sharpe(net), "sharpe_train": _sharpe(train),
            "sharpe_test": _sharpe(test), "mdd": _mdd(net),
            "beta": float(np.cov(net, bench)[0, 1] / np.var(bench, ddof=1)), **(extra or {})}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tickers = pit_price_tickers(config.START, index=INDEX, max_names=SAMPLE, include_unknown_sector=True)
    prices = load_cached_frame(tickers + [BENCH], config.START, config.END)
    raw = load_cached_frame(tickers, config.START, config.END, field="close")
    bench_px = prices.pop(BENCH)
    prices = prices.reindex(bench_px.index).ffill(limit=5)
    tickers = list(prices.columns)
    print(f"Priced {len(tickers)}/{SAMPLE} sampled names; as-traded closes for {raw.shape[1]}")
    membership = membership_matrix(prices.index, tickers, index=INDEX)
    month_ends = list(prices.index.to_series().groupby(prices.index.to_period("M")).max())[:-1]
    rebal = month_ends[::3]

    hist = _history(config.START, prices.index[-1].date().isoformat(), INDEX).copy()
    hist["t"] = hist["ticker"].map(normalize_ticker)
    names = hist.sort_values("as_of").drop_duplicates("t", keep="last").set_index("t")["name"].to_dict()
    ciks = fill_ciks({t: names.get(t) for t in tickers}, ticker_ciks(hist))
    sic = sic_codes([ciks[t] for t in tickers if t in ciks])
    financial = {t for t in tickers if t in ciks and FIN_SIC[0] <= (sic.get(ciks[t]) or 0) <= FIN_SIC[1]}
    print(f"CIK for {sum(t in ciks for t in tickers)}/{len(tickers)}; financials excluded: {len(financial)}")

    path = OUT / "metrics_quarterly.csv"
    if path.exists():
        metrics = pd.read_csv(path, parse_dates=["date"])
    else:
        metrics = value_metrics_table(rebal, tickers, prices, raw, ciks)
        metrics.to_csv(path, index=False)

    books = {f"Q{k}": {} for k in range(1, 6)}
    books.update({"arithmetic top 20%": {}, "eligible EW": {}, "all members EW": {}})
    counts = []
    for d in rebal:
        members = [t for t in tickers if bool(membership.at[d, t]) and np.isfinite(prices.at[d, t])]
        m = metrics[metrics["date"] == d].set_index("ticker").reindex(members)
        m = m[~m.index.isin(financial)]
        geo = composite_scores(m, VALUE_METRICS, method="geometric")
        ari = composite_scores(m, VALUE_METRICS, method="arithmetic")
        for k, s in enumerate(quantile_sets(geo), 1):
            books[f"Q{k}"][d] = s
        books["arithmetic top 20%"][d] = quantile_sets(ari)[-1]
        books["eligible EW"][d] = set(geo.index)
        books["all members EW"][d] = set(members)
        counts.append({"date": d, "members": len(members), "non_fin": len(m), "eligible": len(geo),
                       "top20": len(books["Q5"][d]),
                       "overlap_geo_ari": len(books["Q5"][d] & books["arithmetic top 20%"][d]),
                       **{f"has_{c}": int(m[c].notna().sum()) for c in VALUE_METRICS}})
    counts = pd.DataFrame(counts)
    counts.to_csv(OUT / "quarterly_counts.csv", index=False)
    print("Names per quarter end:")
    print(counts.drop(columns="date").describe().loc[["mean", "min", "max"]].round(1).to_string())

    live = prices.index > rebal[0]
    bench = bench_px.pct_change().fillna(0.0)[live]
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    rows, series = [], {}
    order = ["Q5", "arithmetic top 20%", "Q4", "Q3", "Q2", "Q1", "eligible EW", "all members EW"]
    for label in order:
        w = basket_weights(prices.index, tickers, books[label], rebal, membership, prices)
        w.loc[~live] = 0.0
        net, turnover, longs, _ = book_returns(prices, w)
        # book_returns charges costs on every daily weight change, drift included;
        # trade_turnover counts only actual trades (target vs drifted weights)
        lag = w.shift(1).fillna(0.0)
        gross = (lag * rets).sum(axis=1)
        drifted = (lag * (1.0 + rets)).div(1.0 + gross, axis=0)
        trades = (w - drifted).abs().sum(axis=1)
        net, turnover, longs = net[live], turnover[live], longs[live]
        series[label] = net
        rows.append(_stats(label, net, bench, {"cagr_gross": _cagr(gross[live]), "avg_long": float(longs.mean()),
                                               "turnover": float(turnover.mean() * ANN),
                                               "trade_turnover": float(trades[live].mean() * ANN)}))
    series[BENCH] = bench
    rows.append(_stats(BENCH, bench, bench, {"cagr_gross": _cagr(bench), "avg_long": 1.0, "turnover": 0.0,
                                             "trade_turnover": 0.0}))
    table = pd.DataFrame(rows).replace({"book": {"Q5": "Q5 = geometric top 20% (primary)"}})
    table.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(series).to_csv(OUT / "daily_returns.csv")

    print(f"\n=== {prices.index[live][0].date()} ~ {prices.index[-1].date()} (train <= {SPLIT.date()} < test) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSpreads (annualized mean, t-stat; full / train / test):")
    spreads = []
    for a, b in (("Q5", "eligible EW"), ("Q5", "Q1"), ("Q5", BENCH), ("arithmetic top 20%", "eligible EW"),
                 ("eligible EW", "all members EW")):
        diff = series[a] - series[b]
        masks = {"full": diff.index == diff.index, "train": diff.index <= SPLIT, "test": diff.index > SPLIT}
        spreads += [{"spread": f"{a} - {b}", "period": k, "ann_mean": diff[m].mean() * ANN, "t": _tstat(diff[m])}
                    for k, m in masks.items()]
        print(f"  {a} - {b}: " + " / ".join(f"{s['ann_mean']:+.2%} (t {s['t']:+.2f})" for s in spreads[-3:]))
    pd.DataFrame(spreads).to_csv(OUT / "spreads.csv", index=False)

    rel = (1 + series["Q5"]).cumprod() / (1 + series["eligible EW"]).cumprod()
    under = rel / rel.cummax() - 1
    spell = (under < 0).astype(int).groupby((under == 0).cumsum()).sum().max()
    print(f"\nQ5 vs eligible EW: worst relative drawdown {under.min():.1%}, "
          f"longest stretch below prior high {spell} trading days (~{spell / ANN:.1f} years)")

    yearly = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in series.items()})
    yearly.to_csv(OUT / "yearly.csv")
    print("\n=== Calendar years ===")
    print(yearly.to_string(float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    for k in ("Q5", "arithmetic top 20%", "Q1", "eligible EW", BENCH):
        v = series[k]
        ax.plot(v.index, ((1 + v).cumprod() - 1) * 100, label="Q5 기하평균 상위 20%" if k == "Q5" else k,
                lw=2 if k == "Q5" else 1.2)
    ax.axvline(SPLIT, color="black", ls="--", lw=1, label=f"학습 | 보류 ({SPLIT.date()})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("S&P400 복합 가치 점수 (누적 수익률 %, 비용 포함)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    q = pd.Series({k: _cagr(series[k]) * 100 for k in ("Q1", "Q2", "Q3", "Q4", "Q5")})
    q.plot.bar(ax=axes[1], color=["#bbb", "#999", "#777", "#555", "#c33"])
    axes[1].axhline(_cagr(series["eligible EW"]) * 100, color="blue", ls="--", lw=1, label="eligible EW")
    axes[1].set_title("점수 5분위별 CAGR (%) — Q1 가장 비쌈, Q5 가장 쌈")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "composite_value.png", dpi=110)


if __name__ == "__main__":
    main()
