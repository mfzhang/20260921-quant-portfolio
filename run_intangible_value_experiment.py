"""
Intangible-adjusted composite value vs the standard composite. Rules fixed in
advance (no sweeps):

  capital    knowledge K_R(t) = 0.85 K_R(t-1) + R&D(t); organization
             K_O(t) = 0.80 K_O(t-1) + 0.30 SG&A(t). Fiscal-year flows from
             10-K / 10-K/A filed by the rebalance date only; each item starts at
             the first fiscal year it is reported with K0 = X0 / (g + delta),
             g = 10% (steady-state perpetual inventory); later missing years
             invest 0, never-reported items give K = 0. SG&A = the combined tag,
             else selling/marketing + G&A tagged separately; if SG&A + R&D +
             cost of revenue overshoots revenue - operating income by > 0.5% of
             revenue while SG&A + cost of revenue does not, SG&A already holds
             R&D and R&D is subtracted from it (no double count)
  yields     standard: E/P, B/M, CF/P, S/P, EBIT/EV (value_metrics_asof)
             adjusted: B/M -> (book + K_R + K_O) / mcap; E/P and EBIT/EV add back
             [R&D - 0.15 K_R(prev)] + [0.30 SG&A - 0.20 K_O(prev)]; CF/P, S/P
             unchanged; no tax adjustment
  score      percentile rank per yield, geometric mean, names with >= 4 of 5
             yields under both scores (same eligible set for both)
  book       top 20% equal weight, quarterly rebalance; quintiles reported
  universe   sp500 (primary): the 185 priced S&P 500 names, benchmark SPY;
             sp400 (secondary): run_composite_value_experiment.py sample, IJH.
             Banks/lenders/brokers/insurers (SEC SIC 6000-6499) excluded
Costs as in run_fscore_experiment.py.
"""

from __future__ import annotations

import argparse
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
from data.fundamentals import ADJ_VALUE_METRICS, VALUE_METRICS, intangible_value_table, ticker_ciks
from data.providers import load_cached_frame, load_price_frame
from data.universe_pit import _history, constituents_asof, membership_matrix, normalize_ticker, pit_price_tickers
from run_composite_value_experiment import FIN_SIC, _stats, _tstat
from run_fscore_experiment import ANN, SPLIT, _cagr, book_returns
from run_pair_cap_sweep import _load_prices
from strategies.composite_value import composite_scores, quantile_sets
from strategies.quality_dip import basket_weights

OUT = ROOT / "output" / "intangible_value"
TECH = ("AAPL", "MSFT", "GOOGL", "GOOG", "META", "FB", "NVDA", "INTC", "ORCL")
IT = "Information Technology"


def _universe(index: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str]:
    if index == "sp500":
        prices = _load_prices()
        bench = load_price_frame(["SPY"], config.START, config.END)["SPY"].reindex(prices.index).ffill()
        name = "SPY"
    else:
        tickers = pit_price_tickers(config.START, index=index, max_names=200, include_unknown_sector=True)
        prices = load_cached_frame(tickers + ["IJH"], config.START, config.END)
        bench = prices.pop("IJH")
        prices = prices.reindex(bench.index).ffill(limit=5)
        name = "IJH"
    raw = load_cached_frame(list(prices.columns), config.START, config.END, field="close")
    return prices, raw, bench, name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", choices=("sp500", "sp400"), default="sp500")
    index = ap.parse_args().index
    OUT.mkdir(parents=True, exist_ok=True)
    prices, raw, bench_px, bench_name = _universe(index)
    tickers = list(prices.columns)
    print(f"[{index}] priced {len(tickers)} names; as-traded closes for {raw.shape[1]}")
    membership = membership_matrix(prices.index, tickers, index=index)
    month_ends = list(prices.index.to_series().groupby(prices.index.to_period("M")).max())[:-1]
    rebal = month_ends[::3]

    hist = _history(config.START, prices.index[-1].date().isoformat(), index).copy()
    hist["t"] = hist["ticker"].map(normalize_ticker)
    names = hist.sort_values("as_of").drop_duplicates("t", keep="last").set_index("t")["name"].to_dict()
    ciks = fill_ciks({t: names.get(t) for t in tickers}, ticker_ciks(hist))
    sic = sic_codes([ciks[t] for t in tickers if t in ciks])
    financial = {t for t in tickers if t in ciks and FIN_SIC[0] <= (sic.get(ciks[t]) or 0) <= FIN_SIC[1]}
    print(f"CIK for {sum(t in ciks for t in tickers)}/{len(tickers)}; financials excluded: {len(financial)}")

    path = OUT / f"metrics_{index}.csv"
    if path.exists():
        metrics = pd.read_csv(path, parse_dates=["date", "fy_end"])
    else:
        metrics = intangible_value_table(rebal, tickers, prices, raw, ciks)
        metrics.to_csv(path, index=False)

    std_books = {f"Q{k}": {} for k in range(1, 6)}
    adj_books = {f"Q{k}": {} for k in range(1, 6)}
    ew, all_ew, counts, sector_rows, scored = {}, {}, [], [], []
    for d in rebal:
        members = [t for t in tickers if bool(membership.at[d, t]) and np.isfinite(prices.at[d, t])]
        m = metrics[metrics["date"] == d].set_index("ticker").reindex(members)
        m = m[~m.index.isin(financial)]
        ok_std = m[list(VALUE_METRICS)].notna().sum(axis=1) >= 4
        ok_adj = m[list(ADJ_VALUE_METRICS)].notna().sum(axis=1) >= 4
        e = m[ok_std & ok_adj]
        std = composite_scores(e, VALUE_METRICS, method="geometric")
        adj = composite_scores(e, ADJ_VALUE_METRICS, method="geometric")
        for k, (s, a) in enumerate(zip(quantile_sets(std), quantile_sets(adj)), 1):
            std_books[f"Q{k}"][d], adj_books[f"Q{k}"][d] = s, a
        ew[d], all_ew[d] = set(e.index), set(members)
        sectors = constituents_asof(d, index=index)
        sectors = dict(zip(sectors["ticker"].map(normalize_ticker), sectors["gics_sector"].fillna("Unknown")))
        sec = pd.Series({t: sectors.get(t, "Unknown") for t in e.index}, dtype=object)
        for label, names_ in (("standard Q5", std_books["Q5"][d]), ("adjusted Q5", adj_books["Q5"][d]),
                              ("eligible", set(e.index))):
            share = sec[list(names_)].value_counts(normalize=True)
            sector_rows += [{"date": d, "book": label, "sector": k, "share": v} for k, v in share.items()]
        known = sec[sec != "Unknown"]
        counts.append({
            "date": d, "members": len(members), "non_fin": len(m), "eligible": len(e),
            "std_only": int((ok_std & ~ok_adj).sum()), "adj_only": int((ok_adj & ~ok_std).sum()),
            "top20": len(adj_books["Q5"][d]), "overlap_q5": len(std_books["Q5"][d] & adj_books["Q5"][d]),
            "has_rd": int(e["has_rd"].sum()), "has_sga": int(e["has_sga"].sum()),
            "rd_or_sga": int(((e["has_rd"] > 0) | (e["has_sga"] > 0)).sum()),
            "sga_incl_rd": int(e["sga_incl_rd"].sum()), "sector_known": len(known),
            "it_eligible": int((known == IT).sum()),
            "it_std_q5": int((sec[list(std_books["Q5"][d])] == IT).sum()),
            "it_adj_q5": int((sec[list(adj_books["Q5"][d])] == IT).sum()),
        })
        pct_s = std.rank() / len(std)
        pct_a = adj.rank() / len(adj)
        for t in e.index:
            scored.append({"date": d, "ticker": t, "sector": sec[t], "std_pct": pct_s[t], "adj_pct": pct_a[t]})
    counts = pd.DataFrame(counts)
    counts.to_csv(OUT / f"quarterly_counts_{index}.csv", index=False)
    scored = pd.DataFrame(scored)
    print("Names per quarter end:")
    print(counts.drop(columns="date").describe().loc[["mean", "min", "max"]].round(1).to_string())
    tot = counts["eligible"].sum()
    print(f"Coverage over eligible name-quarters: R&D reported {counts['has_rd'].sum() / tot:.1%}, "
          f"SG&A reported {counts['has_sga'].sum() / tot:.1%}, either {counts['rd_or_sga'].sum() / tot:.1%}, "
          f"SG&A taken to include R&D {counts['sga_incl_rd'].sum() / tot:.1%}")
    flagged = metrics[metrics["sga_incl_rd"] == 1]["ticker"].unique()
    print(f"Names ever flagged SG&A-includes-R&D ({len(flagged)}): {sorted(flagged)}")
    print(f"Q5 overlap standard vs adjusted: {counts['overlap_q5'].mean():.1f} of {counts['top20'].mean():.1f}")

    sector_tab = (pd.DataFrame(sector_rows).pivot_table(index="sector", columns="book", values="share",
                                                        aggfunc="sum", fill_value=0.0) / len(rebal))
    sector_tab = sector_tab[["eligible", "standard Q5", "adjusted Q5"]].sort_values("eligible", ascending=False)
    sector_tab.to_csv(OUT / f"sector_share_{index}.csv")
    print("\nAverage sector share (all quarters):")
    print(sector_tab.to_string(float_format=lambda v: f"{v:.1%}"))
    known = counts["sector_known"].sum()
    print(f"IT names per quarter: eligible {counts['it_eligible'].mean():.1f}, standard Q5 "
          f"{counts['it_std_q5'].mean():.1f}, adjusted Q5 {counts['it_adj_q5'].mean():.1f} "
          f"(sector known for {known / tot:.0%} of eligible name-quarters)")
    it_move = scored[scored["sector"] == IT].assign(d=lambda x: x["adj_pct"] - x["std_pct"])
    print(f"IT names: mean score percentile standard {it_move['std_pct'].mean():.2f} -> adjusted "
          f"{it_move['adj_pct'].mean():.2f}")

    checks = []
    snap_dates = [rebal[0], min(rebal, key=lambda x: abs(x - pd.Timestamp("2022-12-30"))), rebal[-1]]
    for t in [t for t in TECH if t in tickers]:
        for d in snap_dates:
            r = metrics[(metrics["ticker"] == t) & (metrics["date"] == d)]
            if r.empty or not np.isfinite(r["mcap"].iloc[0]):
                continue
            r = r.iloc[0]
            s = scored[(scored["ticker"] == t) & (scored["date"] == d)]
            checks.append({"ticker": t, "date": d.date(), "mcap_bn": r["mcap"] / 1e9, "k_rd_bn": r["k_rd"] / 1e9,
                           "k_org_bn": r["k_org"] / 1e9, "bm": r["bm"], "bm_adj": r["bm_adj"], "ep": r["ep"],
                           "ep_adj": r["ep_adj"], "ebit_ev": r["ebit_ev"], "ebit_ev_adj": r["ebit_ev_adj"],
                           "std_pct": s["std_pct"].iloc[0] if not s.empty else np.nan,
                           "adj_pct": s["adj_pct"].iloc[0] if not s.empty else np.nan})
    checks = pd.DataFrame(checks)
    checks.to_csv(OUT / f"tech_check_{index}.csv", index=False)
    if not checks.empty:
        print("\nTech sanity check ($bn; score percentile 1 = cheapest):")
        print(checks.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    live = prices.index > rebal[0]
    bench = bench_px.pct_change().fillna(0.0)[live]
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    books = {**{f"standard {k}": v for k, v in std_books.items()}, **{f"adjusted {k}": v for k, v in adj_books.items()},
             "eligible EW": ew, "all members EW": all_ew}
    order = ["standard Q5", "adjusted Q5", *[f"adjusted Q{k}" for k in (4, 3, 2, 1)],
             *[f"standard Q{k}" for k in (4, 3, 2, 1)], "eligible EW", "all members EW"]
    rows, series = [], {}
    for label in order:
        w = basket_weights(prices.index, tickers, books[label], rebal, membership, prices)
        w.loc[~live] = 0.0
        net, turnover, longs, _ = book_returns(prices, w)
        lag = w.shift(1).fillna(0.0)
        gross = (lag * rets).sum(axis=1)
        drifted = (lag * (1.0 + rets)).div(1.0 + gross, axis=0)
        trades = (w - drifted).abs().sum(axis=1)
        net, turnover, longs = net[live], turnover[live], longs[live]
        series[label] = net
        rows.append(_stats(label, net, bench, {"cagr_gross": _cagr(gross[live]), "avg_long": float(longs.mean()),
                                               "turnover": float(turnover.mean() * ANN),
                                               "trade_turnover": float(trades[live].mean() * ANN)}))
    series[bench_name] = bench
    rows.append(_stats(bench_name, bench, bench, {"cagr_gross": _cagr(bench), "avg_long": 1.0, "turnover": 0.0,
                                                  "trade_turnover": 0.0}))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"comparison_{index}.csv", index=False)
    pd.DataFrame(series).to_csv(OUT / f"daily_returns_{index}.csv")

    print(f"\n=== {index}: {prices.index[live][0].date()} ~ {prices.index[-1].date()} "
          f"(train <= {SPLIT.date()} < test) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSpreads (annualized mean, t-stat; full / train / test):")
    spreads = []
    for a, b in (("adjusted Q5", "eligible EW"), ("adjusted Q5", "standard Q5"), ("adjusted Q5", "adjusted Q1"),
                 ("standard Q5", "eligible EW"), ("standard Q5", "standard Q1"), ("adjusted Q5", bench_name)):
        diff = series[a] - series[b]
        masks = {"full": diff.index == diff.index, "train": diff.index <= SPLIT, "test": diff.index > SPLIT}
        spreads += [{"spread": f"{a} - {b}", "period": k, "ann_mean": diff[m].mean() * ANN, "t": _tstat(diff[m])}
                    for k, m in masks.items()]
        print(f"  {a} - {b}: " + " / ".join(f"{s['ann_mean']:+.2%} (t {s['t']:+.2f})" for s in spreads[-3:]))
    pd.DataFrame(spreads).to_csv(OUT / f"spreads_{index}.csv", index=False)

    print("\nQuintile CAGR (Q1 dearest -> Q5 cheapest) and Spearman rho(quintile, CAGR):")
    mono = []
    for score in ("standard", "adjusted"):
        for period, mask in (("full", lambda i: i == i), ("train", lambda i: i <= SPLIT), ("test", lambda i: i > SPLIT)):
            c = [_cagr(series[f"{score} Q{k}"][mask(series[f"{score} Q{k}"].index)]) for k in range(1, 6)]
            rho = pd.Series(c).rank().corr(pd.Series(range(1, 6), dtype=float))
            mono.append({"score": score, "period": period, **{f"Q{k}": c[k - 1] for k in range(1, 6)}, "rho": rho})
    mono = pd.DataFrame(mono)
    mono.to_csv(OUT / f"quintiles_{index}.csv", index=False)
    print(mono.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    yearly = pd.DataFrame({k: (1 + series[k]).groupby(series[k].index.year).prod() - 1
                           for k in ("standard Q5", "adjusted Q5", "eligible EW", bench_name)})
    yearly.to_csv(OUT / f"yearly_{index}.csv")
    print("\n=== Calendar years ===")
    print(yearly.to_string(float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    for k, lab, lw in (("adjusted Q5", "무형자산 보정 Q5", 2.2), ("standard Q5", "기존 복합 가치 Q5", 1.6),
                       ("eligible EW", "적격 종목 동일가중", 1.2), (bench_name, bench_name, 1.2)):
        v = series[k]
        ax.plot(v.index, ((1 + v).cumprod() - 1) * 100, label=lab, lw=lw)
    ax.axvline(SPLIT, color="black", ls="--", lw=1, label=f"학습 | 보류 ({SPLIT.date()})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"{'S&P500' if index == 'sp500' else 'S&P400'} 무형자산 보정 복합 가치 점수 (누적 수익률 %, 비용 포함)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    q = pd.DataFrame({"기존 점수": [_cagr(series[f"standard Q{k}"]) * 100 for k in range(1, 6)],
                      "무형자산 보정 점수": [_cagr(series[f"adjusted Q{k}"]) * 100 for k in range(1, 6)]},
                     index=[f"Q{k}" for k in range(1, 6)])
    q.plot.bar(ax=axes[1], color=["#999", "#c33"])
    axes[1].axhline(_cagr(series["eligible EW"]) * 100, color="blue", ls="--", lw=1, label="적격 종목 동일가중")
    axes[1].set_title("점수 5분위별 CAGR (%) — Q1 가장 비쌈, Q5 가장 쌈")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / f"intangible_value_{index}.png", dpi=110)
    print(f"\nSaved {OUT / f'intangible_value_{index}.png'}")


if __name__ == "__main__":
    main()
