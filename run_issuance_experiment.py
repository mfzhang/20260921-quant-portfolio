"""
Net share issuance long-short across the full cached history (2018-01 ~),
checked by year and by market regime. Rules fixed in advance:
  signal    YoY change in weighted-average diluted shares, latest quarter or
            fiscal year, both values from the same SEC filing (split-restated),
            only filings dated on or before each month end
  book      monthly; long the 30% with the lowest share growth, short the 30%
            with the highest, equal weight, 0.5 long / 0.5 short (gross 1.0)
  primary   ranked within each GICS sector (sector-neutral); market-wide shown
            for reference
Costs: 15bps per unit turnover + 1.5%/yr borrow on shorts.
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
from data.fundamentals import issuance_table, ticker_ciks
from data.providers import load_price_frame, normalize_ticker
from data.universe_pit import _history, constituents_asof, membership_matrix
from run_pair_cap_sweep import _load_prices
from strategies.issuance import long_short_weights
from strategies.quality_dip import basket_weights

OUT = ROOT / "output" / "issuance"
SPLIT = pd.Timestamp("2023-05-10")
ANN = config.ANNUALIZATION
TRADE_BPS = (config.COMMISSION_BPS + config.SLIPPAGE_BPS) / 10_000.0
BORROW_DAILY = config.BORROW_FEE_ANNUAL / ANN
EPISODES = {
    "2018 Q4 selloff": ("2018-10-01", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "COVID recovery": ("2020-03-24", "2021-12-31"),
    "2022 rate-hike bear": ("2022-01-03", "2022-10-12"),
    "2022-10 ~ end recovery": ("2022-10-13", "2026-12-31"),
}

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def _cagr(r: pd.Series) -> float:
    return float((1.0 + r).prod() ** (ANN / max(len(r), 1)) - 1.0)


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else np.nan


def _mdd(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def book_returns(prices: pd.DataFrame, w: pd.DataFrame) -> dict[str, pd.Series]:
    rets = prices.pct_change().fillna(0.0)
    lag = w.shift(1).fillna(0.0)
    gross = (lag * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    short = lag.clip(upper=0.0)
    net = gross - turnover * TRADE_BPS - short.abs().sum(axis=1) * BORROW_DAILY
    long_leg = (lag.clip(lower=0.0) * rets).sum(axis=1) / lag.clip(lower=0.0).sum(axis=1).replace(0, np.nan)
    short_leg = (short * rets).sum(axis=1) / short.sum(axis=1).replace(0, np.nan)
    return {"net": net, "gross": gross, "turnover": turnover,
            "long_leg": long_leg.fillna(0.0), "short_leg": short_leg.fillna(0.0)}


def summary_row(label: str, r: pd.Series, spy: pd.Series, extra: dict | None = None) -> dict:
    train, test = r[r.index <= SPLIT], r[r.index > SPLIT]
    return {
        "book": label,
        "cagr": _cagr(r), "cagr_train": _cagr(train), "cagr_test": _cagr(test),
        "vol": float(r.std() * np.sqrt(ANN)), "sharpe": _sharpe(r),
        "sharpe_train": _sharpe(train), "sharpe_test": _sharpe(test),
        "mdd": _mdd(r), "beta": float(np.cov(r, spy)[0, 1] / np.var(spy, ddof=1)),
        "t_stat": float(r.mean() / r.std() * np.sqrt(len(r))),
        **(extra or {}),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    tickers = list(prices.columns)
    membership = membership_matrix(prices.index, tickers, index=config.PIT_INDEX)
    spy_px = load_price_frame(["SPY"], config.START, config.END)["SPY"].reindex(prices.index).ffill()

    month_ends = list(prices.index.to_series().groupby(prices.index.to_period("M")).max())[:-1]
    table_path = OUT / "issuance_monthly.csv"
    if table_path.exists():
        table = pd.read_csv(table_path, parse_dates=["date"])
    else:
        ciks = ticker_ciks(_history(config.START, prices.index[-1].date().isoformat(), config.PIT_INDEX))
        table = issuance_table(month_ends, tickers, ciks)
        table.to_csv(table_path, index=False)
    signals = {d: g.set_index("ticker")["share_growth"] for d, g in table.groupby("date")}
    sectors = {}
    for d in month_ends:
        c = constituents_asof(d, index=config.PIT_INDEX)
        sectors[d] = dict(zip(c["ticker"].map(normalize_ticker), c["gics_sector"]))

    cov = pd.DataFrame({
        d: {"members": int(membership.loc[d].sum()),
            "with_signal": int(s[membership.loc[d].reindex(s.index).fillna(False).to_numpy()].notna().sum()),
            "median_growth": float(s.median())}
        for d, s in signals.items()
    }).T
    print("Signal coverage per month end (members with prices / with share growth):")
    print(cov.describe().loc[["mean", "min", "max"]].round(3).to_string())

    start = month_ends[0]
    live = prices.index > start
    spy = spy_px.pct_change().fillna(0.0)[live]
    ew = basket_weights(prices.index, tickers, {d: set(s.index) for d, s in signals.items()},
                        month_ends, membership, prices)
    ew_ret = book_returns(prices, ew)["net"][live]

    books, rows = {}, []
    for label, sec in (("sector-neutral (primary)", sectors), ("market-wide", None)):
        w = long_short_weights(prices.index, tickers, signals, frac=0.3, sectors_by_date=sec,
                               membership=membership, prices=prices)
        res = {k: v[live] for k, v in book_returns(prices, w).items()}
        books[label] = res
        rows.append(summary_row(label, res["net"], spy, {
            "gross_cagr": _cagr(res["gross"]),
            "turnover": float(res["turnover"].mean() * ANN),
            "long_leg_cagr": _cagr(res["long_leg"]),
            "short_leg_cagr": _cagr(res["short_leg"]),
        }))
    rows.append(summary_row("equal weight members (long only)", ew_ret, spy))
    rows.append(summary_row("SPY", spy, spy))
    table_out = pd.DataFrame(rows)
    table_out.to_csv(OUT / "comparison.csv", index=False)

    primary = books["sector-neutral (primary)"]["net"]
    wide = books["market-wide"]["net"]
    yearly = pd.DataFrame({
        "sector-neutral": (1 + primary).groupby(primary.index.year).prod() - 1,
        "market-wide": (1 + wide).groupby(wide.index.year).prod() - 1,
        "long leg - EW": (1 + books["sector-neutral (primary)"]["long_leg"]).groupby(primary.index.year).prod()
                         - (1 + ew_ret).groupby(ew_ret.index.year).prod(),
        "short leg - EW": (1 + books["sector-neutral (primary)"]["short_leg"]).groupby(primary.index.year).prod()
                          - (1 + ew_ret).groupby(ew_ret.index.year).prod(),
        "SPY": (1 + spy).groupby(spy.index.year).prod() - 1,
    })
    yearly.to_csv(OUT / "yearly.csv")

    regime_rows = []
    for name, (a, b) in EPISODES.items():
        m = (primary.index >= pd.Timestamp(a)) & (primary.index <= pd.Timestamp(b))
        regime_rows.append({"regime": name, "days": int(m.sum()),
                            "sector-neutral": float((1 + primary[m]).prod() - 1),
                            "market-wide": float((1 + wide[m]).prod() - 1),
                            "SPY": float((1 + spy[m]).prod() - 1)})
    spy_month = (1 + spy).groupby(spy.index.to_period("M")).prod() - 1
    up_months = set(spy_month[spy_month > 0].index)
    month_key = primary.index.to_period("M")
    vol60 = spy_px.pct_change().rolling(60).std().shift(1)[live]
    high_vol = vol60 > vol60.median()
    for name, mask in (
        ("SPY up months", month_key.isin(up_months)),
        ("SPY down months", ~month_key.isin(up_months)),
        ("high-vol regime (SPY 60d vol > median)", high_vol.to_numpy()),
        ("low-vol regime", ~high_vol.to_numpy()),
    ):
        regime_rows.append({"regime": name, "days": int(mask.sum()),
                            "sector-neutral": float(primary[mask].mean() * ANN),
                            "market-wide": float(wide[mask].mean() * ANN),
                            "SPY": float(spy[mask].mean() * ANN)})
    regimes = pd.DataFrame(regime_rows)
    regimes.to_csv(OUT / "regimes.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    for label, r in (("섹터 중립 (주 버전)", primary), ("시장 전체", wide)):
        ax.plot(r.index, ((1 + r).cumprod() - 1) * 100, label=label, lw=2 if "주" in label else 1.2)
    for name, (a, b) in EPISODES.items():
        if "recovery" not in name:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="red", alpha=0.12)
    ax.axvline(SPLIT, color="black", ls="--", lw=1, label=f"학습 | 보류 ({SPLIT.date()})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("순주식발행 롱숏 누적 수익률 (%, 비용 포함, 총노출 1배) — 빨간 구간: 시장 급락기")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax = axes[1]
    (yearly[["sector-neutral", "market-wide", "SPY"]] * 100).plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("연도별 수익률 (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "issuance.png", dpi=110)

    fmt = lambda v: f"{v:.4f}"
    print(f"\n=== {prices.index[live][0].date()} ~ {prices.index[-1].date()} (train <= {SPLIT.date()} < test) ===")
    print(table_out.to_string(index=False, float_format=fmt))
    print("\n=== Calendar years ===")
    print(yearly.to_string(float_format=fmt))
    print("\n=== Regimes (episodes: cumulative return; month/vol splits: annualized mean) ===")
    print(regimes.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
