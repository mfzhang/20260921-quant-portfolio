"""Cumulative, yearly and drawdown charts from the latest run_backtest.py equity files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SPLIT = pd.Timestamp("2023-05-10")
BOOKS = {
    "Stat Arb + Shock + BookDD (기본)": "stat_arb_plus_shock_plus_bookdd_equity.csv",
    "Stat Arb (가드 없음)": "stat_arb_equity.csv",
    "Cross-Sectional MR": "cross-sectional_mr_equity.csv",
}

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    eq = pd.DataFrame({
        name: pd.read_csv(OUT / f, index_col=0, parse_dates=True).iloc[:, 0]
        for name, f in BOOKS.items()
    })
    rets = eq.pct_change().fillna(0.0)
    main_book = next(iter(BOOKS))
    live = rets.index[rets[main_book].abs() > 0][0]
    yearly = (1.0 + rets).groupby(rets.index.year).prod() - 1.0
    dd = eq[main_book] / eq[main_book].cummax() - 1.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [3, 2, 1.5]})
    ax = axes[0]
    for name in eq:
        ax.plot(eq.index, (eq[name] / eq[name].iloc[0] - 1.0) * 100, label=name,
                lw=2.2 if name == main_book else 1.2)
    ax.axvspan(eq.index[0], live, color="grey", alpha=0.15, label="Stat Arb 페어 형성 기간 (매매 없음)")
    ax.axvline(SPLIT, color="black", ls="--", lw=1, label=f"학습 | 보류 구간 ({SPLIT.date()})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("누적 수익률 (%, 비용 포함)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    (yearly * 100).plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("연도별 수익률 (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.fill_between(dd.index, dd * 100, 0, color="tab:red", alpha=0.4)
    ax.axvline(SPLIT, color="black", ls="--", lw=1)
    ax.set_title(f"낙폭 (%) — {main_book}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT / "period_returns.png"
    fig.savefig(path, dpi=110)
    print(f"Saved {path}")
    print((yearly * 100).round(2).to_string())


if __name__ == "__main__":
    main()
