"""
Stat Arb restricted to low-leverage, low-P/E names, decided point-in-time at
every pair refresh. Fundamentals come from SEC filings dated on or before the
refresh date; P/E uses as-traded prices and TTM diluted EPS. Pairs are
rescreened inside the passing names with the production caps and traded with
the default pair book. Variants (fixed in advance):
  intersection   debt ratio <= median AND P/E <= median of profitable names
  relative       composite score = mean of debt-ratio and P/E percentile ranks
                 (losses / negative equity / missing rank last); keep the best
                 share across all members or within each GICS sector
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.fundamentals import ticker_ciks, value_screen_table
from data.providers import load_cached_frame, normalize_ticker
from data.universe_pit import _history, constituents_asof, membership_matrix, sector_groups_asof
from run_pair_book_experiment import SPLIT, evaluate, trade_pnl
from run_pair_cap_sweep import _load_prices
from run_sub_industry_experiment import trade_stats
from strategies.pair_book import build_pair_book
from strategies.rolling_pairs import build_rolling_pair_pool

OUT = ROOT / "output" / "value_filter"


def screen_passing(table: pd.DataFrame, membership: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    passing = {}
    for d, g in table.groupby("date"):
        g = g[g["ticker"].map(lambda t: bool(membership.at[d, t]))]
        dr_ok = g["debt_ratio"].notna()
        pe_ok = g["pe"].notna()
        med_dr = g.loc[dr_ok, "debt_ratio"].median()
        med_pe = g.loc[pe_ok, "pe"].median()
        keep = dr_ok & pe_ok & (g["debt_ratio"] <= med_dr) & (g["pe"] <= med_pe)
        passing[d] = set(g.loc[keep, "ticker"])
    return passing


def _pct_rank(s: pd.Series) -> pd.Series:
    """0 = lowest (best), 1 = highest; missing (losses, negative equity, no filing) ranks last."""
    r = s.rank(pct=True, method="average")
    return r.fillna(1.0)


def _sectors(d: pd.Timestamp) -> dict[str, str]:
    df = constituents_asof(d, index=config.PIT_INDEX)
    return dict(zip(df["ticker"].map(normalize_ticker), df["gics_sector"]))


def score_passing(
    table: pd.DataFrame,
    membership: pd.DataFrame,
    keep_frac: float,
    by_sector: bool,
) -> dict[pd.Timestamp, set[str]]:
    """Composite relative value: mean of debt-ratio and P/E percentile ranks,
    ranked across all members (or within each GICS sector); keep the best keep_frac."""
    passing = {}
    for d, g in table.groupby("date"):
        g = g[g["ticker"].map(lambda t: bool(membership.at[d, t]))].copy()
        sectors = _sectors(d)
        g["sector"] = g["ticker"].map(lambda t: sectors.get(t, "Unknown"))
        groups = g.groupby("sector") if by_sector else [("all", g)]
        keep = []
        for _, part in groups:
            score = (_pct_rank(part["debt_ratio"]) + _pct_rank(part["pe"])) / 2.0
            n_keep = int(round(len(part) * keep_frac))
            keep += list(part.loc[score.sort_values(kind="stable").index[:n_keep], "ticker"])
        passing[d] = set(keep)
    return passing


def _screen_pairs(prices, membership, passing, path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["refresh_date"])
    print(f"Screening pairs -> {path.name}")
    log = build_rolling_pair_pool(
        prices,
        sector_groups_fn=lambda asof, names: sector_groups_asof(
            asof, tickers=[t for t in names if t in passing.get(asof, set())],
            index=config.PIT_INDEX, min_names=2, level=config.PAIR_GROUP_LEVEL,
        ),
        membership=membership,
        formation_days=config.COINT_FORMATION_DAYS,
        refresh_days=config.COINT_REFRESH_DAYS,
        screen_kwargs=dict(
            pvalue_max=config.COINT_PVALUE_MAX,
            min_half_life=config.COINT_MIN_HALFLIFE,
            max_half_life=config.COINT_MAX_HALFLIFE,
            target_half_life=config.COINT_TARGET_HALFLIFE,
            min_corr=config.COINT_MIN_CORR,
            stability_window=config.COINT_STABILITY_WINDOW,
            stability_step=config.COINT_STABILITY_STEP,
            min_stability=config.COINT_MIN_STABILITY,
        ),
        max_names_per_sector=config.COINT_MAX_NAMES_PER_SECTOR,
        pool_max_pairs=config.COINT_MAX_PAIRS,
        pool_max_per_sector=config.COINT_MAX_PER_SECTOR,
        verbose=False,
    )
    log.to_csv(path, index=False)
    return log


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    tickers = list(prices.columns)
    raw = load_cached_frame(tickers, config.START, config.END, field="close")
    missing_raw = sorted(set(tickers) - set(raw.columns))
    print(f"As-traded closes cached for {raw.shape[1]}/{len(tickers)} names; missing: {missing_raw}")
    membership = membership_matrix(prices.index, tickers, index=config.PIT_INDEX)

    points = list(range(config.COINT_FORMATION_DAYS, len(prices), config.COINT_REFRESH_DAYS))
    dates = [prices.index[i - 1] for i in points]
    table_path = OUT / "screen_table.csv"
    if table_path.exists():
        table = pd.read_csv(table_path, parse_dates=["date"])
    else:
        ciks = ticker_ciks(_history(config.START, prices.index[-1].date().isoformat(), config.PIT_INDEX))
        table = value_screen_table(dates, tickers, prices, raw, ciks)
        table.to_csv(table_path, index=False)

    variants = {
        "intersection (both <= median)": ("pair_log_value.csv", screen_passing(table, membership)),
        "relative top 50% (all names)": ("pair_log_rel_all50.csv", score_passing(table, membership, 0.5, False)),
        "relative top 50% (in sector)": ("pair_log_rel_sec50.csv", score_passing(table, membership, 0.5, True)),
        "relative top 75% (in sector)": ("pair_log_rel_sec75.csv", score_passing(table, membership, 0.75, True)),
    }

    summary = []
    for label, (_, passing) in variants.items():
        counts, sec2 = [], []
        for d in dates:
            sectors = _sectors(d)
            by_sector = pd.Series([sectors.get(t, "Unknown") for t in passing[d]]).value_counts()
            counts.append(len(passing[d]))
            sec2.append(int((by_sector >= 2).sum()))
        summary.append({"variant": label, "names_avg": sum(counts) / len(counts),
                        "names_min": min(counts), "names_max": max(counts),
                        "sectors_with_2+_avg": sum(sec2) / len(sec2)})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "screen_summary.csv", index=False)
    print("\n=== Names passing per refresh ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    logs = {"all names (current)": pd.read_csv(ROOT / "output" / "rolling_pairs_log.csv",
                                               parse_dates=["refresh_date"])}
    for label, (fname, passing) in variants.items():
        logs[label] = _screen_pairs(prices, membership, passing, OUT / fname)

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
        hl_max_ratio=config.STATARB_HL_MAX_RATIO,
        fallback_pairs=[(normalize_ticker(a), normalize_ticker(b)) for a, b in config.PAIRS],
    )
    rows, stat_rows = [], []
    for k, (label, log) in enumerate(logs.items()):
        w, t = build_pair_book(prices, log, **common)
        t = trade_pnl(prices, t)
        t.to_csv(OUT / f"trades_{k}.csv", index=False)
        row = evaluate(prices, w, label)
        row["pairs_per_refresh"] = float(log.groupby("refresh_date").size().mean()) if len(log) else 0.0
        row["trades"] = len(t)
        rows.append(row)
        for period, part in (("train", t[t["entry"] <= SPLIT]), ("test", t[t["entry"] > SPLIT])):
            st = trade_stats(part)
            st["gross_bps_se"] = (
                float(part["gross_pnl"].std() / len(part) ** 0.5 * config.COINT_MAX_PAIRS * 1e4)
                if len(part) > 1 else float("nan")
            )
            stat_rows.append({"book": label, "period": period, **st})

    table_out, stats = pd.DataFrame(rows), pd.DataFrame(stat_rows)
    table_out.to_csv(OUT / "comparison.csv", index=False)
    stats.to_csv(OUT / "trade_stats.csv", index=False)
    print(f"\n=== Book comparison (train <= {SPLIT.date()} < test, raw book, reversion filter on) ===")
    print(table_out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Per-trade stats (bps of pair capital; se = standard error of gross_bps) ===")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
