"""
Pair candidates grouped by GICS sector (production) vs GICS sub-industry,
both rescreened on the same schedule/caps and traded with the pair book at
the default parameters. Also splits the sector book's trades into
same-sub-industry vs cross-sub-industry pairs. Nothing is tuned here.
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
from data.universe_pit import constituents_asof, membership_matrix, sector_groups_asof
from run_pair_book_experiment import SPLIT, evaluate, trade_pnl
from run_pair_cap_sweep import _load_prices
from strategies.pair_book import build_pair_book
from strategies.rolling_pairs import build_rolling_pair_pool

OUT = ROOT / "output" / "sub_industry"


def _screen(prices, membership, level: str) -> pd.DataFrame:
    path = OUT / f"pair_log_{level}_n{prices.shape[1]}.csv"
    if path.exists():
        print(f"Reusing {path.name}")
        return pd.read_csv(path, parse_dates=["refresh_date"])
    print(f"Screening pairs grouped by {level} ...")
    log = build_rolling_pair_pool(
        prices,
        sector_groups_fn=lambda asof, tickers: sector_groups_asof(
            asof, tickers=tickers, index=config.PIT_INDEX, min_names=2, level=level
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


def _sub_industry_lookup(dates) -> dict[pd.Timestamp, dict[str, str]]:
    out = {}
    for d in sorted(set(dates)):
        df = constituents_asof(d, index=config.PIT_INDEX)
        out[d] = dict(zip(df["ticker"].map(normalize_ticker), df["gics_sub_industry"]))
    return out


def trade_stats(t: pd.DataFrame) -> dict:
    budget = 1.0 / config.COINT_MAX_PAIRS
    return {
        "trades": len(t),
        "hit": float((t["gross_pnl"] > 0).mean()) if len(t) else float("nan"),
        "stop_rate": float((t["reason"] == "stop").mean()) if len(t) else float("nan"),
        "gross_bps": float(t["gross_pnl"].mean() / budget * 1e4) if len(t) else float("nan"),
        "net_bps": float(t["net_pnl"].mean() / budget * 1e4) if len(t) else float("nan"),
        "days": float(t["days"].mean()) if len(t) else float("nan"),
    }


def _split_rows(label: str, t: pd.DataFrame) -> list[dict]:
    rows = []
    for period, part in (
        ("train", t[t["entry"] <= SPLIT]),
        ("test", t[t["entry"] > SPLIT]),
    ):
        rows.append({"group": label, "period": period, **trade_stats(part)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    membership = membership_matrix(prices.index, list(prices.columns), index=config.PIT_INDEX)
    fallback = [(normalize_ticker(a), normalize_ticker(b)) for a, b in config.PAIRS]
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
        fallback_pairs=fallback,
    )

    rows, trade_rows, logs = [], [], {}
    for level in ("sector", "sub_industry"):
        log = _screen(prices, membership, level)
        logs[level] = log
        per_refresh = log.groupby("refresh_date").size()
        w, trades = build_pair_book(prices, log, **common)
        t = trade_pnl(prices, trades)
        t.to_csv(OUT / f"trades_{level}.csv", index=False)
        row = evaluate(prices, w, level)
        row["pairs_per_refresh"] = float(per_refresh.reindex(
            sorted(set(log["refresh_date"]))).mean())
        row["unique_pairs"] = int(log[["y", "x"]].drop_duplicates().shape[0])
        rows.append(row)
        trade_rows += _split_rows(level, t)

        if level == "sector":
            lookup = _sub_industry_lookup(log["refresh_date"])
            same = log.apply(
                lambda r: lookup[r.refresh_date].get(r.y) is not None
                and lookup[r.refresh_date].get(r.y) == lookup[r.refresh_date].get(r.x),
                axis=1,
            )
            print(f"Sector-grouped log: {same.mean():.1%} of pair rows share a sub-industry")
            same_pairs = set(zip(log.loc[same, "y"], log.loc[same, "x"]))
            mask = [(a, b) in same_pairs for a, b in zip(t["y"], t["x"])]
            t_same, t_cross = t[mask], t[[not m for m in mask]]
            trade_rows += _split_rows("sector book: same sub-industry", t_same)
            trade_rows += _split_rows("sector book: cross sub-industry", t_cross)

    overlap = set(zip(logs["sector"]["y"], logs["sector"]["x"])) & set(
        zip(logs["sub_industry"]["y"], logs["sub_industry"]["x"])
    )
    print(f"Unique pairs in both logs: {len(overlap)}")

    sub = trade_pnl(prices, build_pair_book(prices, logs["sub_industry"], **common)[1])
    print("\nSub-industry book, trades by group (all periods):")
    print(sub.groupby("sector").agg(
        n=("net_pnl", "size"), gross=("gross_pnl", "sum"), net=("net_pnl", "sum"),
        hit=("gross_pnl", lambda s: (s > 0).mean()),
    ).sort_values("gross").to_string(float_format=lambda v: f"{v:.4f}"))

    table = pd.DataFrame(rows)
    trades_table = pd.DataFrame(trade_rows)
    table.to_csv(OUT / "comparison.csv", index=False)
    trades_table.to_csv(OUT / "trade_stats.csv", index=False)
    print(f"\n=== Book comparison (train <= {SPLIT.date()} < test, raw book) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Per-trade stats (bps of pair capital) ===")
    print(trades_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
