"""Run mean-reversion portfolio backtest: stat-arb + cross-sectional."""

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
from backtest.engine import combine_weights, run_backtest
from backtest.resilience import resilience_row
from data.loader import download_prices
from data.providers import cached_tiingo_tickers, normalize_ticker
from data.universe_pit import (
    coverage_report,
    historical_tickers,
    membership_matrix,
    pit_price_tickers,
    sector_groups_asof,
)
from strategies.hybrid import (
    apply_agree_book_gross,
    apply_agreement_scale,
    build_hybrid_portfolio,
    compare_lookbacks,
)
from strategies.mean_reversion import build_cross_sectional_weights
from strategies.pair_screener import pairs_from_screen, screen_cointegrated_pairs
from strategies.regime import (
    apply_gross_scaler,
    book_drawdown_scaler,
    macro_shock_scaler,
    realized_vol,
    rebalance_breakeven_pct,
    regime_gross_scaler,
)
from strategies.pair_book import build_pair_book
from strategies.rolling_pairs import build_rolling_stat_arb_weights
from strategies.stat_arb import build_stat_arb_weights
from strategies.universe_structure import make_primary_grouping_fn


def format_stats(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name}\n"
        f"  CAGR              {stats.get('cagr', float('nan')):>8.2%}\n"
        f"  Sharpe            {stats.get('sharpe', float('nan')):>8.2f}\n"
        f"  Vol               {stats.get('vol', float('nan')):>8.2%}\n"
        f"  Max DD            {stats.get('max_drawdown', float('nan')):>8.2%}\n"
        f"  Hit rate          {stats.get('hit_rate', float('nan')):>8.2%}\n"
        f"  Ann. turnover     {stats.get('ann_turnover', float('nan')):>8.1f}x\n"
        f"  Avg short gross   {stats.get('avg_short_gross', float('nan')):>8.2%}\n"
        f"  Trading cost/yr   {stats.get('trading_cost_drag', float('nan')):>8.2%}\n"
        f"  Borrow cost/yr    {stats.get('borrow_cost_drag', float('nan')):>8.2%}\n"
        f"  Total cost/yr     {stats.get('total_cost_drag', float('nan')):>8.2%}\n"
    )


def _renorm_l1(weights: pd.DataFrame) -> pd.DataFrame:
    l1 = weights.abs().sum(axis=1).replace(0, np.nan)
    return weights.div(l1, axis=0).fillna(0.0)


def main() -> None:
    end = config.END or pd.Timestamp.today().date().isoformat()
    membership = None

    if config.USE_PIT_UNIVERSE:
        print(f"Building PIT {config.PIT_INDEX} universe {config.START} -> {end} ...")
        if getattr(config, "PIT_PRICE_UNIVERSE", "asof_start") == "union":
            tickers = historical_tickers(config.START, end, index=config.PIT_INDEX)
            print(f"Historical member tickers (union): {len(tickers)}")
        else:
            tickers = pit_price_tickers(
                config.START,
                index=config.PIT_INDEX,
                max_names=int(config.PIT_MAX_PRICED_NAMES),
                seed=config.UNIVERSE,
            )
            cached = set(cached_tiingo_tickers(config.START, config.END))
            have = [t for t in tickers if t in cached]
            need = [t for t in tickers if t not in cached]
            print(
                f"PIT priced names as of {config.START}: {len(tickers)} "
                f"(cached {len(have)}, to_fetch {len(need)}; "
                f"membership filter still uses full index)"
            )
    else:
        tickers = sorted(
            set(config.UNIVERSE)
            | {t for p in config.PAIRS for t in p}
            | {t for names in config.SECTOR_GROUPS.values() for t in names}
        )
        tickers = [normalize_ticker(t) for t in tickers]

    provider = "tiingo"
    print(f"Downloading {len(tickers)} names from {config.START} ...")
    prices = download_prices(tickers, config.START, config.END, provider=provider)
    # Keep names with enough history; do NOT dropna(how='any') across all
    # (that would re-introduce survivorship by requiring full-period survival)
    min_obs = 252
    prices = prices.loc[:, prices.notna().sum() >= min_obs]
    print(f"Price matrix: {prices.shape[0]} days x {prices.shape[1]} tickers (min {min_obs} obs)")

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    if config.USE_PIT_UNIVERSE:
        membership = membership_matrix(
            prices.index,
            list(prices.columns),
            index=config.PIT_INDEX,
        )
        cov = coverage_report(membership, prices)
        print(
            f"PIT coverage: avg {cov['avg_member_coverage']:.1%} of members priced/day "
            f"({int(cov['priced_tickers'])} priced tickers)"
        )
        pd.Series(cov).to_csv(out_dir / "pit_coverage.csv")

    trade_start = None
    screen = None
    pair_log = None
    sector_groups_for_cs: dict[str, list[str]] | None = None

    if config.USE_PIT_UNIVERSE:
        asof_cs = prices.index[min(len(prices) - 1, config.COINT_FORMATION_DAYS)]
        sector_groups_for_cs = sector_groups_asof(
            asof_cs,
            tickers=list(prices.columns),
            index=config.PIT_INDEX,
            min_names=1,
        )
    else:
        sector_groups_for_cs = config.SECTOR_GROUPS

    screen_kwargs = dict(
        pvalue_max=config.COINT_PVALUE_MAX,
        min_half_life=config.COINT_MIN_HALFLIFE,
        max_half_life=config.COINT_MAX_HALFLIFE,
        target_half_life=config.COINT_TARGET_HALFLIFE,
        min_corr=config.COINT_MIN_CORR,
        stability_window=config.COINT_STABILITY_WINDOW,
        stability_step=config.COINT_STABILITY_STEP,
        min_stability=config.COINT_MIN_STABILITY,
        max_pairs=config.COINT_MAX_PAIRS,
        max_per_sector=getattr(config, "COINT_MAX_PER_SECTOR", 3),
    )
    fallback_pairs = [(normalize_ticker(a), normalize_ticker(b)) for a, b in config.PAIRS]

    def _groups_asof(asof, tickers):
        if config.USE_PIT_UNIVERSE:
            return sector_groups_asof(
                asof, tickers=tickers, index=config.PIT_INDEX, min_names=2,
                level=config.PAIR_GROUP_LEVEL,
            )
        return config.SECTOR_GROUPS

    if config.USE_AUTO_PAIRS and getattr(config, "USE_ROLLING_PAIRS", False):
        print(
            f"Building ROLLING stat-arb pairs "
            f"(formation={config.COINT_FORMATION_DAYS}d, refresh={config.COINT_REFRESH_DAYS}d) ..."
        )
        w_stat, pair_log = build_rolling_stat_arb_weights(
            prices,
            sector_groups_fn=_groups_asof,
            membership=membership,
            formation_days=config.COINT_FORMATION_DAYS,
            refresh_days=config.COINT_REFRESH_DAYS,
            lookback=config.STATARB_LOOKBACK,
            entry_z=config.STATARB_ENTRY_Z,
            exit_z=config.STATARB_EXIT_Z,
            stop_z=config.STATARB_STOP_Z,
            timeout_mult=config.STATARB_HL_TIMEOUT_MULT,
            screen_kwargs=screen_kwargs,
            max_names_per_sector=config.COINT_MAX_NAMES_PER_SECTOR,
            fallback_pairs=fallback_pairs,
            primary_grouping_fn=make_primary_grouping_fn(
                config.PRIMARY_GROUPING, min_corr=config.PRIMARY_CLUSTER_MIN_CORR
            ),
            verbose=True,
        )
        pair_log.to_csv(out_dir / "rolling_pairs_log.csv", index=False)
        if config.STATARB_BOOK == "pair_book":
            w_stat, pair_trades = build_pair_book(
                prices,
                pair_log,
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
                excluded_sectors=tuple(config.STATARB_EXCLUDED_SECTORS),
                fallback_pairs=fallback_pairs,
                hl_max_ratio=config.STATARB_HL_MAX_RATIO,
            )
            pair_trades.to_csv(out_dir / "pair_trades.csv", index=False)
            print(f"Pair book trades: {len(pair_trades)}")
        trade_start = prices.index[config.COINT_FORMATION_DAYS]
        # flat during initial formation
        w_stat.loc[prices.index < trade_start] = 0.0
        if not pair_log.empty:
            print(
                f"Rolling pair refreshes: {pair_log['refresh_date'].nunique()}, "
                f"pair-rows: {len(pair_log)}"
            )

    elif config.USE_AUTO_PAIRS:
        formation_n = min(config.COINT_FORMATION_DAYS, len(prices) - 60)
        formation = prices.iloc[:formation_n]
        trade_start = prices.index[formation_n]
        formation_end = formation.index[-1]

        groups = _groups_asof(formation_end, list(formation.columns))
        max_n = getattr(config, "COINT_MAX_NAMES_PER_SECTOR", 12)
        avail = formation.notna().sum()
        groups = {
            s: sorted(
                [n for n in names if n in formation.columns],
                key=lambda t: float(avail.get(t, 0)),
                reverse=True,
            )[:max_n]
            for s, names in groups.items()
        }
        groups = {s: n for s, n in groups.items() if len(n) >= 2}
        if membership is not None:
            mem_end = membership.loc[formation_end]
            formation = formation.loc[:, mem_end.reindex(formation.columns).fillna(False)]

        print(
            f"Screening static pairs on formation "
            f"{formation.index[0].date()} ~ {formation.index[-1].date()} ..."
        )
        screen = screen_cointegrated_pairs(formation, groups, **screen_kwargs)
        screen.to_csv(out_dir / "coint_pairs.csv", index=False)
        pairs = pairs_from_screen(screen)
        if not pairs:
            pairs = fallback_pairs
            trade_start = None
        else:
            print(f"Auto-selected {len(pairs)} pairs:")
            print(screen.to_string(index=False))

        half_lives = {
            (str(r.y), str(r.x)): float(r.half_life)
            for r in screen.itertuples(index=False)
        } if screen is not None and not screen.empty else {}

        print("Building statistical arbitrage weights ...")
        w_stat = build_stat_arb_weights(
            prices,
            pairs,
            lookback=config.STATARB_LOOKBACK,
            entry_z=config.STATARB_ENTRY_Z,
            exit_z=config.STATARB_EXIT_Z,
            stop_z=config.STATARB_STOP_Z,
            half_lives=half_lives,
            timeout_mult=config.STATARB_HL_TIMEOUT_MULT,
        )
        if trade_start is not None:
            w_stat.loc[prices.index < trade_start] = 0.0
    else:
        pairs = fallback_pairs
        print(f"Using {len(pairs)} static pairs from config")
        print("Building statistical arbitrage weights ...")
        w_stat = build_stat_arb_weights(
            prices,
            pairs,
            lookback=config.STATARB_LOOKBACK,
            entry_z=config.STATARB_ENTRY_Z,
            exit_z=config.STATARB_EXIT_Z,
            stop_z=config.STATARB_STOP_Z,
            timeout_mult=config.STATARB_HL_TIMEOUT_MULT,
        )

    # pair_book closes both legs itself when a leg leaves the index and must keep fixed sizing
    if membership is not None and not (config.USE_ROLLING_PAIRS and config.STATARB_BOOK == "pair_book"):
        w_stat = _renorm_l1(w_stat.where(membership, 0.0))

    print("Building cross-sectional mean-reversion weights ...")
    w_cs = build_cross_sectional_weights(
        prices,
        lookback=config.CS_LOOKBACK,
        holding=config.CS_HOLDING,
        long_pct=config.CS_LONG_PCT,
        short_pct=config.CS_SHORT_PCT,
        membership=membership,
        sector_groups=sector_groups_for_cs,
        sector_neutral=getattr(config, "CS_SECTOR_NEUTRAL", True),
    )
    if getattr(config, "CS_SECTOR_NEUTRAL", True):
        print(f"CS sector-neutral: {len(sector_groups_for_cs or {})} sectors")

    # --- Cost breakeven (what edge you need per rebalance) ---
    print("\n=== Rebalance breakeven (cost hurdle) ===")
    for tau, label in [(0.5, "partial ~0.5"), (1.0, "typical ~1.0"), (2.0, "full flip ~2.0")]:
        be = rebalance_breakeven_pct(
            commission_bps=config.COMMISSION_BPS,
            slippage_bps=config.SLIPPAGE_BPS,
            turnover=tau,
            holding_days=config.CS_HOLDING,
            borrow_fee_annual=config.BORROW_FEE_ANNUAL,
            avg_short_gross=0.5,
            ann=config.ANNUALIZATION,
        )
        print(
            f"  turnover {label}: need >= {be['breakeven_hold_pct']:.3f}% "
            f"over {config.CS_HOLDING}d "
            f"(trade {be['trade_cost_pct']:.3f}% + borrow {be['borrow_cost_pct']:.3f}%) "
            f"~ {be['breakeven_annualized_pct']:.1f}%/yr if always paid"
        )

    combined = combine_weights(
        {"stat_arb": w_stat, "cross_section": w_cs},
        {"stat_arb": config.STATARB_WEIGHT, "cross_section": config.CS_WEIGHT},
    )

    bt_kwargs = dict(
        initial_capital=config.INITIAL_CAPITAL,
        commission_bps=config.COMMISSION_BPS,
        slippage_bps=config.SLIPPAGE_BPS,
        borrow_fee_annual=config.BORROW_FEE_ANNUAL,
        ann=config.ANNUALIZATION,
    )

    # Conservative book: agreement scale + agree-only gross expansion + regime
    w_stat_a, w_cs_a, agree_scale = apply_agreement_scale(
        w_stat,
        w_cs,
        agree_mult=config.AGREE_MULT,
        solo_mult=config.SOLO_MULT,
        conflict_mult=config.CONFLICT_MULT,
    )
    agree_combined = combine_weights(
        {"stat_arb": w_stat_a, "cross_section": w_cs_a},
        {"stat_arb": config.STATARB_WEIGHT, "cross_section": config.CS_WEIGHT},
    )
    agree_book_mult = apply_agree_book_gross(
        agree_combined,
        agree_scale,
        agree_mult=config.AGREE_MULT,
        book_gross_max=getattr(config, "AGREE_BOOK_GROSS_MAX", 1.25),
    )
    agree_boosted = agree_combined.mul(agree_book_mult, axis=0)
    print(
        f"Agree book gross: max={getattr(config, 'AGREE_BOOK_GROSS_MAX', 1.25):.2f}x, "
        f"avg mult={float(agree_book_mult.mean()):.3f}, "
        f"max mult={float(agree_book_mult.max()):.3f}"
    )
    agree_book_mult.to_csv(out_dir / "agree_book_gross.csv", header=["gross_mult"])

    regime_scale = None
    if getattr(config, "USE_REGIME_OVERLAY", False):
        vol = realized_vol(
            prices,
            lookback=config.REGIME_VOL_LOOKBACK,
            ann=config.ANNUALIZATION,
        )
        regime_scale = regime_gross_scaler(
            vol,
            target_vol=config.REGIME_VOL_TARGET,
            floor=config.REGIME_GROSS_FLOOR,
            ceil=config.REGIME_GROSS_CEIL,
        )
        regime_scale.to_csv(out_dir / "regime_gross_scale.csv", header=["gross_scale"])
        print(
            f"Regime overlay: vol lookback={config.REGIME_VOL_LOOKBACK}d, "
            f"target={config.REGIME_VOL_TARGET:.0%}, "
            f"avg scale={float(regime_scale.mean()):.2f}, "
            f"min={float(regime_scale.min()):.2f}"
        )
        agree_regime = apply_gross_scaler(agree_boosted, regime_scale)
    else:
        agree_regime = agree_boosted

    # Stat Arb primary + macro shock hard flatten (fast cut, then cooldown)
    w_stat_protected = w_stat
    shock_scale = None
    if getattr(config, "USE_SHOCK_BREAKER", False):
        shock_scale = macro_shock_scaler(
            prices,
            short_vol_lb=config.SHOCK_SHORT_VOL_LB,
            long_vol_lb=config.SHOCK_LONG_VOL_LB,
            vol_spike_ratio=config.SHOCK_VOL_SPIKE_RATIO,
            crash_days=config.SHOCK_CRASH_DAYS,
            crash_ret=config.SHOCK_CRASH_RET,
            day1_crash=getattr(config, "SHOCK_DAY1_CRASH", -0.035),
            cooldown_days=config.SHOCK_COOLDOWN_DAYS,
            shock_floor=config.SHOCK_FLOOR,
            ann=config.ANNUALIZATION,
        )
        shock_scale.to_csv(out_dir / "shock_scale.csv", header=["shock_scale"])
        n_flat = int((shock_scale < 0.5).sum())
        print(
            f"Shock breaker: vol_spike>={config.SHOCK_VOL_SPIKE_RATIO:.1f}x or "
            f"{config.SHOCK_CRASH_DAYS}d basket<={config.SHOCK_CRASH_RET:.0%} or "
            f"1d<={getattr(config, 'SHOCK_DAY1_CRASH', -0.035):.1%}, "
            f"cooldown={config.SHOCK_COOLDOWN_DAYS}d, "
            f"flat days={n_flat}/{len(shock_scale)} "
            f"({n_flat / max(len(shock_scale), 1):.1%})"
        )
        w_stat_protected = apply_gross_scaler(w_stat, shock_scale)

    book_dd_scale = None
    if getattr(config, "USE_BOOK_DD_STOP", False):
        book_dd_scale = book_drawdown_scaler(
            w_stat_protected,
            prices,
            max_dd=config.BOOK_DD_MAX,
            cooldown_days=config.BOOK_DD_COOLDOWN,
            floor=config.BOOK_DD_FLOOR,
        )
        book_dd_scale.to_csv(out_dir / "book_dd_scale.csv", header=["book_dd_scale"])
        n_dd = int((book_dd_scale < 0.5).sum())
        print(
            f"Book DD stop: max_dd={config.BOOK_DD_MAX:.0%}, "
            f"cooldown={config.BOOK_DD_COOLDOWN}d, "
            f"flat days={n_dd}/{len(book_dd_scale)} "
            f"({n_dd / max(len(book_dd_scale), 1):.1%})"
        )
        w_stat_protected = apply_gross_scaler(w_stat_protected, book_dd_scale)

    w_stat_regime_shock = w_stat_protected
    if regime_scale is not None:
        w_stat_regime_shock = apply_gross_scaler(w_stat_protected, regime_scale)

    results = {
        "Stat Arb": run_backtest(prices, w_stat, **bt_kwargs),
        "Stat Arb + Shock": run_backtest(
            prices,
            apply_gross_scaler(w_stat, shock_scale) if shock_scale is not None else w_stat,
            **bt_kwargs,
        ),
        "Stat Arb + Shock + BookDD": run_backtest(prices, w_stat_protected, **bt_kwargs),
        "Stat Arb + All Guards": run_backtest(prices, w_stat_regime_shock, **bt_kwargs),
        "Cross-Sectional MR": run_backtest(prices, w_cs, **bt_kwargs),
        "Combined 50/50": run_backtest(prices, combined, **bt_kwargs),
        "Agree + Gross + Regime": run_backtest(prices, agree_regime, **bt_kwargs),
    }

    hybrid_lookback = config.SLEEVE_SCORE_LOOKBACK
    # Default book: Stat Arb with shock + book DD (profitability + fast cut)
    combined_for_plot = w_stat_protected

    if getattr(config, "USE_HYBRID_ALLOCATION", True) and getattr(
        config, "USE_DYNAMIC_SLEEVE_MIX", False
    ):
        print("Building hybrid dynamic sleeve mix (optional) ...")
        lb_table = compare_lookbacks(
            prices,
            w_stat,
            w_cs,
            list(config.SLEEVE_LOOKBACK_GRID),
            run_backtest_fn=run_backtest,
            bt_kwargs=bt_kwargs,
            agree_mult=config.AGREE_MULT,
            solo_mult=config.SOLO_MULT,
            conflict_mult=config.CONFLICT_MULT,
        )
        lb_table.to_csv(out_dir / "sleeve_lookback_sensitivity.csv", index=False)
        print(lb_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        hybrid = build_hybrid_portfolio(
            prices,
            w_stat,
            w_cs,
            lookback=hybrid_lookback,
            agree_mult=config.AGREE_MULT,
            solo_mult=config.SOLO_MULT,
            conflict_mult=config.CONFLICT_MULT,
            mix_floor=config.SLEEVE_MIX_FLOOR,
            mix_ceil=config.SLEEVE_MIX_CEIL,
            ann=config.ANNUALIZATION,
        )
        dyn = hybrid["combined_dynamic"]
        if regime_scale is not None:
            dyn = apply_gross_scaler(dyn, regime_scale)
        results["Hybrid Dynamic"] = run_backtest(prices, dyn, **bt_kwargs)

    print("\n=== Backtest results ===")
    for name, res in results.items():
        print(format_stats(name, res.stats))
        safe = name.lower().replace(" ", "_").replace("/", "_").replace("+", "plus")
        res.equity.to_csv(out_dir / f"{safe}_equity.csv")
        res.weights.tail(1).T.to_csv(out_dir / f"{safe}_last_weights.csv")

    # Macro-resilience ranking (not just point Sharpe)
    res_table = pd.DataFrame(
        [
            resilience_row(
                name,
                res,
                prices,
                vol_lookback=config.REGIME_VOL_LOOKBACK,
                ann=config.ANNUALIZATION,
            )
            for name, res in results.items()
        ]
    ).sort_values("resilience_score", ascending=False)
    res_table.to_csv(out_dir / "macro_resilience.csv", index=False)
    print("\n=== Macro resilience ranking (rolling pairs) ===")
    cols = [
        "name", "cagr", "sharpe", "max_dd", "calmar",
        "high_vol_sharpe", "worst_year", "ann_turnover", "resilience_score",
    ]
    print(res_table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = str(res_table.iloc[0]["name"])
    print(
        f"\nRecommended for macro survival: {best}\n"
        f"  (ranks by Calmar + high-vol Sharpe - turnover/DD penalties; "
        f"not max CAGR chase)"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=name)
    ax.set_title("Rolling pairs + sector-neutral CS (macro resilience view)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "equity_curves.png"
    fig.savefig(fig_path, dpi=140)
    print(f"Saved equity chart -> {fig_path}")

    latest = combined_for_plot.iloc[-1]
    active = latest[latest.abs() > 1e-6].sort_values()
    print("\nLatest Stat Arb (Shock+BookDD) weights (nonzero):")
    print(active.to_string() if len(active) else "  (flat - shock/book-DD cooldown active)")
    pd.DataFrame({"weight": active}).to_csv(out_dir / "latest_combined_weights.csv")
    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
