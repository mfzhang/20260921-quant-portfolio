"""
Diagnose why the raw Stat Arb book loses money (no shock / book-DD guards).

  1. Gross vs net: is the signal losing before costs, or do costs flip it?
  2. Turnover sources: daily beta re-hedging vs book-wide L1 re-normalization
     vs entries/exits, by rebuilding the book under counterfactual sizing rules.
  3. Pair / trade P&L: concentration, exit reasons, hedge balance, naked legs.

Uses output/rolling_pairs_log.csv (written by run_backtest.py) so the pair
schedule matches the production book.
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
from data.providers import normalize_ticker
from data.universe_pit import membership_matrix
from run_pair_cap_sweep import _load_prices
from strategies.pair_screener import _half_life
from strategies.rolling_pairs import weights_from_pair_log
from strategies.stat_arb import pair_positions, pair_spread

ANN = config.ANNUALIZATION
TRADE_BPS = (config.COMMISSION_BPS + config.SLIPPAGE_BPS) / 10_000.0
BORROW_DAILY = config.BORROW_FEE_ANNUAL / ANN


def _cagr(r: pd.Series) -> float:
    r = r.fillna(0.0)
    return float((1.0 + r).prod() ** (ANN / max(len(r), 1)) - 1.0)


def _segments(prices: pd.DataFrame, log: pd.DataFrame, fallback):
    """Yield (seg_index, [(y, x, half_life)]) exactly as weights_from_pair_log schedules them."""
    schedule = {
        pd.Timestamp(d).normalize(): [(str(r.y), str(r.x), float(r.half_life)) for r in g.itertuples()]
        for d, g in log.groupby("refresh_date")
    }
    n = len(prices)
    points = list(range(config.COINT_FORMATION_DAYS, n, config.COINT_REFRESH_DAYS))
    for k, i in enumerate(points):
        j = points[k + 1] if k + 1 < len(points) else n
        rows = schedule.get(prices.index[i - 1].normalize()) or [(y, x, np.nan) for y, x in fallback]
        yield prices.index[i:j], rows


def build_pair_legs(prices, log, fallback, *, beta_mode: str, hedge: str):
    """
    Per-trade unscaled leg weights. One record per (segment, pair).
      beta_mode: "daily" (production) | "entry" (beta frozen when the trade opens)
      hedge:     "beta_units" (production: w_x = -pos*beta) |
                 "dollar" (w_x = -pos*beta*Px/Py, i.e. the traded spread y - beta*x in dollars)
    """
    cache: dict = {}
    legs = []
    for seg, rows in _segments(prices, log, fallback):
        for y, x, hl in rows:
            if y not in prices.columns or x not in prices.columns:
                continue
            key = (y, x)
            if key not in cache:
                cache[key] = pair_spread(prices, y, x, config.STATARB_LOOKBACK)
            diag = cache[key]
            z = diag["z"].reindex(seg)
            beta = diag["beta"].reindex(seg).ffill().fillna(1.0)
            if not np.isfinite(hl):
                hl = _half_life(diag["spread"].reindex(seg).dropna())
            pos = pair_positions(
                z, config.STATARB_ENTRY_Z, config.STATARB_EXIT_Z, config.STATARB_STOP_Z,
                half_life=float(hl) if np.isfinite(hl) else None,
                timeout_mult=config.STATARB_HL_TIMEOUT_MULT,
            )
            if beta_mode == "entry":
                trade_id = (pos != pos.shift(1)).cumsum()
                beta = beta.groupby(trade_id).transform("first").where(pos != 0, beta)
            hx = beta
            if hedge == "dollar":
                hx = beta * prices[x].reindex(seg) / prices[y].reindex(seg)
            legs.append({"seg": seg, "y": y, "x": x, "pos": pos, "z": z, "wy": pos, "wx": -pos * hx})
    return legs


def book_from_legs(prices, legs, membership, *, sizing: str):
    """
    sizing: "l1_book" (production: whole book re-normalized to gross 1 every day) |
            "fixed"   (each scheduled pair gets gross 1/n_pairs_in_segment, no re-normalization)
    Returns book weights and the per-leg scaled weights (list aligned with legs).
    """
    idx, cols = prices.index, prices.columns
    raw = pd.DataFrame(0.0, index=idx, columns=cols)
    per_seg_n = {}
    for leg in legs:
        per_seg_n[leg["seg"][0]] = per_seg_n.get(leg["seg"][0], 0) + 1
    scaled = []
    for leg in legs:
        wy, wx = leg["wy"], leg["wx"]
        if sizing == "fixed":
            g = (wy.abs() + wx.abs()).replace(0, np.nan)
            k = 1.0 / per_seg_n[leg["seg"][0]]
            wy, wx = (wy / g * k).fillna(0.0), (wx / g * k).fillna(0.0)
        scaled.append((wy, wx))
        raw.loc[leg["seg"], leg["y"]] += wy
        raw.loc[leg["seg"], leg["x"]] += wx

    mask = membership.reindex_like(raw).fillna(False)
    if sizing == "l1_book":
        s1 = 1.0 / raw.abs().sum(axis=1).replace(0, np.nan)
        book = raw.mul(s1, axis=0).fillna(0.0).where(mask, 0.0)
        s2 = 1.0 / book.abs().sum(axis=1).replace(0, np.nan)
        book = book.mul(s2, axis=0).fillna(0.0)
        scale = (s1 * s2).fillna(0.0)
    else:
        book = raw.where(mask, 0.0)
        scale = pd.Series(1.0, index=idx)
    out = []
    for leg, (wy, wx) in zip(legs, scaled):
        sc = scale.reindex(leg["seg"])
        my = mask[leg["y"]].reindex(leg["seg"])
        mx = mask[leg["x"]].reindex(leg["seg"])
        out.append((wy * sc * my, wx * sc * mx))
    return book, out


def book_stats(prices, book, label):
    rets = prices.pct_change().fillna(0.0)
    lag = book.shift(1).fillna(0.0)
    gross = (lag * rets).sum(axis=1)
    turnover = book.diff().abs().sum(axis=1).fillna(0.0)
    short = lag.clip(upper=0.0).abs().sum(axis=1)
    net = gross - turnover * TRADE_BPS - short * BORROW_DAILY
    live = book.index >= prices.index[config.COINT_FORMATION_DAYS]
    g, nt = gross[live], net[live]
    gross_exp = book.abs().sum(axis=1)[live]
    return {
        "book": label,
        "gross_cagr": _cagr(g),
        "net_cagr": _cagr(nt),
        "ann_turnover": float(turnover[live].mean() * ANN),
        "trade_cost_yr": float((turnover[live] * TRADE_BPS).mean() * ANN),
        "borrow_cost_yr": float((short[live] * BORROW_DAILY).mean() * ANN),
        "avg_gross": float(gross_exp.mean()),
        "turnover_per_gross": float(turnover[live].mean() * ANN / max(gross_exp.mean(), 1e-9)),
        "gross_sharpe": float(g.mean() / g.std(ddof=0) * np.sqrt(ANN)) if g.std() > 0 else 0.0,
        "avg_net_exposure": float(book.sum(axis=1)[live].abs().mean()),
    }, gross, net


def trade_table(prices, legs, scaled):
    rets = prices.pct_change().fillna(0.0)
    rows = []
    for leg, (wy, wx) in zip(legs, scaled):
        seg, pos, z = leg["seg"], leg["pos"], leg["z"]
        pnl = (wy.shift(1).fillna(0.0) * rets[leg["y"]].reindex(seg)
               + wx.shift(1).fillna(0.0) * rets[leg["x"]].reindex(seg))
        cost = ((wy.diff().abs().fillna(wy.abs()) + wx.diff().abs().fillna(wx.abs())) * TRADE_BPS)
        trade_id = ((pos != 0) & (pos.shift(1, fill_value=0) == 0)).cumsum().where(pos != 0)
        for tid, days in pos.groupby(trade_id):
            d = days.index
            last = d[-1]
            nxt = seg.get_loc(last) + 1
            if nxt >= len(seg):
                reason = "refresh_cut"
                exit_day = last
            else:
                exit_day = seg[nxt]
                zv = abs(z.loc[exit_day]) if np.isfinite(z.loc[exit_day]) else np.nan
                if zv >= config.STATARB_STOP_Z:
                    reason = "stop"
                elif zv <= config.STATARB_EXIT_Z:
                    reason = "take_profit"
                else:
                    reason = "timeout"
            span = seg[(seg > d[0]) & (seg <= exit_day)]
            hedge_ratio = (wx.loc[d].abs() / wy.loc[d].abs().replace(0, np.nan)).median()
            rows.append({
                "y": leg["y"], "x": leg["x"], "entry": d[0], "exit": exit_day, "days": len(d),
                "side": float(days.iloc[0]), "reason": reason,
                "gross_pnl": float(pnl.reindex(span).sum()),
                "cost": float(cost.reindex(d.union(span)).sum()),
                "hedge_ratio": float(hedge_ratio) if np.isfinite(hedge_ratio) else np.nan,
                "naked_days": int(((wy.loc[d] != 0) ^ (wx.loc[d] != 0)).sum()),
            })
    t = pd.DataFrame(rows)
    t["net_pnl"] = t["gross_pnl"] - t["cost"]
    return t


def main() -> None:
    out_dir = ROOT / "output" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    membership = membership_matrix(prices.index, list(prices.columns), index=config.PIT_INDEX)
    log = pd.read_csv(ROOT / "output" / "rolling_pairs_log.csv", parse_dates=["refresh_date"])
    fallback = [(normalize_ticker(a), normalize_ticker(b)) for a, b in config.PAIRS]

    legs = build_pair_legs(prices, log, fallback, beta_mode="daily", hedge="beta_units")
    book, scaled = book_from_legs(prices, legs, membership, sizing="l1_book")
    prod = weights_from_pair_log(
        prices, log, lookback=config.STATARB_LOOKBACK, entry_z=config.STATARB_ENTRY_Z,
        exit_z=config.STATARB_EXIT_Z, stop_z=config.STATARB_STOP_Z,
        timeout_mult=config.STATARB_HL_TIMEOUT_MULT, membership=membership,
        formation_days=config.COINT_FORMATION_DAYS, refresh_days=config.COINT_REFRESH_DAYS,
        fallback_pairs=fallback,
    )
    print(f"Reconstruction max |diff| vs production weights: {float((book - prod).abs().max().max()):.2e}")

    # 1 + 2: gross vs net and turnover sources under counterfactual books
    print("\n=== 1-2. Gross vs net, turnover by construction rule ===")
    stats, yearly = [], {}
    variants = [
        ("production: daily beta, beta units, L1 book", "daily", "beta_units", "l1_book"),
        ("entry beta, beta units, L1 book", "entry", "beta_units", "l1_book"),
        ("daily beta, beta units, fixed per pair", "daily", "beta_units", "fixed"),
        ("entry beta, beta units, fixed per pair", "entry", "beta_units", "fixed"),
        ("daily beta, dollar hedge, L1 book", "daily", "dollar", "l1_book"),
        ("entry beta, dollar hedge, fixed per pair", "entry", "dollar", "fixed"),
    ]
    for label, bm, hd, sz in variants:
        lg = legs if (bm, hd) == ("daily", "beta_units") else build_pair_legs(
            prices, log, fallback, beta_mode=bm, hedge=hd)
        bk = book if (bm, hd, sz) == ("daily", "beta_units", "l1_book") else book_from_legs(
            prices, lg, membership, sizing=sz)[0]
        s, g, n = book_stats(prices, bk, label)
        stats.append(s)
        if sz == "l1_book" and bm == "daily" and hd == "beta_units":
            live = g.index >= prices.index[config.COINT_FORMATION_DAYS]
            yearly["gross"] = (1 + g[live]).groupby(g[live].index.year).prod() - 1
            yearly["net"] = (1 + n[live]).groupby(n[live].index.year).prod() - 1
    st = pd.DataFrame(stats)
    st.to_csv(out_dir / "construction_variants.csv", index=False)
    print(st.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nProduction book by year (gross vs net, %):")
    print((pd.DataFrame(yearly) * 100).round(1).T.to_string())

    # 3: trade-level P&L of the production book
    print("\n=== 3. Trades of the production book ===")
    trades = trade_table(prices, legs, scaled)
    trades.to_csv(out_dir / "trades.csv", index=False)
    print(f"trades {len(trades)}, gross hit rate {(trades.gross_pnl > 0).mean():.1%}, "
          f"avg days {trades.days.mean():.1f}")
    by_reason = trades.groupby("reason").agg(
        n=("net_pnl", "size"), gross=("gross_pnl", "sum"), cost=("cost", "sum"),
        net=("net_pnl", "sum"), hit=("gross_pnl", lambda s: (s > 0).mean()), days=("days", "mean"))
    print("\nBy exit reason (P&L in book-return units, summed):")
    print(by_reason.to_string(float_format=lambda v: f"{v:.4f}"))

    pairs = trades.groupby(["y", "x"]).agg(
        n=("net_pnl", "size"), gross=("gross_pnl", "sum"), cost=("cost", "sum"), net=("net_pnl", "sum")
    ).sort_values("net")
    pairs.to_csv(out_dir / "pair_pnl.csv")
    total_loss = pairs.loc[pairs.net < 0, "net"].sum()
    worst10 = pairs.head(max(1, len(pairs) // 10))["net"].sum()
    print(f"\nPairs {len(pairs)}: losing {int((pairs.net < 0).sum())}, "
          f"worst 10% of pairs = {worst10 / total_loss:.0%} of all pair losses")
    print("Worst 8 pairs:\n" + pairs.head(8).to_string(float_format=lambda v: f"{v:.4f}"))
    print("Best 5 pairs:\n" + pairs.tail(5).to_string(float_format=lambda v: f"{v:.4f}"))

    hr = trades["hedge_ratio"].dropna()
    print(f"\nHedge |w_x|/|w_y| per trade: median {hr.median():.2f}, "
          f"5% {hr.quantile(0.05):.2f}, 95% {hr.quantile(0.95):.2f}")
    print(f"Trades with a naked leg: {(trades.naked_days > 0).mean():.1%} "
          f"(total naked leg-days {int(trades.naked_days.sum())})")


if __name__ == "__main__":
    main()
