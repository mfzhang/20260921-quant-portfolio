"""
Event-driven pair book: one state machine per pair over the whole timeline.

Differences from rolling_pairs.weights_from_pair_log (kept for comparison):
  - spread on log prices (default): log y - beta * log x, so beta is the
    dollar hedge ratio and the traded legs match the spread being scored
  - beta, spread mean and spread std are frozen at entry; exits are judged on
    that frozen spread, so the position never re-hedges while open
  - fixed dollar budget per pair (no book-wide re-normalization)
  - an open trade is carried across refreshes until its own exit rule fires
  - if either leg leaves the index, both legs are closed (no naked leg)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADE_KEYS = (
    "y", "x", "sector", "side", "beta", "wy", "wx", "entry_abs_z", "peak_abs_z",
    "trend_score", "jump_score", "hl_ratio",
)


def _ar1_half_life(s: np.ndarray) -> float:
    """Half-life of ds_t = a + b * s_{t-1}; inf when the spread does not revert (b >= 0)."""
    lag, ds = s[:-1], np.diff(s)
    dev = lag - lag.mean()
    sxx = float(dev @ dev)
    if sxx <= 0:
        return np.inf
    b = float(dev @ (ds - ds.mean())) / sxx
    if b >= 0 or b <= -1:
        return np.inf if b >= 0 else 0.0
    return float(-np.log(2.0) / np.log1p(b))


def _idio_return_z(prices: pd.DataFrame, vol_window: int) -> np.ndarray:
    """Daily log return minus the equal-weight universe return, in units of the
    name's trailing (through yesterday) std of that excess return."""
    rets = np.log(prices).diff()
    excess = rets.sub(rets.mean(axis=1), axis=0)
    sd = excess.rolling(vol_window, min_periods=vol_window // 2).std().shift(1)
    return (excess / sd).to_numpy()


def _trend_score(s: np.ndarray, z_sign: float) -> float:
    """R^2 of a linear time trend on spread `s`, signed + when the trend points
    the same way as the divergence (spread drifting away, not snapping out)."""
    tt = np.arange(len(s), dtype=float)
    tt -= tt.mean()
    dev = s - s.mean()
    sxx, syy = float(tt @ tt), float(dev @ dev)
    if sxx <= 0 or syy <= 0:
        return 0.0
    slope = float(tt @ dev) / sxx
    r2 = slope * slope * sxx / syy
    return float(r2 * np.sign(slope) * z_sign)


def _spread_frame(prices: pd.DataFrame, y: str, x: str, lookback: int, spread: str) -> pd.DataFrame:
    py, px = prices[y], prices[x]
    sy, sx = (np.log(py), np.log(px)) if spread == "log" else (py, px)
    # z = today's residual of the regression fitted on the trailing window, so a
    # single beta defines the whole window (matches the spread frozen at entry)
    roll_y, roll_x = sy.rolling(lookback), sx.rolling(lookback)
    var_x = roll_x.var(ddof=0)
    cov = roll_y.cov(sx, ddof=0)
    beta = cov / var_x.replace(0, np.nan)
    resid_var = (roll_y.var(ddof=0) - beta * cov).clip(lower=0)
    resid = (sy - roll_y.mean()) - beta * (sx - roll_x.mean())
    z = resid / np.sqrt(resid_var).replace(0, np.nan)
    return pd.DataFrame({"sy": sy, "sx": sx, "beta": beta, "z": z, "py": py, "px": px})


def build_pair_book(
    prices: pd.DataFrame,
    pair_log: pd.DataFrame,
    *,
    lookback: int = 60,
    entry_z: float = 2.5,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    timeout_mult: float = 2.0,
    membership: pd.DataFrame | None = None,
    formation_days: int = 504,
    refresh_days: int = 126,
    pair_budget: float = 1.0 / 24,
    spread: str = "log",
    excluded_sectors: tuple[str, ...] = (),
    fallback_pairs: list[tuple[str, str]] | None = None,
    entry_mode: str = "cross",
    trend_window: int | None = None,
    trend_max_r2: float | None = None,
    jump_days: int = 5,
    jump_max_z: float | None = None,
    hl_window: int | None = None,
    hl_max_ratio: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (weights, trades). timeout_mult <= 0 disables the time stop
    (hold until the spread reverts or hits stop_z).

    Every trade records trend_score: signed R^2 of a linear trend on the entry
    spread over trend_window days (default 2 * lookback). If trend_max_r2 is
    set, entries with trend_score >= trend_max_r2 are skipped (the divergence
    is a steady drift in its own direction rather than a deviation to fade).

    jump_score: largest |idiosyncratic return z| of either leg over the last
    jump_days (news-like single-name move). Skipped if >= jump_max_z.
    hl_ratio: AR(1) half-life of the entry spread over hl_window days (default
    2 * lookback) divided by the screening half-life (inf if not reverting).
    Skipped if > hl_max_ratio.

    entry_mode:
      "cross"   enter the day |z| first reaches entry_z
      "turn"    arm at entry_z, enter the first day |z| falls versus the prior day
      "recross" arm at entry_z, enter when |z| falls back below entry_z
    An armed signal is dropped if |z| reaches stop_z, flips sign, or decays to exit_z.
    """
    if entry_mode not in ("cross", "turn", "recross"):
        raise ValueError(f"unknown entry_mode {entry_mode!r}")
    trend_window = int(trend_window or 2 * lookback)
    hl_window = int(hl_window or 2 * lookback)
    idio_z = np.abs(_idio_return_z(prices, lookback))
    idx = prices.index
    n = len(idx)
    log = pair_log.copy() if pair_log is not None else pd.DataFrame(columns=["refresh_date", "y", "x"])
    if excluded_sectors and "sector" in log.columns:
        log = log[~log["sector"].isin(excluded_sectors)]
    log["refresh_date"] = pd.to_datetime(log["refresh_date"]).dt.normalize()
    by_date = {d: g for d, g in log.groupby("refresh_date")}

    # day index -> pairs eligible for new entries on that day
    eligible: list[dict[tuple[str, str], tuple[float, str]]] = [dict() for _ in range(n)]
    points = list(range(formation_days, n, refresh_days))
    for k, i in enumerate(points):
        j = points[k + 1] if k + 1 < len(points) else n
        grp = by_date.get(idx[i - 1].normalize())
        if grp is not None and len(grp):
            rows = {
                (str(r.y), str(r.x)): (float(getattr(r, "half_life", np.nan)), str(getattr(r, "sector", "")))
                for r in grp.itertuples(index=False)
            }
        else:
            rows = {(y, x): (np.nan, "fallback") for y, x in (fallback_pairs or [])}
        rows = {p: v for p, v in rows.items() if p[0] in prices.columns and p[1] in prices.columns}
        for t in range(i, j):
            eligible[t] = rows

    pairs = sorted({p for day in eligible for p in day})
    frames = {p: _spread_frame(prices, p[0], p[1], lookback, spread) for p in pairs}
    member = (
        membership.reindex(index=idx, columns=prices.columns).fillna(False).to_numpy()
        if membership is not None else np.ones(prices.shape, dtype=bool)
    )
    col = {c: k for k, c in enumerate(prices.columns)}
    weights = np.zeros(prices.shape)
    trades: list[dict] = []

    for p in pairs:
        y, x = p
        f = frames[p]
        sy, sx, beta, z = (f[c].to_numpy() for c in ("sy", "sx", "beta", "z"))
        py, px = f["py"].to_numpy(), f["px"].to_numpy()
        cy, cx = col[y], col[x]
        open_trade = None
        armed_sign, peak, prev_abs = 0.0, 0.0, 0.0
        for t in range(n):
            both_members = member[t, cy] and member[t, cx]
            if open_trade is not None:
                tr = open_trade
                zt = (sy[t] - tr["beta"] * sx[t] - tr["mu"]) / tr["sd"]
                held = t - tr["t0"]
                reason = None
                if not both_members:
                    reason = "delisted_leg"
                elif not np.isfinite(zt):
                    reason = None
                elif abs(zt) >= stop_z:
                    reason = "stop"
                elif tr["side"] * zt >= -exit_z:
                    reason = "take_profit"
                elif tr["max_hold"] is not None and held >= tr["max_hold"]:
                    reason = "timeout"
                if reason is None:
                    weights[t, cy] += tr["wy"]
                    weights[t, cx] += tr["wx"]
                    continue
                trades.append({**{k: tr[k] for k in TRADE_KEYS},
                               "entry": idx[tr["t0"]], "exit": idx[t], "days": held, "reason": reason})
                open_trade = None
                continue  # no same-day re-entry

            info = eligible[t].get(p)
            if info is None or not both_members or not np.isfinite(z[t]):
                armed_sign = 0.0
                continue
            az = abs(z[t])
            if entry_mode == "cross":
                if not entry_z <= az < stop_z:
                    continue
                peak = az
            elif armed_sign == 0.0:
                if entry_z <= az < stop_z:
                    armed_sign, peak, prev_abs = np.sign(z[t]), az, az
                continue
            else:
                if az >= stop_z or np.sign(z[t]) != armed_sign or az <= exit_z:
                    armed_sign = 0.0
                    continue
                peak = max(peak, az)
                fire = az < prev_abs if entry_mode == "turn" else az < entry_z
                prev_abs = az
                if not fire:
                    continue
                armed_sign = 0.0
            b0 = beta[t]
            if not np.isfinite(b0) or b0 <= 0 or t < max(lookback, trend_window, hl_window):
                continue
            window = sy[t - lookback + 1:t + 1] - b0 * sx[t - lookback + 1:t + 1]
            mu, sd = float(np.mean(window)), float(np.std(window))
            if not np.isfinite(sd) or sd <= 0:
                continue
            long_window = sy[t - trend_window + 1:t + 1] - b0 * sx[t - trend_window + 1:t + 1]
            if not np.all(np.isfinite(long_window)):
                continue
            trend = _trend_score(long_window, float(np.sign(z[t])))
            if trend_max_r2 is not None and trend >= trend_max_r2:
                continue
            recent = idio_z[t - jump_days + 1:t + 1, [cy, cx]]
            jump = float(np.nanmax(recent)) if np.isfinite(recent).any() else 0.0
            if jump_max_z is not None and jump >= jump_max_z:
                continue
            hl, sector = info
            hl_spread = sy[t - hl_window + 1:t + 1] - b0 * sx[t - hl_window + 1:t + 1]
            hl_ratio = _ar1_half_life(hl_spread) / hl if np.isfinite(hl) and hl > 0 else np.nan
            if hl_max_ratio is not None and np.isfinite(hl) and not hl_ratio <= hl_max_ratio:
                continue
            side = -1.0 if z[t] > 0 else 1.0  # +1 = long y / short x
            hedge = b0 if spread == "log" else b0 * px[t] / py[t]
            gross = 1.0 + abs(hedge)
            max_hold = None
            if timeout_mult > 0 and np.isfinite(hl) and hl > 0:
                max_hold = max(1, int(round(hl * timeout_mult)))
            open_trade = {
                "y": y, "x": x, "sector": sector, "side": side, "beta": b0, "mu": mu, "sd": sd,
                "t0": t, "max_hold": max_hold, "entry_abs_z": az, "peak_abs_z": peak, "trend_score": trend,
                "jump_score": jump, "hl_ratio": hl_ratio,
                "wy": side * pair_budget / gross, "wx": -side * hedge * pair_budget / gross,
            }
            weights[t, cy] += open_trade["wy"]
            weights[t, cx] += open_trade["wx"]

        if open_trade is not None:
            tr = open_trade
            trades.append({**{k: tr[k] for k in TRADE_KEYS},
                           "entry": idx[tr["t0"]], "exit": idx[-1], "days": n - 1 - tr["t0"],
                           "reason": "open"})

    return pd.DataFrame(weights, index=idx, columns=prices.columns), pd.DataFrame(trades)
