"""
Point-in-time fundamentals from SEC EDGAR XBRL companyfacts (free).

Only facts whose filing date is on or before the as-of date are used, so a
screen at date d sees exactly what had been filed by d. Requires
SEC_USER_AGENT (e.g. "name research you@example.com") in the environment or .env.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from data.pit_cik_overrides import CIK_OVERRIDES
from data.providers import env_value, normalize_ticker

SEC_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "sec"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
TAGS = (
    "Liabilities",
    "LiabilitiesAndStockholdersEquity",
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
    "NetIncomeLoss",
    "ProfitLoss",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "Assets",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "LongTermDebtNoncurrent",
    "LongTermDebt",
    "NetCashProvidedByUsedInOperatingActivities",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "GrossProfit",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "OperatingIncomeLoss",
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "LongTermDebtCurrent",
    "DebtCurrent",
    "ShortTermBorrowings",
)
_CACHE_VERSION = "v5"  # bump when TAGS change so cached extracts are refetched
_PAUSE_SEC = 0.15  # SEC fair-access limit is 10 requests/second


def _fetch_facts(cik: int) -> pd.DataFrame:
    agent = env_value("SEC_USER_AGENT")
    if not agent:
        raise RuntimeError('SEC_USER_AGENT is not set (e.g. "name research you@example.com")')
    resp = requests.get(
        FACTS_URL.format(cik=cik),
        headers={"User-Agent": agent, "Accept-Encoding": "gzip, deflate"},
        timeout=60,
    )
    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()
    gaap = resp.json().get("facts", {}).get("us-gaap", {})
    rows = []
    for tag in TAGS:
        for unit, facts in gaap.get(tag, {}).get("units", {}).items():
            for f in facts:
                if f.get("form") in FORMS:
                    rows.append({"tag": tag, "unit": unit, "start": f.get("start"), "end": f["end"],
                                 "val": f["val"], "filed": f["filed"], "form": f["form"]})
    return pd.DataFrame(rows)


def load_facts(cik: int) -> pd.DataFrame:
    SEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = SEC_CACHE_DIR / f"CIK{cik:010d}_{_CACHE_VERSION}.csv"
    if path.exists():
        df = pd.read_csv(path) if path.stat().st_size > 1 else pd.DataFrame()
    else:
        df = _fetch_facts(cik)
        df.to_csv(path, index=False)
        time.sleep(_PAUSE_SEC)
    if df.empty:
        return df
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def _instant_asof(facts: pd.DataFrame, tag: str, asof: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    """Latest balance-sheet value (by period end) filed on or before `asof`."""
    f = facts[(facts["tag"] == tag) & facts["start"].isna() & (facts["filed"] <= asof)]
    if f.empty:
        return np.nan, None
    f = f.sort_values(["end", "filed"])
    last = f.iloc[-1]
    return float(last["val"]), last["end"]


def _equity_asof(facts: pd.DataFrame, asof: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    """Total equity (incl. minority interest where reported), latest period end."""
    eq, eq_end = _instant_asof(facts, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", asof)
    eq_p, eq_p_end = _instant_asof(facts, "StockholdersEquity", asof)
    if eq_end is None or (eq_p_end is not None and eq_p_end > eq_end):
        eq, eq_end = eq_p, eq_p_end
    return eq, eq_end


def debt_ratio_asof(facts: pd.DataFrame, asof: pd.Timestamp) -> float:
    """Total liabilities / total equity (incl. minority interest); NaN if equity <= 0."""
    eq, eq_end = _equity_asof(facts, asof)
    if eq_end is None or not eq > 0:
        return np.nan
    liab, liab_end = _instant_asof(facts, "Liabilities", asof)
    if liab_end != eq_end:
        total, total_end = _instant_asof(facts, "LiabilitiesAndStockholdersEquity", asof)
        if total_end != eq_end:
            return np.nan
        liab = total - eq
    return float(liab / eq)


def ttm_eps_asof(facts: pd.DataFrame, asof: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    """
    Trailing-12-month diluted EPS from filings on or before `asof`.
    Returns (eps, filing date of the newest fact used) in that filing's share basis.
    """
    # diluted where reported, basic fills periods that only carry basic
    return _ttm_flow(facts, ("EarningsPerShareDiluted", "EarningsPerShareBasic"), asof)


def roe_asof(facts: pd.DataFrame, asof: pd.Timestamp) -> float:
    """TTM net income (parent, else consolidated) / latest total equity; NaN if equity <= 0."""
    ni, _ = _ttm_flow(facts, ("NetIncomeLoss", "ProfitLoss"), asof)
    eq, eq_end = _equity_asof(facts, asof)
    if eq_end is None or not eq > 0 or not np.isfinite(ni):
        return np.nan
    return float(ni / eq)


def _ttm_flow(facts: pd.DataFrame, tags: tuple[str, ...], asof: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    """
    Trailing-12-month value of a flow item from filings on or before `asof`:
    last fiscal year + current year-to-date - prior year-to-date. Earlier tags
    in `tags` win for the same period; later ones fill periods the first lacks.
    """
    f = facts[facts["tag"].isin(tags) & facts["start"].notna() & (facts["filed"] <= asof)].copy()
    if f.empty:
        return np.nan, None
    f["days"] = (f["end"] - f["start"]).dt.days
    f["pref"] = f["tag"].map({t: len(tags) - k for k, t in enumerate(tags)})
    f = f.sort_values(["pref", "filed"]).drop_duplicates(["start", "end"], keep="last")
    annual = f[f["days"].between(340, 380)].sort_values("end")
    if annual.empty:
        return np.nan, None
    a = annual.iloc[-1]
    if (asof - a["end"]).days > 550:
        return np.nan, None
    eps, filed = float(a["val"]), a["filed"]
    ytd = f[(f["end"] > a["end"]) & (f["days"] < 340)
            & ((f["start"] - a["end"]).dt.days.abs() <= 10)].sort_values("end")
    if not ytd.empty:
        y = ytd.iloc[-1]
        prior = f[((f["end"] - (y["end"] - pd.Timedelta(days=365))).dt.days.abs() <= 10)
                  & ((f["days"] - y["days"]).abs() <= 10)]
        if not prior.empty:
            eps = eps + float(y["val"]) - float(prior.iloc[-1]["val"])
            filed = max(filed, y["filed"])
    return eps, filed


def share_growth_asof(facts: pd.DataFrame, asof: pd.Timestamp, max_age_days: int = 200) -> float:
    """
    Year-over-year change in weighted-average diluted (else basic) shares for the
    latest reported quarter or fiscal year, both values taken from the same
    filing so the prior-year comparative is already restated for splits.
    """
    tags = ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic")
    f = facts[facts["tag"].isin(tags) & facts["start"].notna() & (facts["filed"] <= asof)].copy()
    if f.empty:
        return np.nan
    f["days"] = (f["end"] - f["start"]).dt.days
    f = f[f["days"].between(80, 100) | f["days"].between(340, 380)]
    for tag in tags:
        g = f[f["tag"] == tag]
        if g.empty:
            continue
        latest = g.sort_values(["end", "filed"]).iloc[-1]
        if (asof - latest["end"]).days > max_age_days or not latest["val"] > 0:
            return np.nan
        prior = g[(g["filed"] == latest["filed"])
                  & ((g["days"] - latest["days"]).abs() <= 10)
                  & ((g["end"] - (latest["end"] - pd.Timedelta(days=365))).dt.days.abs() <= 15)]
        if not prior.empty and prior.iloc[-1]["val"] > 0:
            return float(latest["val"] / prior.iloc[-1]["val"] - 1.0)
    return np.nan


_SHARES = ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic")
_REVENUE = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
_COGS = ("CostOfRevenue", "CostOfGoodsAndServicesSold")
_LTD = ("LongTermDebtNoncurrent", "LongTermDebt")
FSCORE_ITEMS = ("roa", "cfo", "d_roa", "accrual", "d_lever", "d_liquid", "neqiss", "d_margin", "d_turn")


def _filing_value(facts: pd.DataFrame, tags: tuple[str, ...], filed: pd.Timestamp,
                  end: pd.Timestamp, annual: bool) -> float:
    """Value for the period ending near `end` as reported in the filing dated
    `filed` (comparatives included). annual=True: ~1-year duration; else instant."""
    f = facts[facts["tag"].isin(tags) & (facts["filed"] == filed)
              & ((facts["end"] - end).dt.days.abs() <= 15)]
    if annual:
        f = f[f["start"].notna() & (f["end"] - f["start"]).dt.days.between(340, 380)]
    else:
        f = f[f["start"].isna()]
    for tag in tags:
        g = f[f["tag"] == tag]
        if not g.empty:
            return float(g.iloc[-1]["val"])
    return np.nan


def fscore_asof(facts: pd.DataFrame, asof: pd.Timestamp, max_age_days: int = 470) -> dict:
    """
    Piotroski F-score from the latest 10-K filed on or before `asof`; current and
    prior fiscal year both come from that filing. The equity-offering item is
    replaced by net share issuance: shares down -> 2, up -> 0, unchanged -> 1,
    so the maximum is 10. Ratios use year-end assets. Missing items score 0.
    """
    out = {k: 0 for k in FSCORE_ITEMS}
    out["score"] = np.nan
    ann = facts[facts["form"].isin(["10-K", "10-K/A"]) & (facts["filed"] <= asof)
                & facts["start"].notna() & (facts["end"] - facts["start"]).dt.days.between(340, 380)]
    if ann.empty:
        return out
    filed = ann["filed"].max()
    end = ann.loc[ann["filed"] == filed, "end"].max()
    if (asof - end).days > max_age_days:
        return out
    prev = end - pd.Timedelta(days=365)

    def flow(tags, e):
        return _filing_value(facts, tags, filed, e, annual=True)

    def stock(tags, e):
        return _filing_value(facts, tags, filed, e, annual=False)

    ni, ni_p = flow(("NetIncomeLoss", "ProfitLoss"), end), flow(("NetIncomeLoss", "ProfitLoss"), prev)
    cfo = flow(("NetCashProvidedByUsedInOperatingActivities",), end)
    assets, assets_p = stock(("Assets",), end), stock(("Assets",), prev)
    rev, rev_p = flow(_REVENUE, end), flow(_REVENUE, prev)
    gp, gp_p = flow(("GrossProfit",), end), flow(("GrossProfit",), prev)
    if not np.isfinite(gp):
        gp = rev - flow(_COGS, end)
    if not np.isfinite(gp_p):
        gp_p = rev_p - flow(_COGS, prev)
    ltd, ltd_p = stock(_LTD, end), stock(_LTD, prev)
    ca, ca_p = stock(("AssetsCurrent",), end), stock(("AssetsCurrent",), prev)
    cl, cl_p = stock(("LiabilitiesCurrent",), end), stock(("LiabilitiesCurrent",), prev)
    sh, sh_p = flow(_SHARES, end), flow(_SHARES, prev)

    def ok(*v):
        return all(np.isfinite(x) for x in v)

    if ok(ni, assets) and assets > 0:
        out["roa"] = int(ni > 0)
        if ok(ni_p, assets_p) and assets_p > 0:
            out["d_roa"] = int(ni / assets > ni_p / assets_p)
    if ok(cfo):
        out["cfo"] = int(cfo > 0)
        if ok(ni):
            out["accrual"] = int(cfo > ni)
    if ok(assets, assets_p) and assets > 0 and assets_p > 0:
        lev = (ltd if ok(ltd) else 0.0) / assets
        lev_p = (ltd_p if ok(ltd_p) else 0.0) / assets_p
        out["d_lever"] = int(lev < lev_p or (lev == 0 and lev_p == 0))
        if ok(rev, rev_p):
            out["d_turn"] = int(rev / assets > rev_p / assets_p)
    if ok(ca, cl, ca_p, cl_p) and cl > 0 and cl_p > 0:
        out["d_liquid"] = int(ca / cl > ca_p / cl_p)
    if ok(sh, sh_p) and sh > 0 and sh_p > 0:
        out["neqiss"] = 2 if sh < sh_p else (0 if sh > sh_p else 1)
    if ok(gp, rev, gp_p, rev_p) and rev > 0 and rev_p > 0:
        out["d_margin"] = int(gp / rev > gp_p / rev_p)
    out["score"] = sum(out[k] for k in FSCORE_ITEMS)
    return out


def share_events(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per 10-Q/10-K filing: filing date and the YoY change in weighted
    diluted (else basic) shares for the latest period in that filing."""
    f = facts[facts["tag"].isin(_SHARES) & facts["start"].notna()].copy()
    if f.empty:
        return pd.DataFrame(columns=["filed", "share_growth"])
    f["days"] = (f["end"] - f["start"]).dt.days
    f = f[f["days"].between(80, 100) | f["days"].between(340, 380)]
    rows = []
    for filed, g in f.groupby("filed"):
        for tag in _SHARES:
            h = g[g["tag"] == tag]
            if h.empty:
                continue
            latest = h.sort_values("end").iloc[-1]
            prior = h[((h["days"] - latest["days"]).abs() <= 10)
                      & ((h["end"] - (latest["end"] - pd.Timedelta(days=365))).dt.days.abs() <= 15)]
            if not prior.empty and prior.iloc[-1]["val"] > 0 and (filed - latest["end"]).days <= 120:
                rows.append({"filed": filed, "share_growth": float(latest["val"] / prior.iloc[-1]["val"] - 1)})
            break
    return pd.DataFrame(rows)


def issuance_table(dates: list[pd.Timestamp], tickers: list[str], ciks: dict[str, int]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        cik = ciks.get(t)
        facts = load_facts(cik) if cik else pd.DataFrame()
        for d in dates:
            g = share_growth_asof(facts, d) if not facts.empty else np.nan
            rows.append({"date": d, "ticker": t, "share_growth": g})
    return pd.DataFrame(rows)


def ticker_ciks(history: pd.DataFrame) -> dict[str, int]:
    h = history.dropna(subset=["cik"]).copy()
    h["ticker"] = h["ticker"].map(normalize_ticker)
    h = h.sort_values("as_of").drop_duplicates("ticker", keep="last")
    return {**CIK_OVERRIDES, **{t: int(c) for t, c in zip(h["ticker"], h["cik"])}}


def _split_factor(adj: pd.Series, raw: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    """Share-count multiple from split(s) between d0 and d1, read off the jump in
    raw/adjusted price (dividends move that ratio only slightly)."""
    ratio = (raw / adj).dropna()
    r0, r1 = ratio.asof(d0), ratio.asof(d1)
    if not (np.isfinite(r0) and np.isfinite(r1)) or r1 <= 0:
        return 1.0
    k = r0 / r1
    if k > 1.3:
        return round(k * 2) / 2 if k < 5 else float(round(k))
    if k < 1 / 1.3:
        inv = 1 / k
        return 1 / (round(inv * 2) / 2 if inv < 5 else float(round(inv)))
    return 1.0


def book_per_share_asof(facts: pd.DataFrame, asof: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    """Latest total equity / weighted diluted (else basic) shares of the latest
    reported period, in the share basis of the filing that carried the shares."""
    eq, eq_end = _equity_asof(facts, asof)
    if eq_end is None or (asof - eq_end).days > 200:
        return np.nan, None
    f = facts[facts["tag"].isin(_SHARES) & facts["start"].notna() & (facts["filed"] <= asof)]
    for tag in _SHARES:
        g = f[f["tag"] == tag]
        if g.empty:
            continue
        last = g.sort_values(["end", "filed"]).iloc[-1]
        if last["val"] > 0 and abs((last["end"] - eq_end).days) <= 100:
            return float(eq / (last["val"] * _share_scale(facts, asof, last["val"]))), last["filed"]
    return np.nan, None


def _share_scale(facts: pd.DataFrame, asof: pd.Timestamp, shares: float) -> float:
    """Some filers tag share counts in thousands or millions; recover the unit
    from net income / shares vs reported EPS (powers of 1000 only)."""
    ni, _ = _ttm_flow(facts, ("NetIncomeLoss", "ProfitLoss"), asof)
    eps, _ = ttm_eps_asof(facts, asof)
    if not (np.isfinite(ni) and np.isfinite(eps)) or abs(eps) < 0.05 or ni * eps <= 0:
        return 1.0
    k = round(np.log(ni / shares / eps) / np.log(1000))
    return float(1000 ** k) if k > 0 else 1.0


def book_to_market_table(
    dates: list[pd.Timestamp],
    tickers: list[str],
    adj_prices: pd.DataFrame,
    raw_prices: pd.DataFrame,
    ciks: dict[str, int],
) -> pd.DataFrame:
    """Per (date, ticker): book equity per share (split-corrected to `date`) / as-traded close."""
    rows = []
    for t in tickers:
        cik = ciks.get(t)
        facts = load_facts(cik) if cik else pd.DataFrame()
        for d in dates:
            bm = np.nan
            if not facts.empty and t in raw_prices and t in adj_prices:
                bps, filed = book_per_share_asof(facts, d)
                price = raw_prices[t].dropna().asof(d)
                if filed is not None and np.isfinite(price) and price > 0:
                    bm = float(bps / _split_factor(adj_prices[t], raw_prices[t], filed, d) / price)
            rows.append({"date": d, "ticker": t, "bm": bm})
    return pd.DataFrame(rows)


VALUE_METRICS = ("ep", "bm", "cfp", "sp", "ebit_ev")


def _recent_instant(facts: pd.DataFrame, tags: tuple[str, ...], asof: pd.Timestamp, max_age_days: int = 200) -> float:
    """First tag with a balance-sheet value at most `max_age_days` old; NaN if none."""
    for tag in tags:
        val, end = _instant_asof(facts, tag, asof)
        if end is not None and (asof - end).days <= max_age_days:
            return val
    return np.nan


def value_metrics_asof(facts: pd.DataFrame, asof: pd.Timestamp, adj: pd.Series, raw: pd.Series) -> dict:
    """
    Yields on market value at `asof`: TTM net income, latest book equity, TTM
    operating cash flow and TTM revenue over market cap, and TTM operating income
    over enterprise value (market cap + debt - cash). Market cap = latest weighted
    diluted shares (split-corrected to `asof`) x as-traded close. Negative values
    stay (they rank last); missing inputs give NaN.
    """
    out = {k: np.nan for k in VALUE_METRICS}
    out["mcap"] = np.nan
    f = facts[facts["tag"].isin(_SHARES) & facts["start"].notna() & (facts["filed"] <= asof)]
    shares, filed = np.nan, None
    for tag in _SHARES:
        g = f[f["tag"] == tag]
        if not g.empty:
            last = g.sort_values(["end", "filed"]).iloc[-1]
            if last["val"] > 0 and (asof - last["end"]).days <= 200:
                shares, filed = float(last["val"]) * _share_scale(facts, asof, last["val"]), last["filed"]
                break
    price = raw.dropna().asof(asof)
    if filed is None or not (np.isfinite(price) and price > 0):
        return out
    mcap = shares * _split_factor(adj, raw, filed, asof) * price
    out["mcap"] = mcap
    ni, _ = _ttm_flow(facts, ("NetIncomeLoss", "ProfitLoss"), asof)
    cfo, _ = _ttm_flow(facts, ("NetCashProvidedByUsedInOperatingActivities",), asof)
    rev, _ = _ttm_flow(facts, _REVENUE, asof)
    ebit, _ = _ttm_flow(facts, ("OperatingIncomeLoss",), asof)
    eq, eq_end = _equity_asof(facts, asof)
    out["ep"], out["cfp"], out["sp"] = ni / mcap, cfo / mcap, rev / mcap
    if eq_end is not None and (asof - eq_end).days <= 200:
        out["bm"] = eq / mcap
    lt = _recent_instant(facts, _LTD, asof)
    cur = _recent_instant(facts, ("LongTermDebtCurrent", "DebtCurrent"), asof)
    st = _recent_instant(facts, ("ShortTermBorrowings",), asof)
    cash = _recent_instant(facts, ("CashAndCashEquivalentsAtCarryingValue",
                                   "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"), asof)
    ev = mcap + sum(x for x in (lt, cur, st) if np.isfinite(x)) - (cash if np.isfinite(cash) else 0.0)
    if ev > 0:
        out["ebit_ev"] = ebit / ev
    return out


def value_metrics_table(
    dates: list[pd.Timestamp],
    tickers: list[str],
    adj_prices: pd.DataFrame,
    raw_prices: pd.DataFrame,
    ciks: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        cik = ciks.get(t)
        facts = load_facts(cik) if cik else pd.DataFrame()
        for d in dates:
            row = {"date": d, "ticker": t, **{k: np.nan for k in VALUE_METRICS}, "mcap": np.nan}
            if not facts.empty and t in raw_prices and t in adj_prices:
                row.update(value_metrics_asof(facts, d, adj_prices[t], raw_prices[t]))
            rows.append(row)
    return pd.DataFrame(rows)


def value_screen_table(
    dates: list[pd.Timestamp],
    tickers: list[str],
    adj_prices: pd.DataFrame,
    raw_prices: pd.DataFrame,
    ciks: dict[str, int],
) -> pd.DataFrame:
    """
    Per (date, ticker): debt_ratio (liabilities/equity), P/E (as-traded price /
    TTM EPS, profitable only), earnings yield ey (TTM EPS / price, any sign) and
    roe (TTM net income / equity).
    """
    rows = []
    for t in tickers:
        cik = ciks.get(t)
        facts = load_facts(cik) if cik else pd.DataFrame()
        for d in dates:
            row = {"date": d, "ticker": t, "debt_ratio": np.nan, "pe": np.nan, "ey": np.nan, "roe": np.nan}
            if not facts.empty:
                row["debt_ratio"] = debt_ratio_asof(facts, d)
                row["roe"] = roe_asof(facts, d)
                eps, filed = ttm_eps_asof(facts, d)
                if filed is not None and t in raw_prices and t in adj_prices:
                    price = raw_prices[t].dropna().asof(d)
                    eps_now = eps / _split_factor(adj_prices[t], raw_prices[t], filed, d)
                    if np.isfinite(price) and price > 0 and np.isfinite(eps_now):
                        row["ey"] = float(eps_now / price)
                        if eps_now > 0:
                            row["pe"] = float(price / eps_now)
            rows.append(row)
    return pd.DataFrame(rows)
