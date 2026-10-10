"""
Ticker -> SEC CIK for names pitindex leaves blank (mostly mid/small caps that
were acquired or delisted): SEC's current ticker map first, then an exact
match of the normalized company name against every EDGAR entity name.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from data.providers import env_value, normalize_ticker

CACHE = Path(__file__).resolve().parent / "cache" / "sec"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
NAMES_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
_SUFFIXES = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC",
             "HOLDINGS", "HOLDING", "GROUP", "THE", "LLC", "LP", "NV", "SA", "AG", "DE", "NEW"}


def _get(url: str, path: Path) -> Path:
    if not path.exists():
        agent = env_value("SEC_USER_AGENT")
        if not agent:
            raise RuntimeError('SEC_USER_AGENT is not set (e.g. "name research you@example.com")')
        CACHE.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, headers={"User-Agent": agent}, timeout=300)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def normalize_name(name: str) -> str:
    words = re.sub(r"[^A-Z0-9 ]", " ", name.upper().replace("&", " AND ").replace("'", "")).split()
    return " ".join(w for w in words if w not in _SUFFIXES)


def sec_ticker_map() -> dict[str, int]:
    data = json.loads(_get(TICKERS_URL, CACHE / "company_tickers.json").read_text(encoding="utf-8"))
    return {normalize_ticker(v["ticker"]): int(v["cik_str"]) for v in data.values()}


def sec_names() -> dict[str, set[int]]:
    """Normalized EDGAR entity name -> every CIK that has used it."""
    out: dict[str, set[int]] = {}
    path = _get(NAMES_URL, CACHE / "cik-lookup-data.txt")
    with path.open(encoding="latin-1") as fh:
        for line in fh:
            name, _, rest = line.rstrip().rstrip(":").rpartition(":")
            if name and rest.isdigit():
                out.setdefault(normalize_name(name), set()).add(int(rest))
    return out


def sec_name_map() -> dict[str, int]:
    """Normalized EDGAR entity name -> CIK; names shared by several CIKs are dropped."""
    return {k: next(iter(v)) for k, v in sec_names().items() if len(v) == 1}


def sic_codes(ciks: list[int]) -> dict[int, int]:
    """SEC SIC industry code per CIK (from the submissions endpoint, cached)."""
    path = CACHE / "sic_codes.json"
    cache = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
    agent = env_value("SEC_USER_AGENT")
    for cik in set(ciks) - set(cache):
        resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                            headers={"User-Agent": agent}, timeout=60)
        sic = resp.json().get("sic") if resp.status_code == 200 else None
        cache[cik] = int(sic) if sic else None
        time.sleep(0.15)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in cache.items()}))
    return {c: cache[c] for c in ciks if cache.get(c) is not None}


def fill_ciks(roster: dict[str, str | None], known: dict[str, int]) -> dict[str, int]:
    """roster: ticker -> company name (or None). Returns known plus whatever an
    exact name match, then SEC's current ticker map (tickers get reused), can add."""
    out = dict(known)
    tick, names = sec_ticker_map(), sec_name_map()
    for t, name in roster.items():
        t = normalize_ticker(t)
        if t in out:
            continue
        cik = names.get(normalize_name(name)) if name else None
        if cik or t in tick:
            out[t] = cik or tick[t]
    return out
