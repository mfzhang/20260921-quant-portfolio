"""
OpenDART (opendart.fss.or.kr) client: corp codes, filing lists, structured
major-event reports and financial statements. Needs DART_API_KEY in .env
(free; 20,000 requests/day). Every response is cached under data/cache/kr/dart/.

The filing list carries the receipt date (rcept_dt) but not the time, and many
filings land after the close, so events should be traded from the next session.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

from data.providers import env_value

BASE = "https://opendart.fss.or.kr/api"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "kr" / "dart"
_PAUSE_SEC = 0.1

# Structured major-event reports (주요사항보고서)
EVENT_ENDPOINTS = {
    "rights_offering": "piicDecsn",       # 유상증자 결정
    "bonus_issue": "fricDecsn",           # 무상증자 결정
    "buyback": "tsstkAqDecsn",            # 자기주식 취득 결정
    "treasury_sale": "tsstkDpDecsn",      # 자기주식 처분 결정
    "buyback_trust": "tsstkAqTrctrCnsDecsn",  # 자기주식취득 신탁계약 체결 결정
    "convertible_bond": "cvbdIsDecsn",    # 전환사채권 발행결정
    "bond_with_warrant": "bdwtIsDecsn",   # 신주인수권부사채권 발행결정
}
REPORT_CODES = {"annual": "11011", "half": "11012", "q1": "11013", "q3": "11014"}


def _key() -> str:
    key = env_value("DART_API_KEY")
    if not key:
        raise RuntimeError("DART_API_KEY is not set; get a free key at https://opendart.fss.or.kr")
    return key


def _get_json(endpoint: str, params: dict) -> dict:
    name = endpoint + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    path = CACHE_DIR / "json" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    resp = requests.get(f"{BASE}/{endpoint}.json", params={"crtfc_key": _key(), **params}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status")
    if status == "020":
        raise RuntimeError("DART daily request limit reached; re-run tomorrow (responses so far are cached)")
    if status not in ("000", "013"):  # 013 = no data
        raise RuntimeError(f"DART {endpoint} error {status}: {data.get('message')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(_PAUSE_SEC)
    return data


def corp_codes() -> pd.DataFrame:
    """corp_code <-> 6-digit stock_code for listed companies."""
    path = CACHE_DIR / "corp_codes.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str)
    resp = requests.get(f"{BASE}/corpCode.xml", params={"crtfc_key": _key()}, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        root = ElementTree.fromstring(z.read(z.namelist()[0]))
    rows = [{c.tag: (c.text or "").strip() for c in item} for item in root.iter("list")]
    df = pd.DataFrame(rows)
    df = df[df["stock_code"].str.len() == 6]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def filings(corp_code: str, start: str, end: str, kind: str | None = None) -> pd.DataFrame:
    """Filing list (rcept_dt = receipt date). kind: DART pblntf_ty, e.g. 'A' periodic,
    'B' major events, 'I' exchange disclosures (e.g. 주식소각결정)."""
    rows, page = [], 1
    while True:
        params = {"corp_code": corp_code, "bgn_de": pd.Timestamp(start).strftime("%Y%m%d"),
                  "end_de": pd.Timestamp(end).strftime("%Y%m%d"), "page_no": page, "page_count": 100}
        if kind:
            params["pblntf_ty"] = kind
        data = _get_json("list", params)
        rows += data.get("list", [])
        if page >= int(data.get("total_page", 1) or 1):
            break
        page += 1
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rcept_dt"] = pd.to_datetime(df["rcept_dt"])
    return df


def event_reports(corp_code: str, event: str, start: str, end: str) -> pd.DataFrame:
    """Structured major-event reports of one type (see EVENT_ENDPOINTS)."""
    data = _get_json(EVENT_ENDPOINTS[event], {
        "corp_code": corp_code,
        "bgn_de": pd.Timestamp(start).strftime("%Y%m%d"),
        "end_de": pd.Timestamp(end).strftime("%Y%m%d"),
    })
    df = pd.DataFrame(data.get("list", []))
    if not df.empty and "rcept_no" in df:
        df["rcept_dt"] = pd.to_datetime(df["rcept_no"].str[:8])
    return df


def financial_statements(corp_code: str, year: int, report: str = "annual", consolidated: bool = True) -> pd.DataFrame:
    """Full financial statements (단일회사 전체 재무제표) for one report, 2015 onward."""
    data = _get_json("fnlttSinglAcntAll", {
        "corp_code": corp_code, "bsns_year": str(year),
        "reprt_code": REPORT_CODES[report], "fs_div": "CFS" if consolidated else "OFS",
    })
    return pd.DataFrame(data.get("list", []))
