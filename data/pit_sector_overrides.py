"""
GICS sectors for S&P 500 members that pitindex lists without sector metadata
(mostly names removed from the index after 2018). Sectors follow the post-2018
GICS layout (Communication Services) to match pitindex's labels for live members.
"""

from __future__ import annotations

_IT = "Information Technology"
_HC = "Health Care"
_FIN = "Financials"
_IND = "Industrials"
_CD = "Consumer Discretionary"
_CS = "Consumer Staples"
_EN = "Energy"
_MAT = "Materials"
_RE = "Real Estate"
_UT = "Utilities"
_COM = "Communication Services"

SECTOR_OVERRIDES: dict[str, str] = {
    "AAL": _IND, "AAP": _CD, "ABC": _HC, "ADS": _IT, "AET": _HC, "AGN": _HC,
    "AIV": _RE, "ALK": _IND, "ALXN": _HC, "AMG": _FIN, "ANDV": _EN, "ANSS": _IT,
    "ANTM": _HC, "APC": _EN, "ARNC": _IND, "ATVI": _COM, "AYI": _IND, "BBT": _FIN,
    "BHF": _FIN, "BHGE": _EN, "BK": _FIN, "BLL": _MAT, "BWA": _CD, "CA": _IT,
    "CAG": _CS, "CBS": _COM, "CELG": _HC, "CERN": _HC, "CHK": _EN, "CMA": _FIN,
    "COG": _EN, "COL": _IND, "COTY": _CS, "CPB": _CS, "CPRI": _CD, "CSRA": _IT,
    "CTL": _COM, "CTXS": _IT, "CXO": _EN, "DFS": _FIN, "DISCA": _COM, "DISCK": _COM,
    "DISH": _COM, "DRE": _RE, "DWDP": _MAT, "DXC": _IT, "EMN": _MAT, "ESRX": _HC,
    "ETFC": _FIN, "EVHC": _HC, "FB": _COM, "FBHS": _IND, "FL": _CD, "FLIR": _IT,
    "FLR": _IND, "FLS": _IND, "FMC": _MAT, "FTI": _EN, "GGP": _RE, "GPS": _CD,
    "GT": _CD, "HBI": _CD, "HCP": _RE, "HES": _EN, "HOG": _CD, "HOLX": _HC,
    "HP": _EN, "HRB": _CD, "HRS": _IND, "ILMN": _HC, "INFO": _IND, "IPG": _COM,
    "JEC": _IND, "JEF": _FIN, "JNPR": _IT, "JWN": _CD, "K": _CS, "KMX": _CD,
    "KORS": _CD, "KSS": _CD, "KSU": _IND, "LB": _CD, "LEG": _CD, "LKQ": _CD,
    "LLL": _IND, "LNC": _FIN, "M": _CD, "MAC": _RE, "MAT": _CD, "MHK": _CD,
    "MMC": _FIN, "MON": _MAT, "MRO": _EN, "MYL": _HC, "NAVI": _FIN, "NBL": _EN,
    "NFX": _EN, "NLSN": _IND, "NOV": _EN, "NWL": _CD, "PBCT": _FIN, "PDCO": _HC,
    "PKI": _HC, "PRGO": _HC, "PVH": _CD, "PX": _MAT, "PXD": _EN, "QRVO": _IT,
    "RE": _FIN, "RHI": _IND, "RHT": _IT, "RRC": _EN, "RTN": _IND, "SCG": _UT,
    "SEE": _MAT, "SIG": _CD, "SLG": _RE, "SNI": _COM, "SRCL": _IND, "STI": _FIN,
    "SYMC": _IT, "TIF": _CD, "TMK": _FIN, "TRIP": _COM, "TSS": _IT, "TWX": _COM,
    "UA": _CD, "UAA": _CD, "UNM": _FIN, "UTX": _IND, "VAR": _HC, "VFC": _CD,
    "VIAB": _COM, "VNO": _RE, "WBA": _CS, "WHR": _CD, "WLTW": _FIN, "WRK": _MAT,
    "WU": _IT, "WYND": _CD, "XEC": _EN, "XL": _FIN, "XLNX": _IT, "XRAY": _HC,
    "XRX": _IT, "ZION": _FIN,
}

# GICS sub-industries for the delisted names inside the default 200-name priced
# universe (names follow pitindex's current labels).
SUB_INDUSTRY_OVERRIDES: dict[str, str] = {
    "AAL": "Passenger Airlines",
    "AAP": "Automotive Retail",
    "AET": "Managed Health Care",
    "AGN": "Pharmaceuticals",
    "AIV": "Multi-Family Residential REITs",
    "ALK": "Passenger Airlines",
    "ALXN": "Biotechnology",
    "AMG": "Asset Management & Custody Banks",
    "ANDV": "Oil & Gas Refining & Marketing",
    "ANSS": "Application Software",
    "APC": "Oil & Gas Exploration & Production",
    "ARNC": "Aerospace & Defense",
    "ATVI": "Interactive Home Entertainment",
    "AYI": "Electrical Components & Equipment",
    "BBT": "Regional Banks",
    "BHF": "Life & Health Insurance",
    "BHGE": "Oil & Gas Equipment & Services",
    "BK": "Asset Management & Custody Banks",
    "BWA": "Automotive Parts & Equipment",
    "CA": "Systems Software",
    "CAG": "Packaged Foods & Meats",
    "CHK": "Oil & Gas Exploration & Production",
    "COL": "Aerospace & Defense",
    "COTY": "Personal Care Products",
    "CPB": "Packaged Foods & Meats",
    "CPRI": "Apparel, Accessories & Luxury Goods",
    "CSRA": "IT Consulting & Other Services",
    "CTXS": "Application Software",
    "CXO": "Oil & Gas Exploration & Production",
    "DISCA": "Broadcasting",
    "DISCK": "Broadcasting",
    "DISH": "Cable & Satellite",
    "DRE": "Industrial REITs",
    "DWDP": "Specialty Chemicals",
    "DXC": "IT Consulting & Other Services",
    "EMN": "Specialty Chemicals",
    "FB": "Interactive Media & Services",
    "FL": "Apparel Retail",
    "FMC": "Fertilizers & Agricultural Chemicals",
    "FTI": "Oil & Gas Equipment & Services",
    "HES": "Oil & Gas Exploration & Production",
}
