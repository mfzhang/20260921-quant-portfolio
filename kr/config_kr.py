"""Korean market settings. Secrets (DART_API_KEY, KRX_ID, KRX_PW) live in .env."""

START = "2018-01-01"
END = None  # latest
INDEX_CODE = "1028"  # KRX KOSPI 200
SPLIT = "2023-05-10"  # same train / held-out boundary as the US work

# Costs: commission + slippage per unit turnover; securities transaction tax on sells only
COMMISSION_BPS = 1.5
SLIPPAGE_BPS = 20.0
# KOSPI sell-side tax incl. special rural tax, effective from each date
SELL_TAX_SCHEDULE = [
    ("2018-01-01", 0.0025),
    ("2021-01-01", 0.0023),
    ("2023-01-01", 0.0020),
    ("2024-01-01", 0.0018),
    ("2025-01-01", 0.0015),
]
BORROW_FEE_ANNUAL = 0.02

# New short sales were not possible in these windows (existing shorts could be covered).
# Between them only KOSPI200 / KOSDAQ150 names could be shorted, which covers this universe.
SHORT_BAN_PERIODS = [
    ("2020-03-16", "2021-05-02"),
    ("2023-11-06", "2025-03-30"),
]
