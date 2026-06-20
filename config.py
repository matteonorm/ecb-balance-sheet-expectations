import os

DUCKDB_PATH = os.path.join(os.path.dirname(__file__), "ecb_bs.duckdb")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5"

DATE_START = "2019-04-01"
DATE_END = "2026-06-20"

SMA_DATES_URL = "https://www.ecb.europa.eu/stats/ecb_surveys/sma/shared/pdf/list_of_sma_survey_round_dates.csv"
SMA_CSV_BASE = "https://www.ecb.europa.eu/stats/ecb_surveys/sma/shared/pdf/"
SMA_PDF_BASE = "https://www.ecb.europa.eu/stats/ecb_surveys/sma/shared/pdf/"

GDELT_KEYWORDS = [
    '"ECB balance sheet"',
    '"ECB asset purchases"',
    '"ECB quantitative easing"',
    '"ECB quantitative tightening"',
    '"PEPP purchases"',
    '"APP purchases"',
    '"Eurosystem balance sheet"',
    '"ECB bond buying"',
]

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DELAY_SECONDS = 5.0
GDELT_MAX_RETRIES = 3

ECB_DATA_API = "https://data-api.ecb.europa.eu/service/data"
