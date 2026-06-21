"""Collect SMA survey results — handles all ECB CSV naming conventions.

Three URL patterns exist across the 2021-2026 rounds:
  1. Early 2021: ecb.smar{pubdate}_{govc_month}_{year}_results.en.csv
  2. Standard:   ecb.smar{pubdate}_{pub_month}.en.csv
  3. Cross-month: ecb.smar{pubdate}_{govc_month}.en.csv (when pub month != govc month)

This script tries all variants and uses the first that returns 200.
It also includes 2026 rounds not yet in the ECB dates CSV.

HARD RULE: does NOT modify existing tables. Writes to sma_raw and sma_expectations
(same tables as collect_sma.py, but replaces all content on each run).
"""

import csv
import io
from datetime import datetime

import duckdb
import requests

from config import DUCKDB_PATH, SMA_DATES_URL
from schema import create_schema

MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}

SMA_CSV_BASE = "https://www.ecb.europa.eu/stats/ecb_surveys/sma/shared/pdf/"

# Rounds not yet in the ECB dates CSV (discovered by probing)
EXTRA_ROUNDS = [
    {"govc_date": "2026-01-29", "results_publication": "2026-02-09"},
    {"govc_date": "2026-03-12", "results_publication": "2026-03-23"},
    {"govc_date": "2026-06-11", "results_publication": "2026-06-15"},
]

# Manually verified URLs for rounds where the pattern is non-standard
MANUAL_OVERRIDES = {
    "2025-04-21": "ecb.smar250422_april.en.csv",
}


def fetch_survey_dates():
    resp = requests.get(SMA_DATES_URL, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rounds = list(reader)
    rounds.extend(EXTRA_ROUNDS)
    return rounds


def find_csv_url(govc_date_str, pub_date_str):
    """Try all naming variants; return the first URL that returns 200."""
    pub_dt = datetime.strptime(pub_date_str.strip(), "%Y-%m-%d")
    govc_dt = datetime.strptime(govc_date_str.strip(), "%Y-%m-%d")
    date_part = pub_dt.strftime("%y%m%d")
    pub_month = MONTH_NAMES[pub_dt.month]
    govc_month = MONTH_NAMES[govc_dt.month]

    if pub_date_str in MANUAL_OVERRIDES:
        candidates = [MANUAL_OVERRIDES[pub_date_str]]
    else:
        candidates = [
            f"ecb.smar{date_part}_{pub_month}.en.csv",
            f"ecb.smar{date_part}_{govc_month}.en.csv",
            f"ecb.smar{date_part}_{govc_month}_{govc_dt.year}_results.en.csv",
        ]
    candidates = list(dict.fromkeys(candidates))

    for fname in candidates:
        url = SMA_CSV_BASE + fname
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                return url
        except requests.RequestException:
            continue

    return None


def download_and_load_csv(url, con):
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        return 0
    resp.raise_for_status()

    reader = csv.reader(io.StringIO(resp.text))
    header = next(reader)
    count = 0
    for row in reader:
        if len(row) < 17:
            continue
        con.execute(
            "INSERT INTO sma_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row[:17],
        )
        count += 1
    return count


def process_expectations(con):
    con.execute("DELETE FROM sma_expectations")

    con.execute("""
        INSERT INTO sma_expectations
        SELECT
            vintage,
            CAST(vintage_date AS DATE) AS vintage_date,
            CAST(time_stamp AS DATE) AS forecast_date,
            measure,
            MAX(CASE WHEN item = 'EUROSYSTEM_APP_HOLDINGS' THEN TRY_CAST(value AS DOUBLE) END) AS app_holdings_eur,
            MAX(CASE WHEN item = 'EUROSYSTEM_PEPP_HOLDINGS' THEN TRY_CAST(value AS DOUBLE) END) AS pepp_holdings_eur,
            COALESCE(MAX(CASE WHEN item = 'EUROSYSTEM_APP_HOLDINGS' THEN TRY_CAST(value AS DOUBLE) END), 0)
            + COALESCE(MAX(CASE WHEN item = 'EUROSYSTEM_PEPP_HOLDINGS' THEN TRY_CAST(value AS DOUBLE) END), 0)
            AS total_holdings_eur
        FROM sma_raw
        WHERE item IN ('EUROSYSTEM_APP_HOLDINGS', 'EUROSYSTEM_PEPP_HOLDINGS')
          AND measure IN ('MEDIAN', 'P25', 'P75')
          AND value IS NOT NULL
          AND value != 'NULL'
          AND value != ''
        GROUP BY vintage, vintage_date, time_stamp, measure
        HAVING total_holdings_eur > 0
        ON CONFLICT DO NOTHING
    """)

    count = con.execute("SELECT COUNT(*) FROM sma_expectations").fetchone()[0]
    vintages = con.execute("""
        SELECT DISTINCT vintage, CAST(vintage_date AS DATE) AS vd
        FROM sma_expectations ORDER BY vd
    """).fetchall()
    print(f"\nProcessed {count} rows into sma_expectations")
    print(f"Vintages: {len(vintages)}")
    print(f"  First: {vintages[0][0]} ({vintages[0][1]})")
    print(f"  Last:  {vintages[-1][0]} ({vintages[-1][1]})")
    return count


def collect_sma_v2(db_path=DUCKDB_PATH):
    create_schema(db_path)
    con = duckdb.connect(db_path)

    con.execute("DELETE FROM sma_raw")

    print("Fetching SMA survey round dates...")
    rounds = fetch_survey_dates()
    print(f"Found {len(rounds)} survey rounds (incl. {len(EXTRA_ROUNDS)} extra 2026 rounds)")

    total_loaded = 0
    found = 0
    missing = []

    for rnd in rounds:
        pub_date_str = rnd.get("results_publication", "").strip()
        govc_date_str = rnd.get("govc_date", "").strip()

        if not pub_date_str or pub_date_str == "NULL":
            print(f"  Skipping {govc_date_str} (no publication date — pre-CSV era)")
            continue

        url = find_csv_url(govc_date_str, pub_date_str)
        if not url:
            missing.append(govc_date_str)
            print(f"  {govc_date_str}: CSV not found (tried all patterns)")
            continue

        fname = url.split("/")[-1]
        try:
            n = download_and_load_csv(url, con)
            total_loaded += n
            found += 1
            if n > 0:
                print(f"  {govc_date_str}: {n} rows ({fname})")
            else:
                print(f"  {govc_date_str}: empty CSV ({fname})")
        except Exception as e:
            print(f"  {govc_date_str}: error downloading {fname}: {e}")

    print(f"\nTotal CSVs downloaded: {found}")
    print(f"Total raw rows: {total_loaded}")
    if missing:
        print(f"Missing rounds: {missing}")

    process_expectations(con)
    con.close()


if __name__ == "__main__":
    collect_sma_v2()
