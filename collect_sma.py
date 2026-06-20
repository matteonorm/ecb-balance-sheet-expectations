import csv
import io
from datetime import datetime

import duckdb
import requests

from config import DUCKDB_PATH, SMA_CSV_BASE, SMA_DATES_URL

MONTH_MAP = {
    "01": "january", "02": "february", "03": "march", "04": "april",
    "05": "may", "06": "june", "07": "july", "08": "august",
    "09": "september", "10": "october", "11": "november", "12": "december",
}


def fetch_survey_dates():
    resp = requests.get(SMA_DATES_URL, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    return list(reader)


def build_csv_url(publication_date_str):
    dt = datetime.strptime(publication_date_str.strip(), "%Y-%m-%d")
    date_part = dt.strftime("%y%m%d")
    month_name = MONTH_MAP[dt.strftime("%m")]
    return f"{SMA_CSV_BASE}ecb.smar{date_part}_{month_name}.en.csv"


def download_and_load_csv(url, con):
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        print(f"    CSV not found: {url.split('/')[-1]}")
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
    print(f"Processed {count} rows into sma_expectations")

    sample = con.execute("""
        SELECT vintage, vintage_date, forecast_date, measure, total_holdings_eur
        FROM sma_expectations
        WHERE measure = 'MEDIAN'
        ORDER BY vintage_date DESC, forecast_date
        LIMIT 5
    """).fetchall()
    for row in sample:
        print(f"  {row}")

    return count


def collect_sma(db_path=DUCKDB_PATH):
    con = duckdb.connect(db_path)

    # Clear previous data to avoid duplicates (no PK on sma_raw)
    con.execute("DELETE FROM sma_raw")

    print("Fetching SMA survey round dates...")
    rounds = fetch_survey_dates()
    print(f"Found {len(rounds)} survey rounds")

    total_loaded = 0

    for rnd in rounds:
        pub_date_str = rnd.get("results_publication", "").strip()
        govc_date_str = rnd.get("govc_date", "").strip()

        if not pub_date_str or pub_date_str == "NULL":
            print(f"  Skipping {govc_date_str} (no publication date — pre-CSV era)")
            continue

        try:
            pub_dt = datetime.strptime(pub_date_str, "%Y-%m-%d")
        except ValueError:
            print(f"  Skipping unparseable date: {pub_date_str}")
            continue

        url = build_csv_url(pub_date_str)
        print(f"  Downloading {pub_dt.strftime('%b %Y')} ({url.split('/')[-1]})...")
        try:
            n = download_and_load_csv(url, con)
            total_loaded += n
            if n > 0:
                print(f"    Loaded {n} rows")
        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nTotal raw rows loaded: {total_loaded}")

    vintages = con.execute("SELECT DISTINCT vintage FROM sma_raw ORDER BY vintage").fetchall()
    print(f"Vintages in database: {[v[0] for v in vintages]}")

    process_expectations(con)
    con.close()


if __name__ == "__main__":
    from schema import create_schema
    create_schema()
    collect_sma()
