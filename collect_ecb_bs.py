"""Collect ECB asset purchase programme holdings (APP + PEPP).
Uses the ECB's published CSV breakdowns rather than total Eurosystem assets."""

import csv
import io
import re
from datetime import datetime

import duckdb
import requests

from config import DUCKDB_PATH

ECB_BASE = "https://www.ecb.europa.eu"
APP_URL = f"{ECB_BASE}/mopo/pdf/APP_breakdown_history.csv"
PEPP_URL = f"{ECB_BASE}/mopo/pdf/PEPP_purchase_history.csv"


def parse_app_holdings(text):
    """Parse APP breakdown CSV. Returns list of (date, holdings_eur_millions)."""
    import csv as csv_mod
    lines = list(csv_mod.reader(text.strip().split("\n")))
    results = []
    current_year = None

    for cols in lines:
        if len(cols) < 14:
            continue

        if cols[0].strip().isdigit():
            current_year = int(cols[0].strip())

        month_str = cols[1].strip().strip('"')
        if not month_str or not current_year:
            continue

        months = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
        }
        month_num = months.get(month_str)
        if not month_num:
            continue

        # Holdings are in columns 10-13 (ABSPP, CBPP3, CSPP, PSPP)
        try:
            holdings = []
            for i in range(10, 14):
                val = cols[i].strip().replace('"', '').replace(',', '')
                holdings.append(float(val) if val else 0.0)
            total_app = sum(holdings)
            if total_app == 0:
                continue

            import calendar
            last_day = calendar.monthrange(current_year, month_num)[1]
            date_str = f"{current_year}-{month_num:02d}-{last_day:02d}"
            results.append((date_str, total_app))
        except (ValueError, IndexError):
            continue

    return results


def parse_pepp_holdings(text):
    """Parse PEPP purchase history CSV. Returns list of (date, cumulative_eur_millions)."""
    lines = text.strip().split("\n")
    results = []
    current_year = None

    for line in lines:
        cols = line.split(",")
        if len(cols) < 4:
            continue

        if cols[0].strip().isdigit():
            current_year = int(cols[0].strip())

        month_str = cols[1].strip().strip('"')
        if not month_str or not current_year:
            continue

        months = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
        }
        month_num = months.get(month_str)
        if not month_num:
            continue

        try:
            cumulative = float(cols[3].strip().replace('"', ''))
            import calendar
            last_day = calendar.monthrange(current_year, month_num)[1]
            date_str = f"{current_year}-{month_num:02d}-{last_day:02d}"
            results.append((date_str, cumulative))
        except (ValueError, IndexError):
            continue

    return results


def collect_ecb_balance_sheet(db_path=DUCKDB_PATH):
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS ecb_app_pepp (
            observation_date DATE PRIMARY KEY,
            app_holdings_eur DOUBLE,
            pepp_holdings_eur DOUBLE,
            total_holdings_eur DOUBLE,
            collected_at TIMESTAMP DEFAULT current_timestamp
        )
    """)

    print("Downloading APP holdings from ECB...", flush=True)
    resp_app = requests.get(APP_URL, timeout=30)
    resp_app.raise_for_status()
    app_data = parse_app_holdings(resp_app.text)
    print(f"  Parsed {len(app_data)} monthly APP observations", flush=True)

    print("Downloading PEPP holdings from ECB...", flush=True)
    resp_pepp = requests.get(PEPP_URL, timeout=30)
    resp_pepp.raise_for_status()
    pepp_data = parse_pepp_holdings(resp_pepp.text)
    print(f"  Parsed {len(pepp_data)} monthly PEPP observations", flush=True)

    app_dict = {d: v for d, v in app_data}
    pepp_dict = {d: v for d, v in pepp_data}

    all_dates = sorted(set(list(app_dict.keys()) + list(pepp_dict.keys())))

    con.execute("DELETE FROM ecb_app_pepp")

    count = 0
    for date_str in all_dates:
        app_val = app_dict.get(date_str, 0.0)
        pepp_val = pepp_dict.get(date_str, 0.0)
        total = app_val + pepp_val

        con.execute(
            """INSERT OR IGNORE INTO ecb_app_pepp
               (observation_date, app_holdings_eur, pepp_holdings_eur, total_holdings_eur)
               VALUES (?, ?, ?, ?)""",
            [date_str, app_val, pepp_val, total],
        )
        count += 1

    print(f"\nLoaded {count} observations into ecb_app_pepp", flush=True)

    sample = con.execute("""
        SELECT observation_date, app_holdings_eur, pepp_holdings_eur, total_holdings_eur
        FROM ecb_app_pepp ORDER BY observation_date DESC LIMIT 5
    """).fetchall()
    print("Latest entries (EUR millions):")
    for row in sample:
        print(f"  {row[0]}: APP={row[1]:,.0f}  PEPP={row[2]:,.0f}  Total={row[3]:,.0f}")

    con.close()
    return count


if __name__ == "__main__":
    from schema import create_schema
    create_schema()
    collect_ecb_balance_sheet()
