import csv
import io

import duckdb
import requests

from config import DUCKDB_PATH, ECB_DATA_API

TOTAL_ASSETS_URL = (
    f"{ECB_DATA_API}/ILM/W.U2.C.T000000.Z5.Z01?format=csvdata"
    "&startPeriod=2019-01"
)


def collect_ecb_balance_sheet(db_path=DUCKDB_PATH):
    con = duckdb.connect(db_path)

    print("Downloading Eurosystem total assets from ECB Data Portal...")
    resp = requests.get(TOTAL_ASSETS_URL, timeout=60)

    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    count = 0
    for row in reader:
        period = row.get("TIME_PERIOD", "")
        value = row.get("OBS_VALUE", "")
        if not period or not value:
            continue

        # Weekly periods like "2019-W02" — convert to a date (Friday of that week)
        if "-W" in period:
            import re
            m = re.match(r"(\d{4})-W(\d{2})", period)
            if m:
                from datetime import datetime, timedelta
                year, week = int(m.group(1)), int(m.group(2))
                dt = datetime.strptime(f"{year}-W{week:02d}-5", "%Y-W%W-%w")
                period = dt.strftime("%Y-%m-%d")
            else:
                continue

        try:
            val = float(value)
        except ValueError:
            continue

        try:
            con.execute(
                """INSERT OR IGNORE INTO ecb_balance_sheet (observation_date, total_assets_eur)
                   VALUES (?, ?)""",
                [period, val],
            )
            count += 1
        except Exception as e:
            print(f"  Insert error: {e}")

    print(f"Loaded {count} observations into ecb_balance_sheet")
    total = con.execute("SELECT COUNT(*) FROM ecb_balance_sheet").fetchone()[0]
    print(f"Total observations in table: {total}")

    sample = con.execute(
        "SELECT * FROM ecb_balance_sheet ORDER BY observation_date DESC LIMIT 3"
    ).fetchall()
    print(f"Latest entries: {sample}")

    con.close()
    return count


if __name__ == "__main__":
    from schema import create_schema
    create_schema()
    collect_ecb_balance_sheet()
