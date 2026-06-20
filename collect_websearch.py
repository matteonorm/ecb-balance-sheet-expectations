"""Collect ECB balance sheet news headlines via web search.
Supplements GDELT articles for months with low coverage."""

import hashlib
import re
import subprocess
import sys
import json
import time
from datetime import datetime

import duckdb
import requests

from config import DUCKDB_PATH


def websearch_headlines(query, timeout=15):
    """Use the GDELT API with quarterly windows and very conservative timing."""
    pass  # Placeholder — we use the CLI approach below


def collect_from_gdelt_quarterly(db_path=DUCKDB_PATH):
    """Collect from GDELT using quarterly windows and 30s delays.
    Much slower but avoids rate limiting."""
    from config import GDELT_API_URL

    con = duckdb.connect(db_path)

    queries = [
        '"ECB balance sheet"',
        '"ECB asset purchases"',
        '"quantitative easing" ECB',
        '"quantitative tightening" ECB',
    ]

    quarters = []
    for year in range(2019, 2027):
        for qm in [(1, 4), (4, 7), (7, 10), (10, 13)]:
            start_m, end_m = qm
            if end_m == 13:
                start = f"{year}-10-01"
                end = f"{year}-12-31"
            else:
                start = f"{year}-{start_m:02d}-01"
                end = f"{year}-{end_m:02d}-01"
            if datetime.strptime(start, "%Y-%m-%d") <= datetime.now():
                quarters.append((start, end))

    total = len(quarters) * len(queries)
    n = 0
    total_inserted = 0

    for query in queries:
        print(f"\nQuery: {query}", flush=True)
        for q_start, q_end in quarters:
            n += 1
            start_dt = datetime.strptime(q_start, "%Y-%m-%d")
            end_dt = datetime.strptime(q_end, "%Y-%m-%d")

            params = {
                "query": f'{query} sourcelang:english',
                "mode": "artlist",
                "maxrecords": "250",
                "format": "json",
                "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
            }

            ok = False
            for attempt in range(3):
                try:
                    resp = requests.get(GDELT_API_URL, params=params, timeout=15)
                    if resp.status_code == 429:
                        wait = 60 * (attempt + 1)
                        print(f"    429 — waiting {wait}s", flush=True)
                        time.sleep(wait)
                        continue
                    if resp.status_code == 200:
                        ok = True
                        break
                    else:
                        print(f"    HTTP {resp.status_code}", flush=True)
                        break
                except Exception as e:
                    print(f"    Error: {e}", flush=True)
                    time.sleep(30)

            if not ok:
                print(f"  [{n}/{total}] {q_start[:7]}: skipped", flush=True)
                time.sleep(30)
                continue

            try:
                articles = resp.json().get("articles", [])
            except Exception:
                articles = []

            inserted = 0
            for art in articles:
                url = art.get("url", "")
                title = art.get("title", "")
                seendate = art.get("seendate", "")
                if not url or not title:
                    continue
                try:
                    seen_dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
                    seendate_fmt = seen_dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    seendate_fmt = seendate
                try:
                    con.execute(
                        """INSERT OR IGNORE INTO gdelt_articles
                           (url, title, seendate, domain, language, sourcecountry, query_keyword)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        [url, title, seendate_fmt, art.get("domain", ""),
                         art.get("language", ""), art.get("sourcecountry", ""), query],
                    )
                    inserted += 1
                except Exception:
                    pass

            total_inserted += inserted
            print(f"  [{n}/{total}] {q_start[:7]}: {len(articles)} found, {inserted} new", flush=True)
            time.sleep(30)

    db_total = con.execute("SELECT COUNT(*) FROM gdelt_articles").fetchone()[0]
    print(f"\nTotal articles in database: {db_total}", flush=True)

    coverage = con.execute("""
        SELECT strftime(seendate, '%Y-%m') AS month, COUNT(*) AS n
        FROM gdelt_articles GROUP BY month ORDER BY month
    """).fetchall()
    for m, c in coverage:
        print(f"  {m}: {c}", flush=True)

    con.close()
    return total_inserted


if __name__ == "__main__":
    from schema import create_schema
    create_schema()
    collect_from_gdelt_quarterly()
