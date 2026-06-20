import json
import time
from datetime import datetime, timedelta

import duckdb
import requests

from config import (
    DATE_START,
    DUCKDB_PATH,
    GDELT_API_URL,
    GDELT_DELAY_SECONDS,
    GDELT_KEYWORDS,
    GDELT_MAX_RETRIES,
)


def fetch_gdelt_articles(keyword, start_date, end_date):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    params = {
        "query": f'{keyword} sourcelang:english',
        "mode": "artlist",
        "maxrecords": "250",
        "format": "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }

    for attempt in range(GDELT_MAX_RETRIES):
        try:
            resp = requests.get(GDELT_API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                wait = (2 ** attempt) * 30
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            try:
                data = resp.json()
            except json.JSONDecodeError:
                return []

            articles = data.get("articles", [])
            return articles

        except requests.exceptions.RequestException as e:
            if attempt < GDELT_MAX_RETRIES - 1:
                time.sleep(10)
            else:
                print(f"    Failed after {GDELT_MAX_RETRIES} attempts: {e}")
                return []

    return []


def generate_monthly_windows(start_date, end_date):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    windows = []
    current = start_dt.replace(day=1)
    while current < end_dt:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        window_end = min(next_month, end_dt)
        windows.append((current.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
        current = next_month
    return windows


def collect_gdelt(db_path=DUCKDB_PATH, start_date=DATE_START):
    con = duckdb.connect(db_path)
    end_date = datetime.now().strftime("%Y-%m-%d")
    windows = generate_monthly_windows(start_date, end_date)

    total_inserted = 0
    total_queries = len(windows) * len(GDELT_KEYWORDS)
    query_num = 0

    for keyword in GDELT_KEYWORDS:
        print(f"\nKeyword: {keyword}")
        for win_start, win_end in windows:
            query_num += 1
            articles = fetch_gdelt_articles(keyword, win_start, win_end)

            inserted = 0
            for art in articles:
                url = art.get("url", "")
                title = art.get("title", "")
                seendate = art.get("seendate", "")
                domain = art.get("domain", "")
                language = art.get("language", "")
                sourcecountry = art.get("sourcecountry", "")

                if not url or not title:
                    continue

                try:
                    seen_dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
                    seendate_formatted = seen_dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    seendate_formatted = seendate

                try:
                    con.execute(
                        """INSERT OR IGNORE INTO gdelt_articles
                           (url, title, seendate, domain, language, sourcecountry, query_keyword)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        [url, title, seendate_formatted, domain, language,
                         sourcecountry, keyword],
                    )
                    inserted += 1
                except Exception:
                    pass

            if len(articles) >= 250:
                print(f"  [{query_num}/{total_queries}] {win_start}: {len(articles)} articles (HIT CAP)")
                mid = datetime.strptime(win_start, "%Y-%m-%d") + timedelta(days=15)
                mid_str = mid.strftime("%Y-%m-%d")
                extra = fetch_gdelt_articles(keyword, mid_str, win_end)
                for art in extra:
                    url = art.get("url", "")
                    title = art.get("title", "")
                    seendate = art.get("seendate", "")
                    if not url or not title:
                        continue
                    try:
                        seen_dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
                        seendate_formatted = seen_dt.strftime("%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        seendate_formatted = seendate
                    try:
                        con.execute(
                            """INSERT OR IGNORE INTO gdelt_articles
                               (url, title, seendate, domain, language, sourcecountry, query_keyword)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            [url, title, seendate_formatted, art.get("domain", ""),
                             art.get("language", ""), art.get("sourcecountry", ""), keyword],
                        )
                        inserted += 1
                    except Exception:
                        pass
                time.sleep(GDELT_DELAY_SECONDS)
            elif articles:
                print(f"  [{query_num}/{total_queries}] {win_start}: {len(articles)} articles")

            total_inserted += inserted
            time.sleep(GDELT_DELAY_SECONDS)

    total = con.execute("SELECT COUNT(*) FROM gdelt_articles").fetchone()[0]
    print(f"\nTotal articles in database: {total} (newly inserted: {total_inserted})")

    coverage = con.execute("""
        SELECT strftime(seendate, '%Y-%m') AS month, COUNT(*) AS n
        FROM gdelt_articles
        GROUP BY month ORDER BY month
    """).fetchall()
    low_coverage = [(m, n) for m, n in coverage if n < 10]
    if low_coverage:
        print(f"\nLow coverage months (<10 articles): {low_coverage}")

    con.close()
    return total_inserted


if __name__ == "__main__":
    from schema import create_schema
    create_schema()
    collect_gdelt()
