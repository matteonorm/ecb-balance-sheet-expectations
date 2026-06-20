"""Ultra-conservative GDELT collector. Sequential keywords, 45s delays."""

import json
import time
from datetime import datetime

import duckdb
import requests

from config import DUCKDB_PATH, GDELT_API_URL
from schema import create_schema

KEYWORDS = [
    '"ECB balance sheet"',
    '"ECB asset purchases"',
    '"quantitative easing" ECB',
    '"quantitative tightening" ECB',
]
DELAY = 45


def collect(db_path=DUCKDB_PATH):
    create_schema()
    con = duckdb.connect(db_path)

    windows = []
    for year in range(2020, 2027):
        for half in [(1, 7), (7, 13)]:
            s, e = half
            start = f"{year}-{s:02d}-01"
            end = f"{year}-12-31" if e == 13 else f"{year}-{e:02d}-01"
            if datetime.strptime(start, "%Y-%m-%d") <= datetime.now():
                windows.append((start, end))

    total = len(windows) * len(KEYWORDS)
    query_num = 0
    inserted_total = 0

    for keyword in KEYWORDS:
        print(f"\nKeyword: {keyword}", flush=True)
        for i, (ws, we) in enumerate(windows):
            query_num += 1
            start_dt = datetime.strptime(ws, "%Y-%m-%d")
            end_dt = datetime.strptime(we, "%Y-%m-%d")

            params = {
                "query": f"{keyword} sourcelang:english",
                "mode": "artlist",
                "maxrecords": "250",
                "format": "json",
                "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
            }

            ok = False
            for attempt in range(5):
                try:
                    resp = requests.get(GDELT_API_URL, params=params, timeout=20)
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            if isinstance(data, dict) and "articles" in data:
                                ok = True
                                break
                            else:
                                print(f"    Bad response: {resp.text[:100]}", flush=True)
                                break
                        except json.JSONDecodeError:
                            print(f"    Non-JSON 200: {resp.text[:100]}", flush=True)
                            time.sleep(DELAY)
                            continue
                    elif resp.status_code == 429:
                        wait = 60 * (attempt + 1)
                        print(f"    429 — waiting {wait}s (attempt {attempt+1}/5)", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"    HTTP {resp.status_code}", flush=True)
                        break
                except Exception as e:
                    print(f"    Error: {e}", flush=True)
                    time.sleep(30)

            if not ok:
                print(f"  [{query_num}/{total}] {ws[:7]}: SKIPPED", flush=True)
                time.sleep(DELAY)
                continue

            articles = data.get("articles", [])

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
                     art.get("language", ""), art.get("sourcecountry", ""), KEYWORD],
                )
                inserted += 1
            except Exception:
                pass

        inserted_total += inserted
        print(f"  [{query_num}/{total}] {ws[:7]}: {len(articles)} found, {inserted} new", flush=True)
        time.sleep(DELAY)

    db_total = con.execute("SELECT COUNT(*) FROM gdelt_articles").fetchone()[0]
    print(f"\nTotal articles in database: {db_total} (+{inserted_total} new)", flush=True)
    con.close()


if __name__ == "__main__":
    print(f"Waiting 600s for GDELT cooldown...", flush=True)
    time.sleep(600)
    print("Starting collection...", flush=True)
    collect()
