"""Collect ECB balance sheet news headlines from Google News RSS feeds."""

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import duckdb
import requests

from config import DUCKDB_PATH
from schema import create_schema

QUERIES = [
    "ECB balance sheet",
    "ECB asset purchases",
    "ECB quantitative tightening",
    "ECB quantitative easing",
    "ECB bond purchases APP",
    "ECB PEPP purchases",
    "Eurosystem balance sheet bonds",
    "ECB bond holdings reduction",
]

RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"


def fetch_rss(query):
    url = RSS_URL.format(query=query.replace(" ", "+"))
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for query: {query}", flush=True)
            return []
        return parse_rss(resp.text, query)
    except Exception as e:
        print(f"  Error fetching {query}: {e}", flush=True)
        return []


def parse_rss(xml_text, query):
    articles = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pubdate_el = item.find("pubDate")
            source_el = item.find("source")

            if title_el is None or link_el is None:
                continue

            title = title_el.text or ""
            url = link_el.text or ""
            pubdate = pubdate_el.text if pubdate_el is not None else ""
            domain = source_el.get("url", "") if source_el is not None else ""
            source_name = source_el.text if source_el is not None else ""

            if not title or not url:
                continue

            seen_dt = None
            if pubdate:
                for fmt in ["%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"]:
                    try:
                        seen_dt = datetime.strptime(pubdate.strip(), fmt)
                        break
                    except ValueError:
                        continue

            seendate = seen_dt.strftime("%Y-%m-%d %H:%M:%S") if seen_dt else ""

            articles.append({
                "url": url,
                "title": title.strip(),
                "seendate": seendate,
                "domain": domain or source_name or "",
                "query": query,
            })
    except ET.ParseError as e:
        print(f"  XML parse error: {e}", flush=True)

    return articles


def collect_gnews(db_path=DUCKDB_PATH):
    create_schema()
    con = duckdb.connect(db_path)

    total_inserted = 0

    for query in QUERIES:
        print(f"Query: {query}", flush=True)
        articles = fetch_rss(query)
        inserted = 0

        for art in articles:
            try:
                con.execute(
                    """INSERT OR IGNORE INTO gdelt_articles
                       (url, title, seendate, domain, language, sourcecountry, query_keyword)
                       VALUES (?, ?, ?, ?, 'English', '', ?)""",
                    [art["url"], art["title"], art["seendate"],
                     art["domain"], art["query"]],
                )
                inserted += 1
            except Exception:
                pass

        total_inserted += inserted
        print(f"  Found {len(articles)}, inserted {inserted} new", flush=True)
        time.sleep(2)

    db_total = con.execute("SELECT COUNT(*) FROM gdelt_articles").fetchone()[0]
    print(f"\nTotal articles in database: {db_total} (+{total_inserted} new)", flush=True)

    coverage = con.execute("""
        SELECT strftime(seendate, '%Y-%m') AS month, COUNT(*) AS n
        FROM gdelt_articles
        WHERE seendate IS NOT NULL AND length(seendate) > 0
        GROUP BY month ORDER BY month
    """).fetchall()
    print("\nMonthly coverage:")
    for m, c in coverage:
        print(f"  {m}: {c}", flush=True)

    con.close()
    return total_inserted


if __name__ == "__main__":
    collect_gnews()
