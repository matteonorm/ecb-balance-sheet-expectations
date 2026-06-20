import json
import time

import anthropic
import duckdb

from config import CLAUDE_MODEL, DUCKDB_PATH

PROMPT_TEMPLATE = """You are an expert analyst of European Central Bank monetary policy.

Given the following news headline, assess whether it implies the ECB's balance sheet (Eurosystem stock of bonds under APP and PEPP) will INCREASE, DECREASE, or is UNCERTAIN in the near future.

Headline: "{title}"

Respond ONLY with a JSON object, no other text:
{{"direction": "increase" | "decrease" | "uncertain", "confidence": <float 0-1>, "magnitude": "small" | "moderate" | "large" | "unspecified", "explanation": "<one sentence, max 25 words>"}}"""


def classify_headline(client, title):
    prompt = PROMPT_TEMPLATE.format(title=title.replace('"', '\\"'))
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        result = json.loads(text)

        direction = result.get("direction", "uncertain")
        if direction not in ("increase", "decrease", "uncertain"):
            direction = "uncertain"

        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        magnitude = result.get("magnitude", "unspecified")
        if magnitude not in ("small", "moderate", "large", "unspecified"):
            magnitude = "unspecified"

        explanation = str(result.get("explanation", ""))[:200]

        return {
            "direction": direction,
            "confidence": confidence,
            "magnitude": magnitude,
            "explanation": explanation,
        }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {
            "direction": "uncertain",
            "confidence": 0.0,
            "magnitude": "unspecified",
            "explanation": f"parse_error: {str(e)[:100]}",
        }
    except anthropic.RateLimitError:
        time.sleep(30)
        return classify_headline(client, title)


def process_unclassified(db_path=DUCKDB_PATH, batch_size=50, max_articles=None):
    con = duckdb.connect(db_path)
    client = anthropic.Anthropic()

    query = """
        SELECT g.url, g.title
        FROM gdelt_articles g
        LEFT JOIN llm_classifications c ON g.url = c.url
        WHERE c.url IS NULL
        ORDER BY g.seendate
    """
    if max_articles:
        query += f" LIMIT {max_articles}"

    unprocessed = con.execute(query).fetchall()
    total = len(unprocessed)
    print(f"Found {total} unclassified headlines")

    processed = 0
    errors = 0

    for i, (url, title) in enumerate(unprocessed):
        result = classify_headline(client, title)

        try:
            con.execute(
                """INSERT OR IGNORE INTO llm_classifications
                   (url, direction, confidence, magnitude, explanation, model_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [url, result["direction"], result["confidence"],
                 result["magnitude"], result["explanation"], CLAUDE_MODEL],
            )
            processed += 1
        except Exception as e:
            errors += 1
            print(f"  Insert error: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{total} "
                  f"(latest: {result['direction']}, conf={result['confidence']:.2f})")

        if (i + 1) % batch_size == 0:
            time.sleep(1)

    dist = con.execute("""
        SELECT direction, COUNT(*) AS n
        FROM llm_classifications
        GROUP BY direction
    """).fetchall()

    print(f"\nDone: {processed} classified, {errors} errors")
    print(f"Distribution: {dict(dist)}")

    con.close()
    return processed


if __name__ == "__main__":
    process_unclassified()
