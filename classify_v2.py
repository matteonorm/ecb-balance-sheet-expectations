"""V2 headline classifier: relevance gate, few-shot anchors, ensemble agreement.

Writes to llm_classifications_v2 — the original llm_classifications is untouched.

Run modes:
  python classify_v2.py validate   # ~80-100 sample, print report, export CSV for hand-labelling
  python classify_v2.py run        # full corpus (only after validate review)
"""

import json
import os
import sys
import time

import anthropic
import duckdb
import pandas as pd

from config import DUCKDB_PATH

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------
ENSEMBLE_K = 5                   # classification runs per headline
VALIDATION_SAMPLE_SIZE = 100     # headlines to validate before full run
MODEL = "claude-haiku-4-5"       # override: set env CLASSIFY_MODEL
BATCH_PAUSE = 1                  # seconds between batches of 50
MAX_ARTICLES = None              # None = all; set for debugging

MODEL = os.environ.get("CLASSIFY_MODEL", MODEL)

# ---------------------------------------------------------------------------
# Few-shot prompt with relevance gate
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert analyst of European Central Bank monetary policy.

You classify news headlines about the ECB/Eurosystem balance sheet — specifically the
stock of bonds held under APP and PEPP, and related programmes (QE, QT, reinvestment,
TLTROs, sovereign bond purchases).

Labels:
- "increase": the headline implies the ECB balance sheet will GROW (new purchases,
  slower runoff, reinvestment extended, expanded QE).
- "decrease": the headline implies the ECB balance sheet will SHRINK (tapering, QT,
  ending reinvestments, faster runoff, active bond sales).
- "uncertain": the headline IS about the ECB balance sheet but the directional
  implication is genuinely ambiguous or mixed.
- "not_relevant": the headline is NOT about the ECB/Eurosystem balance sheet, APP,
  PEPP, QE, QT, reinvestment policy, or TLTROs. This includes: general ECB interest
  rate news without balance-sheet implications, non-ECB central banks, corporate
  earnings, unrelated EU politics, or articles that merely mention "ECB" in passing.

IMPORTANT: most headlines in this corpus were retrieved via keyword search and many are
off-topic. Be aggressive about labelling not_relevant — only classify as
increase/decrease/uncertain if the headline specifically concerns ECB asset purchases,
bond holdings, QE/QT, reinvestment, or TLTRO operations."""

FEW_SHOT_EXAMPLES = [
    {
        "title": "ECB to end net asset purchases under APP in July, signals rate hikes",
        "output": {"direction": "decrease", "confidence": 0.9, "magnitude": "large",
                   "explanation": "Ending net purchases directly shrinks APP inflows."},
    },
    {
        "title": "ECB extends PEPP reinvestments until at least end of 2024",
        "output": {"direction": "increase", "confidence": 0.85, "magnitude": "moderate",
                   "explanation": "Extended reinvestment maintains PEPP holdings longer than expected."},
    },
    {
        "title": "ECB bond portfolio shrinks at fastest pace since QT began",
        "output": {"direction": "decrease", "confidence": 0.92, "magnitude": "large",
                   "explanation": "Accelerating portfolio runoff implies faster balance sheet reduction."},
    },
    {
        "title": "Markets divided on whether ECB will accelerate or slow bond runoff",
        "output": {"direction": "uncertain", "confidence": 0.7, "magnitude": "unspecified",
                   "explanation": "Explicit disagreement on pace with no clear directional signal."},
    },
    {
        "title": "ECB policymakers hint at possible bond-buying restart amid recession fears",
        "output": {"direction": "increase", "confidence": 0.8, "magnitude": "moderate",
                   "explanation": "Potential restart of purchases would expand holdings."},
    },
    {
        "title": "ECB raises interest rates by 50 basis points as inflation persists",
        "output": {"direction": "not_relevant", "confidence": 0.9, "magnitude": "unspecified",
                   "explanation": "Rate decision without balance sheet or asset purchase implications."},
    },
    {
        "title": "German lawmakers consider expropriating private apartments to house the poor",
        "output": {"direction": "not_relevant", "confidence": 0.95, "magnitude": "unspecified",
                   "explanation": "Unrelated to ECB monetary policy or balance sheet."},
    },
    {
        "title": "Euro zone bond yields fall as investors bet on ECB easing",
        "output": {"direction": "not_relevant", "confidence": 0.85, "magnitude": "unspecified",
                   "explanation": "Generic easing reference without specific balance sheet implications."},
    },
]

CLASSIFY_PROMPT = """Classify this headline:
"{title}"

Respond ONLY with a JSON object:
{{"direction": "increase" | "decrease" | "uncertain" | "not_relevant", "confidence": <float 0-1>, "magnitude": "small" | "moderate" | "large" | "unspecified", "explanation": "<one sentence, max 25 words>"}}"""


def _build_messages(title):
    messages = []
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": CLASSIFY_PROMPT.format(title=ex["title"])})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"])})
    messages.append({"role": "user", "content": CLASSIFY_PROMPT.format(title=title)})
    return messages


# ---------------------------------------------------------------------------
# Single classification call
# ---------------------------------------------------------------------------
def _classify_once(client, title):
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=_build_messages(title),
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        result = json.loads(text)

        direction = result.get("direction", "uncertain")
        if direction not in ("increase", "decrease", "uncertain", "not_relevant"):
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
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"direction": "uncertain", "confidence": 0.0,
                "magnitude": "unspecified", "explanation": "parse_error"}
    except anthropic.RateLimitError:
        time.sleep(30)
        return _classify_once(client, title)


# ---------------------------------------------------------------------------
# Ensemble classification
# ---------------------------------------------------------------------------
def classify_ensemble(client, title, k=ENSEMBLE_K):
    votes = [_classify_once(client, title) for _ in range(k)]

    directions = [v["direction"] for v in votes]
    from collections import Counter
    counts = Counter(directions)
    majority_label, majority_count = counts.most_common(1)[0]
    agreement = majority_count / k

    self_confs = [v["confidence"] for v in votes]
    avg_self_conf = sum(self_confs) / len(self_confs)

    magnitudes = [v["magnitude"] for v in votes if v["direction"] == majority_label]
    from collections import Counter as C2
    mag_counts = C2(magnitudes)
    majority_mag = mag_counts.most_common(1)[0][0] if mag_counts else "unspecified"

    explanations = [v["explanation"] for v in votes if v["direction"] == majority_label]
    best_explanation = explanations[0] if explanations else ""

    return {
        "direction": majority_label,
        "ensemble_confidence": agreement,
        "self_confidence": avg_self_conf,
        "magnitude": majority_mag,
        "explanation": best_explanation,
        "vote_distribution": dict(counts),
    }


# ---------------------------------------------------------------------------
# Schema: create v2 table
# ---------------------------------------------------------------------------
def ensure_v2_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS llm_classifications_v2 (
            url                 VARCHAR PRIMARY KEY,
            direction           VARCHAR NOT NULL,
            ensemble_confidence DOUBLE,
            self_confidence     DOUBLE,
            magnitude           VARCHAR,
            explanation         VARCHAR,
            vote_distribution   VARCHAR,
            model_id            VARCHAR NOT NULL,
            ensemble_k          INTEGER NOT NULL,
            processed_at        TIMESTAMP DEFAULT current_timestamp
        )
    """)


# ---------------------------------------------------------------------------
# Validate mode: sample, classify, report, export for hand-labelling
# ---------------------------------------------------------------------------
def run_validate(db_path=DUCKDB_PATH):
    con = duckdb.connect(db_path)
    ensure_v2_table(con)
    client = anthropic.Anthropic()

    sample = con.execute(f"""
        SELECT g.url, g.title, c.direction AS v1_direction, c.confidence AS v1_confidence
        FROM gdelt_articles g
        JOIN llm_classifications c ON g.url = c.url
        USING SAMPLE {VALIDATION_SAMPLE_SIZE}
    """).fetchdf()

    print(f"Validating {len(sample)} headlines with k={ENSEMBLE_K} ensemble runs")
    print(f"Model: {MODEL}")
    print(f"Total API calls: ~{len(sample) * ENSEMBLE_K}")
    print()

    results = []
    for i, row in sample.iterrows():
        r = classify_ensemble(client, row["title"])
        r["url"] = row["url"]
        r["title"] = row["title"]
        r["v1_direction"] = row["v1_direction"]
        r["v1_confidence"] = row["v1_confidence"]
        results.append(r)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample)} done (latest: {r['direction']}, "
                  f"agreement={r['ensemble_confidence']:.1%})")

        if (i + 1) % 50 == 0:
            time.sleep(BATCH_PAUSE)

    df = pd.DataFrame(results)

    # --- Report ---
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print(f"\nV2 class distribution (N={len(df)}):")
    dist = df["direction"].value_counts()
    for label, n in dist.items():
        print(f"  {label:14s}: {n:3d} ({n/len(df):.1%})")

    nr_share = dist.get("not_relevant", 0) / len(df)
    print(f"\nNot-relevant share: {nr_share:.1%}")
    print(f"Mean ensemble agreement: {df['ensemble_confidence'].mean():.3f}")
    print(f"Mean self-reported confidence: {df['self_confidence'].mean():.3f}")

    # V1 vs V2 comparison
    df["changed"] = df["direction"] != df["v1_direction"]
    n_changed = df["changed"].sum()
    print(f"\nV1→V2 disagreements: {n_changed}/{len(df)} ({n_changed/len(df):.1%})")

    # Show some disagreements
    disagree = df[df["changed"]].head(15)
    if not disagree.empty:
        print("\nSample V1→V2 disagreements:")
        for _, row in disagree.iterrows():
            title_short = row["title"][:80]
            print(f"  V1={row['v1_direction']:12s} → V2={row['direction']:14s} "
                  f"(agree={row['ensemble_confidence']:.0%}) | {title_short}")

    # Export for hand-labelling
    export_cols = ["url", "title", "v1_direction", "v1_confidence",
                   "direction", "ensemble_confidence", "self_confidence",
                   "magnitude", "explanation", "vote_distribution"]
    export_df = df[export_cols].copy()
    export_df.rename(columns={"direction": "v2_direction"}, inplace=True)
    export_df["hand_label"] = ""

    out_path = os.path.join(os.path.dirname(__file__), "output",
                            "validation_sample_v2.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_df.to_csv(out_path, index=False)
    print(f"\nExported to {out_path}")
    print("Fill the 'hand_label' column and rerun to measure accuracy.")
    print("\n⚠  STOP HERE — review the sample before running the full corpus.")

    con.close()
    return df


# ---------------------------------------------------------------------------
# Full run mode
# ---------------------------------------------------------------------------
def run_full(db_path=DUCKDB_PATH):
    con = duckdb.connect(db_path)
    ensure_v2_table(con)
    client = anthropic.Anthropic()

    query = """
        SELECT g.url, g.title
        FROM gdelt_articles g
        LEFT JOIN llm_classifications_v2 v ON g.url = v.url
        WHERE v.url IS NULL
        ORDER BY g.seendate
    """
    if MAX_ARTICLES:
        query += f" LIMIT {MAX_ARTICLES}"

    unprocessed = con.execute(query).fetchall()
    total = len(unprocessed)
    print(f"Found {total} unclassified headlines (v2)")
    print(f"Model: {MODEL}, k={ENSEMBLE_K}")
    print(f"Total API calls: ~{total * ENSEMBLE_K}")

    processed = 0
    errors = 0

    for i, (url, title) in enumerate(unprocessed):
        r = classify_ensemble(client, title)

        try:
            con.execute(
                """INSERT OR IGNORE INTO llm_classifications_v2
                   (url, direction, ensemble_confidence, self_confidence,
                    magnitude, explanation, vote_distribution, model_id, ensemble_k)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [url, r["direction"], r["ensemble_confidence"], r["self_confidence"],
                 r["magnitude"], r["explanation"], json.dumps(r["vote_distribution"]),
                 MODEL, ENSEMBLE_K],
            )
            processed += 1
        except Exception as e:
            errors += 1
            print(f"  Insert error: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{total} "
                  f"(latest: {r['direction']}, agree={r['ensemble_confidence']:.0%})")

        if (i + 1) % 50 == 0:
            time.sleep(BATCH_PAUSE)

    dist = con.execute("""
        SELECT direction, COUNT(*) AS n, AVG(ensemble_confidence) AS avg_agree
        FROM llm_classifications_v2
        GROUP BY direction ORDER BY n DESC
    """).fetchdf()
    print(f"\nDone: {processed} classified, {errors} errors")
    print(dist.to_string(index=False))

    con.close()
    return processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        run_validate()
    elif mode == "run":
        run_full()
    else:
        print(f"Usage: {sys.argv[0]} [validate|run]")
        sys.exit(1)
