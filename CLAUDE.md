# Balance Sheet Expectations — Handoff

Bybee (2025) "Ghost in the Machine" replication: LLM-derived belief index ($F_t$) from news vs NY Fed professional forecaster survey expectations. Main result: Spearman rho = 0.51, p = 0.005, N = 29.

## Current state

- **Working tree is clean**, all pushed to `origin/main` at `dbd71ce`.
- **Repo**: `https://github.com/matteonorm/fed-balance-sheet-expectations`
- **Local dir**: `/Users/matteoangelonormanno/ecb-balance-sheet-expectations` (folder name still says "ecb" but the GitHub repo was renamed to `fed-balance-sheet-expectations`).

## Stack

Python, DuckDB (`fed_bs.duckdb`), Claude Haiku (`claude-haiku-4-5`) for classification, matplotlib.

## Key files

- `config.py` — paths, regimes, API config
- `schema.py` — DuckDB tables
- `collect_nyfed_survey.py` — NY Fed Excel surveys (Jul 2023+)
- `extract_pdf_surveys.py` — NY Fed PDF surveys (2011–2023), Claude Haiku extraction, cached JSONs in `data/pdf_extractions/`
- `collect_fred.py` — FRED weekly actuals (WALCL, TREAST, WSHOMCB, WRESBAL)
- `collect_gdelt.py` — GDELT DOC 2.0 articles
- `collect_gnews.py` — Google News RSS articles
- `collect_nyt.py` — NYT Article Search API (backfills to 2011), caches raw JSON to `data/nyt_cache/`
- `classify.py` — 4-class classifier (increase/decrease/uncertain/not_relevant), k=5 ensemble majority vote
- `aggregate.py` — monthly $F_t = (n_{increase} - n_{decrease}) / (n_{increase} + n_{decrease})$
- `leadlag_analysis.py` — differenced cross-correlation with HAC/Newey-West SEs
- `visualize.py` — 2 figures: fig1 (belief index + regime shading), fig2 (F_t vs survey, single chart, percentile-rank y-axis)
- `run_pipeline.py` — end-to-end runner

## Database (`fed_bs.duckdb`)

| Table | Rows | Notes |
|-------|------|-------|
| gdelt_articles | 4,749 | All news sources (gdelt/gnews/nyt) |
| llm_classifications | 4,749 | 4-class, k=5 ensemble |
| llm_expectations | 182 | Monthly F_t |
| nyfed_survey_bs | 4,999 | Excel (3,766) + PDF (1,233) |
| fed_balance_sheet | 964 | FRED weekly actuals |
| nyfed_survey_runoff | 410 | Runoff timing/size |

## Six lessons baked in (from prior ECB project)

1. Relevance gate (4-class classification, not binary)
2. Confidence = ensemble agreement (k=5 majority vote)
3. Few-shot anchoring in classifier prompt
4. Survey object is PACE/PATH (directional, not levels)
5. First differences with HAC standard errors for lead-lag
6. Report nulls as nulls

## Known issues and quirks

- **DuckDB single-writer constraint**: never run two write scripts simultaneously. GDELT collector holds the lock for entire run.
- **Survey signal in fig2** combines two data types: pace/change data (2013–2022, in $bn/mo) and first-differenced level expectations (2018–2026, in $bn). The percentile-rank transform on the y-axis makes this visually comparable.
- **Gaps in fig2** (2011-2012, 2015-2017): real data gaps — either too few relevant articles for F_t or no survey rounds in those months. The Fed was in reinvestment mode 2015-2017 with minimal BS news.
- **F_t is noisy** when n_relevant is small (many months have F_t = +1 or -1 from just 2-3 articles).
- **PDF extraction**: all 86 PDFs cached in `data/pdf_extractions/*.json`. Re-extraction requires `ANTHROPIC_API_KEY`. Ingestion can be re-run from cache without the key (call `ingest_pdf_extractions()` directly).
- **NYT API**: 5 req/min rate limit (12s between calls). Key stored in `.env`. The `fq` source filter doesn't work (returns 0 results) — queries run unfiltered. The `hits` field is unreliable.
- **Classification**: 88.8% of articles classified as not_relevant. Full run takes ~1 hour with Haiku.
- **`.gitignore`**: `output/` is excluded; use `git add -f` for output files.

## Rules

- API keys in `.env` (gitignored): `ANTHROPIC_API_KEY`, `NYT_API_KEY`
- FRED data arrives in millions; collectors divide by 1e3 for billions
- Validate classifier on ~100-item sample before full runs (`python classify.py validate`)
- User preference: no interpretation or honest assessment — stick to Bybee's framing ("professional balance-sheet beliefs track the news contemporaneously")
- User preference: minimal charts, no unnecessary files

## Suggested skills

- `/commit` — for staging and committing changes
- `/pr` — if creating a pull request
- `/code-review` — for reviewing changes before committing
