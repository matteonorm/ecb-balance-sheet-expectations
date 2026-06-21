# ECB Balance Sheet Expectations — Handoff

## Project summary

Compares ECB balance sheet expectations from two sources:
1. **LLM-derived beliefs** from news headlines (Bybee 2025 "Ghost in the Machine" method): classify headlines → compute monthly balance statistic F_t = (n_increase − n_decrease) / (n_increase + n_decrease).
2. **ECB Survey of Monetary Analysts (SMA)**: professional forecasters' expected APP+PEPP holdings at multiple horizons.

Stack: Python, DuckDB (`ecb_bs.duckdb`), Claude Haiku for classification, matplotlib (Kieran Healy style). GitHub: `matteonorm/ecb-balance-sheet-expectations`.

## Current state and key findings

### What works
- **Full pipeline** runs end-to-end: collect articles (Google News RSS, 2,895 headlines) → classify with LLM → aggregate monthly F_t → compare with SMA.
- **SMA collector fixed** (`collect_sma_v2.py`): now pulls **40 vintages** (June 2021 – June 2026), up from 30. Handles three different ECB CSV naming conventions. One round missing (APR25 was found but needed manual URL override, now included).
- **ECB balance sheet data**: securities held for monetary policy (asset 7.1) from ECB Data Portal ILM series + APP/PEPP monthly holdings from ECB CSVs.

### Key empirical results
- **Dispersion analysis: clean null.** LLM headline disagreement does not track SMA cross-forecaster IQR (r = −0.04, p = 0.84). Headlines carry first-moment signal, not heterogeneity. See `dispersion_analysis.py`, figs 7-8.
- **Lead-lag (differenced): suggestive but fragile.** dF_t at lag +1 predicts SMA pace revision (ρ = +0.32, p = 0.056, N = 37). Sign is economically correct (positive). But: fails HAC/Newey-West (t = 0.38, p = 0.71 — autocorrelated pace inflates naive Spearman), and 27/37 jackknife drops flip significance. Signal is concentrated in the 2021-22 QT transition. See `leadlag_analysis.py` and `robustness_leadlag.py`, figs 9-11.
- **Level correlogram was a shared-trend mirage** — all level significance vanishes after differencing.
- **V1 classification quality is poor**: 90% of headlines are off-topic noise (rate decisions, unrelated politics, other central banks). The v1 classifier forced everything into increase/decrease/uncertain without a relevance gate.

### Binding constraints
- **Too few survey observations**: 40 SMA rounds, one policy regime (QT). Not enough variation to establish causal lead-lag.
- **Too few relevant articles**: ~290 out of 2,895 are genuinely about the ECB balance sheet. Many months have 0-2 relevant headlines.
- **Too little regime variation**: ECB has been doing QT monotonically since mid-2022; F_t is stuck near −1 for most of the sample.

## What's in progress / next steps

### V2 classifier (Part A — ready for hand-labelling)
- `classify_v2.py` implements: relevance gate (`not_relevant` label), few-shot anchors (8 worked examples), k=5 ensemble agreement confidence.
- **Validation sample done**: 100 headlines classified, exported to `output/validation_sample_v2.csv` with empty `hand_label` column.
- The user needs to fill `hand_label`, then run accuracy comparison (v1 vs v2 against hand labels).
- After approval: `python classify_v2.py run` to classify all 2,895 headlines into `llm_classifications_v2` table. Then rebuild F_t from relevance-gated articles and rerun lead-lag.
- The key test: does relevance-gating move ρ above v1's +0.32 (signal was attenuated by noise) or below (noise-driven)?

### Theoretical framing
- The user wants to connect this to **Brunnermeier's optimal unconventional monetary policy** — specifically the optimal *pace* of balance sheet normalization.
- The expected 1yr-ahead pace (from SMA multi-horizon structure) is the right empirical object for this.
- The reframing opportunity: the contribution may be methodological (relevance gate matters, dispersion doesn't work, signal concentrates at regime transitions) rather than a clean lead-lag result.

## Database schema

```
gdelt_articles        — 2,895 rows. PK: url. Cols: title, seendate, domain, language, sourcecountry, query_keyword
llm_classifications   — 2,895 rows (v1). PK: url. Cols: direction, confidence, magnitude, explanation, model_id
llm_classifications_v2 — created by classify_v2.py. PK: url. Cols: direction, ensemble_confidence, self_confidence, magnitude, explanation, vote_distribution, model_id, ensemble_k
llm_expectations      — 160 rows. PK: period (YYYY-MM). Cols: n_increase, n_decrease, n_uncertain, n_total, f_statistic
sma_raw               — ~35,000 rows. 17 VARCHAR columns from ECB CSVs.
sma_expectations      — ~2,200 rows. PK: (vintage, forecast_date, measure). Cols: vintage_date, app/pepp/total_holdings_eur
ecb_app_pepp          — ~140 rows. PK: observation_date. Cols: app/pepp/total_holdings_eur
ecb_policy_bs         — ~650 rows. PK: observation_date. Cols: securities_eur, lending_eur, total_policy_eur
```

No body/snippet column in `gdelt_articles` — only titles.

## Script inventory

| Script | Purpose |
|---|---|
| `config.py` | Paths, API config, constants |
| `schema.py` | DuckDB table creation |
| `collect_gnews.py` | Google News RSS collection (19 queries, semi-annual historical windows) |
| `collect_sma.py` | Original SMA collector (30 vintages, broken URL pattern) |
| `collect_sma_v2.py` | **Fixed SMA collector** (40 vintages, all naming conventions) |
| `collect_ecb_bs.py` | APP+PEPP holdings + policy balance sheet from ECB |
| `process_headlines.py` | V1 classifier (single Haiku call, no relevance gate) |
| `classify_v2.py` | **V2 classifier** (relevance gate, few-shot, k=5 ensemble). Modes: `validate` / `run` |
| `aggregate.py` | Builds monthly F_t in `llm_expectations` from v1 |
| `compare.py` | V1 comparison: F_t vs SMA next-quarter change |
| `visualize.py` | Figures 1-6 (main comparison, classification shares, coverage, scatter, confidence, regimes) |
| `dispersion_analysis.py` | Figures 7-8 (dispersion null result) |
| `leadlag_analysis.py` | Figures 9-10 (lead-lag correlograms, levels vs differences) |
| `robustness_leadlag.py` | Figure 11 (scatter + jackknife + HAC + construction sensitivity) |
| `run_pipeline.py` | End-to-end pipeline runner |

## Hard rules
- **Never overwrite v1 tables or results.** V2 classifications go to `llm_classifications_v2`.
- **API key** is in `.env` (gitignored). Export before running classifiers: `export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2)`.
- **Output PNGs** are gitignored by default; use `git add -f output/filename.png` to commit.
- DuckDB does not support concurrent writers — don't run two write scripts simultaneously.
- DuckDB `strftime` requires single quotes (`'%Y-%m'`), not double quotes.

## Known gotchas
- APP CSV has quoted numbers with commas (e.g. `"-3,564"`). `collect_ecb_bs.py` uses `csv.reader` to handle this.
- ECB Data Portal ILM series keys use `U2.EUR` suffix, not `Z5.Z01`.
- SMA data is in EUR billion (values ~4000-5000); ECB policy BS data is in EUR millions. Divide ECB by 1e3 for figures.
- The `collect_sma_v2.py` has `EXTRA_ROUNDS` for 2026 rounds not yet in the ECB dates CSV — update this list as new rounds are published.

## Suggested skills

- `/brainstorming` — if the user wants to reframe the research contribution (e.g., pivoting from lead-lag to methodological comparison, or designing a cross-central-bank extension).
- `/plan` — before implementing the v2 full-run → rebuild F_t → rerun lead-lag pipeline, or before adding new data sources (Reuters, Bloomberg, FT).
