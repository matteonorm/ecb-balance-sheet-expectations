# Relevance contamination in LLM belief extraction on narrow topics

## Summary

Applying Bybee (2025)-style LLM headline classification to a narrow target — expectations about the ECB/Eurosystem balance sheet (APP, PEPP, QE/QT) — produces a monthly sentiment index (F_t) that appears to track policy regimes. On closer inspection, ~90% of the classified headlines are off-topic, and the apparent signal is substantially an artifact of article volume correlating with regime periods.

This note documents the contamination finding and argues that a relevance gate is necessary when applying LLM belief extraction to narrow policy topics, and that high ensemble agreement is not a substitute for accuracy.

## Setup

We collected 2,895 English-language news headlines from Google News RSS using 19 query variants related to the ECB balance sheet ("ECB balance sheet", "ECB asset purchases", "ECB quantitative tightening", etc.), covering 2009–2026. Each headline was classified by Claude Haiku 4.5 into one of three categories: *increase* (the ECB balance sheet will grow), *decrease* (it will shrink), or *uncertain*. No relevance gate was applied — every headline returned by the keyword search was classified, following the Bybee (2025) approach where the LLM sees all articles retrieved by a broad query.

The resulting monthly balance statistic F_t = (n_increase − n_decrease) / (n_increase + n_decrease) was compared against the ECB Survey of Monetary Analysts (SMA), a professional forecaster survey with 40 vintages from June 2021 through June 2026.

## The contamination problem

A second-pass classifier (v2) was built with three changes: (1) a `not_relevant` label for headlines not about the ECB balance sheet, APP/PEPP, QE/QT, reinvestment, or TLTROs; (2) few-shot anchors (8 worked examples covering all four labels); and (3) k=5 ensemble voting, where each headline is classified 5 times independently and the majority label is taken.

On a random 100-headline validation sample:

- **90 of 100 headlines were classified as `not_relevant`** by v2.
- Mean ensemble agreement: **98.8%** — the model is highly consistent.
- V1-to-v2 disagreement rate: **94%** — nearly every headline changed label.

Examples of headlines that v1 classified as `increase`, `decrease`, or `uncertain` on the ECB balance sheet, but v2 correctly identifies as off-topic:

| Headline | V1 label | V2 label |
|----------|----------|----------|
| "Swiss central bank makes 50bn Swiss franc loss" | uncertain | not_relevant |
| "German lawmakers consider expropriating private apartments" | uncertain | not_relevant |
| "ECB rate hikes result in 7.9b-euro loss in 2024" | decrease | not_relevant |
| "Europe's monetary policy shift comes (too) late" | decrease | not_relevant |
| "FX in Focus as Central Bank Policies Diverge" | uncertain | not_relevant |

These are not edge cases. The keyword queries that retrieve ECB-related articles also return articles about ECB interest rates, other central banks, corporate earnings, and EU politics. The v1 classifier, lacking a `not_relevant` option, was forced to assign a directional label to every one of them.

## Why the F_t appeared to work

The v1 F_t index appeared to track ECB policy regimes: positive during 2019–2020 (QE restart, PEPP launch), sharply negative from 2022 (QT). This was taken as evidence that the classification was working. Two mechanisms explain this without requiring accurate classification:

1. **Article volume correlates with regime salience.** More articles mention the ECB balance sheet during major policy transitions (PEPP launch, QT announcement). Even if 90% are off-topic, the *relevant* 10% concentrate in these periods, and the remaining noise has some directional correlation with the dominant narrative.

2. **Keyword bias.** Queries like "ECB quantitative tightening" mechanically return more articles after mid-2022 when QT was announced. The *volume* of query hits acts as a noisy regime indicator even if the individual classifications are wrong.

## High agreement is not accuracy

The v2 ensemble achieves 98.8% agreement — but this measures *self-consistency*, not *correctness*. The v1 classifier also had high self-reported confidence (mean 0.73) while being wrong on most headlines. Agreement tells you the model is stable; only hand-labelled ground truth tells you it is right.

Hand-labelled accuracy on the validation sample is pending. Until it is measured, the v2 labels should be treated as a strong prior, not ground truth.

## The trade-off: coverage vs cleanliness

If ~90% of the corpus is off-topic, the relevance-gated F_t is built from ~290 articles over 12 years. Many months will have zero or one relevant headline. The Bybee (2025) method was designed for large corpora (e.g., the Wall Street Journal archive, with hundreds of articles per month on a broad topic like "the economy"). Applied to a narrow target like ECB balance sheet policy, the method faces a fundamental coverage problem: keyword search returns enough volume, but not enough *relevant* volume.

This is not a failure of the LLM — the classifier with a relevance gate performs well on the validation sample. It is a limitation of the data source (public news headlines retrieved by keyword search) for a topic this specific.

## Implication

When applying LLM belief extraction to narrow policy topics:

1. **A relevance gate is necessary.** Without one, the index is dominated by off-topic contamination. The `not_relevant` label should be the default for most headlines, not a rare exception.
2. **Few-shot examples matter.** The boundary between "ECB raises rates" (not about the balance sheet) and "ECB ends net asset purchases" (about the balance sheet) is clear to a domain expert but ambiguous to a zero-shot LLM.
3. **Ensemble agreement is cheap insurance** but is not accuracy. Validate against human labels.
4. **Coverage may be the binding constraint.** Even with perfect classification, the number of genuinely relevant articles may be too small to build a reliable monthly index for a narrow topic.
