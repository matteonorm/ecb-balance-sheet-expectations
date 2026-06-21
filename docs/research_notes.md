# Research notes: stylized facts for theory motivation

These are empirical observations from the ECB balance-sheet expectations project, documented as potential motivation for theoretical work on optimal unconventional monetary policy (cf. Brunnermeier). They are **not** claimed as robust empirical findings — the underlying data is thin and the sample spans a single regime. They are recorded here because they describe features of the data that a theory might want to match.

## 1. The SMA expected pace series

The ECB Survey of Monetary Analysts asks professional forecasters to project APP+PEPP holdings at quarterly horizons out to ~10 years. From the multi-horizon structure, we compute the expected annualized pace of balance-sheet change: the percentage change in median projected total holdings from the current quarter to ~4 quarters ahead.

This series runs from **+12% in December 2021** (forecasters still expected balance-sheet expansion under ongoing net asset purchases) to **−12% in December 2025** (deep quantitative tightening, with runoff accelerating as PEPP reinvestments ended).

The transition is not monotonic. Key features:

- **Rapid downward revision in early 2022**: the expected pace drops from +12% to near 0% between Dec 2021 and Jun 2022 (6 months), coinciding with the ECB's announcement of APP net purchase end and the first rate hikes.
- **Plateau through mid-2022**: the pace hovers near 0% from Jun–Sep 2022, reflecting uncertainty about whether the ECB would reinvest maturing securities fully.
- **Steady decline from late 2022**: once APP partial reinvestment was announced (Oct 2022) and PEPP reinvestment end dates became clearer, the expected pace moved steadily more negative, reaching −12% to −13% by late 2025.

This series is a clean, directly observed measure of the expected speed of balance-sheet normalization at the 1-year horizon, available at ~bimonthly frequency. It could serve as a dependent variable or calibration target in a model of optimal QT pace.

Source: 40 SMA vintages (Jun 2021 – Jun 2026), MEDIAN measure, `sma_expectations` table. The April 2019 – May 2021 pilot-phase data (which would cover the QE-expansion regime) was not released publicly.

## 2. Signal concentrates at regime transitions

The (noisy, contaminated) LLM balance statistic F_t and the SMA expected pace both show a pattern: variation concentrates at policy regime transitions, not during the continuation of a regime.

In the data:

- **2021-22 QT onset**: F_t swings sharply, SMA pace revisions are large (−3% to −5% per vintage). This is the period where the differenced lead-lag shows its (fragile) positive signal.
- **2023-25 QT continuation**: F_t is stuck near −1 (headlines unanimously say "decrease"), SMA pace revisions are small (−0.5% to −1% per vintage). Both series are essentially flat in differences.

The lead-lag finding, to the extent it exists, is carried entirely by the transition period. During steady-state QT, neither the news nor the survey moves much.

This is consistent with rational inattention or state-dependent information acquisition: the pace of balance-sheet normalization is newsworthy when it *changes* (policy announcements, unexpected acceleration/deceleration), not while it *continues* at a roughly known rate. In a Brunnermeier-type framework, this would suggest that the information structure around the central bank's exit strategy is itself state-dependent — market participants invest attention in learning the pace only when priors are disrupted.

This is an observation about the data, not a tested hypothesis. The sample contains exactly one regime transition (QE → QT), so the claim that "signal concentrates at transitions" is a description of one event, not a general pattern.
