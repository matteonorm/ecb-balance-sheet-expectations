"""
Revision-direction test: does the sign of round-to-round survey belief
revisions align with the LLM-derived news index F_t?

Design: 7 non-redundant blocks of NY Fed survey BS-size expectations,
each using its native variable. Within each block, the round-to-round
change of median (pctl50) at a FIXED absolute horizon is signed:
  +1 = revised toward expansion (larger BS / more purchases)
  -1 = revised toward contraction
   0 = no change (exact tie only, by default)

Each block is classified as NEAR-TERM or LONG-RUN based on the offset
between the selected horizon and the block's last survey round
(threshold: 1.5 years). The near-term class is the primary test object
— conceptually comparable to a high-frequency news index. The long-run
class captures terminal/settle-point beliefs at multi-year horizons.

F_t predictor is kept continuous; dF_t is the change aligned to each
block's survey-round spacing.

Tests report BOTH unclustered and period-clustered SEs (by year-quarter),
since blocks C/D/E overlap in calendar time 2018-2020.
"""

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from config import DUCKDB_PATH, OUTPUT_DIR
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Block definitions ────────────────────────────────────────────────
BLOCKS = [
    {
        "id": "A",
        "variable": "total_soma_soma_size_dist",
        "panel": "SPD",
        "start": "2011-01-01",
        "end": "2012-08-01",
        "label": "SOMA size (total)",
        "regime": "Pre-taper / QE2-OT",
    },
    {
        "id": "B",
        "variable": "treasury_soma_change_path",
        "panel": "SPD",
        "start": "2012-11-01",
        "end": "2014-11-01",
        "label": "SOMA change path (Tsy)",
        "regime": "QE3 / taper tantrum",
    },
    {
        "id": "C",
        "variable": "reserves_reserve_path",
        "panel": "SPD",
        "start": "2017-05-01",
        "end": "2019-10-01",
        "label": "Reserves path",
        "regime": "QT1 / reinvestment end",
    },
    {
        "id": "D",
        "variable": "total_soma_soma_size_dist",
        "panel": "SPD",
        "start": "2018-12-01",
        "end": "2020-07-01",
        "label": "SOMA size (total)",
        "regime": "End-QT1 / COVID QE",
    },
    {
        "id": "E",
        "variable": "treasury_purchase_pace",
        "panel": "SPD",
        "start": "2019-10-01",
        "end": "2022-02-01",
        "label": "Purchase pace (Tsy)",
        "regime": "COVID QE / taper",
    },
    {
        "id": "F",
        "variable": "treasury_soma_change_path",
        "panel": "SPD",
        "start": "2022-02-01",
        "end": "2022-08-01",
        "label": "SOMA change path (Tsy)",
        "regime": "QT2 onset",
    },
    {
        "id": "G",
        "variable": "total_assets",
        "panel": "Combined",
        "start": "2023-09-01",
        "end": "2026-05-01",
        "label": "Total assets (level)",
        "regime": "QT2 / current",
    },
]

MIN_N_RELEVANT = 3
HORIZON_CLASS_THRESHOLD_YEARS = 1.5


def load_data():
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    survey = con.execute(
        "SELECT survey_date, panel_type, variable, horizon_date, pctl50 "
        "FROM nyfed_survey_bs"
    ).fetchdf()
    ft = con.execute(
        "SELECT period, f_statistic, n_relevant FROM llm_expectations"
    ).fetchdf()
    con.close()
    survey["survey_date"] = pd.to_datetime(survey["survey_date"])
    survey["horizon_date"] = pd.to_datetime(survey["horizon_date"])
    ft["period"] = pd.to_datetime(ft["period"])
    return survey, ft


def pick_fixed_horizon(block_data):
    """Pick the absolute horizon_date that maximises informative revision pairs.

    Algorithm — score each horizon by:
      coverage (n pairs) × fraction of round-to-round changes that are non-zero
    This penalises horizons where pctl50 is flat across rounds. Among ties,
    prefer the horizon closest to median(survey_date) + 6 months.

    Returns (chosen horizon_date, DataFrame of rows at that horizon)."""
    scores = []
    for hz, grp in block_data.groupby("horizon_date"):
        deduped = grp.drop_duplicates("survey_date").sort_values("survey_date")
        n_rounds = len(deduped)
        if n_rounds < 2:
            scores.append({"horizon_date": hz, "n_rounds": n_rounds,
                           "n_pairs": 0, "nonzero_frac": 0.0, "score": 0.0})
            continue
        diffs = deduped["pctl50"].diff().iloc[1:]
        n_pairs = len(diffs)
        nonzero_frac = (diffs != 0).sum() / n_pairs if n_pairs > 0 else 0.0
        score = n_pairs * nonzero_frac
        scores.append({"horizon_date": hz, "n_rounds": n_rounds,
                        "n_pairs": n_pairs, "nonzero_frac": nonzero_frac, "score": score})
    hz_scores = pd.DataFrame(scores)

    max_score = hz_scores["score"].max()
    if max_score > 0:
        candidates = hz_scores[hz_scores["score"] == max_score]["horizon_date"].values
    else:
        max_pairs = hz_scores["n_pairs"].max()
        candidates = hz_scores[hz_scores["n_pairs"] == max_pairs]["horizon_date"].values

    median_sd = block_data["survey_date"].sort_values().iloc[
        len(block_data["survey_date"].unique()) // 2
    ]
    target = median_sd + pd.DateOffset(months=6)
    best_hz = min(candidates, key=lambda h: abs((pd.Timestamp(h) - target).days))
    best_hz = pd.Timestamp(best_hz)

    subset = block_data[block_data["horizon_date"] == best_hz].copy()
    subset = subset.sort_values("survey_date").drop_duplicates("survey_date")
    return best_hz, subset


def classify_horizon(hz_date, last_survey_date):
    """Classify a block's horizon as NEAR-TERM or LONG-RUN based on the
    offset between the selected horizon and the block's last survey round."""
    offset_years = (hz_date - last_survey_date).days / 365.25
    if offset_years <= HORIZON_CLASS_THRESHOLD_YEARS:
        return "NEAR-TERM", offset_years
    else:
        return "LONG-RUN", offset_years


def build_revision_signs(survey_df):
    """Build revision-sign series for all blocks, with horizon_class tags."""
    rows = []
    horizon_info = {}
    block_meta = {}

    for blk in BLOCKS:
        mask = (
            (survey_df["variable"] == blk["variable"])
            & (survey_df["panel_type"] == blk["panel"])
            & (survey_df["survey_date"] >= blk["start"])
            & (survey_df["survey_date"] <= blk["end"])
        )
        bdata = survey_df[mask].copy()
        if bdata.empty:
            print(f"  Block {blk['id']}: NO DATA")
            continue

        hz_date, subset = pick_fixed_horizon(bdata)
        horizon_info[blk["id"]] = hz_date

        subset = subset.sort_values("survey_date").reset_index(drop=True)
        last_round = subset["survey_date"].max()
        hz_class, offset_yr = classify_horizon(hz_date, last_round)

        block_meta[blk["id"]] = {
            "horizon_date": hz_date,
            "horizon_class": hz_class,
            "offset_years": offset_yr,
            "first_round": subset["survey_date"].min(),
            "last_round": last_round,
        }

        for i in range(1, len(subset)):
            prev = subset.iloc[i - 1]
            curr = subset.iloc[i]
            delta = curr["pctl50"] - prev["pctl50"]
            if delta > 0:
                sign = 1
            elif delta < 0:
                sign = -1
            else:
                sign = 0
            rows.append(
                {
                    "block_id": blk["id"],
                    "survey_date": curr["survey_date"],
                    "prev_date": prev["survey_date"],
                    "pctl50": curr["pctl50"],
                    "prev_pctl50": prev["pctl50"],
                    "delta_pctl50": delta,
                    "revision_sign": sign,
                    "horizon_date": hz_date,
                    "horizon_class": hz_class,
                    "variable": blk["variable"],
                    "regime": blk["regime"],
                    "label": blk["label"],
                }
            )

    revisions = pd.DataFrame(rows)
    return revisions, horizon_info, block_meta


def align_ft(revisions, ft_df):
    """Align F_t to survey rounds. Compute dF_t between consecutive rounds.
    Inner-join, requiring n_relevant >= MIN_N_RELEVANT for BOTH months."""

    def survey_month(d):
        return pd.Timestamp(year=d.year, month=d.month, day=1)

    ft_lookup = {}
    for _, r in ft_df.iterrows():
        ft_lookup[r["period"]] = (r["f_statistic"], r["n_relevant"])

    rows = []
    for _, rev in revisions.iterrows():
        curr_month = survey_month(rev["survey_date"])
        prev_month = survey_month(rev["prev_date"])

        curr_ft_val, curr_nrel = ft_lookup.get(curr_month, (np.nan, 0))
        prev_ft_val, prev_nrel = ft_lookup.get(prev_month, (np.nan, 0))

        rows.append(
            {
                **rev.to_dict(),
                "ft_curr": curr_ft_val,
                "ft_prev": prev_ft_val,
                "ft_curr_nrel": curr_nrel if curr_nrel else 0,
                "ft_prev_nrel": prev_nrel if prev_nrel else 0,
                "dft": curr_ft_val - prev_ft_val if pd.notna(curr_ft_val) and pd.notna(prev_ft_val) else np.nan,
            }
        )

    merged = pd.DataFrame(rows)

    survived = merged[
        (merged["ft_curr_nrel"] >= MIN_N_RELEVANT)
        & (merged["ft_prev_nrel"] >= MIN_N_RELEVANT)
        & merged["dft"].notna()
    ].copy()

    return merged, survived


def print_horizon_classification(block_meta):
    """Print the horizon-class classification table."""
    print("\n" + "=" * 95)
    print("HORIZON-CLASS CLASSIFICATION (threshold: {:.1f} years from last survey round)".format(
        HORIZON_CLASS_THRESHOLD_YEARS))
    print("=" * 95)
    print(f"  {'Block':<6} {'Survey range':<28} {'Horizon':<14} {'Offset':>8} {'Class':<12} {'Regime'}")
    print("  " + "-" * 90)
    for blk in BLOCKS:
        bid = blk["id"]
        if bid not in block_meta:
            continue
        m = block_meta[bid]
        print(f"  {bid:<6} "
              f"{str(m['first_round'].date()) + ' to ' + str(m['last_round'].date()):<28} "
              f"{str(m['horizon_date'].date()):<14} "
              f"{m['offset_years']:+.1f} yr   "
              f"{m['horizon_class']:<12} "
              f"{blk['regime']}")
    near = [b for b, m in block_meta.items() if m["horizon_class"] == "NEAR-TERM"]
    far = [b for b, m in block_meta.items() if m["horizon_class"] == "LONG-RUN"]
    print(f"\n  NEAR-TERM blocks: {', '.join(near)} ({len(near)} blocks)")
    print(f"  LONG-RUN  blocks: {', '.join(far)} ({len(far)} blocks)")
    print()


def attrition_table(revisions, merged, survived, block_meta):
    """Print attrition table with horizon_class column."""
    print("=" * 95)
    print("ATTRITION TABLE: Survey revision pairs -> F_t inner join survival")
    print("=" * 95)
    print(f"  {'Block':<6} {'Class':<11} {'Regime':<26} {'Survey':>7} {'Joined':>7} {'Surv%':>6}  Notes")
    print("  " + "-" * 90)

    totals = {"NEAR-TERM": [0, 0], "LONG-RUN": [0, 0]}

    for blk in BLOCKS:
        bid = blk["id"]
        hc = block_meta.get(bid, {}).get("horizon_class", "?")
        n_survey = len(revisions[revisions["block_id"] == bid])
        n_survived = len(survived[survived["block_id"] == bid])
        pct = f"{100 * n_survived / n_survey:.0f}%" if n_survey > 0 else "—"

        note = ""
        if n_survived == 0:
            note = "LOST (sparse news)"
        elif n_survived < n_survey * 0.5:
            note = "heavy attrition"

        print(f"  {bid:<6} {hc:<11} {blk['regime']:<26} {n_survey:>7} {n_survived:>7} {pct:>6}  {note}")
        totals[hc][0] += n_survey
        totals[hc][1] += n_survived

    print("  " + "-" * 90)
    for hc in ["NEAR-TERM", "LONG-RUN"]:
        ns, nj = totals[hc]
        pct = f"{100 * nj / ns:.0f}%" if ns > 0 else "—"
        print(f"  {'sub':<6} {hc:<11} {'subtotal':<26} {ns:>7} {nj:>7} {pct:>6}")
    total_s = sum(t[0] for t in totals.values())
    total_j = sum(t[1] for t in totals.values())
    pct_t = f"{100 * total_j / total_s:.0f}%" if total_s > 0 else "—"
    print(f"  {'ALL':<6} {'—':<11} {'POOLED':<26} {total_s:>7} {total_j:>7} {pct_t:>6}")

    near_lost = totals["NEAR-TERM"][0] - totals["NEAR-TERM"][1]
    long_lost = totals["LONG-RUN"][0] - totals["LONG-RUN"][1]
    print(f"\n  News-join attrition by class:")
    print(f"    NEAR-TERM: {near_lost} of {totals['NEAR-TERM'][0]} pairs lost "
          f"({100*near_lost/totals['NEAR-TERM'][0]:.0f}%)" if totals["NEAR-TERM"][0] > 0 else "")
    print(f"    LONG-RUN:  {long_lost} of {totals['LONG-RUN'][0]} pairs lost "
          f"({100*long_lost/totals['LONG-RUN'][0]:.0f}%)" if totals["LONG-RUN"][0] > 0 else "")
    print()


def clustered_se_ols(y, x, clusters):
    """OLS slope with cluster-robust (CR1) standard errors.
    Returns (beta, se_unclustered, se_clustered, t_clustered, p_clustered, n)."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sigma2 = np.sum(resid ** 2) / (n - 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_uncl = np.sqrt(sigma2 * XtX_inv[1, 1])

    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    meat = np.zeros((2, 2))
    for c in unique_clusters:
        mask = clusters == c
        Xc = X[mask]
        ec = resid[mask]
        score = Xc.T @ ec
        meat += np.outer(score, score)
    scale = G / (G - 1) * (n - 1) / (n - 2)
    V_cl = XtX_inv @ meat @ XtX_inv * scale
    se_cl = np.sqrt(V_cl[1, 1])
    t_cl = beta[1] / se_cl if se_cl > 0 else np.nan
    p_cl = 2 * stats.t.sf(abs(t_cl), df=G - 1) if not np.isnan(t_cl) else np.nan
    return beta[1], se_uncl, se_cl, t_cl, p_cl, n


def _run_test_on_subset(data, label, indent="  "):
    """Run sign-agreement, Spearman, and OLS on a subset. Returns a results dict."""
    nonzero = data[data["revision_sign"] != 0].copy()
    n_zeros = len(data) - len(nonzero)
    print(f"{indent}Observations: {len(data)} total, {n_zeros} exact zeros excluded, "
          f"{len(nonzero)} used.")

    result = {"n_total": len(data), "n_zeros": n_zeros, "n_nonzero": len(nonzero)}

    if len(nonzero) < 4:
        print(f"{indent}N < 4, insufficient for tests.\n")
        return result

    if nonzero["revision_sign"].nunique() < 2:
        print(f"{indent}Revision sign is constant ({nonzero['revision_sign'].iloc[0]:+d} in all "
              f"{len(nonzero)} obs) — Spearman undefined.\n")
        result["rho_dft"] = np.nan
        result["p_dft"] = np.nan
        result["constant_sign"] = True
        return result

    rho_dft, p_dft = stats.spearmanr(nonzero["dft"], nonzero["revision_sign"])
    print(f"{indent}Spearman (dF_t vs revision sign): rho = {rho_dft:+.3f}, p = {p_dft:.4f}, N = {len(nonzero)}")
    result["rho_dft"] = rho_dft
    result["p_dft"] = p_dft

    rho_lvl, p_lvl = stats.spearmanr(nonzero["ft_curr"], nonzero["revision_sign"])
    print(f"{indent}Spearman (F_t level vs rev sign): rho = {rho_lvl:+.3f}, p = {p_lvl:.4f}, N = {len(nonzero)}")
    result["rho_lvl"] = rho_lvl
    result["p_lvl"] = p_lvl

    nonzero_dft = nonzero[nonzero["dft"] != 0].copy()
    if len(nonzero_dft) > 0:
        agree = int((np.sign(nonzero_dft["dft"]) == nonzero_dft["revision_sign"]).sum())
        n_agree = len(nonzero_dft)
        rate = agree / n_agree
        binom_p = stats.binomtest(agree, n_agree, 0.5).pvalue
        print(f"{indent}Sign agreement: {agree}/{n_agree} = {rate:.1%}, binomial p = {binom_p:.4f}")
        result["agree"] = agree
        result["n_agree"] = n_agree
        result["agree_rate"] = rate
        result["binom_p"] = binom_p
    else:
        result["agree_rate"] = np.nan
        result["binom_p"] = np.nan

    if len(nonzero) >= 5:
        y = nonzero["revision_sign"].values.astype(float)
        x = nonzero["dft"].values
        clusters = nonzero["survey_date"].dt.to_period("Q").astype(str).values
        beta, se_uncl, se_cl, t_cl, p_cl, n = clustered_se_ols(y, x, clusters)
        t_uncl = beta / se_uncl if se_uncl > 0 else np.nan
        p_uncl = 2 * stats.t.sf(abs(t_uncl), df=n - 2) if not np.isnan(t_uncl) else np.nan
        n_cl = len(np.unique(clusters))
        print(f"{indent}OLS: beta = {beta:+.4f}")
        print(f"{indent}  Unclustered: SE = {se_uncl:.4f}, t = {t_uncl:+.3f}, p = {p_uncl:.4f}")
        print(f"{indent}  Clustered (Q): SE = {se_cl:.4f}, t = {t_cl:+.3f}, p = {p_cl:.4f}  "
              f"[{n_cl} clusters, N = {n}]")
        result["beta"] = beta
        result["se_cl"] = se_cl
        result["p_cl"] = p_cl
        result["n_clusters"] = n_cl

    print()
    return result


def run_tests(survived):
    """Run tests split by horizon class, then pooled."""
    near = survived[survived["horizon_class"] == "NEAR-TERM"]
    long = survived[survived["horizon_class"] == "LONG-RUN"]

    print("=" * 95)
    print("TEST A — PRIMARY: NEAR-TERM class only")
    print("  (pace/level revisions within ~1 year of survey; conceptually comparable to")
    print("   a high-frequency news index)")
    print("=" * 95)
    near_results = _run_test_on_subset(near, "NEAR-TERM", indent="  ")

    print("=" * 95)
    print("TEST B — SECONDARY: LONG-RUN class only")
    print("  (terminal/settle-point beliefs at multi-year horizons; a different, slower-moving")
    print("   object that a daily news index should not be expected to track the same way)")
    print("=" * 95)
    long_results = _run_test_on_subset(long, "LONG-RUN", indent="  ")

    print("=" * 95)
    print("TEST C — POOLED (all blocks, MIXES horizon classes — for completeness only)")
    print("=" * 95)
    pooled_results = _run_test_on_subset(survived, "POOLED", indent="  ")

    return {"near": near_results, "long": long_results, "pooled": pooled_results}


def make_figure(survived, revisions, all_results, block_meta):
    """Two-panel figure: scatter colored by horizon_class, and attrition bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # ── Panel 1: dF_t vs revision sign, colored by horizon_class ──
    nonzero = survived[survived["revision_sign"] != 0].copy()

    class_styles = {
        "NEAR-TERM": {"color": "#1f77b4", "marker": "o", "label": "Near-term"},
        "LONG-RUN": {"color": "#ff7f0e", "marker": "s", "label": "Long-run"},
    }

    rng = np.random.default_rng(42)
    for hc, style in class_styles.items():
        subset = nonzero[nonzero["horizon_class"] == hc]
        if subset.empty:
            continue
        jitter = rng.uniform(-0.12, 0.12, len(subset))
        ax1.scatter(
            subset["dft"],
            subset["revision_sign"] + jitter,
            c=style["color"],
            marker=style["marker"],
            alpha=0.7,
            s=60,
            label=style["label"],
            edgecolors="white",
            linewidth=0.5,
        )

    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("$\\Delta F_t$ (news index change between survey rounds)")
    ax1.set_ylabel("Survey revision sign (+1 = expansion, −1 = contraction)")
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(["−1 (down)", "0", "+1 (up)"])
    ax1.legend(loc="upper left", fontsize=9)

    nr = all_results.get("near", {})
    lr = all_results.get("long", {})
    lines = []
    if "agree_rate" in nr and not np.isnan(nr.get("agree_rate", np.nan)):
        lines.append(f"Near-term: agree {nr['agree_rate']:.0%} "
                     f"(ρ={nr.get('rho_dft', 0):+.2f}, N={nr['n_nonzero']})")
    if "agree_rate" in lr and not np.isnan(lr.get("agree_rate", np.nan)):
        lines.append(f"Long-run: agree {lr['agree_rate']:.0%} "
                     f"(ρ={lr.get('rho_dft', 0):+.2f}, N={lr['n_nonzero']})")
    if lines:
        ax1.text(0.98, 0.02, "\n".join(lines), transform=ax1.transAxes,
                 fontsize=8, va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))

    ax1.set_title("$\\Delta F_t$ vs survey revision sign (by horizon class)")

    # ── Panel 2: attrition bars per block, shaded by class ────────
    block_ids = [b["id"] for b in BLOCKS]
    n_survey = [len(revisions[revisions["block_id"] == bid]) for bid in block_ids]
    n_surv = [len(survived[survived["block_id"] == bid]) for bid in block_ids]
    bar_colors = ["#1f77b4" if block_meta.get(bid, {}).get("horizon_class") == "NEAR-TERM"
                  else "#ff7f0e" for bid in block_ids]

    x_pos = np.arange(len(block_ids))
    width = 0.35
    ax2.bar(x_pos - width / 2, n_survey, width, color=bar_colors, alpha=0.35,
            label="Survey pairs")
    ax2.bar(x_pos + width / 2, n_surv, width, color=bar_colors, alpha=0.85,
            label="After F_t join")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(block_ids)
    ax2.set_xlabel("Block")
    ax2.set_ylabel("Revision pairs")
    ax2.legend(fontsize=8)
    ax2.set_title("Join attrition per block\n(blue = near-term, orange = long-run)")

    for i, (ns, nsv) in enumerate(zip(n_survey, n_surv)):
        if ns > 0:
            pct = f"{100 * nsv / ns:.0f}%"
            ax2.text(i + width / 2, nsv + 0.3, pct, ha="center", fontsize=7)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "revision_direction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}\n")
    return path


def print_verdict(revisions, survived, all_results, block_meta):
    """Print factual, descriptive verdict split by horizon class."""
    near_rev = revisions[revisions["horizon_class"] == "NEAR-TERM"]
    long_rev = revisions[revisions["horizon_class"] == "LONG-RUN"]
    near_surv = survived[survived["horizon_class"] == "NEAR-TERM"]
    long_surv = survived[survived["horizon_class"] == "LONG-RUN"]
    nr = all_results.get("near", {})
    lr = all_results.get("long", {})

    print("=" * 95)
    print("VERDICT (proof-of-concept, revision-direction design)")
    print("=" * 95)
    print(f"""
  The revision-direction design pools 7 native survey measures across
  {len(revisions)} round-to-round revision pairs (2011-2026), classified into two
  horizon classes:

  NEAR-TERM (blocks {''.join(b['id'] for b in BLOCKS if block_meta.get(b['id'], {}).get('horizon_class') == 'NEAR-TERM')}):
    {len(near_rev)} survey pairs, {len(near_surv)} surviving F_t join ({100*len(near_surv)/len(near_rev):.0f}% survival).
    This is the PRIMARY test — pace/level revisions within ~1 year of the survey,
    conceptually comparable to a high-frequency news index.""")
    if "rho_dft" in nr:
        print(f"    At N={nr['n_nonzero']}, Spearman rho={nr['rho_dft']:+.2f} (p={nr['p_dft']:.3f}).", end="")
        if "agree_rate" in nr and not np.isnan(nr.get("agree_rate", np.nan)):
            print(f" Sign agreement: {nr['agree_rate']:.0%} (binom p={nr['binom_p']:.3f}).", end="")
        if "p_cl" in nr:
            print(f"\n    Clustered OLS: p={nr['p_cl']:.3f}.", end="")
        print("\n    The test is underpowered at this N.")
    else:
        print("    Insufficient N for tests.")

    print(f"""
  LONG-RUN (blocks {''.join(b['id'] for b in BLOCKS if block_meta.get(b['id'], {}).get('horizon_class') == 'LONG-RUN')}):
    {len(long_rev)} survey pairs, {len(long_surv)} surviving F_t join ({100*len(long_surv)/len(long_rev):.0f}% survival).
    Terminal/settle-point beliefs at multi-year horizons — a separate belief object
    that a daily news index should not be expected to track the same way.""")
    if lr.get("constant_sign"):
        print(f"    At N={lr['n_nonzero']}, all surviving revisions are in the same direction —")
        print(f"    Spearman undefined (constant input). Sign-agreement not meaningful.")
    elif "rho_dft" in lr:
        print(f"    At N={lr['n_nonzero']}, Spearman rho={lr['rho_dft']:+.2f} (p={lr['p_dft']:.3f}).")
    else:
        print("    Insufficient N for tests.")

    near_lost = len(near_rev) - len(near_surv)
    long_lost = len(long_rev) - len(long_surv)
    total_lost = near_lost + long_lost
    print(f"""
  ATTRITION SUMMARY:
    Near-term: {near_lost} of {len(near_rev)} pairs lost to news sparsity ({100*near_lost/len(near_rev):.0f}%).
    Long-run:  {long_lost} of {len(long_rev)} pairs lost ({100*long_lost/len(long_rev):.0f}%).
    Total:     {total_lost} of {len(revisions)} pairs lost ({100*total_lost/len(revisions):.0f}%).

  A denser full-text corpus (e.g. full NYT/WSJ via Factiva/DNA) would recover
  up to {near_lost} additional near-term observations — the class that matters for
  validating F_t against high-frequency belief revision. This is necessary
  infrastructure; it is not sufficient (denser news raises N but does not by
  itself imply the test will be significant).
""")


def print_block_details(revisions, horizon_info, block_meta):
    """Print per-block revision details for transparency."""
    print("=" * 95)
    print("BLOCK DETAILS: horizon selection and revision signs")
    print("=" * 95)
    for blk in BLOCKS:
        bid = blk["id"]
        brev = revisions[revisions["block_id"] == bid]
        hz = horizon_info.get(bid, "N/A")
        hc = block_meta.get(bid, {}).get("horizon_class", "?")
        n_zeros = (brev["revision_sign"] == 0).sum()
        print(f"\n  Block {bid} ({blk['label']}, {blk['regime']}) [{hc}]")
        print(f"    Fixed horizon: {hz}")
        print(f"    Revision pairs: {len(brev)} (of which {n_zeros} exact zeros)")
        if len(brev) > 0:
            for _, r in brev.iterrows():
                arrow = "↑" if r["revision_sign"] > 0 else ("↓" if r["revision_sign"] < 0 else "→")
                print(f"      {str(r['prev_date'].date())} -> {str(r['survey_date'].date())}: "
                      f"p50 {r['prev_pctl50']:.0f} -> {r['pctl50']:.0f} (d={r['delta_pctl50']:+.0f}) {arrow}")


def main():
    print("Loading data...")
    survey, ft = load_data()

    print("Building revision signs per block...")
    revisions, horizon_info, block_meta = build_revision_signs(survey)
    print(f"  Total revision pairs: {len(revisions)}")
    n_zeros = (revisions["revision_sign"] == 0).sum()
    print(f"  Exact zeros: {n_zeros}")

    print_horizon_classification(block_meta)
    print_block_details(revisions, horizon_info, block_meta)

    print("\nAligning with F_t and computing dF_t...")
    merged, survived = align_ft(revisions, ft)
    print(f"  Merged (all): {len(merged)}")
    print(f"  Survived (n_relevant >= {MIN_N_RELEVANT} both months): {len(survived)}")

    attrition_table(revisions, merged, survived, block_meta)

    all_results = run_tests(survived)

    fig_path = make_figure(survived, revisions, all_results, block_meta)

    print_verdict(revisions, survived, all_results, block_meta)


if __name__ == "__main__":
    main()
