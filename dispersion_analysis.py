"""Dispersion analysis: does news-implied disagreement track SMA cross-forecaster dispersion?

Tests whether the cross-sectional dispersion of LLM headline classifications
(a proxy for belief heterogeneity in the news) co-moves with the IQR of
professional forecasters in the ECB Survey of Monetary Analysts.

An honest null is a valid and useful outcome — the LLM dispersion measure may
capture extraction/outlet noise rather than genuine belief heterogeneity.
"""

import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config import DUCKDB_PATH, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------
USE_CONFIDENCE_WEIGHTING = True
INCLUDE_UNCERTAIN = True        # if False, uncertain articles excluded from dispersion
MAGNITUDE_SCALING = False       # 75% of magnitudes are "unspecified" — adds noise
MIN_ARTICLES_PER_MONTH = 5
SMA_HORIZON_MONTHS = 12         # look-ahead for SMA IQR (nearest quarter ≈ 0)

MAGNITUDE_MAP = {"small": 0.33, "moderate": 0.66, "large": 1.0, "unspecified": 1.0}
DIRECTION_MAP = {"increase": 1.0, "decrease": -1.0, "uncertain": 0.0}

# ---------------------------------------------------------------------------
# Healy-style plot setup (mirrors visualize.py)
# ---------------------------------------------------------------------------
COLORS = {
    "increase": "#2a9d8f",
    "decrease": "#e76f51",
    "uncertain": "#adb5bd",
    "ecb_assets": "#264653",
    "sma": "#e9c46a",
    "light_grid": "#e9ecef",
    "annotation": "#6c757d",
    "dispersion": "#7209b7",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def _add_light_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    if axis in ("y", "both"):
        ax.yaxis.grid(True, color=COLORS["light_grid"], linewidth=0.5)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, color=COLORS["light_grid"], linewidth=0.5)


# ---------------------------------------------------------------------------
# LLM dispersion
# ---------------------------------------------------------------------------
def compute_llm_dispersion(con):
    rows = con.execute("""
        SELECT
            strftime(g.seendate, '%Y-%m') AS month,
            c.direction,
            c.confidence,
            c.magnitude
        FROM llm_classifications c
        JOIN gdelt_articles g ON c.url = g.url
        WHERE g.seendate IS NOT NULL
    """).fetchdf()

    rows["signal"] = rows["direction"].map(DIRECTION_MAP)
    if MAGNITUDE_SCALING:
        rows["mag_scale"] = rows["magnitude"].map(MAGNITUDE_MAP).fillna(1.0)
        rows["signal"] = rows["signal"] * rows["mag_scale"]

    if not INCLUDE_UNCERTAIN:
        rows = rows[rows["direction"] != "uncertain"].copy()

    weight_col = "confidence" if USE_CONFIDENCE_WEIGHTING else None

    results = []
    for month, group in rows.groupby("month"):
        n = len(group)
        if n < MIN_ARTICLES_PER_MONTH:
            continue

        signals = group["signal"].values
        weights = group["confidence"].values if weight_col else np.ones(n)
        weights = np.where(np.isnan(weights) | (weights <= 0), 1e-6, weights)

        w_sum = weights.sum()
        w_mean = np.average(signals, weights=weights)
        w_var = np.average((signals - w_mean) ** 2, weights=weights)
        w_std = np.sqrt(w_var)

        shares = group["direction"].value_counts(normalize=True)
        probs = shares.values
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        results.append({
            "month": month,
            "n": n,
            "mean_signal": w_mean,
            "dispersion": w_std,
            "entropy": entropy,
        })

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["month"] + "-15")
    return df


# ---------------------------------------------------------------------------
# SMA dispersion
# ---------------------------------------------------------------------------
def compute_sma_dispersion(con):
    """SMA provides only aggregated percentiles (P25/MEDIAN/P75), not individual
    forecaster responses. IQR at ~1-year-ahead horizon is the best available
    cross-forecaster dispersion proxy."""

    horizon_interval = f"{SMA_HORIZON_MONTHS} MONTH"

    df = con.execute(f"""
        WITH target_horizon AS (
            SELECT vintage, vintage_date,
                   MIN(forecast_date) FILTER (
                       WHERE forecast_date >= vintage_date + INTERVAL '{horizon_interval}'
                   ) AS fc_date
            FROM sma_expectations
            WHERE measure = 'MEDIAN'
            GROUP BY vintage, vintage_date
        ),
        pivoted AS (
            SELECT
                t.vintage,
                CAST(t.vintage_date AS DATE) AS vintage_date,
                t.fc_date,
                MAX(CASE WHEN e.measure = 'P25' THEN e.total_holdings_eur END) AS p25,
                MAX(CASE WHEN e.measure = 'MEDIAN' THEN e.total_holdings_eur END) AS median_val,
                MAX(CASE WHEN e.measure = 'P75' THEN e.total_holdings_eur END) AS p75
            FROM target_horizon t
            JOIN sma_expectations e
                ON e.vintage = t.vintage AND e.forecast_date = t.fc_date
            WHERE e.measure IN ('P25', 'MEDIAN', 'P75')
            GROUP BY t.vintage, t.vintage_date, t.fc_date
        )
        SELECT vintage, vintage_date, fc_date,
               p75 - p25 AS iqr,
               median_val,
               (p75 - p25) / median_val * 100 AS iqr_pct
        FROM pivoted
        WHERE p25 IS NOT NULL AND p75 IS NOT NULL AND median_val > 0
        ORDER BY vintage_date
    """).fetchdf()

    df["vintage_date"] = pd.to_datetime(df["vintage_date"])
    df["vintage_month"] = df["vintage_date"].dt.strftime("%Y-%m")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_dispersion_timeseries(llm_df, sma_df, output_dir):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(llm_df["date"], llm_df["dispersion"],
             color=COLORS["dispersion"], linewidth=1.5, alpha=0.8,
             label="LLM dispersion (weighted std)")
    ax1.set_ylabel("LLM dispersion", color=COLORS["dispersion"])
    ax1.tick_params(axis="y", labelcolor=COLORS["dispersion"])
    _add_light_grid(ax1)

    ax2 = ax1.twinx()
    ax2.scatter(sma_df["vintage_date"], sma_df["iqr"],
                color=COLORS["sma"], edgecolors="#b8860b", s=50,
                zorder=5, linewidths=0.5, label="SMA IQR (1yr ahead)")
    ax2.set_ylabel("SMA IQR (EUR billion)", color="#b8860b")
    ax2.tick_params(axis="y", labelcolor="#b8860b")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#b8860b")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", frameon=False, fontsize=9)

    ax1.set_title("Belief disagreement: LLM headline dispersion vs SMA forecaster IQR",
                   loc="left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    path = os.path.join(output_dir, "fig7_dispersion_timeseries.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_dispersion_scatter(merged, output_dir):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(merged["dispersion"], merged["iqr"],
               color=COLORS["ecb_assets"], s=55, alpha=0.7,
               edgecolors="white", linewidths=0.5)

    r_pearson, p_pearson = scipy_stats.pearsonr(merged["dispersion"], merged["iqr"])
    r_spearman, p_spearman = scipy_stats.spearmanr(merged["dispersion"], merged["iqr"])

    z = np.polyfit(merged["dispersion"], merged["iqr"], 1)
    x_line = np.linspace(merged["dispersion"].min(), merged["dispersion"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line),
            color=COLORS["decrease"], linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_xlabel("LLM dispersion (weighted std)")
    ax.set_ylabel("SMA IQR (EUR billion)")
    ax.set_title(
        f"LLM vs SMA disagreement (r={r_pearson:.2f}, p={p_pearson:.2f}, N={len(merged)})",
        loc="left")
    _add_light_grid(ax, axis="both")

    ax.text(0.97, 0.03,
            f"Pearson: r={r_pearson:.3f}, p={p_pearson:.3f}\n"
            f"Spearman: ρ={r_spearman:.3f}, p={p_spearman:.3f}",
            transform=ax.transAxes, fontsize=8, color=COLORS["annotation"],
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["light_grid"]))

    path = os.path.join(output_dir, "fig8_dispersion_scatter.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_dispersion_analysis(db_path=DUCKDB_PATH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    con = duckdb.connect(db_path, read_only=True)

    print("=" * 60)
    print("DISPERSION ANALYSIS: LLM disagreement vs SMA forecaster IQR")
    print("=" * 60)

    # --- LLM ---
    llm = compute_llm_dispersion(con)
    total_months = len(llm)
    print(f"\nLLM dispersion: {total_months} months (after dropping <{MIN_ARTICLES_PER_MONTH} articles)")
    print(f"  Confidence weighting: {USE_CONFIDENCE_WEIGHTING}")
    print(f"  Include uncertain: {INCLUDE_UNCERTAIN}")
    print(f"  Magnitude scaling: {MAGNITUDE_SCALING}")

    # --- SMA ---
    sma = compute_sma_dispersion(con)
    print(f"\nSMA dispersion proxy: IQR (P75-P25) at {SMA_HORIZON_MONTHS}-month-ahead horizon")
    print(f"  Vintages: {len(sma)} (Dec 2021 – Dec 2025)")
    print(f"  NOTE: SMA provides only aggregated percentiles, not individual responses.")
    print(f"  NOTE: Cannot extend to 2019 — pre-June-2021 results were never published.")

    con.close()

    # --- Align ---
    # Match SMA vintage month to LLM month (concurrent information environment)
    merged = pd.merge(llm, sma, left_on="month", right_on="vintage_month", how="inner")
    n_overlap = len(merged)

    dropped_thin = len(sma) - n_overlap
    print(f"\nOverlapping observations: {n_overlap}")
    if dropped_thin > 0:
        missing = set(sma["vintage_month"]) - set(llm["month"])
        print(f"  SMA vintages with no LLM match (thin/missing month): {sorted(missing)}")

    if n_overlap < 3:
        print("\nToo few overlapping observations for meaningful correlation. Aborting.")
        return

    # --- Correlations ---
    r_pearson, p_pearson = scipy_stats.pearsonr(merged["dispersion"], merged["iqr"])
    r_spearman, p_spearman = scipy_stats.spearmanr(merged["dispersion"], merged["iqr"])

    print(f"\n{'='*40}")
    print(f"RESULTS (N={n_overlap})")
    print(f"{'='*40}")
    print(f"  Pearson:  r = {r_pearson:+.4f},  p = {p_pearson:.4f}")
    print(f"  Spearman: ρ = {r_spearman:+.4f},  p = {p_spearman:.4f}")

    # --- Figures ---
    print()
    plot_dispersion_timeseries(llm, sma, OUTPUT_DIR)
    plot_dispersion_scatter(merged, OUTPUT_DIR)

    # --- Verdict ---
    print(f"\n{'='*60}")
    if p_spearman < 0.05 and p_pearson < 0.05:
        print("VERDICT: The series co-move — news-implied disagreement may carry")
        print("genuine belief-heterogeneity signal, though N is small.")
    elif p_spearman < 0.10 or p_pearson < 0.10:
        print("VERDICT: Weak/marginal evidence of co-movement. With N=%d," % n_overlap)
        print("this is suggestive but not conclusive — could be noise.")
    else:
        print("VERDICT: No significant co-movement detected (p > 0.10).")
        print("LLM headline dispersion likely reflects extraction/outlet noise")
        print("rather than genuine cross-forecaster belief disagreement.")
    print("=" * 60)


if __name__ == "__main__":
    run_dispersion_analysis()
