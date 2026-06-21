"""Lead-lag analysis: does news F_t lead the SMA expected pace of balance-sheet change?

Primary test: FIRST-DIFFERENCED cross-correlation (trend-robust).
Secondary: level cross-correlation (labelled trend-vulnerable).
Revision-prediction: does lagged dF_t predict dPace (SMA revision)?

Sign convention printed next to every number:
  "positive lag k = F_t leads pace by k survey rounds"
  i.e. lag=+2 means F_t from 2 rounds ago is paired with today's pace.

Uses v1 classifications by default. Set USE_V2=True after v2 is approved.
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
# Config
# ---------------------------------------------------------------------------
USE_V2 = False                   # switch to True after v2 classifications are approved
MIN_ARTICLES_PER_MONTH = 5
MAX_LAG = 5                      # max lead/lag in survey rounds

# Healy-style setup (mirrors visualize.py)
COLORS = {
    "f_t": "#264653",
    "sma_pace": "#e76f51",
    "sig_band": "#e9ecef",
    "annotation": "#6c757d",
    "light_grid": "#e9ecef",
    "bar_pos": "#2a9d8f",
    "bar_neg": "#e76f51",
    "bar_insig": "#adb5bd",
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
# Load data
# ---------------------------------------------------------------------------
def load_f_t(con):
    """Monthly F_t from llm_classifications (v1 or v2)."""
    table = "llm_classifications_v2" if USE_V2 else "llm_classifications"

    if USE_V2:
        # v2: relevance-gated, ensemble-weighted
        df = con.execute(f"""
            SELECT
                strftime(g.seendate, '%Y-%m') AS period,
                SUM(CASE WHEN c.direction = 'increase' THEN 1 ELSE 0 END) AS n_increase,
                SUM(CASE WHEN c.direction = 'decrease' THEN 1 ELSE 0 END) AS n_decrease,
                SUM(CASE WHEN c.direction = 'uncertain' THEN 1 ELSE 0 END) AS n_uncertain,
                SUM(CASE WHEN c.direction = 'not_relevant' THEN 1 ELSE 0 END) AS n_not_relevant,
                COUNT(*) AS n_total
            FROM {table} c
            JOIN gdelt_articles g ON c.url = g.url
            GROUP BY period
            ORDER BY period
        """).fetchdf()
        df["n_relevant"] = df["n_increase"] + df["n_decrease"] + df["n_uncertain"]
        df["n_directional"] = df["n_increase"] + df["n_decrease"]
        df["f_statistic"] = np.where(
            df["n_directional"] > 0,
            (df["n_increase"] - df["n_decrease"]) / df["n_directional"],
            np.nan,
        )
    else:
        # v1: original classifications
        df = con.execute("""
            SELECT period, n_increase, n_decrease, n_uncertain, n_total, f_statistic
            FROM llm_expectations
            WHERE f_statistic IS NOT NULL
            ORDER BY period
        """).fetchdf()

    df = df[df["n_total"] >= MIN_ARTICLES_PER_MONTH].copy()
    df["date"] = pd.to_datetime(df["period"] + "-15")
    return df


def load_sma_pace(con):
    """Expected ~1yr-ahead pace: % change in total holdings from current quarter
    to the forecast date ~4 quarters out.

    Horizon choice: we pick the 5th row per vintage (rn=5) when sorted by forecast_date.
    Each vintage starts with the current/nearest quarter, so rn=5 is ~4 quarters ahead
    (~1 year). This captures the expected pace of balance-sheet change over the next year,
    which is the object relevant to Brunnermeier's optimal-pace question.
    """
    df = con.execute("""
        WITH ranked AS (
            SELECT vintage, CAST(vintage_date AS DATE) AS vd,
                   CAST(forecast_date AS DATE) AS fd,
                   total_holdings_eur,
                   ROW_NUMBER() OVER (PARTITION BY vintage ORDER BY forecast_date) AS rn
            FROM sma_expectations
            WHERE measure = 'MEDIAN'
        )
        SELECT
            a.vintage, a.vd AS vintage_date,
            a.total_holdings_eur AS current_val,
            b.total_holdings_eur AS yr1_val,
            b.fd AS yr1_date,
            (b.total_holdings_eur - a.total_holdings_eur)
                / a.total_holdings_eur * 100 AS pace_1yr
        FROM ranked a
        JOIN ranked b ON a.vintage = b.vintage AND b.rn = 5
        WHERE a.rn = 1
        ORDER BY a.vd
    """).fetchdf()

    df["vintage_date"] = pd.to_datetime(df["vintage_date"])
    df["vintage_month"] = df["vintage_date"].dt.strftime("%Y-%m")
    return df


# ---------------------------------------------------------------------------
# Align F_t to SMA survey rounds
# ---------------------------------------------------------------------------
def align_series(f_df, sma_df):
    """Match F_t month to SMA vintage month."""
    merged = pd.merge(
        sma_df, f_df[["period", "f_statistic", "n_total"]],
        left_on="vintage_month", right_on="period", how="inner",
    ).sort_values("vintage_date").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Cross-correlation at survey-round frequency
# ---------------------------------------------------------------------------
def cross_correlogram(x, y, max_lag, label_x="F_t", label_y="pace"):
    """Compute Spearman cross-correlation at integer lags.

    Convention: positive lag k means x LEADS y by k steps.
    lag=+2 pairs x[t-2] with y[t].
    """
    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x_shifted = x.iloc[:-lag].values
            y_aligned = y.iloc[lag:].values
        elif lag < 0:
            x_shifted = x.iloc[-lag:].values
            y_aligned = y.iloc[:lag].values
        else:
            x_shifted = x.values
            y_aligned = y.values

        n = len(x_shifted)
        if n < 5:
            continue

        rho, p = scipy_stats.spearmanr(x_shifted, y_aligned)
        results.append({"lag": lag, "rho": rho, "p": p, "n": n})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_timeseries(merged, output_dir, suffix=""):
    """F_t and sign-aligned SMA pace on one axis so phase offset is visible."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(merged["vintage_date"], merged["f_statistic"],
             color=COLORS["f_t"], linewidth=1.8, label="F_t (LLM balance statistic)",
             marker="o", markersize=4)
    ax1.set_ylabel("F_t", color=COLORS["f_t"])
    ax1.tick_params(axis="y", labelcolor=COLORS["f_t"])
    _add_light_grid(ax1)

    ax2 = ax1.twinx()
    ax2.plot(merged["vintage_date"], merged["pace_1yr"],
             color=COLORS["sma_pace"], linewidth=1.8, linestyle="--",
             label="SMA expected 1yr pace (%)", marker="s", markersize=4)
    ax2.set_ylabel("SMA expected 1yr pace (%)", color=COLORS["sma_pace"])
    ax2.tick_params(axis="y", labelcolor=COLORS["sma_pace"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COLORS["sma_pace"])

    ax2.axhline(0, color=COLORS["sma_pace"], linewidth=0.5, alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", frameon=False, fontsize=9)

    ax1.set_title("F_t and SMA expected balance-sheet pace", loc="left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    path = os.path.join(output_dir, f"fig9_leadlag_timeseries{suffix}.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_correlogram(ccf_levels, ccf_diffs, output_dir, suffix=""):
    """Side-by-side: level correlogram (trend-vulnerable) and differenced (robust)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Significance band (approximate 95% for Spearman under null)
    for ax, ccf, title in [
        (ax1, ccf_levels, "Levels (trend-vulnerable)"),
        (ax2, ccf_diffs, "First differences (trend-robust)"),
    ]:
        n_eff = ccf["n"].median()
        sig_band = 1.96 / np.sqrt(n_eff)

        ax.axhspan(-sig_band, sig_band, color=COLORS["sig_band"], alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.5)

        bar_colors = []
        for _, row in ccf.iterrows():
            if abs(row["rho"]) > sig_band and row["p"] < 0.05:
                bar_colors.append(COLORS["bar_pos"] if row["rho"] > 0 else COLORS["bar_neg"])
            else:
                bar_colors.append(COLORS["bar_insig"])

        ax.bar(ccf["lag"], ccf["rho"], color=bar_colors, width=0.7, edgecolor="white",
               linewidth=0.5)
        ax.set_xlabel("Lag (positive = F_t leads)")
        ax.set_title(title, loc="left")
        ax.set_xticks(range(-MAX_LAG, MAX_LAG + 1))
        _add_light_grid(ax)

    ax1.set_ylabel("Spearman ρ")

    fig.suptitle("Cross-correlogram: F_t vs SMA expected pace", fontweight="bold",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, f"fig10_correlogram{suffix}.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_leadlag(db_path=DUCKDB_PATH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    con = duckdb.connect(db_path, read_only=True)

    version = "v2" if USE_V2 else "v1"
    suffix = f"_{version}"

    print("=" * 70)
    print(f"LEAD-LAG ANALYSIS: F_t vs SMA expected pace (using {version} classifications)")
    print("=" * 70)

    # --- Load ---
    f_df = load_f_t(con)
    sma_df = load_sma_pace(con)
    con.close()

    merged = align_series(f_df, sma_df)
    N = len(merged)
    print(f"\nAligned observations: N = {N}")
    print(f"Period: {merged['vintage_date'].min():%Y-%m} to {merged['vintage_date'].max():%Y-%m}")
    print(f"F_t range: [{merged['f_statistic'].min():.2f}, {merged['f_statistic'].max():.2f}]")
    print(f"Pace range: [{merged['pace_1yr'].min():.1f}%, {merged['pace_1yr'].max():.1f}%]")

    if N < 8:
        print("\nToo few observations. Aborting.")
        return

    # --- First differences ---
    merged["dF"] = merged["f_statistic"].diff()
    merged["dPace"] = merged["pace_1yr"].diff()
    diff_df = merged.dropna(subset=["dF", "dPace"]).reset_index(drop=True)
    N_diff = len(diff_df)

    # =====================================================================
    # LEVELS cross-correlogram (trend-vulnerable)
    # =====================================================================
    print(f"\n{'='*50}")
    print("LEVEL CROSS-CORRELOGRAM (⚠ trend-vulnerable)")
    print(f"{'='*50}")
    print("Convention: positive lag k = F_t leads pace by k survey rounds")
    print()

    ccf_levels = cross_correlogram(
        merged["f_statistic"], merged["pace_1yr"], MAX_LAG,
    )
    for _, row in ccf_levels.iterrows():
        lag_int = int(row["lag"])
        sig = "***" if row["p"] < 0.01 else ("**" if row["p"] < 0.05 else ("*" if row["p"] < 0.10 else ""))
        sign_label = "F_t leads" if lag_int > 0 else ("contemporaneous" if lag_int == 0 else "pace leads")
        print(f"  lag={lag_int:+d} ({sign_label:>16s}): ρ={row['rho']:+.3f}, "
              f"p={row['p']:.3f} {sig:3s}  N={row['n']:.0f}")

    # =====================================================================
    # DIFFERENCED cross-correlogram (primary test)
    # =====================================================================
    print(f"\n{'='*50}")
    print("DIFFERENCED CROSS-CORRELOGRAM (trend-robust, primary test)")
    print(f"{'='*50}")
    print("Convention: positive lag k = dF_t leads dPace by k survey rounds")
    print()

    ccf_diffs = cross_correlogram(diff_df["dF"], diff_df["dPace"], MAX_LAG)
    for _, row in ccf_diffs.iterrows():
        lag_int = int(row["lag"])
        sig = "***" if row["p"] < 0.01 else ("**" if row["p"] < 0.05 else ("*" if row["p"] < 0.10 else ""))
        sign_label = "dF leads" if lag_int > 0 else ("contemporaneous" if lag_int == 0 else "dPace leads")
        print(f"  lag={lag_int:+d} ({sign_label:>16s}): ρ={row['rho']:+.3f}, "
              f"p={row['p']:.3f} {sig:3s}  N={row['n']:.0f}")

    # =====================================================================
    # REVISION-PREDICTION TEST
    # =====================================================================
    print(f"\n{'='*50}")
    print("REVISION-PREDICTION: does lagged dF_t predict dPace (survey revision)?")
    print(f"{'='*50}")
    print("dPace = change in SMA 1yr-ahead expected pace between consecutive vintages")
    print("dF_t  = change in F_t between corresponding months")
    print()

    for lag in [0, 1, 2]:
        if lag == 0:
            x, y = diff_df["dF"].values, diff_df["dPace"].values
        else:
            x = diff_df["dF"].iloc[:-lag].values
            y = diff_df["dPace"].iloc[lag:].values

        n = len(x)
        if n < 5:
            print(f"  lag={lag}: insufficient data (N={n})")
            continue

        rho, p = scipy_stats.spearmanr(x, y)
        r_p, p_p = scipy_stats.pearsonr(x, y)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        print(f"  lag={lag} (dF_t leads dPace by {lag} rounds): "
              f"Spearman ρ={rho:+.3f}, p={p:.3f} {sig}  "
              f"Pearson r={r_p:+.3f}, p={p_p:.3f}  N={n}")

    # =====================================================================
    # DETRENDING ROBUSTNESS
    # =====================================================================
    print(f"\n{'='*50}")
    print("DETRENDING ROBUSTNESS: do level results survive differencing?")
    print(f"{'='*50}")

    level_sig = ccf_levels[ccf_levels["p"] < 0.05]
    diff_sig = ccf_diffs[ccf_diffs["p"] < 0.05]

    level_sig_lags = set(level_sig["lag"])
    diff_sig_lags = set(diff_sig["lag"])

    survivors = level_sig_lags & diff_sig_lags
    lost = level_sig_lags - diff_sig_lags
    new = diff_sig_lags - level_sig_lags

    print(f"  Significant at p<0.05 in levels: lags {sorted(level_sig_lags) if level_sig_lags else 'none'}")
    print(f"  Significant at p<0.05 in diffs:  lags {sorted(diff_sig_lags) if diff_sig_lags else 'none'}")
    print(f"  Survive differencing:            lags {sorted(survivors) if survivors else 'none'}")
    if lost:
        print(f"  Lost after differencing:         lags {sorted(lost)} ← likely shared trend, not causal")
    if new:
        print(f"  New in differences:              lags {sorted(new)}")

    # =====================================================================
    # FIGURES
    # =====================================================================
    print()
    plot_timeseries(merged, OUTPUT_DIR, suffix)
    plot_correlogram(ccf_levels, ccf_diffs, OUTPUT_DIR, suffix)

    # =====================================================================
    # VERDICT
    # =====================================================================
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    has_diff_lead = any(
        row["lag"] > 0 and row["p"] < 0.05
        for _, row in ccf_diffs.iterrows()
    )
    has_level_lead = any(
        row["lag"] > 0 and row["p"] < 0.05
        for _, row in ccf_levels.iterrows()
    )

    if has_diff_lead:
        lead_lags = ccf_diffs[(ccf_diffs["lag"] > 0) & (ccf_diffs["p"] < 0.05)]
        best = lead_lags.loc[lead_lags["p"].idxmin()]
        print(f"F_t LEADS the SMA pace in first differences at lag {int(best['lag'])} "
              f"(ρ={best['rho']:+.3f}, p={best['p']:.3f}).")
        print(f"This survives detrending → real lead, not shared trend.")
    elif has_level_lead:
        print("F_t appears to lead in LEVELS but NOT in first differences.")
        print("The level result is likely a shared downward trend (both F_t and")
        print("pace track the QT regime), not genuine news-leads-survey causation.")
    else:
        print("No significant lead of F_t over SMA pace in either levels or differences.")

    print(f"\n⚠  Fragility: N={N} survey rounds, N_diff={N_diff} after differencing.")
    print(f"   Results should be interpreted with caution at this sample size.")
    print("=" * 70)


if __name__ == "__main__":
    run_leadlag()
