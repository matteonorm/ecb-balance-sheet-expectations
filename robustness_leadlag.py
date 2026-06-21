"""Robustness checks on the differenced lead-lag finding.

Main result under test: dF_{t-1} predicts dPace_t (Spearman rho=+0.437, p=0.023, N=27).
This is a ONE-SIDED pre-specified test: sign (+), direction (F leads), and lag (1 round)
were fixed before looking at the differenced correlogram. It is NOT the product of
scanning across lags or signs.

Tests:
  1. Scatter of dF_{t-1} vs dPace_t with date labels → visual influential-obs check
  2. Drop-one jackknife → does any single obs flip significance?
  3. HAC/Newey-West regression → robust t-stat for autocorrelated pace
  4. Construction sensitivity → variants of F_t (conf-weighted, magnitude, uncertain)
"""

import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config import DUCKDB_PATH, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Style (mirrors visualize.py)
# ---------------------------------------------------------------------------
COLORS = {
    "ecb_assets": "#264653",
    "decrease": "#e76f51",
    "increase": "#2a9d8f",
    "light_grid": "#e9ecef",
    "annotation": "#6c757d",
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
# Load and align (reproduces leadlag_analysis.py logic)
# ---------------------------------------------------------------------------
def load_aligned(con):
    """Load v1 F_t and SMA 1yr-ahead pace, align at survey frequency, difference."""
    f_df = con.execute("""
        SELECT period, f_statistic, n_total
        FROM llm_expectations
        WHERE f_statistic IS NOT NULL AND n_total >= 5
        ORDER BY period
    """).fetchdf()

    sma_df = con.execute("""
        WITH ranked AS (
            SELECT vintage, CAST(vintage_date AS DATE) AS vd,
                   total_holdings_eur,
                   ROW_NUMBER() OVER (PARTITION BY vintage ORDER BY forecast_date) AS rn
            FROM sma_expectations WHERE measure = 'MEDIAN'
        )
        SELECT a.vintage, a.vd AS vintage_date,
               a.total_holdings_eur AS current_val,
               b.total_holdings_eur AS yr1_val,
               (b.total_holdings_eur - a.total_holdings_eur) / a.total_holdings_eur * 100 AS pace_1yr
        FROM ranked a
        JOIN ranked b ON a.vintage = b.vintage AND b.rn = 5
        WHERE a.rn = 1 ORDER BY a.vd
    """).fetchdf()
    sma_df["vintage_date"] = pd.to_datetime(sma_df["vintage_date"])
    sma_df["vintage_month"] = sma_df["vintage_date"].dt.strftime("%Y-%m")

    merged = pd.merge(sma_df, f_df, left_on="vintage_month", right_on="period",
                       how="inner").sort_values("vintage_date").reset_index(drop=True)
    merged["dF"] = merged["f_statistic"].diff()
    merged["dPace"] = merged["pace_1yr"].diff()
    return merged


def build_diff_pairs(merged):
    """Build the (dF_{t-1}, dPace_t) pairs for the lag-1 test."""
    diff = merged.dropna(subset=["dF", "dPace"]).reset_index(drop=True)
    pairs = pd.DataFrame({
        "date": diff["vintage_date"].iloc[1:].values,
        "vintage": diff["vintage"].iloc[1:].values,
        "dF_lag1": diff["dF"].iloc[:-1].values,
        "dPace": diff["dPace"].iloc[1:].values,
    }).reset_index(drop=True)
    return pairs


# ---------------------------------------------------------------------------
# F_t variants for sensitivity
# ---------------------------------------------------------------------------
def compute_f_variants(con):
    """Build F_t under different construction choices."""
    raw = con.execute("""
        SELECT strftime(g.seendate, '%Y-%m') AS period,
               c.direction, c.confidence, c.magnitude
        FROM llm_classifications c
        JOIN gdelt_articles g ON c.url = g.url
        WHERE g.seendate IS NOT NULL
    """).fetchdf()

    dir_map = {"increase": 1.0, "decrease": -1.0, "uncertain": 0.0}
    mag_map = {"small": 0.33, "moderate": 0.66, "large": 1.0, "unspecified": 1.0}
    raw["signal"] = raw["direction"].map(dir_map)
    raw["mag_scale"] = raw["magnitude"].map(mag_map).fillna(1.0)

    variants = {}

    # Baseline: simple count F_t (same as v1)
    def count_f(df):
        n_inc = (df["direction"] == "increase").sum()
        n_dec = (df["direction"] == "decrease").sum()
        denom = n_inc + n_dec
        return (n_inc - n_dec) / denom if denom > 0 else np.nan

    # Variant 1: confidence-weighted F_t
    def conf_weighted_f(df):
        inc_w = df.loc[df["direction"] == "increase", "confidence"].sum()
        dec_w = df.loc[df["direction"] == "decrease", "confidence"].sum()
        denom = inc_w + dec_w
        return (inc_w - dec_w) / denom if denom > 0 else np.nan

    # Variant 2: magnitude-scaled F_t
    def mag_scaled_f(df):
        inc_w = (df.loc[df["direction"] == "increase", "mag_scale"]).sum()
        dec_w = (df.loc[df["direction"] == "decrease", "mag_scale"]).sum()
        denom = inc_w + dec_w
        return (inc_w - dec_w) / denom if denom > 0 else np.nan

    # Variant 3: exclude uncertain headlines entirely
    def no_uncertain_f(df):
        df2 = df[df["direction"] != "uncertain"]
        return count_f(df2)

    # Variant 4: confidence-weighted + exclude uncertain
    def conf_no_unc_f(df):
        df2 = df[df["direction"] != "uncertain"]
        return conf_weighted_f(df2)

    variant_fns = {
        "baseline (v1)": count_f,
        "conf-weighted": conf_weighted_f,
        "mag-scaled": mag_scaled_f,
        "no uncertain": no_uncertain_f,
        "conf + no unc": conf_no_unc_f,
    }

    for name, fn in variant_fns.items():
        monthly = raw.groupby("period").apply(
            lambda g: pd.Series({"f_statistic": fn(g), "n_total": len(g)}),
            include_groups=False,
        ).reset_index()
        monthly = monthly[monthly["n_total"] >= 5].copy()
        variants[name] = monthly

    return variants


# ---------------------------------------------------------------------------
# Newey-West HAC regression
# ---------------------------------------------------------------------------
def newey_west_regression(y, x, max_lags=None):
    """OLS of y on x with Newey-West HAC standard errors.
    Returns beta, se_hac, t_stat, p_value."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta

    if max_lags is None:
        max_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        max_lags = max(1, min(max_lags, n // 3))

    # Meat: sum of autocovariance-weighted outer products
    S = np.zeros((2, 2))
    for lag in range(max_lags + 1):
        weight = 1.0 if lag == 0 else 1.0 - lag / (max_lags + 1)
        for t in range(lag, n):
            outer = np.outer(X[t] * resid[t], X[t - lag] * resid[t - lag])
            if lag == 0:
                S += weight * outer
            else:
                S += weight * (outer + outer.T)

    # Bread
    XtX_inv = np.linalg.inv(X.T @ X)
    V_hac = n * XtX_inv @ S @ XtX_inv

    se_hac = np.sqrt(np.diag(V_hac))
    t_stat = beta[1] / se_hac[1]
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - 2))

    return {
        "beta": beta[1],
        "se_hac": se_hac[1],
        "t_stat": t_stat,
        "p_value": p_value,
        "nw_lags": max_lags,
        "n": n,
        "r_squared": 1 - np.sum(resid**2) / np.sum((y - y.mean())**2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_robustness(db_path=DUCKDB_PATH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    con = duckdb.connect(db_path, read_only=True)

    merged = load_aligned(con)
    pairs = build_diff_pairs(merged)
    N = len(pairs)

    print("=" * 70)
    print("ROBUSTNESS CHECKS: dF_{t-1} → dPace_t (lag +1 differenced result)")
    print("=" * 70)
    print(f"\nN = {N} observation pairs")
    print()
    print("NOTE: sign (+), direction (F leads), and lag (1 survey round) were")
    print("fixed before the differenced correlogram was run. This is a ONE-SIDED")
    print("pre-specified test, not a correlogram scan.")

    # Reproduce baseline
    rho_base, p_base = scipy_stats.spearmanr(pairs["dF_lag1"], pairs["dPace"])
    print(f"\nBaseline: Spearman ρ = {rho_base:+.4f}, p = {p_base:.4f}, N = {N}")

    # =====================================================================
    # 1. SCATTER with date labels
    # =====================================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pairs["dF_lag1"], pairs["dPace"],
               color=COLORS["ecb_assets"], s=55, alpha=0.7,
               edgecolors="white", linewidths=0.5)

    for _, row in pairs.iterrows():
        label = pd.Timestamp(row["date"]).strftime("%b%y")
        ax.annotate(label, (row["dF_lag1"], row["dPace"]),
                    fontsize=6.5, color=COLORS["annotation"],
                    textcoords="offset points", xytext=(4, 4))

    z = np.polyfit(pairs["dF_lag1"], pairs["dPace"], 1)
    x_line = np.linspace(pairs["dF_lag1"].min(), pairs["dF_lag1"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line),
            color=COLORS["decrease"], linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_xlabel(r"$\Delta F_{t-1}$ (change in F_t, lagged one round)")
    ax.set_ylabel(r"$\Delta Pace_t$ (SMA pace revision this round)")
    ax.set_title(f"Revision prediction: dF(t-1) -> dPace(t) (rho={rho_base:+.3f}, p={p_base:.3f})",
                 loc="left")
    _add_light_grid(ax, axis="both")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.3)

    path = os.path.join(OUTPUT_DIR, "fig11_robustness_scatter.png")
    fig.savefig(path)
    plt.close()
    print(f"\nSaved: {path}")

    # =====================================================================
    # 2. DROP-ONE JACKKNIFE
    # =====================================================================
    print(f"\n{'='*50}")
    print("DROP-ONE JACKKNIFE")
    print(f"{'='*50}")

    jack_results = []
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        rho_i, p_i = scipy_stats.spearmanr(
            pairs["dF_lag1"].values[mask], pairs["dPace"].values[mask],
        )
        jack_results.append({
            "dropped_idx": i,
            "dropped_date": pd.Timestamp(pairs["date"].iloc[i]).strftime("%Y-%m"),
            "dropped_vintage": pairs["vintage"].iloc[i],
            "rho": rho_i,
            "p": p_i,
        })

    jdf = pd.DataFrame(jack_results)
    rho_min_row = jdf.loc[jdf["rho"].idxmin()]
    rho_max_row = jdf.loc[jdf["rho"].idxmax()]
    most_influential = jdf.loc[(jdf["rho"] - rho_base).abs().idxmax()]
    any_flip = (jdf["p"] >= 0.05).any()

    print(f"  ρ range: [{jdf['rho'].min():+.4f}, {jdf['rho'].max():+.4f}]")
    print(f"  Baseline ρ: {rho_base:+.4f}")
    print(f"  Lowest ρ:   {rho_min_row['rho']:+.4f} (drop {rho_min_row['dropped_date']})")
    print(f"  Highest ρ:  {rho_max_row['rho']:+.4f} (drop {rho_max_row['dropped_date']})")
    print(f"  Most influential: drop {most_influential['dropped_date']} "
          f"→ ρ={most_influential['rho']:+.4f} (Δρ={most_influential['rho'] - rho_base:+.4f})")
    print()

    n_insig = (jdf["p"] >= 0.05).sum()
    print(f"  Observations where drop flips p ≥ 0.05: {n_insig}/{N}")
    if any_flip:
        flippers = jdf[jdf["p"] >= 0.05].sort_values("p", ascending=False)
        for _, row in flippers.iterrows():
            print(f"    Drop {row['dropped_date']}: ρ={row['rho']:+.4f}, p={row['p']:.4f}")

    if not any_flip:
        print("  ✓ No single observation flips significance — result is not driven by outliers.")
    else:
        print(f"  ⚠ {n_insig} observation(s) can flip significance — check the scatter for leverage.")

    # =====================================================================
    # 3. NEWEY-WEST HAC REGRESSION
    # =====================================================================
    print(f"\n{'='*50}")
    print("NEWEY-WEST HAC REGRESSION: dPace_t = α + β·dF_{t-1} + ε")
    print(f"{'='*50}")

    nw = newey_west_regression(pairs["dPace"].values, pairs["dF_lag1"].values)
    print(f"  β       = {nw['beta']:+.4f}")
    print(f"  SE(HAC) = {nw['se_hac']:.4f}  (Newey-West, {nw['nw_lags']} lags)")
    print(f"  t-stat  = {nw['t_stat']:+.3f}")
    print(f"  p-value = {nw['p_value']:.4f}")
    print(f"  R²      = {nw['r_squared']:.4f}")
    print(f"  N       = {nw['n']}")

    if nw["p_value"] < 0.05:
        print("  ✓ Significant under HAC — autocorrelation does not kill the result.")
    else:
        print("  ⚠ Insignificant under HAC — autocorrelation inflated the naive test.")

    # =====================================================================
    # 4. CONSTRUCTION SENSITIVITY
    # =====================================================================
    print(f"\n{'='*50}")
    print("CONSTRUCTION SENSITIVITY: lag-+1 ρ under F_t variants")
    print(f"{'='*50}")

    sma_df = con.execute("""
        WITH ranked AS (
            SELECT vintage, CAST(vintage_date AS DATE) AS vd,
                   total_holdings_eur,
                   ROW_NUMBER() OVER (PARTITION BY vintage ORDER BY forecast_date) AS rn
            FROM sma_expectations WHERE measure = 'MEDIAN'
        )
        SELECT a.vintage, a.vd AS vintage_date,
               (b.total_holdings_eur - a.total_holdings_eur) / a.total_holdings_eur * 100 AS pace_1yr
        FROM ranked a
        JOIN ranked b ON a.vintage = b.vintage AND b.rn = 5
        WHERE a.rn = 1 ORDER BY a.vd
    """).fetchdf()
    sma_df["vintage_date"] = pd.to_datetime(sma_df["vintage_date"])
    sma_df["vintage_month"] = sma_df["vintage_date"].dt.strftime("%Y-%m")
    sma_df["dPace"] = sma_df["pace_1yr"].diff()

    variants = compute_f_variants(con)
    con.close()

    print(f"  {'Variant':<20s}  {'ρ':>7s}  {'p':>7s}  {'N':>4s}  sig?")
    print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*4}  ----")

    for name, fdf in variants.items():
        m = pd.merge(sma_df, fdf, left_on="vintage_month", right_on="period",
                      how="inner").sort_values("vintage_date").reset_index(drop=True)
        m["dF"] = m["f_statistic"].diff()
        d = m.dropna(subset=["dF", "dPace"]).reset_index(drop=True)

        if len(d) < 4:
            print(f"  {name:<20s}  {'---':>7s}  {'---':>7s}  {len(d):>4d}  insufficient")
            continue

        # Build lag-1 pairs
        x = d["dF"].iloc[:-1].values
        y = d["dPace"].iloc[1:].values
        rho_v, p_v = scipy_stats.spearmanr(x, y)
        sig = "***" if p_v < 0.01 else ("**" if p_v < 0.05 else ("*" if p_v < 0.10 else ""))
        print(f"  {name:<20s}  {rho_v:+.4f}  {p_v:.4f}  {len(x):>4d}  {sig}")

    # =====================================================================
    # VERDICT
    # =====================================================================
    print(f"\n{'='*70}")
    print("ROBUSTNESS VERDICT")
    print(f"{'='*70}")

    jack_ok = not any_flip
    hac_ok = nw["p_value"] < 0.05

    if jack_ok and hac_ok:
        print("The lag-+1 differenced result SURVIVES both the jackknife and HAC.")
        print("No single observation drives significance. Autocorrelation does not")
        print("inflate the test. The finding is bankable-with-caveats (N=27 remains small).")
    elif jack_ok and not hac_ok:
        print("The result survives the jackknife but NOT HAC standard errors.")
        print("Autocorrelation in the multi-horizon pace may inflate the naive Spearman.")
        print("Treat as suggestive, not confirmed.")
    elif not jack_ok and hac_ok:
        n_flip = n_insig
        print(f"HAC is fine, but {n_flip} observation(s) can flip the jackknife.")
        print("The result is fragile to individual observations — check the scatter")
        print("for high-leverage points.")
    else:
        print("The result fails BOTH the jackknife and HAC. It is fragile.")

    print("=" * 70)


if __name__ == "__main__":
    run_robustness()
