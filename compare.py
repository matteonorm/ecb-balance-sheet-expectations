import os

import duckdb
import pandas as pd

from config import DUCKDB_PATH, OUTPUT_DIR


def load_data(db_path=DUCKDB_PATH):
    con = duckdb.connect(db_path, read_only=True)

    llm_df = con.execute("""
        SELECT period, n_total, f_statistic
        FROM llm_expectations
        WHERE f_statistic IS NOT NULL
        ORDER BY period
    """).fetchdf()

    sma_df = con.execute("""
        SELECT
            vintage,
            vintage_date,
            forecast_date,
            measure,
            app_holdings_eur,
            pepp_holdings_eur,
            total_holdings_eur
        FROM sma_expectations
        WHERE measure = 'MEDIAN'
        ORDER BY vintage_date, forecast_date
    """).fetchdf()

    ecb_df = con.execute("""
        SELECT observation_date, total_assets_eur
        FROM ecb_balance_sheet
        ORDER BY observation_date
    """).fetchdf()

    con.close()
    return llm_df, sma_df, ecb_df


def compute_sma_change(sma_df):
    """For each survey vintage, compute the expected change in total holdings
    relative to the nearest past quarter."""
    if sma_df.empty:
        return pd.DataFrame()

    sma_df = sma_df.copy()
    sma_df["vintage_date"] = pd.to_datetime(sma_df["vintage_date"])
    sma_df["forecast_date"] = pd.to_datetime(sma_df["forecast_date"])
    sma_df["vintage_month"] = sma_df["vintage_date"].dt.to_period("M").astype(str)

    nearest = []
    for vintage, group in sma_df.groupby("vintage"):
        group = group.sort_values("forecast_date")
        if len(group) < 2:
            continue
        current = group.iloc[0]["total_holdings_eur"]
        next_q = group.iloc[1]["total_holdings_eur"]
        if current > 0:
            pct_change = (next_q - current) / current * 100
        else:
            pct_change = 0
        nearest.append({
            "vintage": vintage,
            "vintage_month": group.iloc[0]["vintage_month"],
            "current_holdings": current,
            "next_q_holdings": next_q,
            "expected_change_pct": pct_change,
            "expected_direction": "increase" if pct_change > 0 else "decrease",
        })

    return pd.DataFrame(nearest)


def compare(db_path=DUCKDB_PATH):
    llm_df, sma_df, ecb_df = load_data(db_path)

    print(f"LLM expectations: {len(llm_df)} months")
    print(f"SMA expectations: {len(sma_df)} rows")
    print(f"ECB balance sheet: {len(ecb_df)} observations")

    if llm_df.empty:
        print("No LLM data to compare.")
        return

    sma_changes = compute_sma_change(sma_df)
    if sma_changes.empty:
        print("No SMA data to compare. Showing LLM stats only.")
        print(llm_df.describe())
        return

    merged = pd.merge(
        llm_df, sma_changes,
        left_on="period", right_on="vintage_month",
        how="inner",
    )

    print(f"\nOverlapping months for comparison: {len(merged)}")

    if len(merged) < 3:
        print("Too few overlapping observations for correlation analysis.")
        print("\nLLM F_t summary:")
        print(llm_df[["period", "n_total", "f_statistic"]].to_string(index=False))
        print("\nSMA expected changes:")
        print(sma_changes[["vintage_month", "current_holdings", "expected_change_pct"]].to_string(index=False))
    else:
        from scipy import stats as scipy_stats
        pearson_r, pearson_p = scipy_stats.pearsonr(
            merged["f_statistic"], merged["expected_change_pct"]
        )
        spearman_r, spearman_p = scipy_stats.spearmanr(
            merged["f_statistic"], merged["expected_change_pct"]
        )
        print(f"\nPearson correlation:  r={pearson_r:.4f}, p={pearson_p:.4f}")
        print(f"Spearman correlation: r={spearman_r:.4f}, p={spearman_p:.4f}")

        results = pd.DataFrame([{
            "metric": "Pearson r",
            "value": pearson_r,
            "p_value": pearson_p,
        }, {
            "metric": "Spearman r",
            "value": spearman_r,
            "p_value": spearman_p,
        }, {
            "metric": "N_observations",
            "value": len(merged),
            "p_value": None,
        }])

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "comparison_stats.csv")
        results.to_csv(out_path, index=False)
        print(f"\nSaved comparison stats to {out_path}")

        merged_out = os.path.join(OUTPUT_DIR, "merged_comparison.csv")
        merged.to_csv(merged_out, index=False)
        print(f"Saved merged data to {merged_out}")


if __name__ == "__main__":
    compare()
