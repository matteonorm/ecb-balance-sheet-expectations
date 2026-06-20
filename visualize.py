import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from config import DUCKDB_PATH, OUTPUT_DIR


def load_all_data(db_path=DUCKDB_PATH):
    con = duckdb.connect(db_path, read_only=True)

    llm_df = con.execute("""
        SELECT period, n_total, n_increase, n_decrease, n_uncertain, f_statistic
        FROM llm_expectations
        ORDER BY period
    """).fetchdf()

    sma_df = con.execute("""
        SELECT vintage, vintage_date, forecast_date, measure,
               app_holdings_eur, pepp_holdings_eur, total_holdings_eur
        FROM sma_expectations
        WHERE measure = 'MEDIAN'
        ORDER BY vintage_date, forecast_date
    """).fetchdf()

    ecb_df = con.execute("""
        SELECT observation_date, total_assets_eur
        FROM ecb_balance_sheet
        ORDER BY observation_date
    """).fetchdf()

    article_counts = con.execute("""
        SELECT strftime(seendate, '%Y-%m') AS month, COUNT(*) AS n
        FROM gdelt_articles
        GROUP BY month ORDER BY month
    """).fetchdf()

    con.close()
    return llm_df, sma_df, ecb_df, article_counts


def plot_dual_axis(llm_df, sma_df, ecb_df, output_dir):
    if llm_df.empty:
        print("No LLM data to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))

    llm_df = llm_df.copy()
    llm_df["date"] = pd.to_datetime(llm_df["period"] + "-15")

    ax1.plot(llm_df["date"], llm_df["f_statistic"], "b-o", markersize=3,
             label="LLM F_t (balance statistic)", linewidth=1.5)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("LLM Balance Statistic (F_t)", color="b")
    ax1.set_ylim(-1.1, 1.1)
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()

    if not ecb_df.empty:
        ecb_df = ecb_df.copy()
        ecb_df["observation_date"] = pd.to_datetime(ecb_df["observation_date"])
        ax2.plot(ecb_df["observation_date"], ecb_df["total_assets_eur"] / 1e6,
                 "r-", alpha=0.6, label="ECB Total Assets (EUR trillion)", linewidth=1)
        ax2.set_ylabel("ECB Total Assets (EUR trillion)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

    if not sma_df.empty:
        sma_near = sma_df.copy()
        sma_near["vintage_date"] = pd.to_datetime(sma_near["vintage_date"])
        sma_near["forecast_date"] = pd.to_datetime(sma_near["forecast_date"])
        first_forecasts = sma_near.groupby("vintage").first().reset_index()
        ax2.scatter(first_forecasts["vintage_date"],
                    first_forecasts["total_holdings_eur"] / 1e3,
                    color="green", marker="^", s=40, zorder=5,
                    label="SMA Median (next Q, EUR trillion)")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.title("ECB Balance Sheet: LLM-Generated vs Survey Expectations")
    plt.tight_layout()
    path = os.path.join(output_dir, "dual_axis_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_classification_distribution(llm_df, output_dir):
    if llm_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    llm_df = llm_df.copy()
    llm_df["date"] = pd.to_datetime(llm_df["period"] + "-15")

    ax.bar(llm_df["date"], llm_df["n_increase"], label="Increase",
           color="green", alpha=0.7, width=20)
    ax.bar(llm_df["date"], llm_df["n_decrease"], bottom=llm_df["n_increase"],
           label="Decrease", color="red", alpha=0.7, width=20)
    ax.bar(llm_df["date"], llm_df["n_uncertain"],
           bottom=llm_df["n_increase"] + llm_df["n_decrease"],
           label="Uncertain", color="gray", alpha=0.5, width=20)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    ax.set_ylabel("Number of Articles")
    ax.set_title("LLM Classification Distribution by Month")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "classification_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_article_coverage(article_counts, output_dir):
    if article_counts.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    article_counts = article_counts.copy()
    article_counts["date"] = pd.to_datetime(article_counts["month"] + "-15")

    ax.bar(article_counts["date"], article_counts["n"], color="steelblue",
           alpha=0.7, width=20)
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Min threshold (10)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    ax.set_ylabel("Article Count")
    ax.set_title("GDELT Article Coverage by Month")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "article_coverage.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def visualize(db_path=DUCKDB_PATH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    llm_df, sma_df, ecb_df, article_counts = load_all_data(db_path)

    print(f"Data loaded: {len(llm_df)} LLM months, {len(sma_df)} SMA rows, "
          f"{len(ecb_df)} ECB observations, {len(article_counts)} article months")

    plot_dual_axis(llm_df, sma_df, ecb_df, OUTPUT_DIR)
    plot_classification_distribution(llm_df, OUTPUT_DIR)
    plot_article_coverage(article_counts, OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    visualize()
