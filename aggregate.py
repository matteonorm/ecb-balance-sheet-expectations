import duckdb

from config import DUCKDB_PATH


def compute_monthly_f_statistic(db_path=DUCKDB_PATH):
    con = duckdb.connect(db_path)

    con.execute("DELETE FROM llm_expectations")

    con.execute("""
        INSERT INTO llm_expectations
        SELECT
            strftime(g.seendate, '%Y-%m') AS period,
            SUM(CASE WHEN c.direction = 'increase' THEN 1 ELSE 0 END) AS n_increase,
            SUM(CASE WHEN c.direction = 'decrease' THEN 1 ELSE 0 END) AS n_decrease,
            SUM(CASE WHEN c.direction = 'uncertain' THEN 1 ELSE 0 END) AS n_uncertain,
            COUNT(*) AS n_total,
            CASE
                WHEN SUM(CASE WHEN c.direction IN ('increase','decrease') THEN 1 ELSE 0 END) = 0
                THEN NULL
                ELSE (
                    SUM(CASE WHEN c.direction = 'increase' THEN 1.0 ELSE 0 END)
                    - SUM(CASE WHEN c.direction = 'decrease' THEN 1.0 ELSE 0 END)
                ) / (
                    SUM(CASE WHEN c.direction = 'increase' THEN 1.0 ELSE 0 END)
                    + SUM(CASE WHEN c.direction = 'decrease' THEN 1.0 ELSE 0 END)
                )
            END AS f_statistic
        FROM llm_classifications c
        JOIN gdelt_articles g ON c.url = g.url
        GROUP BY period
        ORDER BY period
    """)

    results = con.execute("""
        SELECT period, n_total, f_statistic
        FROM llm_expectations
        ORDER BY period
    """).fetchall()

    print(f"Computed F_t for {len(results)} months")
    for period, n, f in results:
        f_str = f"{f:.3f}" if f is not None else "N/A"
        print(f"  {period}: n={n}, F_t={f_str}")

    con.close()
    return len(results)


if __name__ == "__main__":
    compute_monthly_f_statistic()
