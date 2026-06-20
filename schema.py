import duckdb
from config import DUCKDB_PATH


def create_schema(db_path=DUCKDB_PATH):
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS sma_raw (
            survey_prov   VARCHAR,
            vintage       VARCHAR,
            vintage_date  VARCHAR,
            measure       VARCHAR,
            question_no   VARCHAR,
            series_key    VARCHAR,
            item          VARCHAR,
            area          VARCHAR,
            frequency     VARCHAR,
            horizon       VARCHAR,
            units         VARCHAR,
            time_stamp    VARCHAR,
            date_str      VARCHAR,
            date_descr    VARCHAR,
            category      VARCHAR,
            bin_descr     VARCHAR,
            value         VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS sma_expectations (
            vintage           VARCHAR NOT NULL,
            vintage_date      DATE NOT NULL,
            forecast_date     DATE NOT NULL,
            measure           VARCHAR NOT NULL,
            app_holdings_eur  DOUBLE,
            pepp_holdings_eur DOUBLE,
            total_holdings_eur DOUBLE,
            PRIMARY KEY (vintage, forecast_date, measure)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS gdelt_articles (
            url            VARCHAR PRIMARY KEY,
            title          VARCHAR NOT NULL,
            seendate       TIMESTAMP NOT NULL,
            domain         VARCHAR,
            language       VARCHAR,
            sourcecountry  VARCHAR,
            query_keyword  VARCHAR,
            collected_at   TIMESTAMP DEFAULT current_timestamp
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS llm_classifications (
            url           VARCHAR PRIMARY KEY,
            direction     VARCHAR NOT NULL,
            confidence    DOUBLE,
            magnitude     VARCHAR,
            explanation   VARCHAR,
            model_id      VARCHAR NOT NULL,
            processed_at  TIMESTAMP DEFAULT current_timestamp
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS llm_expectations (
            period        VARCHAR PRIMARY KEY,
            n_increase    INTEGER,
            n_decrease    INTEGER,
            n_uncertain   INTEGER,
            n_total       INTEGER,
            f_statistic   DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS ecb_balance_sheet (
            observation_date DATE PRIMARY KEY,
            total_assets_eur DOUBLE,
            collected_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)

    con.close()
    print(f"Schema created in {db_path}")


if __name__ == "__main__":
    create_schema()
