import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="ECB Balance Sheet Expectations Pipeline")
    parser.add_argument("steps", nargs="*", default=["all"],
                        help="Pipeline steps to run: schema, sma, gdelt, ecb, classify, aggregate, compare, visualize, all")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Limit number of articles to classify (for testing)")
    args = parser.parse_args()

    steps = [s.lower() for s in args.steps]
    run_all = "all" in steps

    if run_all or "schema" in steps:
        print("\n" + "=" * 60)
        print("STEP 1: Creating DuckDB schema")
        print("=" * 60)
        from schema import create_schema
        create_schema()

    if run_all or "sma" in steps:
        print("\n" + "=" * 60)
        print("STEP 2: Collecting ECB SMA survey data")
        print("=" * 60)
        from collect_sma import collect_sma
        collect_sma()

    if run_all or "gdelt" in steps:
        print("\n" + "=" * 60)
        print("STEP 3: Collecting GDELT news articles")
        print("=" * 60)
        from collect_gdelt import collect_gdelt
        collect_gdelt()

    if run_all or "ecb" in steps:
        print("\n" + "=" * 60)
        print("STEP 4: Collecting ECB balance sheet actuals")
        print("=" * 60)
        from collect_ecb_bs import collect_ecb_balance_sheet
        collect_ecb_balance_sheet()

    if run_all or "classify" in steps:
        print("\n" + "=" * 60)
        print("STEP 5: Classifying headlines with Claude")
        print("=" * 60)
        from process_headlines import process_unclassified
        process_unclassified(max_articles=args.max_articles)

    if run_all or "aggregate" in steps:
        print("\n" + "=" * 60)
        print("STEP 6: Computing balance statistic")
        print("=" * 60)
        from aggregate import compute_monthly_f_statistic
        compute_monthly_f_statistic()

    if run_all or "compare" in steps:
        print("\n" + "=" * 60)
        print("STEP 7: Comparing LLM vs survey expectations")
        print("=" * 60)
        from compare import compare
        compare()

    if run_all or "visualize" in steps:
        print("\n" + "=" * 60)
        print("STEP 8: Generating visualizations")
        print("=" * 60)
        from visualize import visualize
        visualize()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
