"""
Contemporaneous co-movement test: does F_t track survey expectations
in the SAME period? Bybee (2025) validation approach.

Reports three tests (N stated for each):
  A. LEVELS (Bybee-style): Spearman + Pearson, F_t vs survey level.
     Regime-confounded — QE/QT structure inflates raw correlation.
  B. FIRST DIFFERENCES: Spearman + Pearson, dF_t vs dSurvey, lag 0,
     HAC/Newey-West SEs. Strips regime/trend — the correct test.
  C. REGIME-DEMEANED LEVELS: subtract QE/QT regime means, correlate.
     Should agree with B.

Survey measure: median expected total assets (total_assets variable),
~12-month horizon (~365 days), averaged across panels (SPD/SMP/SME).
Single consistent measure — no splicing.

Usage:
    python contemporaneous_analysis.py
"""

import os
import numpy as np
import pandas as pd
import duckdb
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import DUCKDB_PATH, OUTPUT_DIR, FED_REGIMES

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

MIN_RELEVANT = 3
TARGET_HORIZON_DAYS = 365
HORIZON_WINDOW = (270, 450)
SURVEY_VARIABLE = 'total_assets'

REGIME_COLORS = {
    'pre_taper': '#e8e8e8',
    'taper_tantrum': '#ffd6d6',
    'reinvestment': '#e8e8e8',
    'qt1_runoff': '#d6e8ff',
    'qe_covid': '#d6ffd6',
    'qt2': '#d6e8ff',
}
REGIME_LABELS = {
    'pre_taper': 'Pre-Taper',
    'taper_tantrum': 'Taper',
    'reinvestment': 'Reinvest.',
    'qt1_runoff': 'QT1',
    'qe_covid': 'QE (COVID)',
    'qt2': 'QT2',
}


def assign_regime(period):
    for regime, (start, end) in FED_REGIMES.items():
        if start[:7] <= period <= end[:7]:
            return regime
    return "other"


def load_data(db_path=DUCKDB_PATH, min_relevant=MIN_RELEVANT,
              survey_variable=SURVEY_VARIABLE):
    con = duckdb.connect(db_path, read_only=True)

    ft = con.execute("""
        SELECT period, f_statistic AS f_t, n_relevant
        FROM llm_expectations
        WHERE f_statistic IS NOT NULL
        ORDER BY period
    """).fetchdf()

    survey_raw = con.execute(f"""
        SELECT strftime(survey_date, '%Y-%m') AS period,
               survey_date,
               horizon_date,
               (horizon_date - survey_date) AS days_ahead,
               pctl50
        FROM nyfed_survey_bs
        WHERE variable = '{survey_variable}'
          AND pctl50 IS NOT NULL
          AND horizon_date > survey_date
        ORDER BY survey_date, horizon_date
    """).fetchdf()

    con.close()

    if survey_raw.empty or ft.empty:
        print("No data. Run pipeline first.")
        return None, {}

    if len(survey_raw[survey_raw['days_ahead'].between(*HORIZON_WINDOW)]) > 0:
        survey_raw = survey_raw[survey_raw['days_ahead'].between(*HORIZON_WINDOW)]
        survey_raw['dist_to_target'] = (survey_raw['days_ahead'] - TARGET_HORIZON_DAYS).abs()
        best_horizon = (survey_raw
                        .sort_values('dist_to_target')
                        .groupby('period')
                        .first()
                        .reset_index())
        best_horizon_dates = best_horizon[['period', 'horizon_date']].rename(
            columns={'horizon_date': 'best_horizon'})
        survey_at_horizon = survey_raw.merge(best_horizon_dates, on='period')
        survey_at_horizon = survey_at_horizon[
            survey_at_horizon['horizon_date'] == survey_at_horizon['best_horizon']]
        survey = (survey_at_horizon
                  .groupby('period')
                  .agg(survey_level=('pctl50', 'mean'),
                       n_panels=('pctl50', 'count'),
                       days_ahead=('days_ahead', 'first'))
                  .reset_index())
    else:
        survey = (survey_raw
                  .groupby('period')
                  .agg(survey_level=('pctl50', 'mean'),
                       n_panels=('pctl50', 'count'),
                       days_ahead=('days_ahead', 'mean'))
                  .reset_index())

    n_ft_before = len(ft)
    ft_filtered = ft[ft['n_relevant'] >= min_relevant].copy()
    n_dropped = n_ft_before - len(ft_filtered)

    merged = pd.merge(ft_filtered, survey, on='period', how='inner')
    merged = merged.sort_values('period').reset_index(drop=True)
    merged['regime'] = merged['period'].apply(assign_regime)

    meta = {
        'variable': survey_variable,
        'horizon_target': f'~{TARGET_HORIZON_DAYS}d ({HORIZON_WINDOW[0]}-{HORIZON_WINDOW[1]}d window)',
        'n_ft_total': n_ft_before,
        'n_ft_after_filter': len(ft_filtered),
        'n_dropped_low_coverage': n_dropped,
        'min_relevant': min_relevant,
        'n_survey_months': len(survey),
        'n_merged': len(merged),
        'period_range': f"{merged['period'].min()} to {merged['period'].max()}" if len(merged) > 0 else "N/A",
    }

    return merged, meta


def hac_regression(x, y):
    if not HAS_STATSMODELS or len(x) < 6:
        return None
    X = add_constant(x)
    model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(1, len(x) // 4)})
    return {
        'beta': model.params[1],
        't_hac': model.tvalues[1],
        'p_hac': model.pvalues[1],
        'se_hac': model.bse[1],
        'r2': model.rsquared,
        'nw_lags': max(1, len(x) // 4),
    }


def test_levels(merged):
    x, y = merged['f_t'].values, merged['survey_level'].values
    n = len(x)
    rho_s, p_s = sp_stats.spearmanr(x, y)
    rho_p, p_p = sp_stats.pearsonr(x, y)
    hac = hac_regression(x, y)
    return {
        'test': 'A. Levels (regime-confounded)',
        'n': n,
        'spearman_rho': rho_s, 'spearman_p': p_s,
        'pearson_r': rho_p, 'pearson_p': p_p,
        'hac': hac,
    }


def test_first_differences(merged):
    df = merged.copy()
    df['d_ft'] = df['f_t'].diff()
    df['d_survey'] = df['survey_level'].diff()
    df = df.dropna(subset=['d_ft', 'd_survey'])
    n = len(df)
    if n < 4:
        return {'test': 'B. First differences', 'n': n, 'error': 'too few obs'}
    x, y = df['d_ft'].values, df['d_survey'].values
    rho_s, p_s = sp_stats.spearmanr(x, y)
    rho_p, p_p = sp_stats.pearsonr(x, y)
    hac = hac_regression(x, y)
    return {
        'test': 'B. First differences (regime-stripped)',
        'n': n,
        'spearman_rho': rho_s, 'spearman_p': p_s,
        'pearson_r': rho_p, 'pearson_p': p_p,
        'hac': hac,
        'd_ft': x, 'd_survey': y,
        'periods': df['period'].values,
    }


def test_regime_demeaned(merged):
    df = merged.copy()
    n_regimes = df['regime'].nunique()
    for col in ['f_t', 'survey_level']:
        regime_means = df.groupby('regime')[col].transform('mean')
        df[f'{col}_dm'] = df[col] - regime_means
    x, y = df['f_t_dm'].values, df['survey_level_dm'].values
    n = len(x)
    rho_s, p_s = sp_stats.spearmanr(x, y)
    rho_p, p_p = sp_stats.pearsonr(x, y)
    hac = hac_regression(x, y)
    uninformative = (n_regimes == 1)
    label = 'C. Regime-demeaned levels'
    if uninformative:
        label += f' (UNINFORMATIVE — all {n} obs in one regime: {df["regime"].iloc[0]})'
    return {
        'test': label,
        'n': n,
        'spearman_rho': rho_s, 'spearman_p': p_s,
        'pearson_r': rho_p, 'pearson_p': p_p,
        'hac': hac,
        'uninformative': uninformative,
    }


def print_result(r):
    print(f"\n  {r['test']}  (N = {r['n']})")
    if 'error' in r:
        print(f"    {r['error']}")
        return
    print(f"    Spearman rho = {r['spearman_rho']:+.3f}  (p = {r['spearman_p']:.4f})")
    print(f"    Pearson  r   = {r['pearson_r']:+.3f}  (p = {r['pearson_p']:.4f})")
    if r.get('hac'):
        h = r['hac']
        print(f"    OLS beta = {h['beta']:.2f}, t(HAC) = {h['t_hac']:.2f}, "
              f"p(HAC) = {h['p_hac']:.4f}, NW lags = {h['nw_lags']}")


def make_figure(merged, result_b, output_dir=OUTPUT_DIR):
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.linewidth': 0.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={'width_ratios': [1, 1.4]})

    # Left panel: scatter of dF_t vs dSurvey
    ax1 = axes[0]
    if 'd_ft' in result_b and 'd_survey' in result_b:
        dx, dy = result_b['d_ft'], result_b['d_survey']
        ax1.scatter(dx, dy, s=40, color='steelblue', alpha=0.7, edgecolors='white', linewidth=0.5)
        if len(dx) > 2:
            m, b = np.polyfit(dx, dy, 1)
            x_line = np.linspace(dx.min(), dx.max(), 50)
            ax1.plot(x_line, m * x_line + b, '--', color='darkred', linewidth=1.2)
        rho_s = result_b['spearman_rho']
        p_s = result_b['spearman_p']
        n = result_b['n']
        ax1.set_title(f'$\\Delta F_t$ vs $\\Delta$Survey\n'
                      f'Spearman $\\rho$ = {rho_s:.2f}, p = {p_s:.3f}, N = {n}')
    else:
        ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                 transform=ax1.transAxes)
    ax1.set_xlabel('$\\Delta F_t$ (news belief change)')
    ax1.set_ylabel('$\\Delta$ Survey (expected total assets change, $bn)')
    ax1.axhline(0, color='gray', linewidth=0.3)
    ax1.axvline(0, color='gray', linewidth=0.3)

    # Right panel: two series over time with regime shading
    ax2 = axes[1]
    dates = pd.to_datetime(merged['period'] + '-15')

    ax2.bar(dates, merged['f_t'], width=25, alpha=0.6, color='steelblue',
            label='$F_t$ (news)')
    ax2.set_ylabel('$F_t$', color='steelblue')
    ax2.set_ylim(-1.3, 1.3)
    ax2.axhline(0, color='black', linewidth=0.3)

    ax2r = ax2.twinx()
    ax2r.plot(dates, merged['survey_level'], 'o-', color='darkred',
              markersize=4, linewidth=1.5, label='Survey (total assets, $bn)')
    ax2r.set_ylabel('Expected total assets ($bn)', color='darkred')

    for regime, (start, end) in FED_REGIMES.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        color = REGIME_COLORS.get(regime, '#f0f0f0')
        ax2.axvspan(s, e, alpha=0.3, color=color, zorder=0)
        mid = s + (e - s) / 2
        if s >= dates.min() - pd.Timedelta(days=60):
            label = REGIME_LABELS.get(regime, regime)
            ax2.text(mid, 1.2, label, ha='center', va='bottom',
                     fontsize=7, color='gray', fontstyle='italic')

    date_pad = pd.Timedelta(days=30)
    ax2.set_xlim(dates.min() - date_pad, dates.max() + date_pad)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left',
               frameon=False, fontsize=8)
    ax2.set_title('$F_t$ and Survey Expectations Over Time')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'fig3_contemporaneous.png')
    plt.savefig(out)
    plt.close()
    print(f"\nSaved {out}")


def main(survey_variable=SURVEY_VARIABLE, min_relevant=MIN_RELEVANT):
    merged, meta = load_data(survey_variable=survey_variable,
                             min_relevant=min_relevant)
    if merged is None or len(merged) < 4:
        print("Insufficient data for contemporaneous analysis.")
        return

    print(f"\n{'='*70}")
    print("CONTEMPORANEOUS CO-MOVEMENT: F_t vs Survey (Bybee-style validation)")
    print(f"{'='*70}")
    print(f"\nSurvey measure: {meta['variable']} (median expected total Fed assets)")
    print(f"Horizon: {meta['horizon_target']}")
    print(f"F_t filter: n_relevant >= {meta['min_relevant']} "
          f"({meta['n_dropped_low_coverage']} months dropped)")
    print(f"Merged observations: N = {meta['n_merged']}")
    print(f"Period: {meta['period_range']}")

    print(f"\n--- Data summary ---")
    print(merged[['period', 'f_t', 'n_relevant', 'survey_level', 'regime']].to_string(index=False))

    result_a = test_levels(merged)
    result_b = test_first_differences(merged)
    result_c = test_regime_demeaned(merged)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print_result(result_a)
    print_result(result_b)
    print_result(result_c)

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    b_sig = ('spearman_p' in result_b and result_b['spearman_p'] < 0.10)
    b_pearson_sig = ('pearson_p' in result_b and result_b['pearson_p'] < 0.10)
    c_uninformative = result_c.get('uninformative', False)
    a_sig = (result_a['spearman_p'] < 0.10)

    if c_uninformative:
        print("\n  Test C is uninformative: all observations fall within a single regime")
        print("  (QT2), so regime-demeaning = level correlation. The binding test is B.")

    if b_sig:
        print("\n  First-differenced correlation (B) is significant (Spearman).")
        print("  Co-movement survives differencing — not purely regime-labeling.")
    elif b_pearson_sig:
        print(f"\n  First-differenced Spearman (p={result_b['spearman_p']:.3f}) is not significant,")
        print(f"  but Pearson (p={result_b['pearson_p']:.3f}) is marginally significant.")
        print("  Suggestive of linear co-movement, but not robust at conventional levels.")
    elif a_sig and not b_sig:
        if c_uninformative:
            print("\n  Level correlation (A) is significant, but differencing (B) kills it.")
            print("  Within this single-regime sample, F_t and the survey co-move in levels")
            print("  but not in period-to-period changes. Could be shared slow drift, not")
            print("  genuine news-tracking.")
        else:
            print("\n  Only the raw level correlation (A) holds — regime-labeling artifact.")
    else:
        print("\n  No significant co-movement detected at any level.")

    print(f"\n  N = {meta['n_merged']} ({meta['n_merged']-1} for first differences)."
          f" Results are suggestive, not definitive.")

    make_figure(merged, result_b)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', default=SURVEY_VARIABLE,
                        help='Survey variable name (default: total_assets)')
    parser.add_argument('--min-relevant', type=int, default=MIN_RELEVANT,
                        help='Minimum n_relevant for F_t (default: 3)')
    args = parser.parse_args()
    main(survey_variable=args.variable, min_relevant=args.min_relevant)
