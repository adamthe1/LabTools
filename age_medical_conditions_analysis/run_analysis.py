#!/usr/bin/env python
"""Run the age-medical conditions analysis."""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
from body_system_loader.load_feature_df import load_body_system_df

OUTPUT_DIR = '/home/adamgab/PycharmProjects/LabTools/age_medical_conditions_analysis'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Disease groupings based on WHO framework
DISEASE_GROUPS = {
    'Cardiovascular': [
        'Hypertension', 'Ischemic Heart Disease', 'Atrial Fibrillation',
        'Heart valve disease', 'AV. Conduction Disorder', 'Myocarditis', 'Atherosclerotic'
    ],
    'Cancer': ['Breast Cancer', 'Melanoma', 'Lymphoma'],
    'Respiratory': ['Asthma', 'COPD'],
    'Metabolic/Endocrine': [
        'Diabetes', 'Prediabetes', 'Hyperlipidemia', 'Hypercholesterolaemia',
        'Obesity', 'Fatty Liver Disease', 'Hashimoto', 'Goiter', 'Hyperparathyroidism', 'G6PD'
    ],
    'Neurodegenerative/Mental': ['Depression', 'Anxiety', 'ADHD', 'PTSD', 'Insomnia', 'Migraine', 'Headache'],
    'Musculoskeletal': ['Osteoarthritis', 'Back Pain', 'Fibromyalgia', 'Fractures', 'Gout', 'Meniscus Tears'],
    'Gastrointestinal': ['IBD', 'IBS', 'Celiac', 'Peptic Ulcer Disease', 'Gallstone Disease', 'Haemorrhoids', 'Anal Fissure'],
    'Autoimmune/Inflammatory': ['Psoriasis', 'Vitiligo', 'Allergy', 'Atopic Dermatitis', 'Uveitis', 'FMF'],
    'Eye/Ear/Sensory': ['Glaucoma', 'Hearing loss', 'Tinnitus', 'Retinal detachment'],
    'Renal/Urological': ['Renal Stones', 'Urinary Tract Stones', 'Urinary tract infection']
}

def get_condition_to_group_mapping():
    mapping = {}
    for group, conditions in DISEASE_GROUPS.items():
        for condition in conditions:
            if condition not in mapping:
                mapping[condition] = group
    return mapping


def calculate_log_odds_regression(df, condition):
    """Calculate log odds ratio per year of age using logistic regression."""
    valid = df[[condition, 'age']].dropna()
    if len(valid) < 100 or valid[condition].sum() < 10:
        return {'coef': np.nan, 'se': np.nan, 'p_value': np.nan,
                'n': len(valid), 'n_positive': int(valid[condition].sum())}

    X = valid['age'].values.reshape(-1, 1)
    y = valid[condition].values
    try:
        X_const = sm.add_constant(X)
        model = sm.Logit(y, X_const).fit(disp=0, maxiter=100)
        return {
            'coef': model.params[1], 'se': model.bse[1], 'p_value': model.pvalues[1],
            'ci_lower': model.conf_int()[1, 0], 'ci_upper': model.conf_int()[1, 1],
            'n': len(valid), 'n_positive': int(valid[condition].sum()),
            'odds_ratio_per_year': np.exp(model.params[1]),
            'odds_ratio_per_decade': np.exp(model.params[1] * 10)
        }
    except Exception as e:
        return {'coef': np.nan, 'se': np.nan, 'p_value': np.nan,
                'n': len(valid), 'n_positive': int(valid[condition].sum()), 'error': str(e)}


def calculate_prevalence_by_age(df, condition, age_bins=None):
    """Calculate prevalence of a condition by age bin."""
    if age_bins is None:
        age_bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]

    df = df.copy()
    df['age_bin'] = pd.cut(df['age'], bins=age_bins, right=False)

    prevalence = df.groupby('age_bin', observed=True).agg({condition: ['sum', 'count']})
    prevalence.columns = ['n_positive', 'n_total']
    prevalence['prevalence'] = prevalence['n_positive'] / prevalence['n_total']

    eps = 1e-10
    prevalence['odds'] = (prevalence['prevalence'] + eps) / (1 - prevalence['prevalence'] + eps)
    prevalence['log_odds'] = np.log(prevalence['odds'])
    prevalence['age_midpoint'] = [(interval.left + interval.right) / 2 for interval in prevalence.index]

    return prevalence.reset_index()


def main():
    print('=' * 60)
    print('AGE AND MEDICAL CONDITIONS ANALYSIS')
    print('HPP 10K Dataset')
    print('=' * 60)

    # Load data
    print('\n1. Loading data...')
    age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
    mc_df = load_body_system_df('medical_conditions')
    df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')

    # Filter valid ages
    df = df[df['age'] >= 18]
    print(f'   Loaded {len(df)} records (age >= 18)')
    print(f'   Age range: {df["age"].min():.1f} - {df["age"].max():.1f} years')
    print(f'   Medical conditions: {len(mc_df.columns)}')

    # Analyze all conditions
    print('\n2. Analyzing all conditions...')
    condition_cols = [c for c in df.columns if c not in ['age', 'gender']]
    mapping = get_condition_to_group_mapping()

    results = []
    for condition in condition_cols:
        result = calculate_log_odds_regression(df, condition)
        result['condition'] = condition
        result['disease_group'] = mapping.get(condition, 'Other')
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('coef', ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'condition_analysis_results.csv'), index=False)
    print(f'   Saved: condition_analysis_results.csv')

    # Analyze disease groups
    print('\n3. Analyzing disease groups...')
    all_conditions = [c for c in df.columns if c not in ['age', 'gender']]
    group_results = []

    for group in DISEASE_GROUPS.keys():
        group_conditions = [c for c in DISEASE_GROUPS[group] if c in all_conditions]
        if not group_conditions:
            continue

        df[f'has_{group}'] = df[group_conditions].max(axis=1)
        result = calculate_log_odds_regression(df, f'has_{group}')
        result['disease_group'] = group
        result['n_conditions'] = len(group_conditions)
        result['conditions'] = ', '.join(group_conditions)
        group_results.append(result)

    group_results_df = pd.DataFrame(group_results)
    group_results_df.to_csv(os.path.join(OUTPUT_DIR, 'group_analysis_results.csv'), index=False)
    print(f'   Saved: group_analysis_results.csv')

    # Generate figures
    print('\n4. Generating figures...')

    # Figure 1: Age coefficients for all conditions
    print('   - Age coefficients bar plot...')
    valid = results_df[(results_df['n_positive'] >= 50) & (results_df['coef'].notna())].copy()
    valid = valid.sort_values('coef', ascending=True)

    groups = valid['disease_group'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = dict(zip(groups, colors))

    fig, ax = plt.subplots(figsize=(14, max(10, len(valid) * 0.35)))
    y_pos = np.arange(len(valid))
    ax.barh(y_pos, valid['coef'], color=[color_map[g] for g in valid['disease_group']], alpha=0.8)
    ax.errorbar(valid['coef'], y_pos, xerr=1.96*valid['se'], fmt='none', color='black', alpha=0.5, capsize=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid['condition'], fontsize=9)
    ax.set_xlabel('Log Odds per Year of Age', fontsize=12)
    ax.set_title('Age-Disease Association (Log Odds Ratio per Year)', fontsize=14)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', alpha=0.3)
    legend_handles = [plt.Rectangle((0,0), 1, 1, color=color_map[g], alpha=0.8) for g in groups]
    ax.legend(legend_handles, groups, loc='lower right', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'age_coefficients_all_conditions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'   Saved: figures/age_coefficients_all_conditions.png')

    # Figure 2: Age coefficients by group
    print('   - Disease group coefficients...')
    valid_groups = group_results_df[group_results_df['coef'].notna()].copy()
    valid_groups = valid_groups.sort_values('coef', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(valid_groups)))
    y_pos = np.arange(len(valid_groups))
    ax.barh(y_pos, valid_groups['coef'], color=colors, alpha=0.8)
    ax.errorbar(valid_groups['coef'], y_pos, xerr=1.96*valid_groups['se'], fmt='none', color='black', alpha=0.5, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid_groups['disease_group'], fontsize=11)
    ax.set_xlabel('Log Odds per Year of Age', fontsize=12)
    ax.set_title('Age-Disease Group Association (Any Condition in Group)', fontsize=14)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', alpha=0.3)
    for idx, row in valid_groups.reset_index().iterrows():
        if not np.isnan(row.get('odds_ratio_per_decade', np.nan)):
            ax.annotate(f"OR/10y: {row['odds_ratio_per_decade']:.2f}", xy=(row['coef'] + 0.002, idx), fontsize=8, va='center')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'age_coefficients_by_group.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'   Saved: figures/age_coefficients_by_group.png')

    # Figure 3: Top conditions with prevalence curves
    print('   - Top age-associated conditions prevalence curves...')
    sig = results_df[(results_df['n_positive'] >= 50) & (results_df['coef'].notna()) & (results_df['p_value'] < 0.05)]
    top_positive = sig.nlargest(6, 'coef')['condition'].tolist()
    top_negative = sig.nsmallest(6, 'coef')['condition'].tolist()

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    for idx, condition in enumerate(top_positive):
        prev = calculate_prevalence_by_age(df, condition)
        ax = axes[0, idx]
        ax.fill_between(prev['age_midpoint'], 0, prev['prevalence'] * 100, alpha=0.3, color='red')
        ax.plot(prev['age_midpoint'], prev['prevalence'] * 100, marker='o', linewidth=2, markersize=6, color='red')
        coef = results_df[results_df['condition'] == condition]['coef'].values[0]
        ax.set_title(f'{condition}\n(coef={coef:.4f})', fontsize=10)
        ax.set_xlabel('Age', fontsize=9)
        ax.set_ylabel('Prevalence (%)', fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx, condition in enumerate(top_negative):
        prev = calculate_prevalence_by_age(df, condition)
        ax = axes[1, idx]
        ax.fill_between(prev['age_midpoint'], 0, prev['prevalence'] * 100, alpha=0.3, color='blue')
        ax.plot(prev['age_midpoint'], prev['prevalence'] * 100, marker='o', linewidth=2, markersize=6, color='blue')
        coef = results_df[results_df['condition'] == condition]['coef'].values[0]
        ax.set_title(f'{condition}\n(coef={coef:.4f})', fontsize=10)
        ax.set_xlabel('Age', fontsize=9)
        ax.set_ylabel('Prevalence (%)', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel('Prevalence (%)\n(Increase with Age)', fontsize=11)
    axes[1, 0].set_ylabel('Prevalence (%)\n(Decrease with Age)', fontsize=11)
    fig.suptitle('Top Age-Associated Medical Conditions', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'top_age_associated_conditions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'   Saved: figures/top_age_associated_conditions.png')

    # Figure 4: Prevalence by disease group
    print('   - Prevalence by disease group...')
    for group in DISEASE_GROUPS.keys():
        group_conditions = [c for c in DISEASE_GROUPS[group] if c in all_conditions]
        if not group_conditions:
            continue

        n_conditions = len(group_conditions)
        n_cols = min(3, n_conditions)
        n_rows = int(np.ceil(n_conditions / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = np.atleast_2d(axes)

        for idx, condition in enumerate(group_conditions):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            prev = calculate_prevalence_by_age(df, condition)
            ax.fill_between(prev['age_midpoint'], 0, prev['prevalence'] * 100, alpha=0.3)
            ax.plot(prev['age_midpoint'], prev['prevalence'] * 100, marker='o', linewidth=2, markersize=6)
            ax.set_title(condition, fontsize=11)
            ax.set_xlabel('Age (years)', fontsize=10)
            ax.set_ylabel('Prevalence (%)', fontsize=10)
            ax.grid(True, alpha=0.3)

        for idx in range(len(group_conditions), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle(f'{group} - Prevalence by Age', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f'prevalence_{group.replace("/", "_")}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'   Saved: figures/prevalence_*.png for each disease group')

    # Print summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)

    sig = results_df[(results_df['p_value'] < 0.05) & (results_df['n_positive'] >= 50)]
    print(f'\nSignificant conditions (p<0.05, n>=50): {len(sig)}')
    print(f'  - Increase with age: {(sig["coef"] > 0).sum()}')
    print(f'  - Decrease with age: {(sig["coef"] < 0).sum()}')

    print('\nTop 10 conditions INCREASING with age:')
    for _, row in sig.nlargest(10, 'coef').iterrows():
        print(f'  {row["condition"]}: coef={row["coef"]:.4f}, OR/decade={np.exp(row["coef"]*10):.2f}, n={row["n_positive"]}')

    print('\nTop 10 conditions DECREASING with age:')
    for _, row in sig.nsmallest(10, 'coef').iterrows():
        print(f'  {row["condition"]}: coef={row["coef"]:.4f}, OR/decade={np.exp(row["coef"]*10):.2f}, n={row["n_positive"]}')

    print('\nDisease Group Summary:')
    for _, row in group_results_df.sort_values('coef', ascending=False).iterrows():
        if not np.isnan(row['coef']):
            print(f'  {row["disease_group"]}: coef={row["coef"]:.4f}, OR/decade={np.exp(row["coef"]*10):.2f}')

    print('\n' + '=' * 60)
    print('ANALYSIS COMPLETE')
    print('=' * 60)

    return results_df, group_results_df


if __name__ == "__main__":
    results_df, group_results_df = main()
