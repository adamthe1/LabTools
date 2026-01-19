"""
Age and Medical Conditions Analysis - HPP 10K Dataset
======================================================

This module analyzes the relationship between age and log probability (log odds)
of medical conditions, based on the WHO framework of major disease categories.

Major Disease Groupings (based on WHO and epidemiological research):
1. Cardiovascular Diseases - Heart disease, hypertension, stroke-related
2. Cancers - Malignant neoplasms
3. Respiratory Diseases - Chronic respiratory conditions
4. Metabolic/Endocrine - Diabetes, thyroid, metabolic disorders
5. Neurodegenerative/Mental Health - Dementia-related, mental health
6. Musculoskeletal - Joint diseases, bone conditions
7. Gastrointestinal - Digestive system disorders
8. Autoimmune/Inflammatory - Autoimmune conditions

References:
- WHO Noncommunicable Diseases Fact Sheet (2023)
- Journal of Gerontology: What Is an Aging-Related Disease?
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')

from body_system_loader.load_feature_df import load_body_system_df, load_columns_as_df

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Disease groupings based on WHO framework and epidemiological research
DISEASE_GROUPS = {
    'Cardiovascular': [
        'Hypertension', 'Ischemic Heart Disease', 'Atrial Fibrillation',
        'Heart valve disease', 'AV. Conduction Disorder', 'Myocarditis',
        'Atherosclerotic'
    ],
    'Cancer': [
        'Breast Cancer', 'Melanoma', 'Lymphoma'
    ],
    'Respiratory': [
        'Asthma', 'COPD'
    ],
    'Metabolic/Endocrine': [
        'Diabetes', 'Prediabetes', 'Hyperlipidemia', 'Hypercholesterolaemia',
        'Obesity', 'Fatty Liver Disease', 'Hashimoto', 'Goiter',
        'Hyperparathyroidism', 'Thyroid Adenoma', 'G6PD'
    ],
    'Neurodegenerative/Mental': [
        'Depression', 'Anxiety', 'ADHD', 'PTSD', 'Insomnia',
        'Migraine', 'Headache'
    ],
    'Musculoskeletal': [
        'Osteoarthritis', 'Back Pain', 'Fibromyalgia', 'Fractures',
        'Gout', 'Meniscus Tears'
    ],
    'Gastrointestinal': [
        'IBD', 'IBS', 'Celiac', 'Peptic Ulcer Disease', 'Gallstone Disease',
        'Fatty Liver Disease', 'Haemorrhoids', 'Anal Fissure'
    ],
    'Autoimmune/Inflammatory': [
        'Psoriasis', 'Vitiligo', 'Hashimoto', 'Allergy', 'Atopic Dermatitis',
        'Uveitis', 'FMF'
    ],
    'Eye/Ear/Sensory': [
        'Glaucoma', 'Hearing loss', 'Tinnitus', 'Retinal detachment',
        'Ocular Hypertension'
    ],
    'Renal/Urological': [
        'Renal Stones', 'Urinary Tract Stones', 'Urinary tract infection'
    ]
}

# Flatten to get all conditions with their groups
def get_condition_to_group_mapping() -> Dict[str, str]:
    """Create mapping from condition to disease group."""
    mapping = {}
    for group, conditions in DISEASE_GROUPS.items():
        for condition in conditions:
            if condition not in mapping:  # First group takes precedence
                mapping[condition] = group
    return mapping


def load_age_and_conditions() -> pd.DataFrame:
    """Load age and all medical conditions data."""
    print("Loading age data...")
    age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])

    print("Loading medical conditions data...")
    mc_df = load_body_system_df('medical_conditions')

    # Merge on index
    df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')

    print(f"Loaded {len(df)} records with {len(mc_df.columns)} conditions")
    return df


def calculate_prevalence_by_age(df: pd.DataFrame, condition: str,
                                 age_bins: List[int] = None) -> pd.DataFrame:
    """Calculate prevalence of a condition by age bin."""
    if age_bins is None:
        age_bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]

    df = df.copy()
    df['age_bin'] = pd.cut(df['age'], bins=age_bins, right=False)

    # Calculate prevalence per bin
    prevalence = df.groupby('age_bin', observed=True).agg({
        condition: ['sum', 'count']
    })
    prevalence.columns = ['n_positive', 'n_total']
    prevalence['prevalence'] = prevalence['n_positive'] / prevalence['n_total']

    # Calculate log odds (add small epsilon to avoid log(0))
    eps = 1e-10
    prevalence['odds'] = (prevalence['prevalence'] + eps) / (1 - prevalence['prevalence'] + eps)
    prevalence['log_odds'] = np.log(prevalence['odds'])

    # Calculate 95% CI using Wilson score interval for proportion
    z = 1.96
    n = prevalence['n_total']
    p = prevalence['prevalence']
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator

    prevalence['ci_lower'] = center - margin
    prevalence['ci_upper'] = center + margin

    prevalence['age_midpoint'] = [(interval.left + interval.right) / 2
                                   for interval in prevalence.index]

    return prevalence.reset_index()


def calculate_log_odds_regression(df: pd.DataFrame, condition: str) -> Dict:
    """
    Calculate log odds ratio per year of age using logistic regression.
    Returns coefficient, p-value, and 95% CI.
    """
    from scipy.special import expit

    # Remove NaN
    valid = df[[condition, 'age']].dropna()
    if len(valid) < 100 or valid[condition].sum() < 10:
        return {'coef': np.nan, 'se': np.nan, 'p_value': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan, 'n': len(valid),
                'n_positive': valid[condition].sum()}

    X = valid['age'].values.reshape(-1, 1)
    y = valid[condition].values

    # Simple logistic regression via statsmodels
    try:
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        model = sm.Logit(y, X_const).fit(disp=0, maxiter=100)

        coef = model.params[1]  # Age coefficient
        se = model.bse[1]
        p_value = model.pvalues[1]
        ci_lower = model.conf_int()[1, 0]
        ci_upper = model.conf_int()[1, 1]

        return {
            'coef': coef,
            'se': se,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(valid),
            'n_positive': int(valid[condition].sum()),
            'odds_ratio_per_year': np.exp(coef),
            'odds_ratio_per_decade': np.exp(coef * 10)
        }
    except Exception as e:
        return {'coef': np.nan, 'se': np.nan, 'p_value': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan, 'n': len(valid),
                'n_positive': int(valid[condition].sum()), 'error': str(e)}


def analyze_all_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze age relationship for all conditions."""
    condition_cols = [c for c in df.columns if c not in ['age', 'gender']]

    results = []
    for condition in condition_cols:
        result = calculate_log_odds_regression(df, condition)
        result['condition'] = condition

        # Get disease group
        mapping = get_condition_to_group_mapping()
        result['disease_group'] = mapping.get(condition, 'Other')

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('coef', ascending=False)

    return results_df


def analyze_grouped_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze age relationship for grouped conditions."""
    mapping = get_condition_to_group_mapping()
    all_conditions = [c for c in df.columns if c not in ['age', 'gender']]

    group_results = []
    for group in DISEASE_GROUPS.keys():
        # Get conditions in this group that exist in our data
        group_conditions = [c for c in DISEASE_GROUPS[group] if c in all_conditions]

        if not group_conditions:
            continue

        # Create binary indicator: 1 if any condition in group is present
        df[f'has_{group}'] = df[group_conditions].max(axis=1)

        result = calculate_log_odds_regression(df, f'has_{group}')
        result['disease_group'] = group
        result['n_conditions'] = len(group_conditions)
        result['conditions'] = ', '.join(group_conditions)

        group_results.append(result)

    return pd.DataFrame(group_results)


def plot_prevalence_by_age(df: pd.DataFrame, condition: str,
                            ax: Optional[plt.Axes] = None,
                            show_log_odds: bool = True) -> plt.Axes:
    """Plot prevalence vs age for a condition."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    prev_data = calculate_prevalence_by_age(df, condition)

    if show_log_odds:
        ax.errorbar(prev_data['age_midpoint'], prev_data['log_odds'],
                   yerr=None, marker='o', linewidth=2, markersize=8)
        ax.set_ylabel('Log Odds', fontsize=12)
    else:
        ax.fill_between(prev_data['age_midpoint'],
                        prev_data['ci_lower'] * 100,
                        prev_data['ci_upper'] * 100,
                        alpha=0.3)
        ax.plot(prev_data['age_midpoint'], prev_data['prevalence'] * 100,
               marker='o', linewidth=2, markersize=8)
        ax.set_ylabel('Prevalence (%)', fontsize=12)

    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_title(f'{condition}', fontsize=14)
    ax.grid(True, alpha=0.3)

    return ax


def plot_grouped_prevalence(df: pd.DataFrame, group: str,
                            save: bool = True) -> plt.Figure:
    """Plot prevalence by age for all conditions in a disease group."""
    all_conditions = [c for c in df.columns if c not in ['age', 'gender']]
    group_conditions = [c for c in DISEASE_GROUPS.get(group, []) if c in all_conditions]

    if not group_conditions:
        print(f"No conditions found for group: {group}")
        return None

    n_conditions = len(group_conditions)
    n_cols = min(3, n_conditions)
    n_rows = int(np.ceil(n_conditions / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes)

    for idx, condition in enumerate(group_conditions):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        plot_prevalence_by_age(df, condition, ax=ax, show_log_odds=False)

    # Hide empty subplots
    for idx in range(len(group_conditions), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f'{group} - Prevalence by Age', fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        filepath = os.path.join(FIGURES_DIR, f'prevalence_{group.replace("/", "_")}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    return fig


def plot_age_coefficients(results_df: pd.DataFrame,
                          min_positives: int = 50,
                          save: bool = True) -> plt.Figure:
    """Plot age coefficients (log odds per year) for all conditions."""
    # Filter by minimum positives and valid coefficients
    valid = results_df[
        (results_df['n_positive'] >= min_positives) &
        (results_df['coef'].notna())
    ].copy()

    # Sort by coefficient
    valid = valid.sort_values('coef', ascending=True)

    # Color by disease group
    groups = valid['disease_group'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = dict(zip(groups, colors))

    fig, ax = plt.subplots(figsize=(12, max(8, len(valid) * 0.3)))

    y_pos = np.arange(len(valid))
    bars = ax.barh(y_pos, valid['coef'],
                   color=[color_map[g] for g in valid['disease_group']],
                   alpha=0.8)

    # Add error bars
    ax.errorbar(valid['coef'], y_pos,
                xerr=1.96*valid['se'], fmt='none',
                color='black', alpha=0.5, capsize=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid['condition'], fontsize=9)
    ax.set_xlabel('Log Odds per Year of Age', fontsize=12)
    ax.set_title('Age-Disease Association\n(Log Odds Ratio per Year)', fontsize=14)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', alpha=0.3)

    # Legend
    legend_handles = [plt.Rectangle((0,0), 1, 1, color=color_map[g], alpha=0.8)
                      for g in groups]
    ax.legend(legend_handles, groups, loc='lower right', fontsize=9)

    plt.tight_layout()

    if save:
        filepath = os.path.join(FIGURES_DIR, 'age_coefficients_all_conditions.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    return fig


def plot_group_coefficients(group_results: pd.DataFrame,
                            save: bool = True) -> plt.Figure:
    """Plot age coefficients for disease groups."""
    valid = group_results[group_results['coef'].notna()].copy()
    valid = valid.sort_values('coef', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(valid)))

    y_pos = np.arange(len(valid))
    bars = ax.barh(y_pos, valid['coef'], color=colors, alpha=0.8)

    # Add error bars
    ax.errorbar(valid['coef'], y_pos,
                xerr=1.96*valid['se'], fmt='none',
                color='black', alpha=0.5, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid['disease_group'], fontsize=11)
    ax.set_xlabel('Log Odds per Year of Age', fontsize=12)
    ax.set_title('Age-Disease Group Association\n(Any Condition in Group)', fontsize=14)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', alpha=0.3)

    # Add odds ratio annotations
    for idx, row in valid.reset_index().iterrows():
        if not np.isnan(row['odds_ratio_per_decade']):
            ax.annotate(f"OR/10y: {row['odds_ratio_per_decade']:.2f}",
                       xy=(row['coef'] + 0.001, idx),
                       fontsize=8, va='center')

    plt.tight_layout()

    if save:
        filepath = os.path.join(FIGURES_DIR, 'age_coefficients_by_group.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    return fig


def plot_top_age_associated_conditions(df: pd.DataFrame, results_df: pd.DataFrame,
                                        n_top: int = 6, save: bool = True) -> plt.Figure:
    """Plot prevalence curves for top age-associated conditions."""
    # Get top positive and negative associations
    valid = results_df[
        (results_df['n_positive'] >= 50) &
        (results_df['coef'].notna()) &
        (results_df['p_value'] < 0.05)
    ].copy()

    top_positive = valid.nlargest(n_top, 'coef')['condition'].tolist()
    top_negative = valid.nsmallest(n_top, 'coef')['condition'].tolist()

    fig, axes = plt.subplots(2, n_top, figsize=(4*n_top, 8))

    # Top positive associations (increase with age)
    for idx, condition in enumerate(top_positive):
        plot_prevalence_by_age(df, condition, ax=axes[0, idx], show_log_odds=False)
        coef = results_df[results_df['condition'] == condition]['coef'].values[0]
        axes[0, idx].set_title(f'{condition}\n(coef={coef:.4f})', fontsize=10)

    # Top negative associations (decrease with age)
    for idx, condition in enumerate(top_negative):
        plot_prevalence_by_age(df, condition, ax=axes[1, idx], show_log_odds=False)
        coef = results_df[results_df['condition'] == condition]['coef'].values[0]
        axes[1, idx].set_title(f'{condition}\n(coef={coef:.4f})', fontsize=10)

    axes[0, 0].set_ylabel('Prevalence (%)\n(Increase with Age)', fontsize=11)
    axes[1, 0].set_ylabel('Prevalence (%)\n(Decrease with Age)', fontsize=11)

    fig.suptitle('Top Age-Associated Medical Conditions', fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        filepath = os.path.join(FIGURES_DIR, 'top_age_associated_conditions.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    return fig


def plot_heatmap_by_group(results_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Create heatmap of age coefficients by disease group."""
    valid = results_df[
        (results_df['n_positive'] >= 50) &
        (results_df['coef'].notna())
    ].copy()

    # Create pivot for heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    # Sort conditions within each group by coefficient
    valid['sort_order'] = valid.groupby('disease_group')['coef'].rank(ascending=False)
    valid = valid.sort_values(['disease_group', 'sort_order'])

    # Create matrix
    groups = valid['disease_group'].unique()

    # Plot grouped bars
    group_starts = []
    current_pos = 0

    for group in groups:
        group_data = valid[valid['disease_group'] == group]
        positions = np.arange(len(group_data)) + current_pos
        group_starts.append((current_pos, group))

        colors = plt.cm.RdBu_r((group_data['coef'] - valid['coef'].min()) /
                                (valid['coef'].max() - valid['coef'].min()))

        ax.barh(positions, group_data['coef'], color=colors, alpha=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(group_data['condition'], fontsize=8)

        current_pos += len(group_data) + 1  # Add gap between groups

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Log Odds per Year of Age', fontsize=12)
    ax.set_title('Age-Disease Association by Category', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    # Add group labels
    for start, group in group_starts:
        ax.axhline(start - 0.5, color='gray', linestyle='-', linewidth=0.5)
        ax.text(-0.12, start + 1, group, fontsize=10, fontweight='bold',
               transform=ax.get_yaxis_transform(), va='bottom')

    plt.tight_layout()

    if save:
        filepath = os.path.join(FIGURES_DIR, 'age_coefficients_heatmap.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    return fig


def generate_report(df: pd.DataFrame, results_df: pd.DataFrame,
                    group_results: pd.DataFrame) -> str:
    """Generate comprehensive markdown report."""

    report = """# Age and Medical Conditions Analysis Report
## HPP 10K Dataset

### Executive Summary

This report analyzes the relationship between age and the log probability (log odds)
of having various medical conditions in the HPP 10K cohort. The analysis is based on
the WHO framework for major disease categories and epidemiological research on
aging-related diseases.

### Background

According to the World Health Organization, four main categories of noncommunicable
diseases account for 80% of all premature NCD deaths:
1. **Cardiovascular diseases** - 19 million deaths annually
2. **Cancers** - 10 million deaths annually
3. **Chronic respiratory diseases** - 4 million deaths annually
4. **Diabetes** - 2+ million deaths annually

Research from the Journal of Gerontology identifies diseases with exponential
increase in incidence with age (Group A diseases) including Alzheimer's disease,
ischemic heart disease, COPD, and stroke.

---

## Dataset Overview

"""
    # Add dataset stats
    n_subjects = df.index.get_level_values(0).nunique()
    n_records = len(df)
    age_stats = df['age'].describe()

    report += f"""- **Total records**: {n_records:,}
- **Unique subjects**: {n_subjects:,}
- **Age range**: {age_stats['min']:.1f} - {age_stats['max']:.1f} years
- **Mean age**: {age_stats['mean']:.1f} +/- {age_stats['std']:.1f} years
- **Medical conditions analyzed**: {len(results_df)}

---

## Key Findings

### 1. Conditions that Increase Most with Age

"""
    # Top increasing conditions
    top_increase = results_df[
        (results_df['n_positive'] >= 50) &
        (results_df['coef'].notna()) &
        (results_df['p_value'] < 0.05)
    ].nlargest(10, 'coef')

    report += "| Condition | Disease Group | Log OR/year | OR per Decade | N Positive | P-value |\n"
    report += "|-----------|---------------|-------------|---------------|------------|----------|\n"

    for _, row in top_increase.iterrows():
        report += f"| {row['condition']} | {row['disease_group']} | {row['coef']:.4f} | "
        report += f"{row.get('odds_ratio_per_decade', np.exp(row['coef']*10)):.2f} | "
        report += f"{row['n_positive']:,} | {row['p_value']:.2e} |\n"

    report += """
**Interpretation**: For each year of age, the log odds of these conditions increase
by the coefficient shown. The "OR per Decade" shows the multiplicative increase in
odds for every 10 years of age.

### 2. Conditions that Decrease with Age

"""
    # Top decreasing conditions
    top_decrease = results_df[
        (results_df['n_positive'] >= 50) &
        (results_df['coef'].notna()) &
        (results_df['p_value'] < 0.05)
    ].nsmallest(10, 'coef')

    report += "| Condition | Disease Group | Log OR/year | OR per Decade | N Positive | P-value |\n"
    report += "|-----------|---------------|-------------|---------------|------------|----------|\n"

    for _, row in top_decrease.iterrows():
        report += f"| {row['condition']} | {row['disease_group']} | {row['coef']:.4f} | "
        report += f"{row.get('odds_ratio_per_decade', np.exp(row['coef']*10)):.2f} | "
        report += f"{row['n_positive']:,} | {row['p_value']:.2e} |\n"

    report += """
**Interpretation**: These conditions show decreasing prevalence with age, possibly
due to survival bias (patients with severe conditions dying earlier), cohort effects
(younger generations having higher diagnosis rates), or genuine age-related decrease.

---

## Disease Group Analysis

### Overall Group Associations with Age

"""
    # Group results
    group_valid = group_results[group_results['coef'].notna()].sort_values('coef', ascending=False)

    report += "| Disease Group | Log OR/year | OR per Decade | N Conditions | N Positive |\n"
    report += "|---------------|-------------|---------------|--------------|------------|\n"

    for _, row in group_valid.iterrows():
        report += f"| {row['disease_group']} | {row['coef']:.4f} | "
        report += f"{row.get('odds_ratio_per_decade', np.exp(row['coef']*10)):.2f} | "
        report += f"{row['n_conditions']} | {row['n_positive']:,} |\n"

    report += """
---

## Detailed Results by Disease Category

"""
    # Add details for each group
    for group in DISEASE_GROUPS.keys():
        group_data = results_df[results_df['disease_group'] == group]
        if len(group_data) == 0:
            continue

        report += f"\n### {group}\n\n"
        report += "| Condition | Log OR/year | N Positive | P-value | Significant |\n"
        report += "|-----------|-------------|------------|---------|-------------|\n"

        for _, row in group_data.sort_values('coef', ascending=False).iterrows():
            sig = "Yes" if row['p_value'] < 0.05 else "No"
            p_val = f"{row['p_value']:.2e}" if not np.isnan(row['p_value']) else "N/A"
            coef = f"{row['coef']:.4f}" if not np.isnan(row['coef']) else "N/A"
            report += f"| {row['condition']} | {coef} | {row['n_positive']:,} | {p_val} | {sig} |\n"

    report += """
---

## Methodology

### Statistical Approach

1. **Log Odds Calculation**: For each condition, we fit a logistic regression model
   with age as the predictor. The coefficient represents the change in log odds of
   the condition per year of age.

2. **Disease Grouping**: Conditions were grouped based on the WHO framework for
   noncommunicable diseases and clinical categorization.

3. **Confidence Intervals**: 95% confidence intervals were calculated using the
   standard errors from the logistic regression.

4. **Group Analysis**: For each disease group, a binary indicator was created
   (1 if any condition in the group is present) and the same logistic regression
   was applied.

### Limitations

- Cross-sectional analysis cannot establish causality
- Survival bias may affect prevalence in older age groups
- Diagnosis rates may vary by age (detection bias)
- Self-reported conditions may have recall bias

---

## References

1. WHO. (2023). Noncommunicable diseases fact sheet.
   https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases

2. Hou, Y., et al. (2022). What Is an Aging-Related Disease? An Epidemiological
   Perspective. The Journals of Gerontology: Series A.
   https://academic.oup.com/biomedgerontology/article/77/11/2168/6528987

3. Cleveland Clinic. (2024). Neurodegenerative Diseases.
   https://my.clevelandclinic.org/health/diseases/24976-neurodegenerative-diseases

---

## Figures

- `figures/age_coefficients_all_conditions.png` - All conditions sorted by age coefficient
- `figures/age_coefficients_by_group.png` - Disease groups by age coefficient
- `figures/top_age_associated_conditions.png` - Prevalence curves for top conditions
- `figures/prevalence_*.png` - Prevalence by age for each disease group

---

*Report generated using HPP 10K dataset*
"""

    return report


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("AGE AND MEDICAL CONDITIONS ANALYSIS")
    print("HPP 10K Dataset")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    df = load_age_and_conditions()

    # Analyze individual conditions
    print("\n2. Analyzing individual conditions...")
    results_df = analyze_all_conditions(df)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'condition_analysis_results.csv'), index=False)
    print(f"   Saved results to condition_analysis_results.csv")

    # Analyze groups
    print("\n3. Analyzing disease groups...")
    group_results = analyze_grouped_conditions(df)
    group_results.to_csv(os.path.join(OUTPUT_DIR, 'group_analysis_results.csv'), index=False)
    print(f"   Saved results to group_analysis_results.csv")

    # Generate figures
    print("\n4. Generating figures...")

    print("   - Age coefficients for all conditions...")
    plot_age_coefficients(results_df)

    print("   - Age coefficients by group...")
    plot_group_coefficients(group_results)

    print("   - Top age-associated conditions...")
    plot_top_age_associated_conditions(df, results_df)

    print("   - Prevalence plots by disease group...")
    for group in DISEASE_GROUPS.keys():
        plot_grouped_prevalence(df, group)

    # Generate report
    print("\n5. Generating report...")
    report = generate_report(df, results_df, group_results)
    report_path = os.path.join(OUTPUT_DIR, 'analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"   Saved report to {report_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    # Print summary
    print("\nSummary of significant findings:")
    sig_results = results_df[
        (results_df['p_value'] < 0.05) &
        (results_df['n_positive'] >= 50)
    ]
    n_increase = (sig_results['coef'] > 0).sum()
    n_decrease = (sig_results['coef'] < 0).sum()

    print(f"  - {n_increase} conditions increase significantly with age")
    print(f"  - {n_decrease} conditions decrease significantly with age")

    print("\nTop 5 conditions increasing with age:")
    for _, row in sig_results.nlargest(5, 'coef').iterrows():
        print(f"  - {row['condition']}: OR/decade = {np.exp(row['coef']*10):.2f}")

    return df, results_df, group_results


if __name__ == "__main__":
    df, results_df, group_results = main()
