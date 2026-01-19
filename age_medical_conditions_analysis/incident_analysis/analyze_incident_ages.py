#!/usr/bin/env python
"""
Analyze age at incident (diagnosis) for medical conditions.
Calculates -log(incidence) by 2-year age bins.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

# Paths
DATA_PATH = '/home/adamgab/PycharmProjects/LabTools/age_medical_conditions_analysis/medical_conditions_with_dates.csv'
OUT_DIR = '/home/adamgab/PycharmProjects/LabTools/age_medical_conditions_analysis/incident_analysis'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'font.family': 'sans-serif'
})

print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Total subjects: {len(df)}")

# Get condition columns (everything except RegistrationCode and yob)
condition_cols = [c for c in df.columns if c not in ['RegistrationCode', 'yob', ' PES PLANUS']]
print(f"Condition columns: {len(condition_cols)}")

# Convert yob to int where possible
df['yob'] = pd.to_numeric(df['yob'], errors='coerce')
df = df[df['yob'].notna() & (df['yob'] >= 1930) & (df['yob'] <= 2010)]
print(f"Subjects with valid yob: {len(df)}")


def calculate_age_at_diagnosis(df, condition):
    """
    Calculate age at diagnosis for a condition.
    Returns: DataFrame with subjects and their age at diagnosis.
    Excludes 2000-01-01 (placeholder date).
    """
    # Get non-null values
    subset = df[['RegistrationCode', 'yob', condition]].copy()
    subset = subset[subset[condition].notna()]

    if len(subset) == 0:
        return pd.DataFrame()

    # Parse dates
    subset['diagnosis_date'] = pd.to_datetime(subset[condition], errors='coerce')
    subset = subset[subset['diagnosis_date'].notna()]

    # Exclude 2000-01-01 placeholder
    default_date = pd.Timestamp('2000-01-01')
    valid = subset[subset['diagnosis_date'] != default_date].copy()

    if len(valid) == 0:
        return pd.DataFrame()

    # Calculate age at diagnosis
    valid['diagnosis_year'] = valid['diagnosis_date'].dt.year
    valid['age_at_diagnosis'] = valid['diagnosis_year'] - valid['yob']

    # Filter reasonable ages (0-100)
    valid = valid[(valid['age_at_diagnosis'] >= 0) & (valid['age_at_diagnosis'] <= 100)]

    return valid[['RegistrationCode', 'yob', 'age_at_diagnosis']]


def calculate_incidence_by_age(ages, age_bins, total_population_per_bin=None):
    """
    Calculate incidence (proportion diagnosed) by age bin.

    If total_population_per_bin is None, uses count distribution.
    Returns: DataFrame with age_mid, count, incidence, neg_log_incidence
    """
    # Create bins
    ages_binned = pd.cut(ages, bins=age_bins, right=False)
    counts = ages_binned.value_counts().sort_index()

    results = []
    for interval in counts.index:
        age_mid = (interval.left + interval.right) / 2
        count = counts[interval]

        if total_population_per_bin is not None:
            # Use actual population at risk
            pop = total_population_per_bin.get(interval, 1)
            incidence = count / pop if pop > 0 else 0
        else:
            # Use proportion of total diagnoses
            incidence = count / len(ages) if len(ages) > 0 else 0

        # Avoid log(0)
        if incidence > 0:
            neg_log_inc = -np.log10(incidence)
        else:
            neg_log_inc = np.nan

        results.append({
            'age_bin': interval,
            'age_mid': age_mid,
            'count': count,
            'incidence': incidence,
            'neg_log_incidence': neg_log_inc
        })

    return pd.DataFrame(results)


# Calculate total population by birth year -> estimate population at each age
# We'll use birth year distribution to estimate how many people could be at each age
print("\nCalculating population distribution by birth year...")
birth_year_counts = df['yob'].value_counts().sort_index()
print(f"Birth years range: {int(birth_year_counts.index.min())} - {int(birth_year_counts.index.max())}")

# For incidence calculation, we need to know "how many people were at risk at each age"
# This is complex for a cross-sectional study. We'll use two approaches:
# 1. Distribution of diagnosis ages (what proportion of cases occurred at each age)
# 2. Rate per person-years at risk (approximation)

# Define age bins (2-year intervals)
AGE_BINS = list(range(0, 102, 2))  # 0-2, 2-4, ..., 98-100

# Process all conditions
print("\nProcessing conditions...")
results = []

for condition in condition_cols:
    diagnosis_data = calculate_age_at_diagnosis(df, condition)

    if len(diagnosis_data) < 20:  # Need at least 20 cases
        continue

    n_total = df[condition].notna().sum()  # Total with condition (any date)
    n_valid_date = len(diagnosis_data)  # With valid non-default date

    ages = diagnosis_data['age_at_diagnosis']

    # Calculate statistics
    mean_age = ages.mean()
    median_age = ages.median()
    std_age = ages.std()

    # Calculate incidence by age bin
    incidence_df = calculate_incidence_by_age(ages, AGE_BINS)

    # Find peak age (mode bin)
    peak_idx = incidence_df['count'].idxmax()
    peak_age = incidence_df.loc[peak_idx, 'age_mid']

    results.append({
        'condition': condition,
        'n_total': n_total,
        'n_valid_date': n_valid_date,
        'pct_valid_date': n_valid_date / n_total * 100 if n_total > 0 else 0,
        'mean_age': mean_age,
        'median_age': median_age,
        'std_age': std_age,
        'peak_age': peak_age
    })

    # Store incidence data for plotting
    incidence_df['condition'] = condition
    incidence_df.to_csv(os.path.join(OUT_DIR, f'incidence_{condition.replace("/", "_").replace(" ", "_")}.csv'), index=False)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('n_valid_date', ascending=False)
results_df.to_csv(os.path.join(OUT_DIR, 'summary_statistics.csv'), index=False)

print(f"\nProcessed {len(results_df)} conditions with sufficient data")
print("\nTop conditions by sample size:")
print(results_df[['condition', 'n_valid_date', 'mean_age', 'median_age', 'peak_age']].head(20).to_string())

# ============================================================
# FIGURE 1: Distribution of mean age at diagnosis
# ============================================================
print("\nCreating figures...")

fig, ax = plt.subplots(figsize=(10, 6))
valid_results = results_df[results_df['n_valid_date'] >= 50]
ax.hist(valid_results['mean_age'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(valid_results['mean_age'].mean(), color='red', linestyle='--',
           label=f'Mean: {valid_results["mean_age"].mean():.1f} years')
ax.set_xlabel('Mean Age at Diagnosis')
ax.set_ylabel('Number of Conditions')
ax.set_title('Distribution of Mean Age at Diagnosis Across Conditions\n(n≥50 valid dates)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'mean_age_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 2: Top conditions by peak diagnosis age (early vs late)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Early onset conditions
early = valid_results.nsmallest(15, 'mean_age')
ax = axes[0]
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(early)))
bars = ax.barh(range(len(early)), early['mean_age'], color=colors)
ax.set_yticks(range(len(early)))
ax.set_yticklabels(early['condition'])
ax.set_xlabel('Mean Age at Diagnosis')
ax.set_title('Earliest Onset Conditions')
ax.invert_yaxis()
for i, (_, row) in enumerate(early.iterrows()):
    ax.text(row['mean_age'] + 0.5, i, f"n={row['n_valid_date']}", va='center', fontsize=9)

# Late onset conditions
late = valid_results.nlargest(15, 'mean_age')
ax = axes[1]
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(late)))
bars = ax.barh(range(len(late)), late['mean_age'], color=colors)
ax.set_yticks(range(len(late)))
ax.set_yticklabels(late['condition'])
ax.set_xlabel('Mean Age at Diagnosis')
ax.set_title('Latest Onset Conditions')
ax.invert_yaxis()
for i, (_, row) in enumerate(late.iterrows()):
    ax.text(row['mean_age'] + 0.5, i, f"n={row['n_valid_date']}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'early_vs_late_onset.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 3: -log(incidence) plots for top conditions
# ============================================================
# Select top 12 conditions by sample size
top_conditions = results_df.nlargest(12, 'n_valid_date')['condition'].tolist()

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, condition in enumerate(top_conditions):
    ax = axes[idx]

    # Recalculate for this condition
    diagnosis_data = calculate_age_at_diagnosis(df, condition)
    ages = diagnosis_data['age_at_diagnosis']
    incidence_df = calculate_incidence_by_age(ages, AGE_BINS)

    # Filter to age range with data
    plot_data = incidence_df[(incidence_df['count'] > 0) & (incidence_df['age_mid'] >= 10)]

    ax.plot(plot_data['age_mid'], plot_data['neg_log_incidence'],
            'o-', color='darkblue', markersize=4, linewidth=1.5)

    # Add count as text for key points
    peak_idx = plot_data['count'].idxmax()
    if peak_idx in plot_data.index:
        peak_row = plot_data.loc[peak_idx]
        ax.annotate(f"n={int(peak_row['count'])}",
                   (peak_row['age_mid'], peak_row['neg_log_incidence']),
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xlabel('Age at Diagnosis')
    ax.set_ylabel('-log₁₀(Incidence)')
    ax.set_title(f'{condition}\n(n={len(ages)})', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)

plt.suptitle('-log₁₀(Incidence Rate) by Age at Diagnosis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'neg_log_incidence_grid.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 4: Age distribution histograms for select conditions
# ============================================================
fig, axes = plt.subplots(4, 3, figsize=(14, 16))
axes = axes.flatten()

for idx, condition in enumerate(top_conditions):
    ax = axes[idx]

    diagnosis_data = calculate_age_at_diagnosis(df, condition)
    ages = diagnosis_data['age_at_diagnosis']

    # Histogram
    ax.hist(ages, bins=range(0, 95, 5), edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(ages.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {ages.mean():.1f}')
    ax.axvline(ages.median(), color='orange', linestyle=':', linewidth=2,
               label=f'Median: {ages.median():.1f}')

    ax.set_xlabel('Age at Diagnosis')
    ax.set_ylabel('Count')
    ax.set_title(f'{condition} (n={len(ages)})', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 90)

plt.suptitle('Age at Diagnosis Distribution', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'age_distribution_histograms.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 5: Heatmap of -log(incidence) across conditions and ages
# ============================================================
# Build matrix for heatmap
age_range = list(range(20, 80, 2))  # Focus on 20-80 years
top20 = results_df[results_df['n_valid_date'] >= 100].nlargest(25, 'n_valid_date')['condition'].tolist()

heatmap_data = []
for condition in top20:
    diagnosis_data = calculate_age_at_diagnosis(df, condition)
    if len(diagnosis_data) == 0:
        continue
    ages = diagnosis_data['age_at_diagnosis']
    incidence_df = calculate_incidence_by_age(ages, AGE_BINS)

    # Map to our age range
    row = {'condition': condition}
    for age in age_range:
        # Find the bin containing this age
        matching = incidence_df[(incidence_df['age_mid'] >= age) & (incidence_df['age_mid'] < age + 2)]
        if len(matching) > 0:
            row[age] = matching.iloc[0]['neg_log_incidence']
        else:
            row[age] = np.nan
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data).set_index('condition')

fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(heatmap_df.values, aspect='auto', cmap='RdYlBu')
ax.set_yticks(range(len(heatmap_df)))
ax.set_yticklabels(heatmap_df.index)
ax.set_xticks(range(0, len(age_range), 5))
ax.set_xticklabels([age_range[i] for i in range(0, len(age_range), 5)])
ax.set_xlabel('Age at Diagnosis')
ax.set_ylabel('Condition')
ax.set_title('-log₁₀(Incidence) Heatmap\n(Blue = Higher -log = Lower Incidence, Red = Lower -log = Higher Incidence)')
cbar = plt.colorbar(im, ax=ax, label='-log₁₀(Incidence)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'incidence_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 6: Comparison of conditions by disease category
# ============================================================
# Disease groups from skill
DISEASE_GROUPS = {
    'Cardiovascular': ['Hypertension', 'Ischemic Heart Disease', 'Atrial Fibrillation',
                       'Heart valve disease', 'AV. Conduction Disorder', 'Myocarditis', 'Atherosclerotic'],
    'Metabolic': ['Diabetes', 'Prediabetes', 'Hyperlipidemia', 'Hypercholesterolaemia',
                  'Obesity', 'Overweight', 'Fatty Liver Disease'],
    'Mental Health': ['Depression', 'Anxiety', 'ADHD', 'PTSD', 'Insomnia', 'Migraine', 'Headache'],
    'Musculoskeletal': ['Osteoarthritis', 'Back Pain', 'Fibromyalgia', 'Fractures', 'Gout', 'Meniscus Tears'],
    'Autoimmune': ['Psoriasis', 'Vitiligo', 'Allergy', 'Atopic Dermatitis'],
}

GROUP_COLORS = {
    'Cardiovascular': '#E74C3C',
    'Metabolic': '#E67E22',
    'Mental Health': '#1ABC9C',
    'Musculoskeletal': '#34495E',
    'Autoimmune': '#F1C40F',
}

fig, ax = plt.subplots(figsize=(12, 8))

for group, conditions in DISEASE_GROUPS.items():
    group_data = results_df[results_df['condition'].isin(conditions)]
    if len(group_data) > 0:
        x = group_data['mean_age']
        y = group_data['std_age']
        ax.scatter(x, y, label=f'{group} (n={len(group_data)})',
                  s=100, alpha=0.7, color=GROUP_COLORS.get(group, 'gray'))

        # Add condition labels
        for _, row in group_data.iterrows():
            ax.annotate(row['condition'], (row['mean_age'], row['std_age']),
                       fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('Mean Age at Diagnosis')
ax.set_ylabel('Std Dev of Age at Diagnosis')
ax.set_title('Age at Diagnosis by Disease Category\n(Higher std = wider age distribution)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'disease_category_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nFigures saved to: {FIG_DIR}")
print("Done!")
