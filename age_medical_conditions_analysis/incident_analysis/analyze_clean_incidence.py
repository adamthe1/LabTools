#!/usr/bin/env python
"""
Clean analysis of age at incident with proper incidence rate calculation.
Focused on ages 40-70. Shows -log(incidence) for all conditions.
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
FIG_DIR = os.path.join(OUT_DIR, 'figures_clean')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'font.family': 'sans-serif'
})

# Age range to analyze
AGE_MIN = 40
AGE_MAX = 70
AGE_BINS = list(range(AGE_MIN, AGE_MAX + 2, 2))  # 40-42, 42-44, ..., 68-70

print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Clean birth year
df['yob'] = pd.to_numeric(df['yob'], errors='coerce')
df = df[df['yob'].notna() & (df['yob'] >= 1940) & (df['yob'] <= 2005)]

# Calculate current age (assuming data ~2024)
DATA_YEAR = 2024
df['current_age'] = DATA_YEAR - df['yob']

# Filter to people who have reached age 40 at minimum
df = df[df['current_age'] >= AGE_MIN]

print(f"Total subjects (age >= {AGE_MIN}): {len(df)}")
print(f"Current age range: {df['current_age'].min():.0f} - {df['current_age'].max():.0f}")

# Get condition columns
condition_cols = [c for c in df.columns if c not in ['RegistrationCode', 'yob', ' PES PLANUS', 'current_age']]


def get_population_at_risk(df, age_bins):
    """Calculate how many people have reached each age bin."""
    pop = {}
    for i in range(len(age_bins) - 1):
        age_start = age_bins[i]
        age_end = age_bins[i + 1]
        # People at risk = those who have reached at least this age
        n_at_risk = (df['current_age'] >= age_end).sum()
        pop[(age_start, age_end)] = n_at_risk
    return pop


population_at_risk = get_population_at_risk(df, AGE_BINS)
print(f"\nPopulation at risk by age bin ({AGE_MIN}-{AGE_MAX}):")
for (start, end), n in population_at_risk.items():
    print(f"  [{start}-{end}): {n:,}")


def calculate_age_at_diagnosis_clean(df, condition):
    """Calculate age at diagnosis with data cleaning."""
    subset = df[['RegistrationCode', 'yob', 'current_age', condition]].copy()
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

    # Filter: must be positive, not greater than current age, within 40-70
    valid = valid[
        (valid['age_at_diagnosis'] >= AGE_MIN) &
        (valid['age_at_diagnosis'] <= AGE_MAX) &
        (valid['age_at_diagnosis'] <= valid['current_age'])
    ]

    return valid


def calculate_incidence_rate(ages, age_bins, population_at_risk):
    """Calculate TRUE incidence rate = diagnoses / population at risk."""
    results = []

    for i in range(len(age_bins) - 1):
        age_start = age_bins[i]
        age_end = age_bins[i + 1]
        age_mid = (age_start + age_end) / 2

        # Count diagnoses in this bin
        n_diagnosed = ((ages >= age_start) & (ages < age_end)).sum()

        # Get population at risk
        n_at_risk = population_at_risk.get((age_start, age_end), 1)

        if n_at_risk > 0 and n_diagnosed > 0:
            incidence_rate = n_diagnosed / n_at_risk
            log_incidence = np.log10(incidence_rate)  # Regular log (not negative)
        else:
            incidence_rate = 0
            log_incidence = np.nan

        results.append({
            'age_start': age_start,
            'age_end': age_end,
            'age_mid': age_mid,
            'n_diagnosed': n_diagnosed,
            'n_at_risk': n_at_risk,
            'incidence_rate': incidence_rate,
            'incidence_per_1000': incidence_rate * 1000,
            'log_incidence': log_incidence  # Intuitive: UP = more incidence
        })

    return pd.DataFrame(results)


# Process ALL conditions
print(f"\nProcessing conditions (ages {AGE_MIN}-{AGE_MAX})...")
results = []

for condition in condition_cols:
    diagnosis_data = calculate_age_at_diagnosis_clean(df, condition)

    if len(diagnosis_data) < 20:  # Minimum sample
        continue

    ages = diagnosis_data['age_at_diagnosis']
    incidence_df = calculate_incidence_rate(ages, AGE_BINS, population_at_risk)

    # Need at least some bins with data
    valid_bins = incidence_df[incidence_df['n_diagnosed'] >= 2]
    if len(valid_bins) < 3:
        continue

    results.append({
        'condition': condition,
        'n_valid': len(diagnosis_data),
        'mean_age': ages.mean(),
        'median_age': ages.median(),
        'std_age': ages.std()
    })

results_df = pd.DataFrame(results).sort_values('n_valid', ascending=False)
results_df.to_csv(os.path.join(OUT_DIR, 'clean_summary_40_70.csv'), index=False)

print(f"\nProcessed {len(results_df)} conditions with sufficient data in {AGE_MIN}-{AGE_MAX} range")
print("\nAll conditions:")
print(results_df[['condition', 'n_valid', 'mean_age']].to_string())

# ============================================================
# Create grid figures showing ALL conditions
# ============================================================
print("\nCreating figures...")

all_conditions = results_df['condition'].tolist()
n_conditions = len(all_conditions)

# Calculate how many pages needed (12 conditions per page)
conditions_per_page = 12
n_pages = (n_conditions + conditions_per_page - 1) // conditions_per_page

for page in range(n_pages):
    start_idx = page * conditions_per_page
    end_idx = min(start_idx + conditions_per_page, n_conditions)
    page_conditions = all_conditions[start_idx:end_idx]

    n_cols = 4
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()

    for idx, condition in enumerate(page_conditions):
        ax = axes[idx]

        diagnosis_data = calculate_age_at_diagnosis_clean(df, condition)
        ages = diagnosis_data['age_at_diagnosis']
        incidence_df = calculate_incidence_rate(ages, AGE_BINS, population_at_risk)

        plot_data = incidence_df[incidence_df['n_diagnosed'] >= 2]

        if len(plot_data) > 0:
            ax.plot(plot_data['age_mid'], plot_data['log_incidence'],
                    'o-', color='darkblue', markersize=5, linewidth=2)

        ax.set_xlabel('Age')
        ax.set_ylabel('log₁₀(Incidence)')
        ax.set_title(f'{condition[:25]}\n(n={len(ages)})', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(AGE_MIN - 1, AGE_MAX + 1)

    # Hide empty subplots
    for idx in range(len(page_conditions), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'log₁₀(Incidence Rate) by Age at Diagnosis (Page {page+1}/{n_pages})\nAges {AGE_MIN}-{AGE_MAX} | Line UP = More incidence',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'neg_log_incidence_page{page+1}.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"Created {n_pages} pages of -log(incidence) plots")

# ============================================================
# HEATMAP of all conditions
# ============================================================
print("Creating heatmap...")

age_mids = [(AGE_BINS[i] + AGE_BINS[i+1])/2 for i in range(len(AGE_BINS)-1)]

heatmap_data = []
for condition in all_conditions:
    diagnosis_data = calculate_age_at_diagnosis_clean(df, condition)
    if len(diagnosis_data) == 0:
        continue

    ages = diagnosis_data['age_at_diagnosis']
    incidence_df = calculate_incidence_rate(ages, AGE_BINS, population_at_risk)

    row = {'condition': condition}
    for _, r in incidence_df.iterrows():
        row[r['age_mid']] = r['log_incidence']
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data).set_index('condition')

fig, ax = plt.subplots(figsize=(14, max(10, len(heatmap_df) * 0.4)))
im = ax.imshow(heatmap_df.values, aspect='auto', cmap='RdYlBu_r')  # Reversed: Red = High
ax.set_yticks(range(len(heatmap_df)))
ax.set_yticklabels(heatmap_df.index, fontsize=8)
ax.set_xticks(range(len(heatmap_df.columns)))
ax.set_xticklabels([f'{int(c)}' for c in heatmap_df.columns])
ax.set_xlabel('Age at Diagnosis')
ax.set_ylabel('Condition')
ax.set_title(f'log₁₀(Incidence Rate) Heatmap (Ages {AGE_MIN}-{AGE_MAX})\nRed = Higher incidence (more common), Blue = Lower incidence (rarer)',
             fontweight='bold')
cbar = plt.colorbar(im, ax=ax, label='log₁₀(Incidence)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'heatmap_all_conditions.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Summary: Early vs Late peak
# ============================================================
print("Creating summary figure...")

fig, ax = plt.subplots(figsize=(12, max(8, len(results_df) * 0.3)))
sorted_df = results_df.sort_values('mean_age')

colors = plt.cm.coolwarm(np.linspace(0, 1, len(sorted_df)))
bars = ax.barh(range(len(sorted_df)), sorted_df['mean_age'], color=colors)
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels(sorted_df['condition'], fontsize=8)
ax.set_xlabel('Mean Age at Diagnosis')
ax.set_title(f'Mean Age at Diagnosis for All Conditions (Ages {AGE_MIN}-{AGE_MAX})', fontweight='bold')
ax.axvline(55, color='black', linestyle='--', alpha=0.5, label='Age 55')
ax.legend()

# Add sample sizes
for i, (_, row) in enumerate(sorted_df.iterrows()):
    ax.text(row['mean_age'] + 0.3, i, f"n={int(row['n_valid'])}", va='center', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'mean_age_all_conditions.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nAll figures saved to: {FIG_DIR}")
print("Done!")
