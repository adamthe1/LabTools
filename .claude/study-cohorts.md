# Skill: Study Cohorts

Understanding the different study cohorts in the HPP dataset for accurate research design and analysis.

## Overview

The HPP dataset contains subjects from multiple study cohorts, not just the main 10K population study. The `study_type` body system maps each subject to their cohort via:
- `study_type`: Numeric identifier
- `study_description`: Human-readable name

**Why this matters:** Different cohorts have different inclusion criteria, disease profiles, and demographics. Analyzing cohorts together without awareness can introduce confounding or dilute effects.

---

## Available Cohorts

| ID | Name | N | Description |
|----|------|---|-------------|
| 10 | 10K | ~23,461 | Main general population cohort (93% of data) |
| 1001 | Beilinson_Cardio | ~605 | Cardiovascular patients from Beilinson hospital |
| 1009 | 10K_young | ~552 | Younger participants (subset of 10K) |
| 1007 | BRCA | ~231 | BRCA gene mutation carriers (breast/ovarian cancer risk) |
| 1008 | Endometriosis | ~194 | Patients with endometriosis |
| 1006 | BreastCancerRecovered | ~182 | Breast cancer survivors |
| 1010 | 10K_T1D | ~72 | Type 1 Diabetes patients |
| 1005 | Colorectal_cancer_recovered | ~25 | Colorectal cancer survivors |
| 1011 | 10K_GLP | ~9 | GLP-1 agonist users |
| 1002 | OvarianCancer | 0* | Defined but no body measures data |
| 1003 | Ichilov-crc-recovered | 0* | Defined but no body measures data |
| 1004 | Ichilov-BreastCancer-recovered | 0* | Defined but no body measures data |

*These cohorts exist in definitions but have no subjects with body measures in current data.

---

## Cohort Categories

### General Population
```python
GENERAL_POPULATION = [10, 1009]  # 10K and 10K_young
```
- **10K**: Main cohort, ages typically 40-70, general population
- **10K_young**: Younger participants, extends age range downward

### Disease-Specific Studies
```python
DISEASE_COHORTS = {
    'cancer_survivors': [1005, 1006],  # Colorectal, Breast
    'cancer_risk': [1007],             # BRCA carriers
    'cardiovascular': [1001],          # Beilinson_Cardio
    'metabolic': [1010, 1011],         # T1D, GLP-1
    'reproductive': [1008]             # Endometriosis
}
```

### Hospital-Based vs Population-Based
```python
HOSPITAL_COHORTS = [1001, 1002, 1003, 1004]  # Beilinson, Ichilov
POPULATION_COHORTS = [10, 1009, 1005, 1006, 1007, 1008, 1010, 1011]
```

---

## Loading Study Type Data

```python
from body_system_loader.load_feature_df import load_body_system_df, load_columns_as_df

# Load study type information
study_type = load_body_system_df('study_type')

# Or load specific columns
study_info = load_columns_as_df(['study_type', 'study_description'])

# Check cohort distribution
print(study_type['study_description'].value_counts())
```

---

## When to Consider Cohort

### Always Check Cohort When:

1. **Studying disease prevalence** - Cancer survivors have 100% cancer history
2. **Age-related analyses** - 10K_young has different age distribution
3. **Cardiovascular research** - Beilinson_Cardio is enriched for CVD
4. **Metabolic research** - T1D cohort has very different metabolic profiles
5. **Sex-specific conditions** - Endometriosis/BRCA cohorts are female-only

### Filter to Main 10K for:

```python
# General population analyses
df_10k = df[df['study_type'] == 10]

# Or include young cohort
df_general = df[df['study_type'].isin([10, 1009])]
```

### Stratify by Cohort When:

```python
# Compare cohorts
for cohort, cohort_df in df.groupby('study_type'):
    # Run analysis per cohort
    pass

# Or explicitly compare
df_10k = df[df['study_type'] == 10]
df_cardio = df[df['study_type'] == 1001]
```

---

## Helper Functions

```python
def get_cohort_name(study_type_id):
    """Get human-readable name for study type ID."""
    COHORT_NAMES = {
        10: '10K',
        1001: 'Beilinson_Cardio',
        1005: 'Colorectal_cancer_recovered',
        1006: 'BreastCancerRecovered',
        1007: 'BRCA',
        1008: 'Endometriosis',
        1009: '10K_young',
        1010: '10K_T1D',
        1011: '10K_GLP'
    }
    return COHORT_NAMES.get(study_type_id, f'Unknown ({study_type_id})')


def filter_to_general_population(df, include_young=False):
    """Filter to general population cohort(s)."""
    cohorts = [10]
    if include_young:
        cohorts.append(1009)
    return df[df['study_type'].isin(cohorts)]


def get_disease_cohorts(df, disease_type):
    """Get subjects from disease-specific cohorts.

    Args:
        disease_type: 'cancer', 'cardiovascular', 'metabolic', 'reproductive'
    """
    DISEASE_COHORTS = {
        'cancer': [1005, 1006, 1007],      # Survivors + BRCA
        'cancer_survivors': [1005, 1006],   # Only survivors
        'cancer_risk': [1007],              # BRCA carriers
        'cardiovascular': [1001],
        'metabolic': [1010, 1011],
        'reproductive': [1008]
    }
    cohort_ids = DISEASE_COHORTS.get(disease_type, [])
    return df[df['study_type'].isin(cohort_ids)]


def check_cohort_distribution(df):
    """Print cohort distribution summary."""
    if 'study_type' not in df.columns:
        print("Warning: study_type column not found")
        return

    print("=== COHORT DISTRIBUTION ===\n")
    counts = df['study_type'].value_counts()
    total = len(df)

    for study_id, count in counts.items():
        name = get_cohort_name(study_id)
        pct = count / total * 100
        print(f"  {name:30s}: n={count:6,} ({pct:5.1f}%)")

    # Check if mostly 10K
    n_10k = df[df['study_type'] == 10].shape[0]
    pct_10k = n_10k / total * 100
    if pct_10k < 90:
        print(f"\n⚠️ Only {pct_10k:.1f}% from main 10K cohort")
        print("   Consider filtering to study_type == 10 for general population analyses")


def add_cohort_labels(df):
    """Add human-readable cohort labels to dataframe."""
    df = df.copy()
    df['cohort_name'] = df['study_type'].apply(get_cohort_name)
    return df
```

---

## Research Design Considerations

### Case-Control Studies

When using disease cohorts as cases:

```python
# Example: BRCA carriers vs general population
brca_carriers = df[df['study_type'] == 1007]
controls = df[df['study_type'] == 10]

# Important: Match on age and gender
# BRCA cohort is female-only, so filter controls
controls_female = controls[controls['gender'] == 0]
```

### Enrichment Studies

Disease cohorts can provide enriched case sets:

```python
# Cardiovascular disease analysis
# Beilinson_Cardio cohort has higher CVD prevalence
cardio_cohort = df[df['study_type'] == 1001]
# But be aware of selection bias - hospital patients differ from general population
```

### Sensitivity Analyses

Always run sensitivity analysis excluding special cohorts:

```python
# Main analysis on all data
result_all = run_analysis(df)

# Sensitivity: 10K only
result_10k = run_analysis(df[df['study_type'] == 10])

# Compare results
print(f"All cohorts: {result_all}")
print(f"10K only: {result_10k}")
```

---

## Cohort-Specific Demographics

### Expected Differences

| Cohort | Age Range | Gender | Key Differences |
|--------|-----------|--------|-----------------|
| 10K | 40-70 | ~50/50 | General population |
| 10K_young | <40 | ~50/50 | Younger, healthier |
| Beilinson_Cardio | Variable | Male-heavy | CVD enriched |
| BRCA | Variable | 100% F | Cancer risk genes |
| Endometriosis | <50 typical | 100% F | Reproductive disorder |
| BreastCancerRecovered | Variable | ~100% F | Cancer history |
| 10K_T1D | Variable | ~50/50 | Autoimmune diabetes |

### Verification Code

```python
def verify_cohort_demographics(df, study_type_col='study_type'):
    """Verify demographic distribution by cohort."""
    for cohort in df[study_type_col].unique():
        cohort_df = df[df[study_type_col] == cohort]
        name = get_cohort_name(cohort)

        print(f"\n=== {name} (n={len(cohort_df)}) ===")

        if 'Age' in cohort_df.columns:
            print(f"Age: {cohort_df['Age'].mean():.1f} ± {cohort_df['Age'].std():.1f}")

        if 'gender' in cohort_df.columns:
            pct_female = (cohort_df['gender'] == 0).mean() * 100
            print(f"Female: {pct_female:.1f}%")
```

---

## Quick Reference

### Filter Commands

```python
# Main 10K only (most common)
df_10k = df[df['study_type'] == 10]

# General population (10K + young)
df_general = df[df['study_type'].isin([10, 1009])]

# Exclude hospital cohorts
df_population = df[~df['study_type'].isin([1001, 1002, 1003, 1004])]

# Cancer-related cohorts only
df_cancer = df[df['study_type'].isin([1005, 1006, 1007])]
```

### Sample Size by Cohort

| Cohort | Approximate N | % of Total |
|--------|---------------|------------|
| 10K | 23,461 | 93% |
| Beilinson_Cardio | 605 | 2.4% |
| 10K_young | 552 | 2.2% |
| BRCA | 231 | 0.9% |
| Endometriosis | 194 | 0.8% |
| BreastCancerRecovered | 182 | 0.7% |
| 10K_T1D | 72 | 0.3% |
| Colorectal_cancer_recovered | 25 | 0.1% |
| 10K_GLP | 9 | <0.1% |

---

## Integration with Other Skills

### With Data Quality Check
```python
# First check cohort distribution
check_cohort_distribution(df)

# Then run demographic checks (from data-quality-check.md)
from data_quality_check import check_age_distribution
check_age_distribution(df[df['study_type'] == 10])  # On main cohort
```

### With Disease Groups
```python
# Cancer disease group analysis should consider cancer cohorts
from disease_groups import DISEASE_GROUPS

# BRCA carriers have elevated breast/ovarian cancer risk
# BreastCancerRecovered have cancer history
# Be explicit about what you're measuring
```
