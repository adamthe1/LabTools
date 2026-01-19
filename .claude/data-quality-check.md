# Skill: Data Quality Check

Use this skill before any analysis to validate data quality and identify potential issues.

## Why This Matters

Poor data quality leads to:
- Unreliable estimates (low n in subgroups)
- Biased results (uneven distributions)
- Misleading conclusions (confounders not addressed)

**Rule:** Always check data quality BEFORE running any analysis.

---

## 1. Check Demographic Distributions

### Age Distribution

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_age_distribution(df, age_col='age'):
    """Check age distribution and identify reliable ranges."""

    print("=== AGE DISTRIBUTION ===\n")

    # Basic stats
    print(f"N = {len(df)}")
    print(f"Range: {df[age_col].min():.1f} - {df[age_col].max():.1f}")
    print(f"Mean: {df[age_col].mean():.1f} ± {df[age_col].std():.1f}")
    print(f"Median: {df[age_col].median():.1f}")

    # Check by 5-year bins
    bins = list(range(0, 105, 5))
    df['age_bin'] = pd.cut(df[age_col], bins=bins, right=False)
    counts = df.groupby('age_bin', observed=True).size()

    print("\nSample size by age bin:")
    for interval, count in counts.items():
        flag = "⚠️ LOW" if count < 200 else "✓" if count >= 500 else ""
        pct = count / len(df) * 100
        print(f"  {interval}: n={count:5d} ({pct:5.1f}%) {flag}")

    # Recommend reliable range
    adequate = counts[counts >= 200]
    if len(adequate) > 0:
        min_age = adequate.index[0].left
        max_age = adequate.index[-1].right
        n_adequate = df[(df[age_col] >= min_age) & (df[age_col] < max_age)].shape[0]
        print(f"\n✓ RECOMMENDED RANGE: {min_age}-{max_age} years")
        print(f"  Contains {n_adequate:,} subjects ({n_adequate/len(df)*100:.1f}%)")

    return counts

# Usage
counts = check_age_distribution(df, 'age')
```

**Decision rules for age range:**
| Sample per bin | Action |
|----------------|--------|
| n < 100 | ❌ Exclude this age range |
| 100 ≤ n < 200 | ⚠️ Include with caution, wide CI |
| 200 ≤ n < 500 | ✓ Acceptable |
| n ≥ 500 | ✓ Good statistical power |

**HPP 10K specific:** Use ages 40-70 (contains 94% of data).

---

### Gender Distribution

```python
def check_gender_distribution(df, gender_col='gender'):
    """Check gender distribution and identify imbalances."""

    print("=== GENDER DISTRIBUTION ===\n")

    counts = df[gender_col].value_counts()
    total = len(df)

    for gender, count in counts.items():
        pct = count / total * 100
        label = "Female" if gender == 0 else "Male" if gender == 1 else str(gender)
        print(f"  {label}: n={count:,} ({pct:.1f}%)")

    # Check balance
    ratio = counts.min() / counts.max()
    if ratio < 0.8:
        print(f"\n⚠️ IMBALANCED: ratio = {ratio:.2f}")
        print("  Consider stratifying by gender or adjusting for gender")
    else:
        print(f"\n✓ BALANCED: ratio = {ratio:.2f}")

    return counts

# Usage
gender_counts = check_gender_distribution(df, 'gender')
```

**When to stratify by gender:**

| Situation | Action |
|-----------|--------|
| Sex-specific conditions (PCOS, prostate) | Always stratify |
| Known biological differences (CVD risk) | Stratify or adjust |
| Imbalanced sample (ratio < 0.8) | Consider stratifying |
| Exploratory analysis | Stratify to check for differences |
| Final confirmatory analysis | Adjust in model if appropriate |

---

### BMI Distribution

```python
def check_bmi_distribution(df, bmi_col='bmi'):
    """Check BMI distribution and identify issues."""

    print("=== BMI DISTRIBUTION ===\n")

    # Remove invalid values
    valid = df[df[bmi_col].between(15, 60)]
    invalid = len(df) - len(valid)
    if invalid > 0:
        print(f"⚠️ Removed {invalid} invalid BMI values (<15 or >60)")

    print(f"N = {len(valid)}")
    print(f"Range: {valid[bmi_col].min():.1f} - {valid[bmi_col].max():.1f}")
    print(f"Mean: {valid[bmi_col].mean():.1f} ± {valid[bmi_col].std():.1f}")

    # WHO categories
    bins = [0, 18.5, 25, 30, 35, 40, 100]
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
    valid['bmi_cat'] = pd.cut(valid[bmi_col], bins=bins, labels=labels, right=False)

    print("\nBMI categories:")
    for cat in labels:
        count = (valid['bmi_cat'] == cat).sum()
        pct = count / len(valid) * 100
        flag = "⚠️ LOW" if count < 100 else ""
        print(f"  {cat:12s}: n={count:5d} ({pct:5.1f}%) {flag}")

    return valid

# Usage
df_valid = check_bmi_distribution(df, 'bmi')
```

**BMI considerations:**
- Extreme values (BMI < 15 or > 60) are likely errors
- Underweight (BMI < 18.5) often has low n
- Consider BMI as confounder for metabolic conditions
- May need to stratify or adjust for obesity studies

---

## 2. Check Outcome Variables

### Binary Outcomes (Medical Conditions)

```python
def check_binary_outcomes(df, outcome_cols, min_n=100):
    """Check binary outcomes for adequate sample sizes."""

    print(f"=== BINARY OUTCOMES (min n={min_n}) ===\n")

    results = []
    for col in outcome_cols:
        n_pos = df[col].sum()
        n_neg = len(df) - n_pos
        prevalence = n_pos / len(df) * 100

        status = "✓" if n_pos >= min_n else "❌ EXCLUDE"
        results.append({
            'condition': col,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'prevalence': prevalence,
            'status': status
        })

    results_df = pd.DataFrame(results).sort_values('n_positive', ascending=False)

    valid = results_df[results_df['n_positive'] >= min_n]
    excluded = results_df[results_df['n_positive'] < min_n]

    print(f"Total conditions: {len(results_df)}")
    print(f"Valid (n≥{min_n}): {len(valid)}")
    print(f"Excluded (n<{min_n}): {len(excluded)}")

    if len(excluded) > 0:
        print(f"\nExcluded conditions:")
        for _, row in excluded.iterrows():
            print(f"  {row['condition']}: n={row['n_positive']}")

    return results_df

# Usage
outcome_cols = [c for c in df.columns if c not in ['age', 'gender', 'bmi']]
outcomes = check_binary_outcomes(df, outcome_cols, min_n=100)
```

---

## 3. Choosing Correct Ranges

### Decision Framework

```
┌─────────────────────────────────────────────────────────┐
│           DATA RANGE SELECTION FRAMEWORK                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. CHECK DISTRIBUTION                                  │
│     └─→ Plot histogram, identify where data exists      │
│                                                         │
│  2. IDENTIFY ADEQUATE SAMPLE SIZES                      │
│     └─→ n ≥ 200 per bin for stable estimates           │
│                                                         │
│  3. CONSIDER BIOLOGICAL PLAUSIBILITY                    │
│     └─→ Does the range make scientific sense?           │
│                                                         │
│  4. REPORT WHAT YOU EXCLUDED                            │
│     └─→ Transparency about data filtering               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Example: Age Range Selection

```python
def select_age_range(df, age_col='age', min_n_per_bin=200, bin_width=5):
    """Automatically select optimal age range."""

    # Create bins
    min_age = int(df[age_col].min() // bin_width * bin_width)
    max_age = int(df[age_col].max() // bin_width * bin_width) + bin_width
    bins = list(range(min_age, max_age + 1, bin_width))

    df['_bin'] = pd.cut(df[age_col], bins=bins, right=False)
    counts = df.groupby('_bin', observed=True).size()

    # Find contiguous range with adequate n
    adequate_bins = counts[counts >= min_n_per_bin].index.tolist()

    if len(adequate_bins) == 0:
        print("⚠️ No bins with adequate sample size")
        return None, None

    # Get range
    recommended_min = adequate_bins[0].left
    recommended_max = adequate_bins[-1].right

    # Calculate what we keep
    n_kept = df[(df[age_col] >= recommended_min) &
                (df[age_col] < recommended_max)].shape[0]
    pct_kept = n_kept / len(df) * 100

    print(f"Recommended age range: {recommended_min}-{recommended_max}")
    print(f"Subjects retained: {n_kept:,} ({pct_kept:.1f}%)")

    return recommended_min, recommended_max

# Usage
min_age, max_age = select_age_range(df, 'age', min_n_per_bin=200)
df_filtered = df[(df['age'] >= min_age) & (df['age'] < max_age)]
```

---

## 4. When to Stratify by Gender

### Decision Tree

```
Should I stratify by gender?
│
├─► Is the condition sex-specific?
│   (PCOS, prostate cancer, endometriosis)
│   └─► YES → Always stratify or analyze one sex only
│
├─► Is there known biological sex difference?
│   (CVD risk, autoimmune diseases, osteoporosis)
│   └─► YES → Stratify to examine differences
│
├─► Is the sample imbalanced?
│   (gender ratio < 0.8)
│   └─► YES → Consider stratifying
│
├─► Is this exploratory analysis?
│   └─► YES → Stratify to check for interactions
│
└─► Is this confirmatory analysis?
    └─► Adjust for gender in model OR stratify if interaction exists
```

### Code Template

```python
def should_stratify_by_gender(df, condition, gender_col='gender'):
    """Determine if gender stratification is needed."""

    # Check sample sizes per gender
    n_male = df[df[gender_col] == 1][condition].sum()
    n_female = df[df[gender_col] == 0][condition].sum()

    # Sex-specific conditions
    female_only = ['Polycystic Ovary Disease', 'Endometriosis', 'Breast Cancer']
    male_only = ['Erectile Dysfunction', 'Prostate Cancer']

    if condition in female_only:
        return "FEMALE_ONLY", "Sex-specific condition"
    if condition in male_only:
        return "MALE_ONLY", "Sex-specific condition"

    # Check if one gender has no cases
    if n_male < 10:
        return "FEMALE_ONLY", f"Too few male cases (n={n_male})"
    if n_female < 10:
        return "MALE_ONLY", f"Too few female cases (n={n_female})"

    # Check ratio
    ratio = min(n_male, n_female) / max(n_male, n_female)
    if ratio < 0.5:
        return "STRATIFY", f"Large imbalance (ratio={ratio:.2f})"

    return "POOLED_OK", f"Balanced (M:{n_male}, F:{n_female})"

# Usage
for condition in conditions:
    decision, reason = should_stratify_by_gender(df, condition)
    print(f"{condition}: {decision} - {reason}")
```

---

## 5. Complete Quality Check Pipeline

```python
def run_data_quality_check(df, age_col='age', gender_col='gender',
                            bmi_col='bmi', outcome_cols=None, min_n=100):
    """Run complete data quality check."""

    print("="*60)
    print("DATA QUALITY CHECK")
    print("="*60)

    # 1. Age
    print("\n" + "-"*40)
    age_counts = check_age_distribution(df, age_col)

    # 2. Gender
    print("\n" + "-"*40)
    gender_counts = check_gender_distribution(df, gender_col)

    # 3. BMI (if available)
    if bmi_col in df.columns:
        print("\n" + "-"*40)
        check_bmi_distribution(df, bmi_col)

    # 4. Outcomes
    if outcome_cols:
        print("\n" + "-"*40)
        outcomes = check_binary_outcomes(df, outcome_cols, min_n)

    # 5. Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    # Age range
    min_age, max_age = select_age_range(df, age_col)

    print(f"\n1. Filter to ages {min_age}-{max_age}")
    print(f"2. Use minimum n={min_n} for outcomes")
    print(f"3. Check gender stratification for sex-related conditions")

    return {
        'recommended_age_range': (min_age, max_age),
        'gender_ratio': gender_counts.min() / gender_counts.max(),
        'n_valid_outcomes': len(outcomes[outcomes['n_positive'] >= min_n]) if outcome_cols else None
    }

# Usage
quality = run_data_quality_check(
    df,
    age_col='age',
    gender_col='gender',
    bmi_col='bmi',
    outcome_cols=condition_cols,
    min_n=100
)
```

---

## Quick Reference Card

| Check | Threshold | Action if Failed |
|-------|-----------|------------------|
| Age bin sample | n < 200 | Exclude age range |
| Gender ratio | < 0.8 | Consider stratifying |
| BMI valid range | 15-60 | Remove outliers |
| Outcome prevalence | n < 100 | Exclude condition |
| Missing data | > 10% | Investigate or impute |

---

## HPP 10K Specific Notes

Based on the HPP 10K dataset:
- **Age:** Use 40-70 years (94% of data)
- **Gender:** Roughly balanced (52% F, 48% M)
- **Medical conditions:** 47 conditions have n ≥ 100
- **Always stratify:** PCOS, Endometriosis, Breast Cancer, Gout
