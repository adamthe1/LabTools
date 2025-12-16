# Statistical Methods

This document explains the statistical approach used in the Biological Age Analysis library.

## Overview: What Are We Trying to Answer?

**Core question:** Which biomarkers differ between people who are "biologically older" vs "biologically younger" than their chronological age?

We use a machine learning model to predict biological age from various features. Subjects whose predicted age is **higher** than their chronological age are considered "biologically older" (accelerated aging), while those predicted **younger** are "biologically younger" (decelerated aging).

By comparing biomarker levels between these groups, we identify features associated with accelerated or decelerated aging.

---

## Step 1: Age-Matched Binning

### The Problem with Raw Comparisons

Simply comparing "high predicted age" vs "low predicted age" subjects is confounded by actual age—older people naturally have higher predicted ages. We need to compare subjects **within the same chronological age range**.

### Solution: Stratified Percentile Selection

1. **Divide subjects into age bins** (e.g., 40-43, 44-47, 48-51, ...)
2. **Within each bin**, identify:
   - **Top percentile**: Subjects with highest predicted age (predicted older than peers)
   - **Bottom percentile**: Subjects with lowest predicted age (predicted younger than peers)
3. **Aggregate** top/bottom groups across all bins

```
Example with 4-year bins and 25% percentile:

Age Bin 40-43 (n=100):
  └─ Top 25%: 25 subjects with highest predicted age
  └─ Bottom 25%: 25 subjects with lowest predicted age

Age Bin 44-47 (n=120):
  └─ Top 25%: 30 subjects with highest predicted age
  └─ Bottom 25%: 30 subjects with lowest predicted age

... (repeat for all bins)

Final groups:
  └─ "Biologically Old": All top percentile subjects aggregated
  └─ "Biologically Young": All bottom percentile subjects aggregated
```

This ensures we're comparing subjects of similar chronological age, isolating the effect of biological aging.

---

## Step 2: Feature Standardization

Before statistical testing, features are **z-score standardized** on the whole population:

```
z = (x - μ) / σ
```

Where:
- `x` = raw feature value
- `μ` = population mean
- `σ` = population standard deviation

### Why Standardize?

1. **Comparable effect sizes**: A 0.3 difference in z-score means the same thing across all features
2. **No unit dependency**: Effect size doesn't depend on whether we measure in mg/dL or mmol/L
3. **Interpretable**: Effect size is in standard deviation units (ΔSD)

---

## Step 3: Statistical Testing

### Mann-Whitney U Test (Wilcoxon Rank-Sum)

For each feature, we compare the "Old" vs "Young" groups using the **Mann-Whitney U test**:

- **Non-parametric**: No assumption of normal distribution
- **Robust**: Works well with outliers and skewed data
- **Two-sided**: Detects differences in either direction

```
H₀: The distributions of the two groups are equal
H₁: The distributions differ
```

The test returns a **p-value** indicating the probability of observing such a difference by chance.

---

## Step 4: Multiple Testing Correction (FDR)

### The Multiple Testing Problem

When testing thousands of features, some will appear significant by chance. With α=0.05 and 1000 tests, we expect ~50 false positives.

### Solution: Benjamini-Hochberg FDR Correction

We control the **False Discovery Rate (FDR)** using the Benjamini-Hochberg procedure:

1. Sort all p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Find the largest k where: pₖ ≤ (k/m) × α
3. Reject all hypotheses with p ≤ pₖ

This controls the **expected proportion of false discoveries** among all discoveries.

### FDR Modes

The library supports two FDR correction modes:

| Mode | Description | Use When |
|------|-------------|----------|
| `per_system` | Apply FDR separately to each body system | Exploratory analysis, want to find top features per system |
| `all` | Apply FDR across all features combined | Strict control, comparing across systems |

```python
# Per-system FDR (default)
config = BiologicalAgeConfig(run_fdr_on='per_system')

# Global FDR
config = BiologicalAgeConfig(run_fdr_on='all')
```

---

## Step 5: Effect Size (ΔSD)

The effect size is the **difference in standardized means** between groups:

```
ΔSD = mean(Old group) - mean(Young group)
```

### Interpretation

| ΔSD | Interpretation |
|-----|----------------|
| +0.2 | Small increase in "Old" group |
| +0.5 | Medium increase in "Old" group |
| +0.8 | Large increase in "Old" group |
| -0.3 | Moderate decrease in "Old" group |

Positive ΔSD means the feature is **elevated** in the biologically older group.

---

## Step 6: Volcano Plot Visualization

The volcano plot displays all features simultaneously:

- **X-axis**: Effect size (ΔSD)
- **Y-axis**: Statistical significance (-log₁₀ p-value)

```
                     ↑ More significant
                     │
         ●           │           ●
    (decreased)      │      (increased)
                     │
    ─────────────────┼─────────────────→
                     │              Effect size
         Not         │
      significant    │
                     │
```

### Significance Thresholds

- **Horizontal dashed line**: Nominal α threshold (p < 0.05)
- **Horizontal dash-dot line**: FDR-corrected threshold
- **Colored points**: Features passing FDR correction
  - 🔴 Red: Higher in "Old" group (upregulated)
  - 🟢 Green: Lower in "Old" group (downregulated)

---

## Summary of Statistical Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA                               │
│  Predictions: (subject, real_age, predicted_age)            │
│  Features: (subject, feature1, feature2, ...)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: AGE-MATCHED BINNING                    │
│  • Divide into chronological age bins                       │
│  • Select top/bottom percentiles within each bin            │
│  • Aggregate across bins                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: STANDARDIZATION                        │
│  • Z-score all features on whole population                 │
│  • Enables comparable effect sizes                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: STATISTICAL TESTING                    │
│  • Mann-Whitney U test for each feature                     │
│  • Compare "Old" vs "Young" distributions                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 4: FDR CORRECTION                         │
│  • Benjamini-Hochberg procedure                             │
│  • Control false discovery rate at α                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 5: EFFECT SIZE                            │
│  • ΔSD = difference in standardized means                   │
│  • Positive = higher in "Old" group                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: VOLCANO PLOT                           │
│  • X: Effect size (ΔSD)                                     │
│  • Y: Significance (-log₁₀ p)                               │
│  • Colored by significance after FDR                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_width` | 4 | Years per age bin |
| `percentile` | 0.25 | Fraction for top/bottom groups (25%) |
| `alpha` | 0.05 | FDR significance threshold |
| `fc_threshold` | 0.0 | Minimum effect size for significance |
| `run_fdr_on` | 'per_system' | FDR correction scope |

---

## References

1. **Mann-Whitney U test**: Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50-60.

2. **Benjamini-Hochberg FDR**: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B*, 57(1), 289-300.

3. **Effect size interpretation**: Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
