# Age and Medical Conditions Analysis Report
## HPP 10K Dataset - Ages 40-70

---

## Executive Summary

This report analyzes the relationship between age and the log probability (log odds) of having various medical conditions in the Human Phenotype Project (HPP) 10K cohort. The analysis was **restricted to ages 40-70** (N=23,473) where 94% of the data resides, ensuring adequate statistical power.

**Key Findings:**
- 47 conditions showed statistically significant (p<0.05) associations with age
- 28 conditions **increase** with age; 19 conditions **decrease** with age
- Cardiovascular and sensory conditions show the strongest positive associations
- Allergic, mental health, and reproductive conditions show negative associations

---

## Background: Age-Disease Relationships

According to the WHO and epidemiological research:

1. **Four major NCD categories** account for 80% of premature deaths:
   - Cardiovascular diseases (19M deaths/year)
   - Cancers (10M deaths/year)
   - Chronic respiratory diseases (4M deaths/year)
   - Diabetes (2M deaths/year)

2. **True aging-related diseases** (exponential increase with age) include:
   - Dementia, stroke, ischemic heart disease (Group A diseases)
   - Characterized by increasing incidence throughout life

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total records | 23,473 |
| Age range | 40-70 years |
| Mean age | 54.3 ± 8.2 years |
| Medical conditions | 79 |
| Significant associations | 47 |

### Age Distribution
The analysis was limited to 40-70 because:
- 94% of subjects fall in this range
- Ages <40 and >75 have very low sample sizes (unreliable estimates)
- See `figures/age_dist.png`

---

## Key Findings

### 1. Conditions that INCREASE Most with Age

| Condition | OR per Decade | n Positive | P-value | Disease Group |
|-----------|---------------|------------|---------|---------------|
| Atrial Fibrillation | 3.82 | 69 | 5.6e-15 | Cardiovascular |
| Atherosclerotic | 3.44 | 72 | 4.7e-14 | Cardiovascular |
| COPD | 3.09 | 31 | 3.7e-06 | Respiratory |
| Erectile Dysfunction | 2.58 | 96 | 1.7e-12 | Other |
| Osteoarthritis | 2.50 | 611 | 1.2e-64 | Musculoskeletal |
| Glaucoma | 2.42 | 161 | 1.1e-17 | Sensory |
| Ischemic Heart Disease | 2.39 | 285 | 2.6e-29 | Cardiovascular |
| Hypertension | 2.36 | 2,494 | <1e-200 | Cardiovascular |
| Retinal Detachment | 2.23 | 200 | 1.8e-18 | Other |
| Hyperlipidemia | 1.91 | 5,013 | <1e-200 | Metabolic |

**Interpretation:** For each decade of age, the odds of atrial fibrillation increase by a factor of 3.82. This aligns with epidemiological literature showing cardiovascular conditions as "true aging-related diseases."

### 2. Conditions that DECREASE with Age

| Condition | OR per Decade | n Positive | P-value | Disease Group |
|-----------|---------------|------------|---------|---------------|
| Polycystic Ovary Disease | 0.53 | 582 | 6.5e-25 | Reproductive |
| Iron Deficiency | 0.55 | 53 | 0.003 | Hematologic |
| Lactose Intolerance | 0.63 | 75 | 0.005 | GI |
| Endometriosis | 0.67 | 278 | 1.7e-06 | Reproductive |
| Celiac | 0.69 | 54 | 0.047 | GI |
| Allergy | 0.71 | 4,220 | 4.9e-51 | Autoimmune |
| Hypercoagulability | 0.72 | 76 | 0.037 | Hematologic |
| Sinusitis | 0.79 | 752 | 2.0e-06 | Other |
| B12 Deficiency | 0.79 | 1,110 | 1.4e-08 | Hematologic |
| Anemia | 0.80 | 1,047 | 1.3e-07 | Hematologic |

---

## Disease Group Analysis

### Cardiovascular Diseases
The strongest age-associated group. Aligns with WHO classification as major NCDs.

| Condition | OR/Decade | Significance |
|-----------|-----------|--------------|
| Atrial Fibrillation | 3.82 | *** |
| Atherosclerotic | 3.44 | *** |
| Ischemic Heart Disease | 2.39 | *** |
| Hypertension | 2.36 | *** |
| Heart valve disease | 0.99 | NS |

### Metabolic/Endocrine
| Condition | OR/Decade | Significance |
|-----------|-----------|--------------|
| Diabetes | 1.88 | *** |
| Prediabetes | 1.77 | *** |
| Hyperlipidemia | 1.91 | *** |
| Fatty Liver Disease | 1.25 | *** |
| Obesity | 1.13 | ** |

### Mental Health
**Notable finding:** Most mental health conditions DECREASE with age in this cohort.

| Condition | OR/Decade | Significance |
|-----------|-----------|--------------|
| Insomnia | 1.39 | *** |
| Depression | 0.83 | *** |
| Migraine | 0.82 | *** |
| Anxiety | 0.81 | *** |
| ADHD | 0.86 | *** |
| Headache | 0.85 | ** |

### Musculoskeletal
| Condition | OR/Decade | Significance |
|-----------|-----------|--------------|
| Osteoarthritis | 2.50 | *** |
| Gout | 1.48 | *** |
| Fibromyalgia | 1.19 | * |
| Back Pain | 1.14 | *** |
| Fractures | 1.05 | NS |

### Sensory (Eye/Ear)
| Condition | OR/Decade | Significance |
|-----------|-----------|--------------|
| Glaucoma | 2.42 | *** |
| Retinal Detachment | 2.23 | *** |
| Hearing Loss | 1.70 | *** |
| Tinnitus | 1.30 | NS |

---

## Critical Analysis: Why These Findings May Be Misleading

### 1. Survival Bias (Major Concern)
- **Problem:** Severe conditions may cause earlier death, artificially reducing prevalence in older groups
- **Affected conditions:** Heart valve disease (OR=0.99), potentially cardiovascular conditions
- **Example:** If people with severe heart disease die before 70, survivors appear "healthier"

### 2. Cohort/Generation Effects
- **Problem:** Different generations may have different diagnosis rates or exposures
- **Affected conditions:** ADHD (OR=0.86), Allergies (OR=0.71)
- **Explanation:** ADHD wasn't widely diagnosed in older generations; increased environmental allergens may affect younger cohorts more

### 3. Detection/Diagnosis Bias
- **Problem:** Older adults may seek more medical care → more diagnoses
- **Affected conditions:** All conditions with high prevalence in older groups
- **Counter-argument:** Some conditions (PCOS, endometriosis) naturally resolve post-menopause

### 4. Healthy Volunteer Bias
- **Problem:** HPP participants may be healthier than general population
- **Impact:** May underestimate true disease prevalence, especially in older groups

### 5. Cross-Sectional Design Limitations
- **Problem:** Cannot establish causality or track individual trajectories
- **Solution needed:** Longitudinal follow-up studies

### 6. Sample Size Concerns for Rare Conditions
| Condition | N Positive | Reliability |
|-----------|------------|-------------|
| Atrial Fibrillation | 69 | Moderate |
| Lymphoma | 17 | Low |
| COPD | 31 | Low |
| Thyroid Adenoma | 13 | Low |

Conditions with <50 cases have unstable estimates and wide confidence intervals.

### 7. Confounding Variables Not Controlled
This analysis did not adjust for:
- Gender (many conditions are sex-specific)
- BMI (obesity correlates with many conditions)
- Socioeconomic status
- Smoking history
- Medication use

---

## Conditions Requiring Special Interpretation

### Polycystic Ovary Disease (OR=0.53)
- **Apparent finding:** Strong decrease with age
- **True explanation:** PCOS symptoms naturally diminish post-menopause; also female-only condition with changing demographics across age

### Allergies (OR=0.71)
- **Apparent finding:** Decreases with age
- **Possible explanations:**
  1. Immune system "calms down" with age
  2. Cohort effect: increased environmental allergens in recent decades
  3. Older adults less likely to seek diagnosis for allergies

### ADHD (OR=0.86)
- **Apparent finding:** Decreases with age
- **True explanation:** Primarily a cohort effect - ADHD wasn't diagnosed in older generations as children; doesn't mean ADHD resolves with age

### Depression (OR=0.83)
- **Apparent finding:** Decreases with age
- **Possible explanations:**
  1. Selection bias: depressed individuals may have higher mortality
  2. Older adults may underreport mental health symptoms
  3. May genuinely improve with life stage changes

---

## Findings That Align with Literature

These findings are well-supported by epidemiological research:

| Condition | Our OR/Decade | Expected | Alignment |
|-----------|---------------|----------|-----------|
| Hypertension | 2.36 | ↑↑ | ✓ |
| Ischemic Heart Disease | 2.39 | ↑↑ | ✓ |
| Atrial Fibrillation | 3.82 | ↑↑ | ✓ |
| Diabetes | 1.88 | ↑↑ | ✓ |
| Osteoarthritis | 2.50 | ↑↑ | ✓ |
| Glaucoma | 2.42 | ↑↑ | ✓ |
| Hearing Loss | 1.70 | ↑↑ | ✓ |

---

## Recommendations for Future Analysis

1. **Stratify by gender** - Many conditions are sex-specific
2. **Adjust for BMI** - Major confounder for metabolic conditions
3. **Use age-period-cohort models** - Separate aging from generational effects
4. **Longitudinal analysis** - Track individuals over time when data available
5. **Increase sample at age extremes** - Current data is sparse <40 and >70
6. **Validate against external cohorts** - Compare with other population studies

---

## Figures

| Figure | Description |
|--------|-------------|
| `figures/coefficients.png` | All conditions ranked by age coefficient |
| `figures/top_conditions.png` | Prevalence curves for top increasing/decreasing conditions |
| `figures/age_dist.png` | Age distribution (40-70) |
| `figures/prevalence_*.png` | Prevalence by age for each disease group |

---

## Methodology

### Statistical Approach
- **Model:** Logistic regression with age as continuous predictor
- **Outcome:** Binary disease indicator (0/1)
- **Coefficient:** Log odds change per year of age
- **OR per decade:** exp(coefficient × 10)

### Inclusion Criteria
- Age 40-70 years
- Complete data for condition
- Minimum 20 positive cases for regression

### Significance Levels
- \* p < 0.05
- \** p < 0.01
- \*** p < 0.001

---

## References

1. WHO. (2023). Noncommunicable diseases fact sheet.
2. Hou et al. (2022). What Is an Aging-Related Disease? An Epidemiological Perspective. J Gerontol A.
3. Chang et al. (2019). Age-related disease burden as a measure of population ageing. Lancet Public Health.

---

*Report generated: 2024*
*Data source: HPP 10K Dataset*
*Analysis limited to ages 40-70 (N=23,473)*
