# Age and Medical Conditions Analysis Report
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

- **Total records**: 24,929
- **Unique subjects**: 13,492
- **Age range**: -1.0 - 81.0 years
- **Mean age**: 52.8 +/- 9.2 years
- **Medical conditions analyzed**: 79

---

## Key Findings

### 1. Conditions that Increase Most with Age

| Condition | Disease Group | Log OR/year | OR per Decade | N Positive | P-value |
|-----------|---------------|-------------|---------------|------------|----------|
| Atrial Fibrillation | Cardiovascular | 0.1280 | 3.60 | 84 | 8.72e-21 |
| Atherosclerotic | Cardiovascular | 0.1190 | 3.29 | 85 | 4.62e-19 |
| Hyperparathyroidism | Metabolic/Endocrine | 0.1003 | 2.73 | 50 | 2.11e-09 |
| Ischemic Heart Disease | Cardiovascular | 0.0886 | 2.43 | 313 | 1.51e-40 |
| Glaucoma | Eye/Ear/Sensory | 0.0861 | 2.37 | 175 | 1.36e-22 |
| Hypertension | Cardiovascular | 0.0854 | 2.35 | 2,713 | 1.03e-263 |
| Erectile Dysfunction | Other | 0.0853 | 2.35 | 108 | 2.14e-14 |
| Osteoarthritis | Musculoskeletal | 0.0841 | 2.32 | 672 | 9.81e-76 |
| Melanoma | Cancer | 0.0826 | 2.28 | 53 | 1.89e-07 |
| Retinal detachment | Eye/Ear/Sensory | 0.0694 | 2.00 | 211 | 1.07e-18 |

**Interpretation**: For each year of age, the log odds of these conditions increase
by the coefficient shown. The "OR per Decade" shows the multiplicative increase in
odds for every 10 years of age.

### 2. Conditions that Decrease with Age

| Condition | Disease Group | Log OR/year | OR per Decade | N Positive | P-value |
|-----------|---------------|-------------|---------------|------------|----------|
| Endometriosis and Adenomyosis | Other | -0.0880 | 0.41 | 410 | 4.07e-71 |
| Celiac | Gastrointestinal | -0.0591 | 0.55 | 66 | 1.26e-06 |
| Lactose Intolerance | Other | -0.0534 | 0.59 | 85 | 1.10e-06 |
| IBD | Gastrointestinal | -0.0484 | 0.62 | 57 | 3.47e-04 |
| Polycystic Ovary Disease | Other | -0.0461 | 0.63 | 619 | 3.59e-27 |
| Iron Defficiency | Other | -0.0416 | 0.66 | 55 | 2.93e-03 |
| Anxiety | Neurodegenerative/Mental | -0.0366 | 0.69 | 786 | 1.82e-21 |
| Allergy | Autoimmune/Inflammatory | -0.0347 | 0.71 | 4,559 | 2.53e-81 |
| IBS | Gastrointestinal | -0.0325 | 0.72 | 807 | 1.53e-17 |
| Anemia | Other | -0.0245 | 0.78 | 1,136 | 7.55e-14 |

**Interpretation**: These conditions show decreasing prevalence with age, possibly
due to survival bias (patients with severe conditions dying earlier), cohort effects
(younger generations having higher diagnosis rates), or genuine age-related decrease.

---

## Disease Group Analysis

### Overall Group Associations with Age

| Disease Group | Log OR/year | OR per Decade | N Conditions | N Positive |
|---------------|-------------|---------------|--------------|------------|
| Cardiovascular | 0.0797 | 2.22 | 7 | 3,351 |
| Eye/Ear/Sensory | 0.0654 | 1.92 | 5 | 1,317 |
| Metabolic/Endocrine | 0.0556 | 1.74 | 11 | 8,111 |
| Cancer | 0.0412 | 1.51 | 3 | 246 |
| Musculoskeletal | 0.0177 | 1.19 | 6 | 6,751 |
| Gastrointestinal | 0.0103 | 1.11 | 8 | 7,726 |
| Renal/Urological | 0.0092 | 1.10 | 3 | 2,965 |
| Respiratory | -0.0122 | 0.88 | 2 | 1,228 |
| Neurodegenerative/Mental | -0.0215 | 0.81 | 7 | 6,238 |
| Autoimmune/Inflammatory | -0.0270 | 0.76 | 7 | 5,793 |

---

## Detailed Results by Disease Category


### Cardiovascular

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| Atrial Fibrillation | 0.1280 | 84 | 8.72e-21 | Yes |
| Atherosclerotic | 0.1190 | 85 | 4.62e-19 | Yes |
| Ischemic Heart Disease | 0.0886 | 313 | 1.51e-40 | Yes |
| AV. Conduction Disorder | 0.0880 | 41 | 1.24e-06 | Yes |
| Hypertension | 0.0854 | 2,713 | 1.03e-263 | Yes |
| Myocarditis | 0.0098 | 39 | 5.78e-01 | No |
| Heart valve disease | 0.0057 | 310 | 3.59e-01 | No |

### Cancer

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| Melanoma | 0.0826 | 53 | 1.89e-07 | Yes |
| Breast Cancer | 0.0312 | 177 | 1.88e-04 | Yes |
| Lymphoma | 0.0141 | 18 | 5.87e-01 | No |

### Respiratory

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| COPD | 0.0948 | 32 | 4.80e-06 | Yes |
| Asthma | -0.0153 | 1,200 | 1.81e-06 | Yes |

### Metabolic/Endocrine

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| Hyperparathyroidism | 0.1003 | 50 | 2.11e-09 | Yes |
| Hyperlipidemia | 0.0635 | 5,318 | 6.00e-261 | Yes |
| Prediabetes | 0.0578 | 2,072 | 3.13e-107 | Yes |
| Thyroid Adenoma | 0.0468 | 13 | 1.30e-01 | No |
| Goiter | 0.0364 | 80 | 3.43e-03 | Yes |
| Diabetes | 0.0317 | 293 | 1.12e-06 | Yes |
| Hypercholesterolaemia | 0.0295 | 107 | 5.86e-03 | Yes |
| Fatty Liver Disease | 0.0272 | 1,341 | 2.09e-18 | Yes |
| Obesity | 0.0220 | 1,343 | 1.26e-12 | Yes |
| Hashimoto | 0.0051 | 136 | 5.88e-01 | No |
| G6PD | -0.0134 | 232 | 6.03e-02 | No |

### Neurodegenerative/Mental

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| Insomnia | 0.0190 | 363 | 1.10e-03 | Yes |
| Headache | -0.0179 | 576 | 8.64e-05 | Yes |
| Migraine | -0.0198 | 1,307 | 1.15e-10 | Yes |
| Depression | -0.0240 | 786 | 6.71e-10 | Yes |
| ADHD | -0.0242 | 3,921 | 4.65e-37 | Yes |
| Anxiety | -0.0366 | 786 | 1.82e-21 | Yes |
| PTSD | -0.0696 | 26 | 1.77e-04 | Yes |

### Musculoskeletal

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| Osteoarthritis | 0.0841 | 672 | 9.81e-76 | Yes |
| Meniscus Tears | 0.0410 | 103 | 1.87e-04 | Yes |
| Gout | 0.0399 | 219 | 1.27e-07 | Yes |
| Back Pain | 0.0129 | 4,529 | 1.01e-12 | Yes |
| Fractures | 0.0064 | 2,051 | 1.11e-02 | Yes |
| Fibromyalgia | -0.0087 | 272 | 1.90e-01 | No |

### Gastrointestinal

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| Peptic Ulcer Disease | 0.0335 | 453 | 1.91e-10 | Yes |
| Gallstone Disease | 0.0306 | 930 | 1.66e-16 | Yes |
| Haemorrhoids | 0.0097 | 4,698 | 4.30e-08 | Yes |
| Anal Fissure | -0.0036 | 1,443 | 2.22e-01 | No |
| IBS | -0.0325 | 807 | 1.53e-17 | Yes |
| IBD | -0.0484 | 57 | 3.47e-04 | Yes |
| Celiac | -0.0591 | 66 | 1.26e-06 | Yes |

### Autoimmune/Inflammatory

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| Uveitis | 0.0304 | 54 | 4.41e-02 | Yes |
| Psoriasis | 0.0106 | 660 | 1.45e-02 | Yes |
| FMF | 0.0083 | 45 | 6.11e-01 | No |
| Vitiligo | -0.0022 | 147 | 8.09e-01 | No |
| Atopic Dermatitis | -0.0142 | 716 | 5.30e-04 | Yes |
| Allergy | -0.0347 | 4,559 | 2.53e-81 | Yes |

### Eye/Ear/Sensory

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| Ocular Hypertension | 0.0954 | 28 | 1.70e-05 | Yes |
| Glaucoma | 0.0861 | 175 | 1.36e-22 | Yes |
| Retinal detachment | 0.0694 | 211 | 1.07e-18 | Yes |
| Hearing loss | 0.0609 | 919 | 3.37e-57 | Yes |
| Tinnitus | 0.0370 | 54 | 1.45e-02 | Yes |

### Renal/Urological

| Condition | Log OR/year | N Positive | P-value | Significant |
|-----------|-------------|------------|---------|-------------|
| Urinary Tract Stones | 0.0234 | 697 | 3.37e-08 | Yes |
| Renal Stones | 0.0195 | 112 | 6.20e-02 | No |
| Urinary tract infection | 0.0033 | 2,266 | 1.70e-01 | No |

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
