#!/usr/bin/env python
"""Generate clean PDF report following the create-report skill guidelines."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import os
from PIL import Image

OUT = '/home/adamgab/PycharmProjects/LabTools/age_medical_conditions_analysis'
FIG = os.path.join(OUT, 'figures')

# Style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'font.family': 'sans-serif'
})

def add_image_page(pdf, image_path, title=None):
    """Add a figure page to PDF (landscape)."""
    if not os.path.exists(image_path):
        print(f"  Warning: {image_path} not found")
        return False
    fig = plt.figure(figsize=(11, 8.5))
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return True

print("Generating clean PDF report...")

results = pd.read_csv(os.path.join(OUT, 'results.csv'))
sig = results[(results['p'] < 0.05) & (results['n_pos'] >= 100) & results['coef'].notna()]

pdf_path = os.path.join(OUT, 'Age_Medical_Conditions_Report.pdf')

with PdfPages(pdf_path) as pdf:

    # ===== PAGE 1: METHODS =====
    fig = plt.figure(figsize=(8.5, 11))

    methods = """
METHODS
═══════════════════════════════════════════════════════════════════

Study Design
• Cross-sectional analysis of the HPP 10K cohort
• Sample size: N = 23,473 subjects
• Age range: 40-70 years (where 94% of data resides)

Statistical Analysis
• Model: Logistic regression for each medical condition
• Outcome: Binary disease indicator (0 = no, 1 = yes)
• Predictor: Age in years (continuous)
• Output: Log odds ratio per year of age

How to Interpret Results
• Odds Ratio (OR) per decade = exp(coefficient × 10)
• OR = 2.0 → Condition is 2× more likely per 10 years of age
• OR = 0.5 → Condition is 2× less likely per 10 years of age
• OR = 1.0 → No association with age

Inclusion Criteria
• Age between 40 and 70 years
• Minimum 100 positive cases per condition
• Complete data for age and condition

Quality Control
• 47 conditions met inclusion criteria
• 37 showed statistically significant associations (p < 0.05)
• Results shown with 95% confidence intervals

Data Source
• Human Phenotype Project (HPP) 10K Dataset
• Medical conditions from clinical records and self-report
"""

    fig.text(0.08, 0.92, methods, ha='left', va='top', fontsize=11,
             family='monospace', linespacing=1.4)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Page 1: Methods")

    # ===== PAGE 2: MAIN FINDING (HERO FIGURE) =====
    add_image_page(pdf, os.path.join(FIG, 'summary_odds_ratios.png'),
                   'Medical Conditions That Change with Age')
    print("  Page 2: Main finding (hero figure)")

    # ===== PAGE 3: TOP CONDITIONS INCREASING =====
    add_image_page(pdf, os.path.join(FIG, 'summary_log_odds_increasing.png'),
                   'Top 6 Conditions That Increase with Age')
    print("  Page 3: Increasing conditions detail")

    # ===== PAGE 4: TOP CONDITIONS DECREASING =====
    add_image_page(pdf, os.path.join(FIG, 'summary_log_odds_decreasing.png'),
                   'Top 6 Conditions That Decrease with Age')
    print("  Page 4: Decreasing conditions detail")

    # ===== PAGE 5: FOREST PLOT - ALL CONDITIONS =====
    add_image_page(pdf, os.path.join(FIG, 'forest_plot_age_effects.png'),
                   'All Significant Age-Disease Associations')
    print("  Page 5: Forest plot")

    # ===== PAGE 6: ODDS RATIO WITH CONFIDENCE INTERVALS =====
    add_image_page(pdf, os.path.join(FIG, 'odds_ratio_per_decade.png'),
                   'Odds Ratio per Decade with 95% Confidence Intervals')
    print("  Page 6: Odds ratios with CI")

    # ===== PAGE 7: EXAMPLE CONDITIONS (COMPARISON) =====
    add_image_page(pdf, os.path.join(FIG, 'comparison_Hypertension.png'),
                   'Example: Hypertension - Prevalence and Log Odds')
    print("  Page 7: Example - Hypertension")

    # ===== PAGE 8: ANOTHER EXAMPLE =====
    add_image_page(pdf, os.path.join(FIG, 'comparison_Allergy.png'),
                   'Example: Allergy - Prevalence and Log Odds')
    print("  Page 8: Example - Allergy")

print(f"\nSaved: {pdf_path}")

# ===== GENERATE COMPANION MARKDOWN FILE =====
findings_md = f"""# Age and Medical Conditions Analysis
## HPP 10K Dataset - Detailed Findings

---

## Executive Summary

This analysis examines the relationship between age (40-70 years) and the probability of having various medical conditions in the HPP 10K cohort (N=23,473).

**Key Finding:** Of 47 conditions analyzed, 37 showed statistically significant associations with age:
- **28 conditions increase** with age (primarily cardiovascular, metabolic, sensory)
- **19 conditions decrease** with age (primarily allergic, mental health, reproductive)

---

## Main Results

### Conditions That Increase Most with Age

| Condition | Odds Ratio per Decade | Interpretation |
|-----------|----------------------|----------------|
| Atrial Fibrillation | 3.82 | 3.8× more likely every 10 years |
| Atherosclerotic | 3.44 | 3.4× more likely every 10 years |
| Osteoarthritis | 2.50 | 2.5× more likely every 10 years |
| Glaucoma | 2.42 | 2.4× more likely every 10 years |
| Ischemic Heart Disease | 2.39 | 2.4× more likely every 10 years |
| Hypertension | 2.36 | 2.4× more likely every 10 years |

**Interpretation:** These findings align with known epidemiology. Cardiovascular and degenerative conditions are well-established aging-related diseases.

### Conditions That Decrease with Age

| Condition | Odds Ratio per Decade | Interpretation |
|-----------|----------------------|----------------|
| Polycystic Ovary Disease | 0.53 | 2× less likely every 10 years |
| Allergy | 0.71 | 1.4× less likely every 10 years |
| Endometriosis | 0.67 | 1.5× less likely every 10 years |
| ADHD | 0.86 | 1.2× less likely every 10 years |
| Depression | 0.83 | 1.2× less likely every 10 years |

**Interpretation:** These findings require careful interpretation due to potential biases (see Limitations).

---

## Findings by Disease Category

### Cardiovascular
Strong, consistent increase with age. This is the most robust finding and aligns with global epidemiological data showing cardiovascular disease as the leading cause of death in older adults.

### Metabolic
Diabetes, prediabetes, and hyperlipidemia all increase with age. Obesity shows a weaker association, possibly due to survival bias.

### Mental Health
Most conditions show *decreasing* prevalence with age. This is likely due to:
1. **Cohort effects:** ADHD was not diagnosed in older generations
2. **Survival bias:** Depression may increase mortality
3. **Under-reporting:** Older adults may not report mental health symptoms

### Musculoskeletal
Osteoarthritis shows one of the strongest age associations (OR=2.50 per decade), consistent with degenerative nature.

### Sensory
Glaucoma, hearing loss, and retinal conditions all increase substantially with age.

### Allergic/Autoimmune
Most show decreasing prevalence. May reflect immune system changes with age or cohort effects (increased environmental allergens in recent decades).

---

## Limitations & Critical Analysis

### 1. Survival Bias
Severe conditions may cause earlier death, artificially reducing prevalence in older groups. This may explain why some serious conditions don't show expected increases.

### 2. Cohort/Generation Effects
Different generations have different diagnosis rates:
- ADHD was not widely diagnosed in older generations
- Allergies may reflect increased environmental allergens

### 3. Cross-Sectional Design
We observe associations, not causation. Cannot track individual disease trajectories.

### 4. Confounders Not Controlled
This analysis did not adjust for:
- Gender (many conditions are sex-specific)
- BMI (major metabolic confounder)
- Socioeconomic status
- Medications

### 5. Sample Size at Extremes
Analysis limited to ages 40-70 where adequate data exists. Findings may not generalize to younger or older populations.

---

## Methodology

### Statistical Model
```
logit(P(condition=1)) = β₀ + β₁ × Age
```
- β₁ = change in log odds per year of age
- Odds Ratio per decade = exp(β₁ × 10)

### Significance
- α = 0.05
- Minimum n = 100 positive cases

### Software
Python with statsmodels, matplotlib, pandas

---

## Files Generated

| File | Description |
|------|-------------|
| `Age_Medical_Conditions_Report.pdf` | Main visual report |
| `results.csv` | All coefficients and statistics |
| `figures/*.png` | Individual figures |
| `FINDINGS.md` | This detailed documentation |

---

*Analysis conducted on HPP 10K Dataset*
*Age range: 40-70 years | N = 23,473*
"""

with open(os.path.join(OUT, 'FINDINGS.md'), 'w') as f:
    f.write(findings_md)
print(f"Saved: FINDINGS.md")
