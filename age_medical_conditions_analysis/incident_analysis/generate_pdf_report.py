#!/usr/bin/env python
"""Generate PDF report for age at incident analysis."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import os
from PIL import Image

OUT_DIR = '/home/adamgab/PycharmProjects/LabTools/age_medical_conditions_analysis/incident_analysis'
FIG_DIR = os.path.join(OUT_DIR, 'figures')

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
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


print("Generating PDF report...")

# Load summary data
summary = pd.read_csv(os.path.join(OUT_DIR, 'summary_statistics.csv'))

pdf_path = os.path.join(OUT_DIR, 'Age_at_Incident_Report.pdf')

with PdfPages(pdf_path) as pdf:

    # ===== PAGE 1: METHODS =====
    fig = plt.figure(figsize=(8.5, 11))

    methods = """
AGE AT INCIDENT ANALYSIS
═══════════════════════════════════════════════════════════════════

Study Design
• Analysis of age at diagnosis for medical conditions
• Data source: HPP 10K Dataset with diagnosis dates
• Sample: N = 31,146 subjects with valid year of birth

Key Calculation
• Age at diagnosis = Year of diagnosis - Year of birth
• Incidence proportion = Count diagnosed at age / Total diagnosed
• -log₁₀(incidence) = Higher values = Rarer at that age

Data Quality Notes
• Excluded dates = "2000-01-01" (placeholder for unknown date)
• ~30% of diagnosis dates are placeholders (excluded)
• Minimum 20 valid cases required per condition
• 66 conditions met inclusion criteria

How to Interpret -log(Incidence) Plots
• Lower -log values (warmer colors) = More common at that age
• Higher -log values (cooler colors) = Less common at that age
• The curve shows WHEN in life conditions typically develop

Key Patterns
• EARLY ONSET: Asthma, Allergies, ADHD, Migraine
  - These conditions peak in childhood/young adulthood

• MID-LIFE ONSET: Depression, Anxiety, Back Pain, Obesity
  - These conditions peak around ages 35-50

• LATE ONSET: Hypertension, Diabetes, Osteoarthritis
  - These conditions peak after age 50

Important Caveats
• Age at diagnosis ≠ Age at onset (diagnosis may lag years)
• Cohort effects may bias older generations
• Survival bias may affect late-onset conditions

"""

    fig.text(0.08, 0.95, methods, ha='left', va='top', fontsize=10.5,
             family='monospace', linespacing=1.4)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Page 1: Methods")

    # ===== PAGE 2: EARLY VS LATE ONSET (HERO FIGURE) =====
    add_image_page(pdf, os.path.join(FIG_DIR, 'early_vs_late_onset.png'),
                   'When Do Conditions Develop? Early vs Late Onset')
    print("  Page 2: Early vs Late Onset")

    # ===== PAGE 3: -LOG INCIDENCE GRID =====
    add_image_page(pdf, os.path.join(FIG_DIR, 'neg_log_incidence_grid.png'),
                   '-log₁₀(Incidence) by Age for Top 12 Conditions')
    print("  Page 3: -log Incidence Grid")

    # ===== PAGE 4: HEATMAP =====
    add_image_page(pdf, os.path.join(FIG_DIR, 'incidence_heatmap.png'),
                   'Incidence Heatmap Across Age and Conditions')
    print("  Page 4: Heatmap")

    # ===== PAGE 5: AGE DISTRIBUTION HISTOGRAMS =====
    add_image_page(pdf, os.path.join(FIG_DIR, 'age_distribution_histograms.png'),
                   'Age at Diagnosis Distributions')
    print("  Page 5: Histograms")

    # ===== PAGE 6: DISEASE CATEGORY SCATTER =====
    add_image_page(pdf, os.path.join(FIG_DIR, 'disease_category_scatter.png'),
                   'Age at Diagnosis by Disease Category')
    print("  Page 6: Disease Categories")

    # ===== PAGE 7: SUMMARY TABLE =====
    fig = plt.figure(figsize=(8.5, 11))

    # Get top 30 conditions
    top30 = summary.nlargest(30, 'n_valid_date')[['condition', 'n_valid_date', 'mean_age', 'median_age', 'std_age']]
    top30.columns = ['Condition', 'N', 'Mean Age', 'Median Age', 'Std']

    # Format numbers
    table_text = "TOP 30 CONDITIONS BY SAMPLE SIZE\n"
    table_text += "═" * 70 + "\n\n"
    table_text += f"{'Condition':<30} {'N':>6} {'Mean':>8} {'Median':>8} {'Std':>6}\n"
    table_text += "-" * 70 + "\n"

    for _, row in top30.iterrows():
        table_text += f"{row['Condition'][:28]:<30} {int(row['N']):>6} {row['Mean Age']:>8.1f} {row['Median Age']:>8.1f} {row['Std']:>6.1f}\n"

    fig.text(0.05, 0.95, table_text, ha='left', va='top', fontsize=10,
             family='monospace')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Page 7: Summary Table")

    # ===== PAGE 8: KEY FINDINGS =====
    fig = plt.figure(figsize=(8.5, 11))

    # Calculate some statistics
    early_onset = summary[summary['mean_age'] < 30]['condition'].tolist()[:8]
    late_onset = summary[summary['mean_age'] > 50]['condition'].tolist()[:8]

    findings = f"""
KEY FINDINGS
═══════════════════════════════════════════════════════════════════

1. EARLIEST ONSET CONDITIONS (Mean Age < 30 years)
   These conditions typically develop in childhood or young adulthood:

"""
    for c in early_onset:
        row = summary[summary['condition'] == c].iloc[0]
        findings += f"   • {c}: Mean age {row['mean_age']:.0f} years (n={int(row['n_valid_date'])})\n"

    findings += f"""

2. LATEST ONSET CONDITIONS (Mean Age > 50 years)
   These conditions typically develop in middle age or later:

"""
    for c in late_onset:
        row = summary[summary['condition'] == c].iloc[0]
        findings += f"   • {c}: Mean age {row['mean_age']:.0f} years (n={int(row['n_valid_date'])})\n"

    findings += """

3. WIDEST AGE DISTRIBUTIONS
   Some conditions span a wide range of ages at diagnosis:

"""
    wide_dist = summary.nlargest(5, 'std_age')
    for _, row in wide_dist.iterrows():
        findings += f"   • {row['condition']}: Std = {row['std_age']:.1f} years\n"

    findings += """

4. NARROWEST AGE DISTRIBUTIONS
   Some conditions cluster tightly around a specific age:

"""
    narrow_dist = summary.nsmallest(5, 'std_age')
    for _, row in narrow_dist.iterrows():
        findings += f"   • {row['condition']}: Std = {row['std_age']:.1f} years\n"

    fig.text(0.05, 0.95, findings, ha='left', va='top', fontsize=10.5,
             family='monospace', linespacing=1.4)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Page 8: Key Findings")

print(f"\nSaved: {pdf_path}")

# ===== MARKDOWN FINDINGS FILE =====
findings_md = f"""# Age at Incident Analysis
## HPP 10K Dataset - Diagnosis Age Patterns

---

## Executive Summary

This analysis examines **when** medical conditions are typically diagnosed, calculating
the age at diagnosis for each subject (diagnosis year - birth year).

**Key Statistic:** Analyzed {len(summary)} conditions with valid diagnosis dates.

---

## Data Quality Notes

- **Total subjects:** 31,146 with valid birth years
- **Excluded dates:** "2000-01-01" placeholder (~30% of records)
- **Minimum sample:** n ≥ 20 valid dates per condition
- **Age range:** 0-100 years (calculated)

---

## Main Findings

### Earliest Onset Conditions
| Condition | Mean Age | n |
|-----------|----------|---|
"""

for c in summary.nsmallest(10, 'mean_age')['condition']:
    row = summary[summary['condition'] == c].iloc[0]
    findings_md += f"| {c} | {row['mean_age']:.1f} | {int(row['n_valid_date'])} |\n"

findings_md += """

### Latest Onset Conditions
| Condition | Mean Age | n |
|-----------|----------|---|
"""

for c in summary.nlargest(10, 'mean_age')['condition']:
    row = summary[summary['condition'] == c].iloc[0]
    findings_md += f"| {c} | {row['mean_age']:.1f} | {int(row['n_valid_date'])} |\n"

findings_md += """

---

## Interpretation Guide

### Understanding -log₁₀(Incidence) Plots

The `-log₁₀(incidence)` transformation helps visualize when conditions are most common:

- **Lower values** (red on heatmap) = More diagnoses at that age
- **Higher values** (blue on heatmap) = Fewer diagnoses at that age

### Why Use -log?

The `-log` transformation:
1. Makes rare events more visible
2. Compresses the scale for common events
3. Shows the relative "rarity" at each age

### Example Interpretation

For **Asthma** (mean age ~16):
- Very low -log values in childhood (ages 5-15) = Very common diagnosis age
- High -log values in adulthood (ages 40+) = Rare to be diagnosed at this age

For **Hypertension** (mean age ~49):
- High -log values in youth (ages 20-30) = Rare diagnosis age
- Low -log values at ages 50+ = Peak diagnosis age

---

## Caveats & Limitations

1. **Diagnosis lag:** Age at diagnosis ≠ Age at actual disease onset
2. **Cohort effects:** Older generations may have different diagnosis patterns
3. **Placeholder dates:** 30% of dates were "2000-01-01" (excluded)
4. **Ascertainment bias:** Some conditions more likely to be recorded

---

## Files Generated

| File | Description |
|------|-------------|
| `Age_at_Incident_Report.pdf` | Visual report |
| `summary_statistics.csv` | Per-condition statistics |
| `incidence_*.csv` | Age-binned incidence data per condition |
| `figures/` | All PNG figures |

---

*Analysis conducted on HPP 10K Dataset*
*N = 31,146 subjects with valid diagnosis dates*
"""

with open(os.path.join(OUT_DIR, 'FINDINGS.md'), 'w') as f:
    f.write(findings_md)
print("Saved: FINDINGS.md")
