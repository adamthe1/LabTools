#!/usr/bin/env python
"""Generate PDF report for Age and Medical Conditions Analysis."""
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

def add_text_page(pdf, title, content, fontsize=11):
    """Add a text page to PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, title, ha='center', fontsize=16, fontweight='bold')
    fig.text(0.05, 0.88, content, ha='left', va='top', fontsize=fontsize,
             family='monospace', wrap=True, transform=fig.transFigure)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_image_page(pdf, image_path, title):
    """Add an image page to PDF."""
    if not os.path.exists(image_path):
        return
    fig = plt.figure(figsize=(11, 8.5))
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_table_page(pdf, df, title, cols=None):
    """Add a table page to PDF."""
    if cols:
        df = df[cols].copy()

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print("Generating PDF report...")

# Load results
results = pd.read_csv(os.path.join(OUT, 'results.csv'))

# Create PDF
pdf_path = os.path.join(OUT, 'Age_Medical_Conditions_Report.pdf')
with PdfPages(pdf_path) as pdf:

    # Title page
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.6, 'Age and Medical Conditions\nAnalysis Report',
             ha='center', fontsize=28, fontweight='bold')
    fig.text(0.5, 0.45, 'HPP 10K Dataset', ha='center', fontsize=18)
    fig.text(0.5, 0.38, 'Ages 40-70  |  N = 23,473', ha='center', fontsize=14)
    fig.text(0.5, 0.25, 'Key Findings:', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.20, '47 conditions with significant age associations', ha='center', fontsize=12)
    fig.text(0.5, 0.16, '28 increase with age  |  19 decrease with age', ha='center', fontsize=12)
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)

    # Executive Summary
    summary = """
EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━

This report analyzes the relationship between age and the log probability
of having various medical conditions in the HPP 10K cohort.

STUDY DESIGN:
• Cross-sectional analysis
• Age range: 40-70 years (94% of cohort)
• Sample size: 23,473 subjects
• 79 medical conditions analyzed

KEY FINDINGS:

Strongest INCREASES with age:
• Atrial Fibrillation (OR/decade: 3.82)
• Atherosclerotic Disease (OR/decade: 3.44)
• Osteoarthritis (OR/decade: 2.50)
• Glaucoma (OR/decade: 2.42)
• Ischemic Heart Disease (OR/decade: 2.39)
• Hypertension (OR/decade: 2.36)

Strongest DECREASES with age:
• Polycystic Ovary Disease (OR/decade: 0.53)
• Allergies (OR/decade: 0.71)
• Endometriosis (OR/decade: 0.67)
• ADHD (OR/decade: 0.86)
• Depression (OR/decade: 0.83)

INTERPRETATION: Cardiovascular and sensory conditions show expected
age-related increases. Mental health decreases may reflect cohort
effects (different diagnosis rates by generation) or survival bias.
"""
    add_text_page(pdf, '', summary, fontsize=10)

    # Top increasing conditions table
    top_inc = results[results['coef'].notna()].nlargest(15, 'coef')[
        ['condition', 'group', 'OR10', 'n_pos', 'p']
    ].copy()
    top_inc['OR10'] = top_inc['OR10'].round(2)
    top_inc['p'] = top_inc['p'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    top_inc.columns = ['Condition', 'Group', 'OR/Decade', 'N Positive', 'P-value']
    add_table_page(pdf, top_inc, 'Top 15 Conditions INCREASING with Age')

    # Top decreasing conditions table
    top_dec = results[(results['coef'].notna()) & (results['coef'] < 0)].nsmallest(15, 'coef')[
        ['condition', 'group', 'OR10', 'n_pos', 'p']
    ].copy()
    top_dec['OR10'] = top_dec['OR10'].round(2)
    top_dec['p'] = top_dec['p'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    top_dec.columns = ['Condition', 'Group', 'OR/Decade', 'N Positive', 'P-value']
    add_table_page(pdf, top_dec, 'Top 15 Conditions DECREASING with Age')

    # Main coefficient figure
    add_image_page(pdf, os.path.join(FIG, 'coefficients.png'),
                   'Age Coefficients for All Conditions')

    # Top conditions prevalence
    add_image_page(pdf, os.path.join(FIG, 'top_conditions.png'),
                   'Prevalence by Age: Top Increasing and Decreasing Conditions')

    # Age distribution
    add_image_page(pdf, os.path.join(FIG, 'age_dist.png'),
                   'Age Distribution (40-70)')

    # Disease group figures
    for group in ['Cardiovascular', 'Metabolic_Endocrine', 'Mental Health',
                  'Musculoskeletal', 'Sensory', 'Autoimmune', 'GI']:
        fpath = os.path.join(FIG, f'prevalence_{group}.png')
        if os.path.exists(fpath):
            add_image_page(pdf, fpath, f'{group}: Prevalence by Age')

    # Limitations page
    limitations = """
CRITICAL ANALYSIS: LIMITATIONS AND BIASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SURVIVAL BIAS (Major Concern)
   • Severe conditions may cause earlier death
   • Reduces apparent prevalence in older groups
   • Affected: Heart valve disease, potentially all severe CVD

2. COHORT/GENERATION EFFECTS
   • Different generations have different diagnosis rates
   • Affected: ADHD (not diagnosed in older generations)
   • Affected: Allergies (more environmental allergens recently)

3. DETECTION/DIAGNOSIS BIAS
   • Older adults may seek more medical care → more diagnoses
   • Younger adults may underreport symptoms
   • Counter: Some conditions (PCOS) naturally resolve with age

4. HEALTHY VOLUNTEER BIAS
   • HPP participants likely healthier than general population
   • May underestimate true disease prevalence

5. CROSS-SECTIONAL DESIGN
   • Cannot establish causality
   • Cannot track individual disease trajectories
   • Solution: Longitudinal follow-up studies needed

6. CONFOUNDING NOT CONTROLLED
   • Gender (many conditions sex-specific)
   • BMI (major confounder)
   • Socioeconomic status
   • Smoking, medications

7. SAMPLE SIZE FOR RARE CONDITIONS
   • Conditions with <50 cases have unstable estimates
   • Wide confidence intervals
   • Results should be interpreted cautiously
"""
    add_text_page(pdf, '', limitations, fontsize=10)

    # Methodology page
    methodology = """
METHODOLOGY
━━━━━━━━━━━━

STATISTICAL APPROACH:
• Model: Logistic regression
• Predictor: Age (continuous, years)
• Outcome: Binary disease indicator (0/1)
• Coefficient: Log odds change per year of age
• OR per decade: exp(coefficient × 10)

INCLUSION CRITERIA:
• Age 40-70 years
• Complete data for condition
• Minimum 20 positive cases for regression
• Restricted to reliable age range (94% of data)

DISEASE GROUPINGS:
Based on WHO framework for noncommunicable diseases:
• Cardiovascular: Hypertension, IHD, AF, etc.
• Cancer: Breast, Melanoma, Lymphoma
• Respiratory: Asthma, COPD
• Metabolic: Diabetes, Hyperlipidemia, Obesity
• Mental Health: Depression, Anxiety, ADHD
• Musculoskeletal: Osteoarthritis, Back Pain
• Gastrointestinal: IBS, IBD, Peptic Ulcer
• Autoimmune: Allergies, Psoriasis

SIGNIFICANCE LEVELS:
* p < 0.05
** p < 0.01
*** p < 0.001

DATA SOURCE:
Human Phenotype Project (HPP) 10K Dataset
Medical conditions from self-report and clinical records
"""
    add_text_page(pdf, '', methodology, fontsize=10)

    # Final summary
    final = """
SUMMARY AND RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY TAKEAWAYS:

1. Cardiovascular conditions (AF, atherosclerosis, hypertension, IHD)
   show the strongest positive associations with age - consistent
   with WHO classification as major aging-related NCDs.

2. Sensory conditions (glaucoma, hearing loss, retinal detachment)
   increase substantially with age - expected findings.

3. Mental health conditions show DECREASING prevalence with age,
   likely due to cohort effects (ADHD) and survival bias (depression).

4. Reproductive conditions (PCOS, endometriosis) decrease due to
   natural resolution post-menopause.


RECOMMENDATIONS FOR FUTURE ANALYSIS:

1. Stratify by gender for sex-specific conditions
2. Adjust for BMI and other confounders
3. Use age-period-cohort models to separate effects
4. Conduct longitudinal analysis when data available
5. Increase sample size at age extremes
6. Validate against external population cohorts


FILES GENERATED:
• REPORT.md - Full markdown report
• Age_Medical_Conditions_Report.pdf - This document
• results.csv - All condition coefficients
• figures/*.png - Individual figures


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Report generated from HPP 10K Dataset
Analysis restricted to ages 40-70 (N=23,473)
"""
    add_text_page(pdf, '', final, fontsize=10)

print(f"PDF saved: {pdf_path}")
