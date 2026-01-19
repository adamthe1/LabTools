#!/usr/bin/env python
"""Generate PDF report focused on -log(incidence) plots for all conditions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import os
from PIL import Image

OUT_DIR = '/home/adamgab/PycharmProjects/LabTools/age_medical_conditions_analysis/incident_analysis'
FIG_DIR = os.path.join(OUT_DIR, 'figures_clean')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
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
        plt.title(title, fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return True


print("Generating PDF report...")

# Load summary data
summary = pd.read_csv(os.path.join(OUT_DIR, 'clean_summary_40_70.csv'))

pdf_path = os.path.join(OUT_DIR, 'Age_Incidence_Report_Clean.pdf')

with PdfPages(pdf_path) as pdf:

    # ===== PAGE 1: METHODS =====
    fig = plt.figure(figsize=(8.5, 11))

    methods = """
AGE AT INCIDENT - CLEANED ANALYSIS
═══════════════════════════════════════════════════════════════════

Data Cleaning Applied
• Excluded placeholder dates (2000-01-01)
• Limited to ages 40-70 years
• Required age at diagnosis ≤ current age
• Required minimum 20 valid diagnoses per condition

Population at Risk Calculation
• Incidence rate = Diagnoses at age X / Population who reached age X
• This corrects for the fact that not everyone in the dataset
  has lived through all age bins yet

Why log₁₀(Incidence)?
• Log transformation makes rates comparable across conditions
• Intuitive interpretation: Line goes UP = incidence goes UP

How to Read the Plots
• X-axis: Age at diagnosis (40-70 years)
• Y-axis: log₁₀(incidence rate)
• HIGH values (top of plot) = HIGH incidence (common at this age)
• LOW values (bottom of plot) = LOW incidence (rare at this age)

• FLAT LINE = Constant incidence across age
• INCREASING LINE = Incidence INCREASES with age (more diagnoses)
• DECREASING LINE = Incidence DECREASES with age (fewer diagnoses)

Sample Sizes by Age Bin (Population at Risk)
• Age 40-42: 28,547 subjects
• Age 50-52: 16,950 subjects
• Age 60-62: 6,128 subjects
• Age 68-70: 1,504 subjects

Conditions Analyzed: 53 (with n≥20 valid diagnoses in 40-70 range)

"""

    fig.text(0.06, 0.95, methods, ha='left', va='top', fontsize=10.5,
             family='monospace', linespacing=1.4)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Page 1: Methods")

    # ===== PAGES 2-6: LOG INCIDENCE PLOTS (ALL CONDITIONS) =====
    n_pages = 5
    for page in range(1, n_pages + 1):
        path = os.path.join(FIG_DIR, f'neg_log_incidence_page{page}.png')
        if os.path.exists(path):
            add_image_page(pdf, path, f'log₁₀(Incidence Rate) by Age (Page {page}/{n_pages})')
            print(f"  Page {page+1}: log Incidence Page {page}")

    # ===== PAGE 7: HEATMAP =====
    add_image_page(pdf, os.path.join(FIG_DIR, 'heatmap_all_conditions.png'),
                   'All Conditions Heatmap')
    print("  Page 7: Heatmap")

    # ===== PAGE 8: SUMMARY BAR CHART =====
    add_image_page(pdf, os.path.join(FIG_DIR, 'mean_age_all_conditions.png'),
                   'Mean Age at Diagnosis Summary')
    print("  Page 8: Summary")

    # ===== PAGE 9: KEY FINDINGS =====
    fig = plt.figure(figsize=(8.5, 11))

    # Calculate some statistics
    early_onset = summary.nsmallest(10, 'mean_age')
    late_onset = summary.nlargest(10, 'mean_age')

    findings = """
KEY FINDINGS: Age at Diagnosis (40-70 years)
═══════════════════════════════════════════════════════════════════

CONDITIONS WITH EARLIEST ONSET (in 40-70 range)
These conditions are diagnosed earlier in life:

"""
    for _, row in early_onset.iterrows():
        findings += f"  {row['condition'][:30]:<32} Mean: {row['mean_age']:.0f} years  (n={int(row['n_valid'])})\n"

    findings += """

CONDITIONS WITH LATEST ONSET (in 40-70 range)
These conditions are diagnosed later in life:

"""
    for _, row in late_onset.iterrows():
        findings += f"  {row['condition'][:30]:<32} Mean: {row['mean_age']:.0f} years  (n={int(row['n_valid'])})\n"

    findings += """

INTERPRETATION NOTES
• Atrial Fibrillation: Strong late-onset (mean 58 years)
• Atherosclerotic disease: Strong late-onset (mean 58 years)
• Ischemic Heart Disease: Late-onset (mean 55 years)
• Endometriosis: Earliest in 40-70 range (diagnosed before 50)
• Migraine: Earlier onset (diagnosed in 40s)

HOW TO READ THE LOG PLOTS:
• Line goes UP = More diagnoses at older ages
• Line goes DOWN = Fewer diagnoses at older ages
• Flat line = Constant risk across ages 40-70

"""

    fig.text(0.05, 0.95, findings, ha='left', va='top', fontsize=10,
             family='monospace', linespacing=1.35)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Page 9: Key Findings")

    # ===== PAGE 10: SUMMARY TABLE =====
    fig = plt.figure(figsize=(8.5, 11))

    table_text = "ALL CONDITIONS SUMMARY (Ages 40-70)\n"
    table_text += "═" * 65 + "\n\n"
    table_text += f"{'Condition':<32} {'N':>6} {'Mean':>6} {'Median':>6} {'Std':>6}\n"
    table_text += "-" * 65 + "\n"

    for _, row in summary.iterrows():
        name = row['condition'][:30]
        table_text += f"{name:<32} {int(row['n_valid']):>6} {row['mean_age']:>6.1f} {row['median_age']:>6.1f} {row['std_age']:>6.1f}\n"

    fig.text(0.03, 0.98, table_text, ha='left', va='top', fontsize=8.5,
             family='monospace')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Page 10: Summary Table")

print(f"\nSaved: {pdf_path}")
