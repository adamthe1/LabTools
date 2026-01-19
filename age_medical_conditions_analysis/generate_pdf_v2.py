#!/usr/bin/env python
"""Generate comprehensive PDF report with clearer figures. Min n=100."""
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

MIN_N = 100  # Minimum positive cases

def add_image_page(pdf, image_path, title=None):
    """Add an image page to PDF."""
    if not os.path.exists(image_path):
        print(f"  Warning: {image_path} not found")
        return
    fig = plt.figure(figsize=(11, 8.5))
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_text_page(pdf, content, fontsize=10):
    """Add a text page to PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, content, ha='left', va='top', fontsize=fontsize,
             family='monospace', wrap=True, transform=fig.transFigure)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print("Generating PDF report (min n=100)...")

# Load results and filter
results = pd.read_csv(os.path.join(OUT, 'results.csv'))
results = results[results['n_pos'] >= MIN_N]  # Filter to n>=100
sig = results[(results['p'] < 0.05) & results['coef'].notna()]

print(f"Conditions with n>={MIN_N}: {len(results)}")
print(f"Significant (p<0.05): {len(sig)}")

pdf_path = os.path.join(OUT, 'Age_Medical_Conditions_Report.pdf')

with PdfPages(pdf_path) as pdf:

    # ===== TITLE PAGE =====
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.65, 'Age and Medical Conditions', ha='center', fontsize=32, fontweight='bold')
    fig.text(0.5, 0.55, 'Analysis Report', ha='center', fontsize=28, fontweight='bold')
    fig.text(0.5, 0.42, 'HPP 10K Dataset', ha='center', fontsize=20)
    fig.text(0.5, 0.35, 'Ages 40-70  |  N = 23,473', ha='center', fontsize=16)
    fig.text(0.5, 0.28, f'Conditions with n ≥ {MIN_N} cases', ha='center', fontsize=14)

    n_inc = (sig['coef'] > 0).sum()
    n_dec = (sig['coef'] < 0).sum()
    fig.text(0.5, 0.18, f'{len(sig)} significant associations', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.14, f'{n_inc} increase with age  |  {n_dec} decrease with age', ha='center', fontsize=12)
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)
    print("  Added: Title page")

    # ===== EXECUTIVE SUMMARY =====
    top_inc = sig[sig['coef'] > 0].nlargest(5, 'coef')
    top_dec = sig[sig['coef'] < 0].nsmallest(5, 'coef')

    summary = f"""
EXECUTIVE SUMMARY
{'='*50}

STUDY DESIGN
• Cross-sectional analysis of HPP 10K cohort
• Age range: 40-70 years (94% of cohort)
• Sample size: 23,473 subjects
• Minimum cases per condition: {MIN_N}

KEY FINDINGS

{len(sig)} conditions show significant age associations (p<0.05):
  • {n_inc} conditions INCREASE with age
  • {n_dec} conditions DECREASE with age


TOP 5 CONDITIONS INCREASING WITH AGE:
"""
    for _, r in top_inc.iterrows():
        summary += f"  • {r['condition']}: {np.exp(r['coef']*10):.2f}x odds per decade (n={int(r['n_pos'])})\n"

    summary += "\n\nTOP 5 CONDITIONS DECREASING WITH AGE:\n"
    for _, r in top_dec.iterrows():
        summary += f"  • {r['condition']}: {np.exp(r['coef']*10):.2f}x odds per decade (n={int(r['n_pos'])})\n"

    summary += """

INTERPRETATION
Cardiovascular conditions (AF, IHD, hypertension) and sensory
conditions (glaucoma, hearing loss) show expected age-related
increases. Mental health conditions show decreasing prevalence,
likely due to cohort effects and survival bias.
"""
    add_text_page(pdf, summary, fontsize=11)
    print("  Added: Executive summary")

    # ===== MAIN SUMMARY FIGURE =====
    add_image_page(pdf, os.path.join(FIG, 'summary_odds_ratios.png'),
                   'Summary: Odds Ratio per Decade of Age')
    print("  Added: Summary odds ratios")

    # ===== LOG ODDS SUMMARY GRIDS =====
    add_image_page(pdf, os.path.join(FIG, 'summary_log_odds_increasing.png'),
                   'Top Conditions Increasing with Age (Log Odds)')
    print("  Added: Log odds increasing grid")

    add_image_page(pdf, os.path.join(FIG, 'summary_log_odds_decreasing.png'),
                   'Top Conditions Decreasing with Age (Log Odds)')
    print("  Added: Log odds decreasing grid")

    # ===== FOREST PLOT =====
    add_image_page(pdf, os.path.join(FIG, 'forest_plot_age_effects.png'),
                   'Forest Plot: Age Effect on All Significant Conditions')
    print("  Added: Forest plot")

    # ===== ODDS RATIO VISUALIZATION =====
    add_image_page(pdf, os.path.join(FIG, 'odds_ratio_per_decade.png'),
                   'Odds Ratio per Decade with 95% Confidence Intervals')
    print("  Added: Odds ratio visualization")

    # ===== COMPARISON PLOTS =====
    comparisons = ['Hypertension', 'Diabetes', 'Osteoarthritis', 'Allergy', 'ADHD', 'Depression']
    for cond in comparisons:
        fpath = os.path.join(FIG, f'comparison_{cond}.png')
        if os.path.exists(fpath):
            add_image_page(pdf, fpath, f'{cond}: Prevalence vs Log Odds')
            print(f"  Added: Comparison - {cond}")

    # ===== INDIVIDUAL LOG ODDS PLOTS (TOP CONDITIONS) =====
    # Top increasing
    for _, r in sig[sig['coef'] > 0].nlargest(8, 'coef').iterrows():
        cond = r['condition']
        fname = f"log_odds_{cond.replace(' ', '_').replace('/', '_')}.png"
        fpath = os.path.join(FIG, fname)
        if os.path.exists(fpath):
            add_image_page(pdf, fpath)
            print(f"  Added: Log odds - {cond}")

    # Top decreasing
    for _, r in sig[sig['coef'] < 0].nsmallest(8, 'coef').iterrows():
        cond = r['condition']
        fname = f"log_odds_{cond.replace(' ', '_').replace('/', '_')}.png"
        fpath = os.path.join(FIG, fname)
        if os.path.exists(fpath):
            add_image_page(pdf, fpath)
            print(f"  Added: Log odds - {cond}")

    # ===== PREVALENCE BY DISEASE GROUP =====
    groups = ['Cardiovascular', 'Metabolic_Endocrine', 'Mental Health',
              'Musculoskeletal', 'Sensory', 'Autoimmune', 'GI']
    for group in groups:
        fpath = os.path.join(FIG, f'prevalence_{group}.png')
        if os.path.exists(fpath):
            add_image_page(pdf, fpath, f'{group}: Prevalence by Age')
            print(f"  Added: Prevalence - {group}")

    # ===== LIMITATIONS PAGE =====
    limitations = f"""
LIMITATIONS AND CRITICAL ANALYSIS
{'='*50}

1. SURVIVAL BIAS
   Severe conditions may cause earlier death, reducing apparent
   prevalence in older age groups. This biases results toward
   finding conditions that "decrease" with age.

2. COHORT/GENERATION EFFECTS
   Different generations have different diagnosis rates:
   • ADHD was not diagnosed in older generations as children
   • Allergies may reflect increased environmental allergens
   These are NOT true age effects but generational differences.

3. DETECTION BIAS
   Older adults may seek more medical care, leading to more
   diagnoses. Younger adults may underreport symptoms.

4. CROSS-SECTIONAL DESIGN
   Cannot establish causality or track individual trajectories.
   We observe associations, not causes.

5. CONFOUNDERS NOT CONTROLLED
   This analysis did NOT adjust for:
   • Gender (many conditions are sex-specific)
   • BMI (major metabolic confounder)
   • Socioeconomic status
   • Smoking, alcohol, medications

6. SAMPLE SIZE CONSIDERATIONS
   Analysis limited to conditions with n ≥ {MIN_N} cases
   to ensure stable estimates.


RECOMMENDATIONS FOR INTERPRETATION
• Cardiovascular findings (↑) align with known epidemiology
• Mental health findings (↓) likely reflect cohort effects
• Reproductive conditions (↓) reflect natural resolution
• Consider these results as hypothesis-generating
"""
    add_text_page(pdf, limitations, fontsize=10)
    print("  Added: Limitations page")

    # ===== METHODOLOGY =====
    methodology = """
METHODOLOGY
{'='*50}

STATISTICAL MODEL
• Logistic regression: log(p/(1-p)) = β₀ + β₁ × Age
• β₁ = change in log odds per year of age
• Odds Ratio per decade = exp(β₁ × 10)

INTERPRETATION GUIDE
• OR = 2.0: Condition is 2x more likely per decade of age
• OR = 0.5: Condition is 2x less likely per decade of age
• OR = 1.0: No association with age

INCLUSION CRITERIA
• Age: 40-70 years
• Minimum positive cases: 100
• Complete data for condition

WHAT THE PLOTS SHOW

Log Odds Plot:
• Y-axis: Log odds of having condition at each age
• Points: Observed log odds per age bin
• Line: Fitted regression trend
• Upward slope = increases with age

Prevalence Plot:
• Y-axis: Percentage with condition
• Shows raw proportions (easier to understand)
• Does not show statistical uncertainty

Odds Ratio Plot:
• Shows multiplicative change per decade
• OR > 1 = more likely with age
• OR < 1 = less likely with age
• Error bars = 95% confidence interval


DATA SOURCE
Human Phenotype Project (HPP) 10K Dataset
Medical conditions from clinical records
"""
    add_text_page(pdf, methodology, fontsize=10)
    print("  Added: Methodology page")

    # ===== RESULTS TABLE =====
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Prepare table data
    table_data = sig.nlargest(20, 'coef')[['condition', 'n_pos', 'coef', 'p']].copy()
    table_data['OR/decade'] = np.exp(table_data['coef'] * 10).round(2)
    table_data['coef'] = table_data['coef'].round(4)
    table_data['p'] = table_data['p'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    table_data['n_pos'] = table_data['n_pos'].astype(int)
    table_data = table_data[['condition', 'n_pos', 'OR/decade', 'coef', 'p']]
    table_data.columns = ['Condition', 'N', 'OR/Decade', 'Log OR/Year', 'P-value']

    table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for j in range(len(table_data.columns)):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Top 20 Conditions Increasing with Age', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Added: Results table (increasing)")

    # Decreasing table
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    table_data = sig.nsmallest(15, 'coef')[['condition', 'n_pos', 'coef', 'p']].copy()
    table_data['OR/decade'] = np.exp(table_data['coef'] * 10).round(2)
    table_data['coef'] = table_data['coef'].round(4)
    table_data['p'] = table_data['p'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    table_data['n_pos'] = table_data['n_pos'].astype(int)
    table_data = table_data[['condition', 'n_pos', 'OR/decade', 'coef', 'p']]
    table_data.columns = ['Condition', 'N', 'OR/Decade', 'Log OR/Year', 'P-value']

    table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for j in range(len(table_data.columns)):
        table[(0, j)].set_facecolor('#3498DB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Top 15 Conditions Decreasing with Age', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Added: Results table (decreasing)")

print(f"\nPDF saved: {pdf_path}")
print(f"Total conditions analyzed (n>={MIN_N}): {len(results)}")
print(f"Significant associations: {len(sig)}")
