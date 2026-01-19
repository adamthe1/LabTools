#!/usr/bin/env python
"""Generate gender-stratified PDF report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import os
from PIL import Image

OUT = '/home/adamgab/PycharmProjects/LabTools/age_medical_conditions_analysis/gender_stratified'

def add_image_page(pdf, image_path, title=None):
    """Add an image page to PDF."""
    if not os.path.exists(image_path):
        print(f"  Warning: {image_path} not found")
        return False
    fig = plt.figure(figsize=(11, 8.5))
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return True

def add_text_page(pdf, content, fontsize=10):
    """Add a text page to PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, content, ha='left', va='top', fontsize=fontsize,
             family='monospace', wrap=True, transform=fig.transFigure)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print("Generating gender-stratified PDF report...")

# Load results
results = pd.read_csv(os.path.join(OUT, 'gender_stratified_results.csv'))
print(f"Loaded {len(results)} conditions")

pdf_path = os.path.join(OUT, 'Gender_Stratified_Report.pdf')

with PdfPages(pdf_path) as pdf:

    # ===== TITLE PAGE =====
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.65, 'Age and Medical Conditions', ha='center', fontsize=32, fontweight='bold')
    fig.text(0.5, 0.55, 'Gender-Stratified Analysis', ha='center', fontsize=28, fontweight='bold', color='#8E44AD')
    fig.text(0.5, 0.42, 'HPP 10K Dataset', ha='center', fontsize=20)
    fig.text(0.5, 0.35, 'Ages 40-70', ha='center', fontsize=16)
    fig.text(0.5, 0.28, 'Female: 12,313  |  Male: 11,152', ha='center', fontsize=14)
    fig.text(0.5, 0.18, 'Blue = Male  |  Red = Female', ha='center', fontsize=14, fontweight='bold')
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)
    print("  Added: Title page")

    # ===== EXECUTIVE SUMMARY =====
    # Find conditions with significant effects in each gender
    sig_male = results[(results['p_male'] < 0.05) & (results['n_male'] >= 50)].dropna(subset=['coef_male'])
    sig_female = results[(results['p_female'] < 0.05) & (results['n_female'] >= 50)].dropna(subset=['coef_female'])

    n_male_inc = (sig_male['coef_male'] > 0).sum()
    n_male_dec = (sig_male['coef_male'] < 0).sum()
    n_female_inc = (sig_female['coef_female'] > 0).sum()
    n_female_dec = (sig_female['coef_female'] < 0).sum()

    summary = f"""
EXECUTIVE SUMMARY: GENDER-STRATIFIED ANALYSIS
{'='*55}

This report examines whether the relationship between age and
medical conditions differs between males and females.

SAMPLE SIZE
• Female: 12,313 subjects (52%)
• Male: 11,152 subjects (48%)
• Age range: 40-70 years

SIGNIFICANT ASSOCIATIONS (p<0.05, n≥50)

MALES:
  • {len(sig_male)} significant conditions
  • {n_male_inc} increase with age
  • {n_male_dec} decrease with age

FEMALES:
  • {len(sig_female)} significant conditions
  • {n_female_inc} increase with age
  • {n_female_dec} decrease with age


KEY FINDINGS

1. CARDIOVASCULAR CONDITIONS
   Generally increase with age in BOTH genders, but may have
   different slopes (effect sizes).

2. SEX-SPECIFIC CONDITIONS
   • Breast Cancer, PCOS, Endometriosis: Female only
   • Erectile Dysfunction: Male only
   • Gout: More common in males

3. MENTAL HEALTH
   Depression, Anxiety: May show different age patterns by gender

4. AUTOIMMUNE CONDITIONS
   Often more prevalent in females with different age trajectories
"""
    add_text_page(pdf, summary, fontsize=11)
    print("  Added: Executive summary")

    # ===== SUMMARY COMPARISON PLOT =====
    if add_image_page(pdf, os.path.join(OUT, 'summary_gender_differences.png'),
                      'Conditions with Largest Gender Differences'):
        print("  Added: Summary gender differences")

    # ===== TOP CONDITIONS BY GENDER - MALES =====
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    male_data = results[results['n_male'] >= 50].dropna(subset=['coef_male', 'p_male'])
    male_data = male_data.nlargest(15, 'coef_male')[['condition', 'n_male', 'OR10_male', 'p_male']].copy()
    male_data['OR10_male'] = male_data['OR10_male'].round(2)
    male_data['p_male'] = male_data['p_male'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    male_data['n_male'] = male_data['n_male'].astype(int)
    male_data.columns = ['Condition', 'N (Male)', 'OR/Decade', 'P-value']

    table = ax.table(cellText=male_data.values, colLabels=male_data.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for j in range(len(male_data.columns)):
        table[(0, j)].set_facecolor('#2980B9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Top 15 Conditions Increasing with Age - MALES', fontsize=14, fontweight='bold',
                 color='#2980B9', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Added: Male top conditions table")

    # ===== TOP CONDITIONS BY GENDER - FEMALES =====
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    female_data = results[results['n_female'] >= 50].dropna(subset=['coef_female', 'p_female'])
    female_data = female_data.nlargest(15, 'coef_female')[['condition', 'n_female', 'OR10_female', 'p_female']].copy()
    female_data['OR10_female'] = female_data['OR10_female'].round(2)
    female_data['p_female'] = female_data['p_female'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    female_data['n_female'] = female_data['n_female'].astype(int)
    female_data.columns = ['Condition', 'N (Female)', 'OR/Decade', 'P-value']

    table = ax.table(cellText=female_data.values, colLabels=female_data.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for j in range(len(female_data.columns)):
        table[(0, j)].set_facecolor('#E74C3C')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Top 15 Conditions Increasing with Age - FEMALES', fontsize=14, fontweight='bold',
                 color='#E74C3C', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Added: Female top conditions table")

    # ===== GENDER COMPARISON TABLE =====
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Compare same conditions across genders
    compare = results.dropna(subset=['coef_male', 'coef_female']).copy()
    compare['diff'] = abs(compare['coef_male'] - compare['coef_female'])
    compare = compare.nlargest(15, 'diff')[['condition', 'n_male', 'OR10_male', 'n_female', 'OR10_female']].copy()
    compare['OR10_male'] = compare['OR10_male'].round(2)
    compare['OR10_female'] = compare['OR10_female'].round(2)
    compare['n_male'] = compare['n_male'].astype(int)
    compare['n_female'] = compare['n_female'].astype(int)
    compare.columns = ['Condition', 'N Male', 'OR/10y Male', 'N Female', 'OR/10y Female']

    table = ax.table(cellText=compare.values, colLabels=compare.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for j in range(len(compare.columns)):
        table[(0, j)].set_facecolor('#8E44AD')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Conditions with Largest Gender Differences in Age Effect', fontsize=14,
                 fontweight='bold', color='#8E44AD', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Added: Gender comparison table")

    # ===== INDIVIDUAL CONDITION PLOTS =====
    # Key conditions to show
    key_conditions = [
        'Hypertension', 'Diabetes', 'Ischemic_Heart_Disease', 'Osteoarthritis',
        'Depression', 'Anxiety', 'Allergy', 'ADHD',
        'Migraine', 'Gout', 'Anemia', 'Sleep_Apnea',
        'Glaucoma', 'Hearing_loss', 'Back_Pain', 'Obesity'
    ]

    for cond in key_conditions:
        fpath = os.path.join(OUT, f'gender_{cond}.png')
        if add_image_page(pdf, fpath):
            print(f"  Added: {cond}")

    # ===== SEX-SPECIFIC CONDITIONS =====
    sex_specific = ['Breast_Cancer', 'Polycystic_Ovary_Disease', 'Endometriosis_and_Adenomyosis']
    for cond in sex_specific:
        fpath = os.path.join(OUT, f'gender_{cond}.png')
        if add_image_page(pdf, fpath, f'{cond.replace("_", " ")} (Sex-Specific)'):
            print(f"  Added: {cond}")

    # ===== METHODOLOGY PAGE =====
    methodology = """
METHODOLOGY
{'='*55}

STATISTICAL MODEL
• Separate logistic regressions for each gender
• log(p/(1-p)) = β₀ + β₁ × Age
• Performed independently for males and females

PLOT INTERPRETATION
• Blue circles/line = Male data and fit
• Red squares/line = Female data and fit
• Dashed lines = Fitted regression
• Legend shows sample size (n) and p-value

ODDS RATIO INTERPRETATION
• OR/Decade = exp(coefficient × 10)
• OR = 2.0 means 2x more likely per decade
• OR = 0.5 means 2x less likely per decade

INCLUSION CRITERIA
• Age: 40-70 years
• Minimum 50 cases per gender for analysis
• Total n ≥ 100 for condition

LIMITATIONS
1. Some conditions are sex-specific (PCOS, breast cancer)
2. Prevalence differences may affect power
3. Hormonal factors not directly measured
4. Cross-sectional design limitations apply

SAMPLE SIZES
• Total: 23,473 subjects
• Female: 12,313 (52%)
• Male: 11,152 (48%)
"""
    add_text_page(pdf, methodology, fontsize=10)
    print("  Added: Methodology page")

    # ===== KEY FINDINGS PAGE =====
    findings = """
KEY FINDINGS BY CONDITION CATEGORY
{'='*55}

CARDIOVASCULAR
Both genders show increased risk with age for:
• Hypertension, Ischemic Heart Disease, Heart Valve Disease
Males may have steeper increase for some conditions.

METABOLIC
• Diabetes, Hyperlipidemia: Increase with age in both
• Obesity: Different patterns by gender
• Fatty Liver: More common in males

MENTAL HEALTH
• Depression: Decreases with age in both genders
• Anxiety: Similar pattern, may differ in magnitude
• ADHD: Cohort effect visible in both genders

MUSCULOSKELETAL
• Osteoarthritis: Increases with age, similar in both
• Gout: Much more common in males, steeper increase

AUTOIMMUNE/INFLAMMATORY
• Allergy: Decreases with age in both
• Psoriasis: Different patterns by gender possible
• Hashimoto's: More common in females

SENSORY
• Hearing loss: Increases in both, may differ in rate
• Glaucoma: Similar pattern in both genders


NOTES ON INTERPRETATION
• Gender differences may reflect:
  - Biological differences (hormones, genetics)
  - Healthcare-seeking behavior
  - Diagnostic patterns
  - Survival differences
• Some apparent differences may not be statistically significant
• Always check sample sizes and p-values
"""
    add_text_page(pdf, findings, fontsize=10)
    print("  Added: Key findings page")

print(f"\nPDF saved: {pdf_path}")
