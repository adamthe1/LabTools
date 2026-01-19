#!/usr/bin/env python
"""Simple age-medical conditions analysis limited to ages 40-70."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import os
import sys

sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
from body_system_loader.load_feature_df import load_body_system_df

OUT = '/home/adamgab/PycharmProjects/LabTools/age_medical_conditions_analysis'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

GROUPS = {
    'Cardiovascular': ['Hypertension', 'Ischemic Heart Disease', 'Atrial Fibrillation',
                       'Heart valve disease', 'Atherosclerotic'],
    'Cancer': ['Breast Cancer', 'Melanoma', 'Lymphoma'],
    'Respiratory': ['Asthma', 'COPD'],
    'Metabolic': ['Diabetes', 'Prediabetes', 'Hyperlipidemia', 'Obesity', 'Fatty Liver Disease'],
    'Mental': ['Depression', 'Anxiety', 'ADHD', 'Insomnia', 'Migraine', 'Headache'],
    'Musculoskeletal': ['Osteoarthritis', 'Back Pain', 'Fibromyalgia', 'Fractures', 'Gout'],
    'GI': ['IBD', 'IBS', 'Peptic Ulcer Disease', 'Gallstone Disease', 'Haemorrhoids'],
    'Autoimmune': ['Psoriasis', 'Allergy', 'Atopic Dermatitis'],
    'Sensory': ['Glaucoma', 'Hearing loss', 'Tinnitus']
}

def get_mapping():
    m = {}
    for g, cs in GROUPS.items():
        for c in cs:
            if c not in m:
                m[c] = g
    return m

def calc_logreg(df, cond):
    v = df[[cond, 'age']].dropna()
    if len(v) < 100 or v[cond].sum() < 20:
        return {'coef': np.nan, 'se': np.nan, 'p': np.nan, 'n_pos': int(v[cond].sum()), 'n': len(v)}
    X = sm.add_constant(v['age'].values.reshape(-1, 1))
    try:
        m = sm.Logit(v[cond].values, X).fit(disp=0, maxiter=100)
        return {'coef': m.params[1], 'se': m.bse[1], 'p': m.pvalues[1],
                'n_pos': int(v[cond].sum()), 'n': len(v), 'OR10': np.exp(m.params[1] * 10)}
    except:
        return {'coef': np.nan, 'se': np.nan, 'p': np.nan, 'n_pos': int(v[cond].sum()), 'n': len(v)}

def calc_prev(df, cond):
    bins = [40, 45, 50, 55, 60, 65, 70]
    d = df.copy()
    d['bin'] = pd.cut(d['age'], bins=bins, right=False)
    p = d.groupby('bin', observed=True).agg({cond: ['sum', 'count']})
    p.columns = ['pos', 'n']
    p['prev'] = p['pos'] / p['n']
    p['mid'] = [(i.left + i.right) / 2 for i in p.index]
    return p.reset_index()

print("=" * 50)
print("AGE & MEDICAL CONDITIONS (Ages 40-70)")
print("=" * 50)

# Load
print("\nLoading data...")
age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
mc_df = load_body_system_df('medical_conditions')
df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')
df = df[(df['age'] >= 40) & (df['age'] <= 70)]
print(f"N = {len(df)}")

# Analyze
print("\nAnalyzing...")
mapping = get_mapping()
conds = [c for c in df.columns if c not in ['age', 'gender']]
results = []
for c in conds:
    r = calc_logreg(df, c)
    r['condition'] = c
    r['group'] = mapping.get(c, 'Other')
    results.append(r)

rdf = pd.DataFrame(results).sort_values('coef', ascending=False)
rdf.to_csv(os.path.join(OUT, 'results.csv'), index=False)

# Summary
sig = rdf[(rdf['p'] < 0.05) & (rdf['n_pos'] >= 50)]
print(f"\nSignificant: {len(sig)} conditions")
print(f"  Increase with age: {(sig['coef'] > 0).sum()}")
print(f"  Decrease with age: {(sig['coef'] < 0).sum()}")

print("\nTop 10 INCREASING:")
for _, r in sig.nlargest(10, 'coef').iterrows():
    print(f"  {r['condition']}: OR/10y={r['OR10']:.2f}, n={r['n_pos']}")

print("\nTop 10 DECREASING:")
for _, r in sig.nsmallest(10, 'coef').iterrows():
    print(f"  {r['condition']}: OR/10y={r['OR10']:.2f}, n={r['n_pos']}")

# FIGURES
print("\nGenerating figures...")

# Fig 1: All coefficients
valid = rdf[(rdf['n_pos'] >= 50) & rdf['coef'].notna()].sort_values('coef')
grps = valid['group'].unique()
cmap = dict(zip(grps, plt.cm.tab10(np.linspace(0, 1, len(grps)))))

fig, ax = plt.subplots(figsize=(14, max(10, len(valid) * 0.35)))
y = np.arange(len(valid))
ax.barh(y, valid['coef'], color=[cmap[g] for g in valid['group']], alpha=0.8)
ax.errorbar(valid['coef'], y, xerr=1.96 * valid['se'], fmt='none', color='k', alpha=0.5, capsize=2)
ax.set_yticks(y)
ax.set_yticklabels(valid['condition'], fontsize=9)
ax.set_xlabel('Log Odds per Year of Age')
ax.set_title('Age-Disease Association (Ages 40-70)')
ax.axvline(0, color='k', ls='--')
ax.grid(True, axis='x', alpha=0.3)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=cmap[g]) for g in grps], grps, loc='lower right', fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'coefficients.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: coefficients.png")

# Fig 2: Top conditions prevalence
top_pos = sig.nlargest(6, 'coef')['condition'].tolist()
top_neg = sig.nsmallest(6, 'coef')['condition'].tolist()

fig, axes = plt.subplots(2, 6, figsize=(24, 8))
for i, c in enumerate(top_pos):
    p = calc_prev(df, c)
    ax = axes[0, i]
    ax.plot(p['mid'], p['prev'] * 100, 'o-', color='red', lw=2)
    ax.fill_between(p['mid'], 0, p['prev'] * 100, alpha=0.2, color='red')
    coef = rdf[rdf['condition'] == c]['coef'].values[0]
    ax.set_title(f'{c}\n(coef={coef:.4f})', fontsize=10)
    ax.set_xlabel('Age')
    ax.set_ylabel('Prevalence (%)')
    ax.grid(True, alpha=0.3)

for i, c in enumerate(top_neg):
    p = calc_prev(df, c)
    ax = axes[1, i]
    ax.plot(p['mid'], p['prev'] * 100, 'o-', color='blue', lw=2)
    ax.fill_between(p['mid'], 0, p['prev'] * 100, alpha=0.2, color='blue')
    coef = rdf[rdf['condition'] == c]['coef'].values[0]
    ax.set_title(f'{c}\n(coef={coef:.4f})', fontsize=10)
    ax.set_xlabel('Age')
    ax.set_ylabel('Prevalence (%)')
    ax.grid(True, alpha=0.3)

axes[0, 0].set_ylabel('Prevalence (%)\nINCREASE with age', fontsize=11)
axes[1, 0].set_ylabel('Prevalence (%)\nDECREASE with age', fontsize=11)
fig.suptitle('Top Age-Associated Medical Conditions', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'top_conditions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: top_conditions.png")

# Fig 3: Age distribution
fig, ax = plt.subplots(figsize=(10, 6))
df['age'].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.set_title('Age Distribution (40-70)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'age_dist.png'), dpi=150)
plt.close()
print("  Saved: age_dist.png")

print("\nDone!")
