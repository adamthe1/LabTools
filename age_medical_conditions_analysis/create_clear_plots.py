#!/usr/bin/env python
"""Create clear log odds plots for age-medical conditions analysis."""
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

print("Loading data...")
age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
mc_df = load_body_system_df('medical_conditions')
df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')
df = df[(df['age'] >= 40) & (df['age'] <= 70)]
print(f"N = {len(df)}")

# Load results
results = pd.read_csv(os.path.join(OUT, 'results.csv'))

def calc_log_odds_by_age(df, condition):
    """Calculate log odds for each age bin."""
    bins = [40, 45, 50, 55, 60, 65, 70]
    d = df.copy()
    d['bin'] = pd.cut(d['age'], bins=bins, right=False)

    stats = d.groupby('bin', observed=True).agg({condition: ['sum', 'count']})
    stats.columns = ['pos', 'n']
    stats['prev'] = stats['pos'] / stats['n']

    # Log odds with small epsilon to avoid log(0)
    eps = 0.5 / stats['n']  # Continuity correction
    stats['odds'] = (stats['pos'] + 0.5) / (stats['n'] - stats['pos'] + 0.5)
    stats['log_odds'] = np.log(stats['odds'])

    # Standard error of log odds
    stats['se'] = np.sqrt(1/(stats['pos']+0.5) + 1/(stats['n']-stats['pos']+0.5))

    stats['age_mid'] = [(i.left + i.right) / 2 for i in stats.index]
    return stats.reset_index()


def plot_log_odds_individual(df, condition, coef, p_val, n_pos, ax=None, show_title=True):
    """Create a clear log odds vs age plot for one condition."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    stats = calc_log_odds_by_age(df, condition)

    # Plot points with error bars
    ax.errorbar(stats['age_mid'], stats['log_odds'], yerr=1.96*stats['se'],
                fmt='o', markersize=10, capsize=5, capthick=2, linewidth=2,
                color='#2E86AB', ecolor='#2E86AB', alpha=0.8)

    # Fit line
    x_fit = np.linspace(40, 70, 100)
    # Use the regression coefficient to draw the line
    intercept = stats['log_odds'].mean() - coef * stats['age_mid'].mean()
    y_fit = intercept + coef * x_fit

    color = 'red' if coef > 0 else 'blue'
    ax.plot(x_fit, y_fit, '--', color=color, linewidth=2, alpha=0.7,
            label=f'Slope: {coef:.4f} per year')

    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Log Odds of Condition', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    if show_title:
        direction = "↑ increases" if coef > 0 else "↓ decreases"
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        ax.set_title(f'{condition}\n{direction} with age (n={n_pos}) {sig}', fontsize=14)

    return ax


# 1. Create individual plots for top 10 increasing and decreasing conditions
print("\nCreating individual log odds plots...")

sig = results[(results['p'] < 0.05) & (results['n_pos'] >= 100) & results['coef'].notna()]
top_inc = sig.nlargest(10, 'coef')
top_dec = sig.nsmallest(10, 'coef')

# Individual plots for increasing conditions
for _, row in top_inc.iterrows():
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_log_odds_individual(df, row['condition'], row['coef'], row['p'], row['n_pos'], ax)
    fname = f"log_odds_{row['condition'].replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")

# Individual plots for decreasing conditions
for _, row in top_dec.iterrows():
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_log_odds_individual(df, row['condition'], row['coef'], row['p'], row['n_pos'], ax)
    fname = f"log_odds_{row['condition'].replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# 2. Create summary grid: Top 6 increasing
print("\nCreating summary grids...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, (_, row) in enumerate(top_inc.head(6).iterrows()):
    plot_log_odds_individual(df, row['condition'], row['coef'], row['p'], row['n_pos'], axes[i])
fig.suptitle('Top 6 Conditions INCREASING with Age\n(Log Odds vs Age)', fontsize=16, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'summary_log_odds_increasing.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: summary_log_odds_increasing.png")

# Top 6 decreasing
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, (_, row) in enumerate(top_dec.head(6).iterrows()):
    plot_log_odds_individual(df, row['condition'], row['coef'], row['p'], row['n_pos'], axes[i])
fig.suptitle('Top 6 Conditions DECREASING with Age\n(Log Odds vs Age)', fontsize=16, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'summary_log_odds_decreasing.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: summary_log_odds_decreasing.png")


# 3. Create a clear "slope" visualization - Forest plot style
print("\nCreating forest plot...")

fig, ax = plt.subplots(figsize=(12, 14))

# Get all significant conditions
all_sig = sig.sort_values('coef', ascending=True).copy()
n = len(all_sig)
y_pos = np.arange(n)

# Color by direction
colors = ['#E74C3C' if c > 0 else '#3498DB' for c in all_sig['coef']]

# Plot points with error bars (horizontal)
ax.errorbarx = ax.errorbar(all_sig['coef'], y_pos,
                            xerr=1.96*all_sig['se'],
                            fmt='o', markersize=8,
                            color='black', ecolor='gray',
                            capsize=3, capthick=1, linewidth=1)

# Add colored bars
for i, (idx, row) in enumerate(all_sig.iterrows()):
    color = '#E74C3C' if row['coef'] > 0 else '#3498DB'
    ax.barh(i, row['coef'], height=0.6, color=color, alpha=0.6)

ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['condition']} (n={r['n_pos']})" for _, r in all_sig.iterrows()], fontsize=9)
ax.set_xlabel('Change in Log Odds per Year of Age\n(negative = decreases with age, positive = increases with age)', fontsize=12)
ax.set_title('Age Effect on Medical Conditions\n(Significant associations only, p<0.05, n≥50)', fontsize=14)
ax.grid(True, axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E74C3C', alpha=0.6, label='Increases with age'),
                   Patch(facecolor='#3498DB', alpha=0.6, label='Decreases with age')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(FIG, 'forest_plot_age_effects.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: forest_plot_age_effects.png")


# 4. Create "odds ratio per decade" visualization - easier to interpret
print("\nCreating odds ratio visualization...")

fig, ax = plt.subplots(figsize=(12, 14))

all_sig['OR10'] = np.exp(all_sig['coef'] * 10)
all_sig = all_sig.sort_values('OR10', ascending=True)
n = len(all_sig)
y_pos = np.arange(n)

# Calculate CI for OR
all_sig['OR10_lo'] = np.exp((all_sig['coef'] - 1.96*all_sig['se']) * 10)
all_sig['OR10_hi'] = np.exp((all_sig['coef'] + 1.96*all_sig['se']) * 10)

# Plot
for i, (idx, row) in enumerate(all_sig.iterrows()):
    color = '#E74C3C' if row['OR10'] > 1 else '#3498DB'
    ax.plot([row['OR10_lo'], row['OR10_hi']], [i, i], color='gray', linewidth=1)
    ax.scatter(row['OR10'], i, color=color, s=100, zorder=5)

ax.axvline(1, color='black', linestyle='-', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['condition']} (n={r['n_pos']})" for _, r in all_sig.iterrows()], fontsize=9)
ax.set_xlabel('Odds Ratio per 10 Years of Age\n(OR<1 = lower odds in older, OR>1 = higher odds in older)', fontsize=12)
ax.set_title('How Much More/Less Likely is Each Condition per Decade of Age?\n(with 95% CI)', fontsize=14)
ax.set_xscale('log')
ax.grid(True, axis='x', alpha=0.3)

# Add annotations for extreme values
for i, (idx, row) in enumerate(all_sig.iterrows()):
    if row['OR10'] > 2 or row['OR10'] < 0.6:
        ax.annotate(f"{row['OR10']:.2f}x", xy=(row['OR10'], i),
                    xytext=(5, 0), textcoords='offset points', fontsize=8, va='center')

legend_elements = [Patch(facecolor='#E74C3C', label='More likely with age (OR>1)'),
                   Patch(facecolor='#3498DB', label='Less likely with age (OR<1)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(FIG, 'odds_ratio_per_decade.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: odds_ratio_per_decade.png")


# 5. Create comparison plot: Prevalence vs Log Odds side by side for key conditions
print("\nCreating side-by-side comparison plots...")

key_conditions = ['Hypertension', 'Osteoarthritis', 'Allergy', 'ADHD', 'Diabetes', 'Depression']

for cond in key_conditions:
    row = results[results['condition'] == cond].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Prevalence
    stats = calc_log_odds_by_age(df, cond)
    axes[0].bar(stats['age_mid'], stats['prev']*100, width=4, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Age (years)', fontsize=12)
    axes[0].set_ylabel('Prevalence (%)', fontsize=12)
    axes[0].set_title(f'{cond}: Prevalence by Age', fontsize=13)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Right: Log Odds
    plot_log_odds_individual(df, cond, row['coef'], row['p'], row['n_pos'], axes[1], show_title=False)
    axes[1].set_title(f'{cond}: Log Odds by Age', fontsize=13)

    direction = "INCREASES" if row['coef'] > 0 else "DECREASES"
    or10 = np.exp(row['coef'] * 10)
    fig.suptitle(f'{cond} {direction} with age\nOdds Ratio per decade: {or10:.2f}x',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    fname = f"comparison_{cond.replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# 6. Summary visualization: All conditions in one interpretable plot
print("\nCreating final summary visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Left: Increasing conditions
inc = sig[sig['coef'] > 0].nlargest(15, 'coef').copy()
inc['OR10'] = np.exp(inc['coef'] * 10)
y = np.arange(len(inc))
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(inc)))[::-1]
axes[0].barh(y, inc['OR10'] - 1, left=1, color=colors, alpha=0.8, edgecolor='black')
axes[0].axvline(1, color='black', linewidth=2)
axes[0].set_yticks(y)
axes[0].set_yticklabels(inc['condition'], fontsize=10)
axes[0].set_xlabel('Odds Ratio per Decade', fontsize=12)
axes[0].set_title('Conditions that INCREASE with Age\n(How many times more likely per 10 years)', fontsize=13)
axes[0].set_xlim(0.5, 4.5)
for i, (_, r) in enumerate(inc.iterrows()):
    axes[0].text(r['OR10'] + 0.05, i, f"{r['OR10']:.2f}x", va='center', fontsize=9)
axes[0].grid(True, axis='x', alpha=0.3)

# Right: Decreasing conditions
dec = sig[sig['coef'] < 0].nsmallest(15, 'coef').copy()
dec['OR10'] = np.exp(dec['coef'] * 10)
y = np.arange(len(dec))
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(dec)))
axes[1].barh(y, 1 - dec['OR10'], left=dec['OR10'], color=colors, alpha=0.8, edgecolor='black')
axes[1].axvline(1, color='black', linewidth=2)
axes[1].set_yticks(y)
axes[1].set_yticklabels(dec['condition'], fontsize=10)
axes[1].set_xlabel('Odds Ratio per Decade', fontsize=12)
axes[1].set_title('Conditions that DECREASE with Age\n(How many times less likely per 10 years)', fontsize=13)
axes[1].set_xlim(0.4, 1.1)
for i, (_, r) in enumerate(dec.iterrows()):
    axes[1].text(r['OR10'] - 0.05, i, f"{r['OR10']:.2f}x", va='center', ha='right', fontsize=9)
axes[1].grid(True, axis='x', alpha=0.3)

fig.suptitle('Age Effect on Medical Conditions: Summary\n(Odds Ratio = 2.0 means twice as likely per decade of age)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'summary_odds_ratios.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: summary_odds_ratios.png")


print("\n" + "="*50)
print("DONE! New figures created:")
print("="*50)
print("""
Individual log odds plots:
  - log_odds_<condition>.png (20 conditions)

Summary grids:
  - summary_log_odds_increasing.png
  - summary_log_odds_decreasing.png

Overall visualizations:
  - forest_plot_age_effects.png
  - odds_ratio_per_decade.png
  - summary_odds_ratios.png

Comparison plots:
  - comparison_<condition>.png (6 key conditions)
""")
