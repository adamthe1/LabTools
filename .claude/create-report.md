# Skill: Create Scientific Report

Use this skill when creating PDF reports for data analysis projects.

## Report Philosophy

**Principle:** Let the data speak through figures. Minimize text, maximize visual clarity.

> "A good figure should tell its story without requiring the reader to hunt through paragraphs of text."

## Report Structure

### Page 1: Methods (1 page max)

Write methods in **clear, simple language with technical precision**:

```
METHODS
═══════════════════════════════════════════════════

Study Design
• Cross-sectional analysis of [dataset name]
• Sample size: N = [number]
• Age range: [range] years

Statistical Analysis
• Model: [e.g., Logistic regression]
• Outcome: [what you're predicting]
• Predictor: [main variable]
• Interpretation: [how to read the results]

Inclusion Criteria
• [criterion 1]
• [criterion 2]
• Minimum sample: n ≥ [threshold]
```

**Guidelines for methods text:**
- Use bullet points, not paragraphs
- One concept per line
- Include the "interpretation guide" so readers know how to read figures
- No jargon without brief explanation

### Page 2: Main Finding Figure (THE KEY PAGE)

This is the **hero figure** - it should communicate the main message at a glance.

**Requirements:**
- Clear, descriptive title stating the finding
- Shows top 6-10 most important results
- Color-coded (e.g., red=increase, blue=decrease)
- Annotations for key values
- Large fonts (readable when printed)

**Example titles that work:**
- "Top 6 Conditions That Increase with Age"
- "Odds Ratio per Decade: Conditions Most Affected by Age"
- "Gender Differences in Age-Disease Associations"

**Bad titles to avoid:**
- "Results" (too vague)
- "Figure 1" (says nothing)
- "Age Coefficients" (too technical for first impression)

### Pages 3-7: Supporting Figures (5 pages)

Each page should have **ONE clear message**:

| Page | Purpose | Example |
|------|---------|---------|
| 3 | Detailed breakdown of main finding | Individual log-odds plots for top conditions |
| 4 | Opposite direction finding | Conditions that DECREASE with age |
| 5 | Subgroup analysis | Gender-stratified results |
| 6 | Validation/robustness | Confidence intervals, sample sizes |
| 7 | Category breakdown | Results by disease group |

**Figure guidelines:**
- 2x3 or 3x2 grid layouts work well
- Consistent color scheme throughout
- Same axis scales when comparing
- Error bars where appropriate

### Final Page: Summary Table (optional)

Only include if it adds value:
- Top 10-20 results with key statistics
- Color-coded by direction (increase/decrease)
- Include n, OR, p-value

## What NOT to Include in PDF

❌ Long paragraphs of text
❌ Detailed methodology (save for .md file)
❌ Literature review
❌ Extensive limitations discussion
❌ Raw data tables

## Companion Markdown File

Create a separate `REPORT.md` or `FINDINGS.md` with:

```markdown
# [Analysis Name] - Detailed Findings

## Executive Summary
[2-3 sentences on main findings]

## Key Results

### Finding 1: [Title]
[Detailed explanation with numbers]
- Interpretation
- Caveats

### Finding 2: [Title]
...

## Limitations & Critical Analysis
[Detailed discussion of biases, confounders, etc.]

## Methodology Details
[Full technical details for reproducibility]

## References
[If applicable]
```

## Code Template for PDF Generation

```python
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def create_report(output_path, figures_dir):
    with PdfPages(output_path) as pdf:

        # Page 1: Methods
        fig = plt.figure(figsize=(8.5, 11))
        methods_text = """
        METHODS
        ═══════════════════════════════════════

        [Your methods here - bullet points]
        """
        fig.text(0.05, 0.95, methods_text, ha='left', va='top',
                 fontsize=11, family='monospace')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Main Finding (Hero Figure)
        add_image_page(pdf, f"{figures_dir}/main_finding.png",
                       title="Main Finding Title")

        # Pages 3-7: Supporting Figures
        supporting_figs = [
            ("detailed_breakdown.png", "Detailed View"),
            ("opposite_direction.png", "Decreasing Conditions"),
            ("subgroup.png", "By Gender"),
            ("validation.png", "Confidence Intervals"),
            ("categories.png", "By Disease Category"),
        ]
        for fname, title in supporting_figs:
            add_image_page(pdf, f"{figures_dir}/{fname}", title=title)

def add_image_page(pdf, image_path, title=None):
    """Add a figure page to PDF."""
    from PIL import Image
    fig = plt.figure(figsize=(11, 8.5))  # Landscape
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
```

## Figure Design Guidelines

### Colors
```python
# Recommended palette
INCREASE_COLOR = '#E74C3C'  # Red
DECREASE_COLOR = '#3498DB'  # Blue
MALE_COLOR = '#2980B9'      # Dark blue
FEMALE_COLOR = '#E74C3C'    # Red
NEUTRAL_COLOR = '#2C3E50'   # Dark gray
```

### Fonts
```python
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'font.family': 'sans-serif'
})
```

### Layout
- Use `figsize=(11, 8.5)` for landscape (most figures)
- Use `figsize=(8.5, 11)` for portrait (text pages)
- Always call `plt.tight_layout()` before saving
- Use `bbox_inches='tight'` when saving

## Checklist Before Finalizing

- [ ] Methods page is ≤1 page and uses bullet points
- [ ] Main figure clearly shows the key finding
- [ ] Each figure has a descriptive title
- [ ] Color scheme is consistent
- [ ] Fonts are readable when printed
- [ ] No orphan figures (every figure has context)
- [ ] Companion .md file has detailed explanations
- [ ] Sample sizes shown where relevant
- [ ] P-values or confidence intervals included
