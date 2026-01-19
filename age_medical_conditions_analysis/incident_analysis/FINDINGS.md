# Age at Incident Analysis
## HPP 10K Dataset - Diagnosis Age Patterns

---

## Executive Summary

This analysis examines **when** medical conditions are typically diagnosed, calculating
the age at diagnosis for each subject (diagnosis year - birth year).

**Key Statistic:** Analyzed 66 conditions with valid diagnosis dates.

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
| G6PD | 14.0 | 177 |
| Asthma | 16.4 | 1144 |
| Thalassemia | 17.8 | 86 |
| Allergy | 18.6 | 328 |
| Polycystic Ovary Disease | 20.6 | 39 |
| Migraine | 25.9 | 1229 |
| Headache | 27.4 | 558 |
| Atopic Dermatitis | 27.5 | 632 |
| FMF | 28.5 | 42 |
| Sinusitis | 29.6 | 23 |


### Latest Onset Conditions
| Condition | Mean Age | n |
|-----------|----------|---|
| Atrial Fibrillation | 59.1 | 55 |
| Atherosclerotic | 58.5 | 39 |
| Ischemic Heart Disease | 55.2 | 237 |
| Ocular Hypertension | 52.8 | 22 |
| Erectile Dysfunction | 51.9 | 104 |
| Prediabetes | 51.8 | 1429 |
| Melanoma | 51.1 | 33 |
| Hyperparathyroidism | 51.1 | 41 |
| Meniscus Tears | 50.9 | 87 |
| Perimenopausal Disorders | 50.6 | 65 |


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
