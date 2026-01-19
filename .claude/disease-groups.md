# Skill: Disease Groupings

Standard disease groupings based on WHO framework and clinical classification for use in HPP analyses.

## WHO Major NCD Categories

The World Health Organization identifies **4 major categories** of noncommunicable diseases that account for 80% of premature NCD deaths:

1. **Cardiovascular diseases** (19 million deaths/year)
2. **Cancers** (10 million deaths/year)
3. **Chronic respiratory diseases** (4 million deaths/year)
4. **Diabetes** (2+ million deaths/year)

---

## Disease Groupings for HPP Medical Conditions

```python
DISEASE_GROUPS = {

    'Cardiovascular': [
        'Hypertension',
        'Ischemic Heart Disease',
        'Atrial Fibrillation',
        'Heart valve disease',
        'AV. Conduction Disorder',
        'Myocarditis',
        'Atherosclerotic'
    ],

    'Cancer': [
        'Breast Cancer',
        'Melanoma',
        'Lymphoma'
    ],

    'Respiratory': [
        'Asthma',
        'COPD'
    ],

    'Metabolic/Endocrine': [
        'Diabetes',
        'Prediabetes',
        'Hyperlipidemia',
        'Hypercholesterolaemia',
        'Obesity',
        'Overweight',
        'Fatty Liver Disease',
        'Hashimoto',
        'Goiter',
        'Hyperparathyroidism',
        'Thyroid Adenoma',
        'G6PD'
    ],

    'Mental Health': [
        'Depression',
        'Anxiety',
        'ADHD',
        'PTSD',
        'Insomnia',
        'Migraine',
        'Headache'
    ],

    'Musculoskeletal': [
        'Osteoarthritis',
        'Back Pain',
        'Fibromyalgia',
        'Fractures',
        'Gout',
        'Meniscus Tears'
    ],

    'Gastrointestinal': [
        'IBD',
        'IBS',
        'Celiac',
        'Peptic Ulcer Disease',
        'Gallstone Disease',
        'Haemorrhoids',
        'Anal Fissure',
        'Anal abscess',
        'Lactose Intolerance'
    ],

    'Autoimmune/Inflammatory': [
        'Psoriasis',
        'Vitiligo',
        'Allergy',
        'Atopic Dermatitis',
        'Uveitis',
        'FMF'
    ],

    'Eye/Ear/Sensory': [
        'Glaucoma',
        'Hearing loss',
        'Tinnitus',
        'Retinal detachment',
        'Ocular Hypertension'
    ],

    'Renal/Urological': [
        'Renal Stones',
        'Urinary Tract Stones',
        'Urinary tract infection'
    ],

    'Hematologic': [
        'Anemia',
        'B12 Deficiency',
        'Iron Defficiency',
        'Thalassemia',
        'Hypercoagulability'
    ],

    'Reproductive (Female)': [
        'Polycystic Ovary Disease',
        'Endometriosis and Adenomyosis',
        'Breast fibrocystic change',
        'Perimenopausal Disorders',
        'Gestational diabetes'
    ],

    'Reproductive (Male)': [
        'Erectile Dysfunction'
    ]
}
```

---

## Helper Functions

```python
def get_condition_to_group_mapping():
    """Create mapping from condition name to disease group."""
    mapping = {}
    for group, conditions in DISEASE_GROUPS.items():
        for condition in conditions:
            if condition not in mapping:  # First group takes precedence
                mapping[condition] = group
    return mapping


def get_group(condition):
    """Get disease group for a condition."""
    mapping = get_condition_to_group_mapping()
    return mapping.get(condition, 'Other')


def get_conditions_in_group(group):
    """Get all conditions in a disease group."""
    return DISEASE_GROUPS.get(group, [])


def get_all_groups():
    """Get list of all disease groups."""
    return list(DISEASE_GROUPS.keys())


def filter_conditions_by_group(df, group):
    """Filter dataframe columns to conditions in a group."""
    conditions = get_conditions_in_group(group)
    available = [c for c in conditions if c in df.columns]
    return df[available]


def create_group_indicator(df, group):
    """Create binary indicator for having ANY condition in group."""
    conditions = get_conditions_in_group(group)
    available = [c for c in conditions if c in df.columns]
    if not available:
        return None
    return df[available].max(axis=1)
```

---

## Usage Examples

### Get group for a condition
```python
>>> get_group('Hypertension')
'Cardiovascular'

>>> get_group('Depression')
'Mental Health'

>>> get_group('Unknown Condition')
'Other'
```

### Analyze by group
```python
# Add group labels to results
results['disease_group'] = results['condition'].apply(get_group)

# Group statistics
group_stats = results.groupby('disease_group').agg({
    'coef': 'mean',
    'n_pos': 'sum'
})
```

### Create composite outcomes
```python
# Does patient have ANY cardiovascular condition?
df['has_cardiovascular'] = create_group_indicator(df, 'Cardiovascular')

# Does patient have ANY mental health condition?
df['has_mental_health'] = create_group_indicator(df, 'Mental Health')
```

### Plot by group
```python
import matplotlib.pyplot as plt

groups = results['disease_group'].unique()
colors = plt.cm.tab10(range(len(groups)))
color_map = dict(zip(groups, colors))

# Color bars by group
results['color'] = results['disease_group'].map(color_map)
```

---

## Sex-Specific Conditions

Some conditions require special handling:

### Female-Only Conditions
```python
FEMALE_ONLY = [
    'Polycystic Ovary Disease',
    'Endometriosis and Adenomyosis',
    'Breast fibrocystic change',
    'Perimenopausal Disorders',
    'Gestational diabetes',
    'Breast Cancer'  # Rare in males, typically analyze females only
]
```

### Male-Predominant Conditions
```python
MALE_PREDOMINANT = [
    'Erectile Dysfunction',  # Male only
    'Gout'  # Much more common in males
]
```

### Check function
```python
def is_sex_specific(condition):
    """Check if condition is sex-specific."""
    if condition in FEMALE_ONLY:
        return 'female'
    if condition in MALE_PREDOMINANT:
        return 'male'
    return None
```

---

## Age-Related Disease Classification

Based on epidemiological research (Hou et al., 2022), diseases can be classified by their relationship with age:

### Group A: True Aging-Related Diseases
Exponential increase in incidence with age throughout life.
```python
TRUE_AGING_DISEASES = [
    'Atrial Fibrillation',
    'Ischemic Heart Disease',
    'Atherosclerotic',
    'Osteoarthritis',
    'Glaucoma',
    'Hearing loss',
    'Hypertension',
    'Diabetes'
]
```

### Group B: Late-Onset with Plateau
Increase with age but plateau or decline in very old age.
```python
LATE_ONSET_PLATEAU = [
    'COPD',
    'Some cancers'
]
```

### Group C/D: Early-Onset or Stable
Onset in earlier life, stable or decreasing in old age.
```python
EARLY_ONSET = [
    'ADHD',
    'Allergy',
    'Asthma',
    'Migraine',
    'Depression'
]
```

---

## Quick Reference Tables

### Conditions by Expected Age Association

| Increases with Age | Decreases with Age | No Clear Pattern |
|--------------------|--------------------| -----------------|
| Hypertension | Allergy | Psoriasis |
| Diabetes | ADHD | Back Pain |
| Atrial Fibrillation | PCOS | IBS |
| Osteoarthritis | Endometriosis | Asthma |
| Glaucoma | Migraine | |
| Hearing loss | Depression | |
| Hyperlipidemia | Anxiety | |

### Sample Sizes in HPP 10K (ages 40-70)

| Group | N Conditions | Total Cases | Largest |
|-------|--------------|-------------|---------|
| Cardiovascular | 7 | ~4,500 | Hypertension (2,494) |
| Metabolic | 12 | ~9,000 | Hyperlipidemia (5,013) |
| Mental Health | 7 | ~4,500 | ADHD (2,257) |
| GI | 9 | ~5,500 | Haemorrhoids (2,475) |
| Autoimmune | 6 | ~7,500 | Allergy (4,220) |

---

## Color Scheme for Visualizations

```python
GROUP_COLORS = {
    'Cardiovascular': '#E74C3C',      # Red
    'Cancer': '#9B59B6',              # Purple
    'Respiratory': '#3498DB',          # Blue
    'Metabolic/Endocrine': '#E67E22', # Orange
    'Mental Health': '#1ABC9C',        # Teal
    'Musculoskeletal': '#34495E',      # Dark gray
    'Gastrointestinal': '#27AE60',     # Green
    'Autoimmune/Inflammatory': '#F1C40F',  # Yellow
    'Eye/Ear/Sensory': '#95A5A6',      # Light gray
    'Renal/Urological': '#2980B9',     # Dark blue
    'Hematologic': '#C0392B',          # Dark red
    'Reproductive (Female)': '#FF69B4', # Pink
    'Reproductive (Male)': '#4169E1',   # Royal blue
    'Other': '#BDC3C7'                 # Silver
}

def get_group_color(group):
    """Get color for a disease group."""
    return GROUP_COLORS.get(group, '#BDC3C7')
```

---

## References

1. WHO. (2023). Noncommunicable diseases fact sheet.
2. Hou et al. (2022). What Is an Aging-Related Disease? An Epidemiological Perspective. J Gerontol A.
3. GBD 2019 Diseases and Injuries Collaborators. Global burden of 369 diseases.
