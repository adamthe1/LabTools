# LabTools

Statistical analysis tools for laboratory research.

## Overview

This repository contains Python libraries for analyzing biological data, with a focus on:

- **Biological Age Analysis** - Statistical methods for age-related biomarker analysis
- **Volcano Plot Analysis** - Differential expression visualization and significance testing
- **Body System Feature Loading** - Utilities for organizing features by biological systems
- **Feature prediction** 

## Structure

```
LabTools/
├── biological_age_lib/    # Core statistical analysis library
│   ├── analyze.py         # Main analysis functions
│   ├── visualization.py   # Plotting and visualization
│   ├── volcano_analysis.py # Volcano plot generation
│   └── utils.py           # Helper utilities
└── body_system_loader/    # Feature organization by body system
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- matplotlib
- seaborn

## Environment Setup

Create a `.env` file in the project root with:

```bash
BODY_SYSTEMS="/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/body_systems"
```

## Usage

Add an .env file
See `biological_age_lib/README.md` for detailed documentation and examples.

