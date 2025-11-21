# I3LUNG_PDSS

### Abstract

Despite a decade of immunotherapy, treatment selection in non-small cell lung cancer (NSCLC) relies on subgroup analyses and clinical scores. I3LUNG (NCT05537922) is the largest international, real-world, multimodal, AI-based trial, enrolling 2365 patients across six countries. We integrated real-world data (RWD), CT images, digital pathology, and genomics into machine learning early-fusion and deep-learning intermediate-fusion models. Multimodal models achieved AUCs up to 0.86, outperforming PD-L1, ECOG PS, NLR, LDH (all p<0.01) and LIPI score. Expert and non-expert physicians improved predictions using the explainable AI tool. The I3LUNG tool is currently under prospective validation in >2,000 patients.

This repository contains all necessary to replicate these results.

## How to Run

### Prerequisites

Before running either pipeline, ensure the `I3LUNG_PDSS/data/` directory contains:

**Required CSV files:**
- `rwd.csv`
- `genomics.csv`
- `digital_pathology.csv`
- `pyradiomics.csv`
- `fmrad.csv`
- `outcomes.csv`

**Required JSON files:**
- `features.json`
- `no_genomics_train.json`
- `no_genomics_test.json`
- `no_genomics_ext_val.json`

---
### Pipelines

The I3LUNG_PDSS repository contains two main pipelines for analysis:

### 1. Machine Learning Early Fusion (MLEF) Pipeline

The MLEF pipeline implements machine learning approaches with early fusion of features.

**To run this pipeline**, please refer to the detailed instructions in:
- [`mlef_pipeline/README.md`](mlef_pipeline/README.md)

### 2. Deep Learning Intermediate Fusion (DLIF) Pipeline

The DLIF pipeline implements deep learning approaches with intermediate fusion of features.

**To run this pipeline**, please refer to the detailed instructions in:
- [`dlif_pipeline/README.md`](dlif_pipeline/README.md)

---

## Visualization Notebooks

In addition to the two main pipelines, the repository includes several Jupyter notebooks for data visualization and analysis:

- **`data_visualization.ipynb`** - General data visualization and exploration
- **`graphs_classification.ipynb`** - Visualization of classification results
- **`graphs_survival_analysis.ipynb`** - Visualization of survival analysis results
- **`Metadata_Extraction.ipynb`** - Metadata extraction and analysis

These notebooks can be run independently to generate visualizations and analyze results from the pipelines.

---

## Repository Structure
```
I3LUNG_PDSS/
├── mlef_pipeline/               # Machine Learning Early Fusion pipeline
│   └── README.md               # Detailed instructions for MLEF pipeline
├── dlif_pipeline/               # Deep Learning Intermediate Fusion pipeline
│   └── README.md               # Detailed instructions for DLIF pipeline
├── data/                        # Data directory (user must populate)
├── data_visualization.ipynb     # Data visualization notebook
├── graphs_classification.ipynb  # Classification results visualization
├── graphs_survival_analysis.ipynb  # Survival analysis visualization
├── Metadata_Extraction.ipynb    # Metadata extraction notebook
└── README.md                    # This file
```