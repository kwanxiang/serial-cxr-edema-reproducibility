# Serial CXR Pulmonary Edema Worsening Reproducibility Package

This repository contains the non-image reproducibility materials for the manuscript:

**Serial Chest Radiograph Comparison for Detecting Pulmonary Edema Worsening With Expert Adjudication**

The study evaluates three matched input representations for detecting interval pulmonary edema worsening on serial frontal chest radiographs:

- `current_only_pixels`: current radiograph only
- `difference_only_pixels`: pixel-level current-minus-prior representation
- `prior_current_temporal_pixels`: prior image, current image, and pixel difference representation

## What Is Included

- Sanitized held-out prediction table for the report-label cohort (`n = 5341`)
- Sanitized expert-reference case table (`n = 500`)
- Aggregate cohort, model, and reader-agreement tables
- ROC and precision-recall figures used for manuscript submission
- Scripts to regenerate ROC/PR figures and verify reported AUC/AP values

## What Is Not Included

This repository does **not** include raw CheXpert or CheXTemporal images, exported case-review images, DICOM files, Stanford AIMI credentials, PhysioNet credentials, or any patient/study/image-path identifiers.

Raw source data must be obtained from the official CheXpert and CheXTemporal data providers under their respective data-use terms.

## Repository Layout

```text
data/derived/
  held_out_predictions_sanitized.csv
  expert_reference_cases_sanitized.csv
  report_label_model_metrics.csv
  expert_reference_model_metrics.csv
  expert_reader_agreement.csv
  report_vs_expert_target_crosstab.csv
  cohort_flow.csv
  cohort_summary.csv
  split_characteristics.csv
  model_specification.csv
figures/
  roc_curves.png
  pr_curves.png
scripts/
  regenerate_figures.py
  verify_metrics.py
docs/
  DATA_AVAILABILITY_STATEMENT.md
  DATA_DICTIONARY.md
```

## Quick Start

Create a fresh Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the reported point estimates:

```bash
python scripts/verify_metrics.py
```

Regenerate the ROC and precision-recall figures:

```bash
python scripts/regenerate_figures.py
```

The regenerated figures are written to `figures/roc_curves.png` and `figures/pr_curves.png`.

## Data-Use Notice

The CSV files in `data/derived/` are non-image derived research outputs. They exclude patient IDs, study IDs, image paths, raw images, exported review images, reader identifiers, free-text reader comments, and per-case report/pathology phrases. They are provided to support manuscript review and reproducibility of the reported statistical summaries.

The source CheXpert/CheXTemporal data are not redistributed. Users who need to rerun image preprocessing or model fitting from raw images should obtain the source datasets from the official providers and comply with their data-use agreements.

## Manuscript Availability Statement

After uploading this folder to GitHub, replace the placeholder repository URL in `docs/DATA_AVAILABILITY_STATEMENT.md`, `CITATION.cff`, and the manuscript declarations.
