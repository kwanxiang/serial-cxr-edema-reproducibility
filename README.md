# Serial Images Improve AI Detection of Pulmonary Edema Worsening: An Expert-Adjudicated Benchmark

Reproducibility materials for manuscript **INSI-D-26-00883** (*Insights into Imaging*).

This repository supports a controlled input-representation benchmark comparing current-only,
difference-only, and temporal (serial-image) inputs for detecting interval pulmonary edema
worsening. The locked primary analysis used the same classifier family and the same retained
PCA dimensionality (128 components) for all three representations.

## Headline results (matched PCA = 128)

| Contrast | Reference | ΔAUC (95% CI) | Interpretation |
|---|---|---|---|
| Temporal − current-only | Report labels, n = 5,341 | +0.084 (0.066–0.102) | Supported |
| Temporal − current-only | Expert reference, n = 500 | +0.141 (0.089–0.192) | Supported |
| Temporal − difference-only | Expert reference, n = 500 | +0.017 (−0.009 to 0.041) | Not established |

The serial-versus-single-image contrast was positive across the tested dimensionalities. The
incremental contrast between full temporal and change-only inputs was small and was not
established at the primary expert-reference analysis.

## Repository contents

```text
reproduce_statistics.py       Primary point estimates and paired AUC contrasts from locked scores
requirements.txt              Python dependencies
DATA_NOTICE.md                Scope and privacy treatment of the released derived data
data/
  unified128_test_scores.csv  Sanitized held-out scores and report-derived targets
  unified128_tune_scores.csv  Sanitized tuning scores and report-derived targets
  final_adjudicated_cases.csv Sanitized reader grades and final expert-reference targets
scripts/
  expert500_unified.py        Three-reference contrasts and clustered bootstrap
  full_metrics.py             Tables 3–5, calibration, agreement, and tipping-point check
  s4_sensitivity.py           Supplementary Table S4 reference definitions
  figures/                    Data-driven figure-generation scripts
outputs/                      Precomputed aggregate results used in the revision
```

The released identifiers (`pair_key` and `patient_cluster_id`) are random repository-specific
keys. They preserve one-to-one joins and patient-clustered resampling without disclosing the
public-dataset patient identifiers or image paths.

## Reproduce the locked-score analyses

Create a Python environment, then run:

```bash
pip install -r requirements.txt
python reproduce_statistics.py
python scripts/expert500_unified.py
python scripts/full_metrics.py
python scripts/s4_sensitivity.py
```

These commands recompute the reported discrimination estimates, paired contrasts, operating-
point metrics, calibration summaries, reader agreement, and sensitivity-reference results from
the sanitized locked scores. Scripts write aggregate files to `outputs/`.

## Model fitting from source images

The source CheXpert images and original CheXTemporal tables are governed by their respective
data-use terms and are not redistributed here. Consequently, this public package verifies the
reported locked-score statistical analysis but is not a one-command redistribution of the image
preprocessing and PCA fitting inputs. The retained PCA dimensionality, split design, and input
representations are fully described in the manuscript and supplement.

## Privacy and data provenance

The score and annotation tables are derived from the public CheXpert and CheXTemporal releases.
Exact image paths, original patient/study identifiers, reader identifiers, demographics, and
free-text adjudication comments have been removed. See `DATA_NOTICE.md`.

## Environment

Python 3.12; numpy, pandas, scipy, scikit-learn, matplotlib, and Pillow. A fixed random seed of
42 is used for stochastic analyses.

## Citation

Xiang K. *Serial Images Improve AI Detection of Pulmonary Edema Worsening: An
Expert-Adjudicated Benchmark.* Insights into Imaging (under revision), manuscript
INSI-D-26-00883.
