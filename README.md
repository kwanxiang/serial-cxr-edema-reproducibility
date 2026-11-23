# serial-cxr-edema-reproducibility

Code for the input-representation benchmark in manuscript INSI-D-26-00883 (*Insights into Imaging*):
current-only, difference-only and temporal inputs for detecting pulmonary edema worsening on serial
chest radiographs, with worsening labels derived from CheXTemporal report annotations.

This repository contains code only. No images, labels, model scores or reader data are distributed.

## Data

Obtain both datasets from Stanford AIMI under their data use agreements:

- **CheXpert-v1.0-small**: images and `train.csv`
- **CheXTemporal** (`StanfordAIMI/chextemporal` on Hugging Face): `chextemporal_chexpert_train.csv`

## Run

```bash
pip install -r requirements.txt
python build_cohort.py --chextemporal chextemporal_chexpert_train.csv \
    --chexpert-csv CheXpert-v1.0-small/train.csv --out pairs.csv
python fit_models.py --pairs pairs.csv --image-root . --out scores.csv
python evaluate.py --scores scores.csv --out results.csv
```

`--image-root` is the folder that contains `CheXpert-v1.0-small/`.

| Script | What it does |
|---|---|
| `build_cohort.py` | Selects edema-related CheXTemporal rows, builds current–prior frontal image pairs, assigns worsening labels, removes conflicting pairs, and splits by patient (train / tuning / test). Prints the selection flow and cohort characteristics. |
| `fit_models.py` | Preprocesses images (32 × 32 by default), builds the three input representations, fits PCA (128 components) + standardization + L2 logistic regression on the training split, and writes tuning and test scores. |
| `evaluate.py` | Test-split AUC, average precision, partial AUC, operating-point metrics at the tuning-selected Youden threshold, calibration, and paired contrasts (bootstrap and patient-clustered intervals, DeLong test). |

Options: `fit_models.py --components 64` or `--components 256` for the PCA-dimensionality grid, and
`--size 96` for the 96 × 96 analysis. The 96 × 96 option needs roughly 16 GB of memory.

## Expected results

| | Current-only | Difference-only | Temporal |
|---|---|---|---|
| Cohort | 26,251 pairs from 10,704 patients | | |
| Split (train / tuning / test pairs) | 18,279 / 2,631 / 5,341 | | |
| Test AUC (95% CI) | 0.584 (0.565–0.602) | 0.654 (0.636–0.672) | 0.668 (0.650–0.685) |
| Test average precision | 0.284 | 0.348 | 0.358 |

Temporal − current-only ΔAUC 0.084 (95% CI 0.066–0.102); temporal − difference-only ΔAUC 0.014
(95% CI 0.006–0.021).

Tested with Python 3.9.6 and the package versions in `requirements.txt`. With other versions,
randomized PCA can shift scores in the fifth decimal place.

## License

MIT
