# Data notice

This repository contains only derived, non-image research tables needed to audit the locked-score
statistical analyses.

Included:

- report-derived binary targets and model scores for the tuning and held-out partitions;
- reader severity grades and binary targets required to recompute agreement and sensitivity
  analyses for the 500-case expert-reference subset;
- random repository-specific pair and patient-cluster keys needed for joins and clustered
  resampling.

Excluded:

- chest radiograph files and exact image paths;
- original CheXpert patient and study identifiers;
- reader and adjudicator identifiers;
- age, sex, projection metadata, and free-text review comments;
- data not required to reproduce the reported aggregate statistics.

The random keys cannot be used to locate source images. Access to CheXpert and CheXTemporal must
be obtained separately under the data providers' terms. No additional right to redistribute the
source datasets is granted by this repository.
