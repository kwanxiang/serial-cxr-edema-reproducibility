# Data Dictionary

## `held_out_predictions_sanitized.csv`

One row per held-out serial radiograph pair evaluated against report-derived CheXTemporal labels.

- `heldout_id`: public row identifier assigned for this release
- `pair_hash`: SHA-256 hash prefix of the internal pair identifier
- `report_target`: report-derived worsening label; 1 = worsening, 0 = non-worsening
- `report_label_set`: CheXTemporal temporal label set after pair-level aggregation
- `current_age_decade`: decade-binned age for the current study
- `current_sex_label`: sex label from the source metadata
- `current_view_label`: projection label from the source metadata
- `*_score`: model score for the named input representation

## `expert_reference_cases_sanitized.csv`

One row per case in the 500-case expert-reference subset.

- `case_id`: public expert-review case identifier
- `pair_hash`: SHA-256 hash prefix of the internal pair identifier
- `report_target`: report-derived worsening label
- `expert_target`: final adjudicated image-based reference label
- `reader*_prior_severity`: reader ordinal edema grade for the prior radiograph
- `reader*_current_severity`: reader ordinal edema grade for the current radiograph
- `reader*_unevaluable`: whether the reader marked the pair unevaluable
- `reader*_target`: reader-specific worsening label
- `agree_*`: agreement indicators between the two primary readers
- `full_agreement`: 1 if no third-reader adjudication was needed
- `adjudicator_used`: 1 if third-reader adjudication determined the final reference
- `adjudicator_*`: adjudicator ordinal grades when adjudication was used
- `final_prior_severity`: final reference prior edema grade
- `final_current_severity`: final reference current edema grade
- `final_unevaluable`: final unevaluable indicator
- `reference_standard`: whether the final reference came from full agreement or adjudication
- `*_score`: model score for the named input representation

Pulmonary edema severity grades use the manuscript scale:

- 0 = none
- 1 = vascular congestion or redistribution
- 2 = interstitial edema
- 3 = alveolar edema

## Metrics Tables

- `report_label_model_metrics.csv`: model performance against report-derived held-out labels
- `expert_reference_model_metrics.csv`: model performance against the final adjudicated expert-reference labels
- `expert_reader_agreement.csv`: reader agreement, adjudication counts, and report-label agreement with the final reference
- `report_vs_expert_target_crosstab.csv`: 2 x 2 table comparing report-derived and expert-reference binary labels

## Cohort Tables

- `cohort_flow.csv`: cohort construction flow counts
- `cohort_summary.csv`: pair and patient counts by split
- `split_characteristics.csv`: demographic and projection summaries by split
- `label_set_frequency.csv`: frequency of aggregated temporal label sets
- `model_specification.csv`: input representation and PCA settings for each model
