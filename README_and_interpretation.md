# Random Forest Regression for Literature-Derived Steel Hardness

## Dataset and target
- Input workbook: `AISI_1040_4140_4340_ML_dataset.xlsx`
- `Include_in_ML` was ignored for record selection.
- Total records retained: **89**
- AISI 1040: **29**; AISI 4140: **12**; AISI 4340: **48**
- Records with numeric `Hardness_HV`: **83**
- Records with missing `Hardness_HV`: **6**. These are retained in the master workbook but cannot be used as labelled observations for supervised training.
- Source papers represented among labelled records: **15**
- Target: `Hardness_HV` (Vickers hardness, HV)

## Validation strategy
The main baseline evaluation used **5-fold GroupKFold cross-validation grouped by `Paper_ID`**. All records from a source paper are kept in one fold, which better tests transfer to an unseen paper than randomly splitting individual rows.

The `Paper_ID`, row identifiers, original hardness text/scale, and hardness conversion method were not used as model predictors. `Include_in_ML` was not used as a filter or predictor. Missing input values were imputed inside the preprocessing pipeline. Numeric predictors were median-imputed; categorical predictors were most-frequent-imputed and one-hot encoded.

## Baseline grouped 5-fold results

| Metric | Result |
|---|---:|
| Pooled out-of-fold R² | 0.163 |
| Pooled out-of-fold MAE | 100.8 HV |
| Pooled out-of-fold MSE | 18323.8 HV² |
| Pooled out-of-fold RMSE | 135.4 HV |
| Mean fold R² ± SD | 0.048 ± 0.324 |
| Mean fold MAE ± SD | 100.7 ± 23.8 HV |
| Mean fold RMSE ± SD | 128.4 ± 45.8 HV |

Fold-wise results are in `grouped_5fold_baseline_metrics.csv`. Performance varies notably between folds, including negative R² in some folds. This indicates that cross-paper generalization is currently weak/unstable; these results should not be presented as a highly accurate model.

## Hyperparameter tuning
A RandomizedSearchCV search used grouped 3-fold CV on the 83 labelled records, optimizing MAE. Best parameters:
```text
{
  "model__n_estimators": 400,
  "model__min_samples_split": 8,
  "model__min_samples_leaf": 1,
  "model__max_features": 0.7,
  "model__max_depth": 5
}
```
Best grouped 3-fold model-selection MAE: **94.6 HV**.

This tuning score is used to select a final model and is **not an independent test score**. The tuned pipeline was then refitted on all 83 labelled records and saved as `random_forest_hardness_model.joblib`.

## Feature importance
`feature_importance.csv` and `feature_importance_top15.png` provide impurity-based feature importance from the final fitted model. These rankings describe how the fitted forest used the available features; they do not prove causation. Some categorical features and phase descriptors may also encode differences in source-paper reporting practices.

## Six records with missing hardness
`predictions_for_missing_hardness.csv` contains model estimates for the six records whose hardness was not reported in the sheet. These values are **predictions only, not measured or recovered hardness values**. Treat them as exploratory estimates, especially because the grouped cross-validation performance is limited.

## Important limitations and next steps
1. **Small, grouped dataset:** 83 labelled rows across 15 papers is small for a model with many process and microstructure variables.
2. **Literature heterogeneity:** Heat-treatment definitions, measurement loads, phase quantification methods, and reporting practices may differ between papers.
3. **Hardness conversion uncertainty:** Approximate HRC-to-HV conversions are mixed with directly reported HV values. Where possible, verify the original hardness scale and use a consistent conversion policy.
4. **Data-quality review:** Review the workbook's `Review_Checklist`, particularly ambiguous treatment labels and grade assignments, before drawing metallurgical conclusions.
5. **No independent test set:** All labelled data were used for grouped cross-validation and final refitting. The reported baseline out-of-fold metrics are the principal performance estimate; the tuned model's 3-fold search score is not a separate test result.
6. **Feature importance is not causation:** Interpret rankings in the context of metallurgy and the quality/availability of each feature.

## Files
- `train_random_forest_hardness.py`: complete reproducible training script.
- `steel_hardness_ml_data_and_results.xlsx`: original sheets plus master records, labelled records, missing-target records, metrics, predictions, feature importance, and estimates for missing targets.
- `random_forest_hardness_model.joblib`: tuned pipeline fitted on all labelled rows.
- CSV files: metrics, fold scores, out-of-fold predictions, feature importance, tuned parameters, and estimates for missing-target rows.
- PNG files: actual-vs-predicted, residual, fold MAE, and feature-importance plots.
