"""
Random Forest regression for literature-derived AISI 1040, AISI 4140, and AISI 4340 hardness data.

Input:
    AISI_1040_4140_4340_ML_dataset.xlsx
Expected sheet:
    ML_Data

Important methodological choices:
- Include_in_ML is deliberately NOT used to filter records.
- All rows are retained in the master dataset.
- Rows without Hardness_HV cannot be used as labelled supervised training examples.
- Paper_ID is used for grouped cross-validation, never as a model input.
- Hardness columns, conversion method, source notes, and record identifiers are excluded from predictors.
- The baseline performance is estimated with 5-fold GroupKFold by Paper_ID.
- Hyperparameter tuning uses grouped 3-fold CV on the labelled dataset, then refits the selected pipeline on all labelled rows.
  The tuning score is for model selection and is not an independent final test score.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

INPUT_NAME = "AISI_1040_4140_4340_ML_dataset.xlsx"
SCRIPT_DIR = Path(__file__).resolve().parent
candidates = [
    Path.cwd() / INPUT_NAME,
    SCRIPT_DIR / INPUT_NAME,
    SCRIPT_DIR.parent / INPUT_NAME,
]
INPUT_FILE = next((candidate for candidate in candidates if candidate.exists()), None)
OUTPUT_DIR = SCRIPT_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

if INPUT_FILE is None:
    raise FileNotFoundError(
        f"Cannot find {INPUT_NAME}. Put the Excel file in the script folder "
        "or its parent folder, or update INPUT_NAME/INPUT_FILE."
    )

# ---------- 1. Load and inspect all records ----------
all_sheets = pd.read_excel(INPUT_FILE, sheet_name=None)
if "ML_Data" not in all_sheets:
    raise ValueError("The workbook must contain a sheet named 'ML_Data'.")

df = all_sheets["ML_Data"].copy()
print("Master dataset shape:", df.shape)
print("Records by steel grade:")
print(df["Steel_Grade"].value_counts(dropna=False))
print("Include_in_ML is ignored; no rows are filtered using this column.")

if "Hardness_HV" not in df or "Paper_ID" not in df:
    raise ValueError("Required columns Hardness_HV and Paper_ID were not found.")

# Keep all records in the master data. Supervised model training requires known targets.
labelled = df.loc[df["Hardness_HV"].notna()].copy()
unlabelled = df.loc[df["Hardness_HV"].isna()].copy()
print(f"All records retained: {len(df)}")
print(f"Labelled rows for training/evaluation: {len(labelled)}")
print(f"Rows with missing hardness retained separately: {len(unlabelled)}")
print(f"Number of source papers among labelled rows: {labelled['Paper_ID'].nunique()}")

# ---------- 2. Select predictors ----------
# Features with too few observations or no variation are excluded to reduce instability.
numeric_candidates = [
    "C_wt_pct", "Mn_wt_pct", "Si_wt_pct", "P_wt_pct", "S_wt_pct",
    "Cr_wt_pct", "Ni_wt_pct", "Mo_wt_pct",
    "Aus_Time_min", "Aus_Temp_C",
    "Tempering_Temp_C", "Tempering_Time_min",
    "Phase_Fraction_pct", "Ferrite_pct", "Pearlite_pct",
    "Martensite_pct", "Retained_Austenite_pct",
]
categorical_candidates = [
    "Steel_Grade", "Quench", "Final_Cooling", "Phase_Description",
]

numeric_features = [
    c for c in numeric_candidates
    if c in labelled.columns
    and labelled[c].notna().sum() >= 10
    and labelled[c].nunique(dropna=True) > 1
]
categorical_features = [
    c for c in categorical_candidates
    if c in labelled.columns
    and labelled[c].notna().sum() >= 10
    and labelled[c].nunique(dropna=True) > 1
]
features = numeric_features + categorical_features

if not features:
    raise ValueError("No usable features were found. Check the workbook columns.")

X = labelled[features].copy()
y = labelled["Hardness_HV"].astype(float).copy()
groups = labelled["Paper_ID"].astype(str).copy()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Target: Hardness_HV (Vickers hardness)")
print("Predictor count before one-hot encoding:", len(features))

# ---------- 3. Preprocessing and model pipeline ----------
def make_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ], remainder="drop")

    return Pipeline([
        ("preprocess", preprocessing),
        ("model", RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

# ---------- 4. Baseline 5-fold grouped cross-validation ----------
# All observations from one paper stay in the same validation fold.
outer_cv = GroupKFold(n_splits=5)
oof_predictions = np.full(len(labelled), np.nan)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups=groups), start=1
):
    baseline = make_pipeline()
    baseline.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = baseline.predict(X.iloc[test_idx])
    oof_predictions[test_idx] = predictions

    fold_results.append({
        "Fold": fold,
        "N_test": len(test_idx),
        "Papers_test": ", ".join(sorted(groups.iloc[test_idx].unique())),
        "R2": r2_score(y.iloc[test_idx], predictions),
        "MAE_HV": mean_absolute_error(y.iloc[test_idx], predictions),
        "MSE_HV2": mean_squared_error(y.iloc[test_idx], predictions),
        "RMSE_HV": mean_squared_error(y.iloc[test_idx], predictions) ** 0.5,
    })

fold_df = pd.DataFrame(fold_results)
baseline_metrics = {
    "Model": "Baseline Random Forest (fixed settings)",
    "Validation": "5-fold GroupKFold by Paper_ID; out-of-fold predictions",
    "N_labelled_records": len(labelled),
    "N_papers": groups.nunique(),
    "Pooled_OOF_R2": r2_score(y, oof_predictions),
    "Pooled_OOF_MAE_HV": mean_absolute_error(y, oof_predictions),
    "Pooled_OOF_MSE_HV2": mean_squared_error(y, oof_predictions),
    "Pooled_OOF_RMSE_HV": mean_squared_error(y, oof_predictions) ** 0.5,
    "Mean_fold_R2": fold_df["R2"].mean(),
    "SD_fold_R2": fold_df["R2"].std(ddof=1),
    "Mean_fold_MAE_HV": fold_df["MAE_HV"].mean(),
    "SD_fold_MAE_HV": fold_df["MAE_HV"].std(ddof=1),
    "Mean_fold_RMSE_HV": fold_df["RMSE_HV"].mean(),
    "SD_fold_RMSE_HV": fold_df["RMSE_HV"].std(ddof=1),
}
overall_df = pd.DataFrame([baseline_metrics])
print("\nGrouped 5-fold baseline results:")
print(fold_df.to_string(index=False))
print(overall_df.to_string(index=False))

# ---------- 5. Hyperparameter tuning with grouped CV ----------
# Keep the search moderate because the labelled dataset is small.
param_distributions = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [None, 3, 5, 8],
    "model__min_samples_split": [2, 4, 6, 8],
    "model__min_samples_leaf": [1, 2, 3, 4],
    "model__max_features": [0.5, 0.7, 1.0],
}
search = RandomizedSearchCV(
    estimator=make_pipeline(),
    param_distributions=param_distributions,
    n_iter=8,
    scoring="neg_mean_absolute_error",
    cv=GroupKFold(n_splits=3),
    random_state=42,
    n_jobs=1,
    refit=True,
)
search.fit(X, y, groups=groups)
final_model = search.best_estimator_
print("\nBest parameters:")
print(search.best_params_)
print("Grouped 3-fold tuning MAE (model-selection score, HV):", -search.best_score_)
print("Caution: this tuning score is not an independent test score.")

# ---------- 6. Feature importance ----------
feature_names = final_model.named_steps["preprocess"].get_feature_names_out()
importances = final_model.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

# ---------- 7. Save tables, plots, and trained model ----------
predictions_df = pd.DataFrame({
    "Record_ID": labelled["Record_ID"].values if "Record_ID" in labelled else labelled.index,
    "Paper_ID": groups.values,
    "Steel_Grade": labelled["Steel_Grade"].values,
    "Actual_Hardness_HV": y.values,
    "Baseline_OOF_Predicted_HV": oof_predictions,
    "OOF_Residual_HV": y.values - oof_predictions,
})

fold_df.to_csv(OUTPUT_DIR / "grouped_5fold_baseline_metrics.csv", index=False)
overall_df.to_csv(OUTPUT_DIR / "overall_baseline_metrics.csv", index=False)
predictions_df.to_csv(OUTPUT_DIR / "baseline_oof_predictions.csv", index=False)
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
pd.DataFrame([
    {"Parameter": key, "Best_Value": str(value)}
    for key, value in search.best_params_.items()
]).to_csv(OUTPUT_DIR / "tuned_model_best_parameters.csv", index=False)
joblib.dump(final_model, OUTPUT_DIR / "random_forest_hardness_model.joblib")

# Plot 1: Actual vs predicted from held-out-paper predictions
plt.figure(figsize=(7, 6))
plt.scatter(y, oof_predictions, alpha=0.8)
low = min(float(y.min()), float(np.min(oof_predictions)))
high = max(float(y.max()), float(np.max(oof_predictions)))
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual hardness (HV)")
plt.ylabel("Out-of-fold predicted hardness (HV)")
plt.title("Grouped 5-fold CV: actual vs predicted")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=200)
plt.close()

# Plot 2: Residuals
residuals = y.to_numpy() - oof_predictions
plt.figure(figsize=(7, 5))
plt.scatter(oof_predictions, residuals, alpha=0.8)
plt.axhline(0, linestyle="--")
plt.xlabel("Out-of-fold predicted hardness (HV)")
plt.ylabel("Residual (actual - predicted, HV)")
plt.title("Grouped 5-fold CV residual plot")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=200)
plt.close()

# Plot 3: Fold-wise MAE
plt.figure(figsize=(7, 5))
plt.bar(fold_df["Fold"].astype(str), fold_df["MAE_HV"])
plt.xlabel("Cross-validation fold")
plt.ylabel("MAE (HV)")
plt.title("Grouped 5-fold CV: fold-wise MAE")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fold_mae.png", dpi=200)
plt.close()

# Plot 4: Top 15 feature importances
top = importance_df.head(15).sort_values("Importance", ascending=True)
plt.figure(figsize=(9, 7))
plt.barh(top["Feature"], top["Importance"])
plt.xlabel("Random Forest impurity-based importance")
plt.title("Top 15 feature importances (final model)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_top15.png", dpi=200)
plt.close()

# Excel output: all original workbook sheets plus complete retained record sets and results.
with pd.ExcelWriter(OUTPUT_DIR / "steel_hardness_ml_data_and_results.xlsx", engine="openpyxl") as writer:
    for sheet_name, sheet_df in all_sheets.items():
        sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    df.to_excel(writer, sheet_name="All_89_Records", index=False)
    labelled.to_excel(writer, sheet_name="Labelled_Training_83", index=False)
    unlabelled.to_excel(writer, sheet_name="Missing_Hardness_6", index=False)
    fold_df.to_excel(writer, sheet_name="5Fold_Baseline_Metrics", index=False)
    overall_df.to_excel(writer, sheet_name="Overall_Metrics", index=False)
    importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
    predictions_df.to_excel(writer, sheet_name="OOF_Predictions", index=False)
    pd.DataFrame([
        {"Parameter": key, "Best_Value": str(value)}
        for key, value in search.best_params_.items()
    ]).to_excel(writer, sheet_name="Tuned_Params", index=False)

print(f"\nDone. Results saved to: {OUTPUT_DIR.resolve()}")
