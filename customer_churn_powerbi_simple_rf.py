"""
CUSTOMER CHURN ML - POWER BI EXPORT
-----------------------------------
Run:
    python customer_churn_powerbi.py

The script opens a file picker for customer_churn_data.csv, trains a
Random Forest model, evaluates it on a stratified 20% holdout test set,
and exports a CSV that can be imported into Power BI.

Expected input columns:
customer_id, credit_score, country, gender, age, tenure, balance,
products_number, credit_card, active_member, estimated_salary, churn

Target:
    churn = 0 (retained), 1 (churned)

Install dependencies:
    pip install pandas numpy scikit-learn matplotlib joblib

Important:
- customer_id is retained only as an identifier; it is not used as a feature.
- The reported metrics are calculated ONLY on the held-out test set.
- Power BI export includes scores for all rows. Rows marked "Train" were
  seen during fitting, so their predictions are in-sample. Use Test rows
  for an unbiased holdout assessment.
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from tkinter import Tk, filedialog, messagebox

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)


EXPECTED_COLUMNS = [
    "customer_id", "credit_score", "country", "gender", "age", "tenure",
    "balance", "products_number", "credit_card", "active_member",
    "estimated_salary", "churn"
]

CATEGORICAL_FEATURES = ["country", "gender"]
NUMERICAL_FEATURES = [
    "credit_score", "age", "tenure", "balance", "products_number",
    "credit_card", "active_member", "estimated_salary"
]
TARGET = "churn"
ID_COL = "customer_id"


def choose_csv():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select your customer churn CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path


def validate_and_prepare(df):
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "The dataset is missing these required columns:\n"
            + ", ".join(missing_cols)
        )

    df = df[EXPECTED_COLUMNS].copy()

    if df[ID_COL].isna().any():
        raise ValueError("customer_id contains missing values.")
    if df[ID_COL].duplicated().any():
        print("WARNING: customer_id contains duplicates. Check the source data.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("The churn column contains missing or non-numeric values.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The churn column must contain only 0 and 1.")
    df[TARGET] = df[TARGET].astype(int)

    # Numeric fields must be numeric. Missing values are handled in the pipeline.
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize categorical values and let the encoder handle unknown categories.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").fillna("Unknown").astype(str)

    if df[NUMERICAL_FEATURES].isna().all().any():
        bad = df[NUMERICAL_FEATURES].columns[
            df[NUMERICAL_FEATURES].isna().all()
        ].tolist()
        raise ValueError("These numeric columns contain no usable values: " + ", ".join(bad))

    return df


def build_model():
    # Imputation is included so the script can handle missing numeric values.
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline

    numeric_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERICAL_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])

    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessing", preprocess),
        ("classifier", classifier)
    ])


def main():
    csv_path = choose_csv()
    if not csv_path:
        print("No file selected. Exiting.")
        return

    csv_path = Path(csv_path)
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    df = validate_and_prepare(df)

    print(f"Dataset shape: {df.shape}")
    print("\nChurn distribution:")
    print(df[TARGET].value_counts().sort_index())

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Need at least two examples in each churn class for stratified split.")

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = build_model()
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # Holdout metrics: only the untouched 20% test partition.
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    auc = roc_auc_score(y_test, test_prob)

    print("\n========== HOLDOUT TEST PERFORMANCE ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, test_pred, target_names=["Retained (0)", "Churned (1)"],
        zero_division=0
    ))
    print("Confusion matrix [rows=actual, columns=predicted]:")
    print(confusion_matrix(y_test, test_pred))

    # Score every row using the model trained on the 80% training partition.
    # Train rows are in-sample; test rows are held out.
    all_prob = model.predict_proba(X)[:, 1]
    all_pred = (all_prob >= 0.5).astype(int)

    split_label = np.where(df.index.isin(test_idx), "Test", "Train")

    powerbi = pd.DataFrame({
        ID_COL: df[ID_COL].values,
        "Actual_Churn": y.values,
        "Predicted_Churn": all_pred,
        "Churn_Probability": all_prob,
        "Risk_Level": pd.cut(
            all_prob,
            bins=[-0.000001, 0.30, 0.60, 1.000001],
            labels=["Low", "Medium", "High"],
            include_lowest=True
        ).astype(str),
        "Data_Split": split_label
    })

    # Keep useful original fields so Power BI can slice the predictions.
    for col in [
        "country", "gender", "age", "tenure", "credit_score",
        "balance", "products_number", "credit_card", "active_member",
        "estimated_salary"
    ]:
        powerbi[col] = df[col].values

    output_dir = csv_path.parent
    pred_path = output_dir / "customer_churn_predictions_powerbi.csv"
    model_path = output_dir / "customer_churn_random_forest.joblib"
    metrics_path = output_dir / "customer_churn_model_metrics.csv"

    powerbi.to_csv(pred_path, index=False)
    joblib.dump(model, model_path)

    metrics = pd.DataFrame([{
        "Model": "Random Forest",
        "Test_Size": 0.20,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "Threshold": 0.50,
        "Random_State": 42
    }])
    metrics.to_csv(metrics_path, index=False)

    print("\n========== FILES EXPORTED ==========")
    print(f"Power BI predictions: {pred_path}")
    print(f"Trained model:        {model_path}")
    print(f"Test metrics:         {metrics_path}")
    print("\nIn Power BI, use Get Data > Text/CSV and import:")
    print(pred_path.name)
    print("\nUse Data_Split = 'Test' when showing holdout model performance.")
    print("Train rows are in-sample scores and should not be used as test results.")

    try:
        root = Tk()
        root.withdraw()
        messagebox.showinfo(
            "Churn ML completed",
            "Model training and evaluation completed.\n\n"
            f"Power BI CSV:\n{pred_path}\n\n"
            f"Holdout accuracy: {accuracy:.3f}\n"
            f"Holdout ROC-AUC: {auc:.3f}"
        )
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR:", exc)
        input("\nPress Enter to close...")
        sys.exit(1)
