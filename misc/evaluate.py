"""Honest evaluation of the trained stateful SVM (ONNX) on the held-out test set.

Reproduces the EXACT patient-level split used in train.py (test_size=0.2,
random_state=42, split on unique PatientID so no patient leaks across
train/test), runs the real model.onnx + scaler.joblib on the test patients,
and reports the genuine confusion matrix, per-class precision/recall/F1, and
a Seaborn heatmap. No numbers are invented: every value comes from the model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "synthetic_vr_dataset.csv"
SCALER_PATH = BASE_DIR / "scaler.joblib"
MODEL_PATH = BASE_DIR / "model.onnx"
HEATMAP_PATH = BASE_DIR / "confusion_matrix.png"

FEATURE_COLUMNS = [
    "ORS", "ERS", "SCS", "RTA", "ER", "SPS",
    "baseline_sps", "slope_sps", "delta_sps",
    "mean_ors", "mean_ers", "mean_er", "mean_rta",
    "std_sps", "session_count",
]
LABEL_MAP = {
    "decrease_difficulty": 0,
    "maintain_difficulty": 1,
    "increase_difficulty": 2,
}
CLASS_NAMES = ["decrease", "maintain", "increase"]


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURE_COLUMNS].astype(np.float64).values
    y = df["Target_Recommendation"].map(LABEL_MAP).values
    patient_ids = df["PatientID"].values

    # Reproduce the identical patient-stratified split from train.py.
    unique_patients = np.unique(patient_ids)
    _, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )
    test_mask = np.isin(patient_ids, test_patients)
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Dataset: {len(df)} sessions / {len(unique_patients)} patients")
    print(f"Held-out test: {len(X_test)} sessions / {len(test_patients)} patients")
    print(f"Test class distribution: "
          f"{pd.Series(y_test).map({v: k for k, v in LABEL_MAP.items()}).value_counts().to_dict()}")

    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    session = ort.InferenceSession(str(MODEL_PATH))
    input_name = session.get_inputs()[0].name
    y_pred = session.run(None, {input_name: X_test_scaled})[0]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"\nTest accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows = real, cols = predicted):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, digits=4
    ))

    # Heatmap: counts + row-normalized percentages.
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
    annot = np.array(
        [[f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%" for j in range(cm.shape[1])]
         for i in range(cm.shape[0])]
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Session count"}, square=True,
        linewidths=0.5, linecolor="white",
    )
    plt.title(f"AREMEC — Confusion Matrix (held-out test, acc={acc:.3f})")
    plt.ylabel("Real difficulty recommendation")
    plt.xlabel("Model prediction")
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=150)
    print(f"\nHeatmap saved to {HEATMAP_PATH}")


if __name__ == "__main__":
    main()
