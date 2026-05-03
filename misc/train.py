"""Training script: fits a stateful SVM (16 features) on the longitudinal synthetic dataset.

Target: difficulty recommendation (decrease / maintain / increase). The model receives
both the 7 current-session metrics and 9 historical-context features, so it consumes the
patient's longitudinal history as part of every inference.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "synthetic_vr_dataset.csv"
SCALER_PATH = BASE_DIR / "scaler.joblib"
MODEL_PATH = BASE_DIR / "model.onnx"

FEATURE_COLUMNS = [
    "ORS", "ERS", "SCS", "RTA", "ATS", "ER", "SPS",
    "baseline_sps", "slope_sps", "delta_sps",
    "mean_ors", "mean_ers", "mean_er", "mean_rta",
    "std_sps", "session_count",
]
LABEL_MAP = {
    "decrease_difficulty": 0,
    "maintain_difficulty": 1,
    "increase_difficulty": 2,
}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def train() -> None:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(df)} rows, {df['PatientID'].nunique()} patients")

    X = df[FEATURE_COLUMNS].astype(np.float64).values
    y = df["Target_Recommendation"].map(LABEL_MAP).values

    # Stratified split by patient to avoid leaking sessions of the same patient
    # across train/test.
    patient_ids = df["PatientID"].values
    unique_patients = np.unique(patient_ids)
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )
    train_mask = np.isin(patient_ids, train_patients)
    test_mask = np.isin(patient_ids, test_patients)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    print(f"\nTrain accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Test accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=[INVERSE_LABEL_MAP[i] for i in sorted(INVERSE_LABEL_MAP)],
        )
    )

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    initial_type = [("features", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}},
    )
    with open(MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to {MODEL_PATH}")

    # Validate ONNX output matches sklearn (labels and probabilities)
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_PATH))
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: X_test_scaled.astype(np.float32)})
    onnx_labels = onnx_outputs[0]
    onnx_probs = onnx_outputs[1]
    label_match = np.mean(onnx_labels == y_test_pred)
    sklearn_probs = model.predict_proba(X_test_scaled)
    prob_match = np.allclose(onnx_probs, sklearn_probs, atol=1e-4)
    print(f"\nONNX vs sklearn label match: {label_match:.4f}")
    print(f"ONNX vs sklearn probability match (atol=1e-4): {prob_match}")


if __name__ == "__main__":
    train()
