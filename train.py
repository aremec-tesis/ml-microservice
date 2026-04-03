"""Training script: fits SVM on the synthetic dataset and exports to ONNX."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "synthetic_vr_dataset.csv"
SCALER_PATH = BASE_DIR / "scaler.joblib"
MODEL_PATH = BASE_DIR / "model.onnx"

FEATURE_COLUMNS = ["ORS", "ERS", "SCS", "RTA", "ATS", "ER", "SPS"]
LABEL_MAP = {"low": 0, "medium": 1, "high": 2}


def train() -> None:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(df)} rows")

    X = df[FEATURE_COLUMNS].values
    y = df["Target_Class"].map(LABEL_MAP).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel="rbf", C=1.0, random_state=42)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    print(f"Training accuracy: {accuracy_score(y, y_pred):.4f}")
    print(classification_report(y, y_pred, target_names=list(LABEL_MAP.keys())))

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    initial_type = [("features", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to {MODEL_PATH}")

    # Validate ONNX output matches sklearn
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_PATH))
    input_name = session.get_inputs()[0].name
    onnx_pred = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
    match = np.mean(onnx_pred == y_pred)
    print(f"ONNX vs sklearn prediction match: {match:.4f}")


if __name__ == "__main__":
    train()
