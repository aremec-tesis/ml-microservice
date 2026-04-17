"""Runtime configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "misc" / "model.onnx"
SCALER_PATH = BASE_DIR / "misc" / "scaler.joblib"

HISTORY_WINDOW = 10


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        return cls(
            host=os.getenv("DATABASE_HOST", ""),
            port=int(os.getenv("DATABASE_PORT", "5432")),
            user=os.getenv("DATABASE_USER", ""),
            password=os.getenv("DATABASE_PASSWORD", ""),
            database=os.getenv("DATABASE_NAME", ""),
        )
