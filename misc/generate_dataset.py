"""Synthetic longitudinal VR dataset generator.

Produces multiple sessions per patient with coherent cognitive trajectories,
the new ORS formula (key vs secondary objects), the 16-feature vector the
stateful ML expects, and a clinically-defendable `recommendation` target
(decrease / maintain / increase).

Output: dataset/synthetic_vr_dataset.csv
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "dataset" / "synthetic_vr_dataset.csv"

NUM_PATIENTS = 500
MIN_SESSIONS = 5
MAX_SESSIONS = 20
HISTORY_WINDOW = 10
SEED = 42

# Recommendation labelling thresholds (mirror personalization_engine reference rules
# so the dataset encodes the same defendable clinical logic the model will learn).
LOW_SPS_THRESHOLD = 0.4
HIGH_SPS_THRESHOLD = 0.7
DELTA_SIGNIFICANT = -0.15
SPS_IMPROVEMENT_FLOOR = 0.6
TREND_THRESHOLD = 0.02

Phenotype = Literal["improving", "stable", "declining"]


def _sample_phenotype(rng: np.random.Generator) -> Phenotype:
    return rng.choice(["improving", "stable", "declining"], p=[0.4, 0.4, 0.2])


def _drift_for(phenotype: Phenotype, rng: np.random.Generator) -> float:
    if phenotype == "improving":
        return float(rng.uniform(0.005, 0.020))
    if phenotype == "declining":
        return float(rng.uniform(-0.020, -0.005))
    return float(rng.uniform(-0.002, 0.002))


def _sample_comprehension(theta: float, rng: np.random.Generator) -> int:
    if theta < 0.4:
        probs = [0.55, 0.35, 0.10]
    elif theta < 0.7:
        probs = [0.20, 0.50, 0.30]
    else:
        probs = [0.05, 0.30, 0.65]
    return int(rng.choice([0, 1, 2], p=probs))


def _sample_session_raw(theta: float, rng: np.random.Generator) -> dict:
    total_key = int(rng.integers(3, 9))
    total_secondary = int(rng.integers(3, 11))
    correct_key = int(rng.binomial(total_key, np.clip(theta, 0.05, 0.99)))
    correct_secondary = int(
        rng.binomial(total_secondary, np.clip(theta * 0.85, 0.05, 0.99))
    )
    missed = (total_key + total_secondary) - (correct_key + correct_secondary)
    incorrect_objects = int(rng.binomial(max(missed, 0), 0.45)) if missed > 0 else 0

    total_events = int(rng.integers(3, 9))
    correct_events = int(rng.binomial(total_events, np.clip(theta, 0.05, 0.99)))

    expected_interactions = int(rng.integers(5, 16))
    interaction_events = int(
        np.clip(
            rng.binomial(expected_interactions, np.clip(theta * 0.95, 0.05, 0.99)),
            0,
            int(expected_interactions * 1.5),
        )
    )

    total_questions = int(rng.integers(5, 13))
    incorrect_answers = int(
        rng.binomial(total_questions, np.clip(1.0 - theta, 0.05, 0.95))
    )

    log_mean = math.log(2.0 + (1.0 - theta) * 4.0)
    response_times = rng.lognormal(mean=log_mean, sigma=0.3, size=total_questions)
    response_times = np.clip(response_times, 0.3, 15.0)

    return {
        "comprehension_score": _sample_comprehension(theta, rng),
        "correct_key_objects": correct_key,
        "correct_secondary_objects": correct_secondary,
        "incorrect_objects": incorrect_objects,
        "total_key_objects": total_key,
        "total_secondary_objects": total_secondary,
        "correct_events": correct_events,
        "total_events": total_events,
        "expected_interactions": expected_interactions,
        "interaction_events": interaction_events,
        "total_questions": total_questions,
        "incorrect_answers": incorrect_answers,
        "response_times": response_times.tolist(),
    }


def _compute_metrics(raw: dict) -> dict:
    den_ors = raw["total_key_objects"] * 2 + raw["total_secondary_objects"]
    num_ors = (
        raw["correct_key_objects"] * 2
        + raw["correct_secondary_objects"]
        - raw["incorrect_objects"]
    )
    ors = num_ors / den_ors if den_ors > 0 else 0.0

    ers = raw["correct_events"] / raw["total_events"] if raw["total_events"] else 0.0
    scs = raw["comprehension_score"] / 2.0
    rta = float(np.mean(raw["response_times"])) if raw["response_times"] else 0.0
    ats = (
        raw["interaction_events"] / raw["expected_interactions"]
        if raw["expected_interactions"]
        else 0.0
    )
    er = (
        raw["incorrect_answers"] / raw["total_questions"]
        if raw["total_questions"]
        else 0.0
    )
    sps = 0.3 * ors + 0.3 * ers + 0.2 * scs + 0.2 * (1 - er)

    return {"ors": ors, "ers": ers, "scs": scs, "rta": rta, "ats": ats, "er": er, "sps": sps}


def _aggregate_history(history: list[dict], current: dict) -> dict:
    """Compute the 10 historical-context features. History is most-recent-first."""
    if not history:
        return {
            "baseline_sps": current["sps"],
            "slope_sps": 0.0,
            "delta_sps": 0.0,
            "mean_ors": current["ors"],
            "mean_ers": current["ers"],
            "mean_er": current["er"],
            "mean_rta": current["rta"],
            "std_sps": 0.0,
            "session_count": 0,
            "cold_start": True,
        }

    sps_vals = [h["sps"] for h in history]
    weights = [1.0 / (i + 1) for i in range(len(sps_vals))]
    baseline = sum(w * v for w, v in zip(weights, sps_vals)) / sum(weights)

    if len(sps_vals) >= 2:
        ordered = list(reversed(sps_vals))
        n = len(ordered)
        mean_x = (n - 1) / 2
        mean_y = sum(ordered) / n
        num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(ordered))
        den = sum((i - mean_x) ** 2 for i in range(n))
        slope = num / den if den else 0.0
    else:
        slope = 0.0

    delta = current["sps"] - baseline
    std_sps = float(np.std(sps_vals)) if len(sps_vals) > 1 else 0.0

    return {
        "baseline_sps": baseline,
        "slope_sps": slope,
        "delta_sps": delta,
        "mean_ors": float(np.mean([h["ors"] for h in history])),
        "mean_ers": float(np.mean([h["ers"] for h in history])),
        "mean_er": float(np.mean([h["er"] for h in history])),
        "mean_rta": float(np.mean([h["rta"] for h in history])),
        "std_sps": std_sps,
        "session_count": len(history),
        "cold_start": False,
    }


def _label_recommendation(current_sps: float, ctx: dict) -> str:
    """Clinical labelling rule used as ground truth for the synthetic target."""
    if current_sps < LOW_SPS_THRESHOLD:
        base = "decrease_difficulty"
    elif current_sps > HIGH_SPS_THRESHOLD:
        base = "increase_difficulty"
    else:
        base = "maintain_difficulty"

    if ctx["cold_start"]:
        return base

    if ctx["delta_sps"] < DELTA_SIGNIFICANT:
        return "decrease_difficulty"

    slope = ctx["slope_sps"]
    if slope < -TREND_THRESHOLD and base == "increase_difficulty":
        return "maintain_difficulty"
    if (
        slope > TREND_THRESHOLD
        and base == "maintain_difficulty"
        and current_sps > SPS_IMPROVEMENT_FLOOR
    ):
        return "increase_difficulty"
    return base


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for patient_id in range(1, NUM_PATIENTS + 1):
        phenotype = _sample_phenotype(rng)
        theta_base = float(rng.uniform(0.25, 0.85))
        drift = _drift_for(phenotype, rng)
        n_sessions = int(rng.integers(MIN_SESSIONS, MAX_SESSIONS + 1))

        per_session_metrics: list[dict] = []

        for t in range(n_sessions):
            theta = float(np.clip(theta_base + drift * t + rng.normal(0, 0.05), 0.05, 0.99))
            raw = _sample_session_raw(theta, rng)
            metrics = _compute_metrics(raw)

            history = list(reversed(per_session_metrics[-HISTORY_WINDOW:]))
            ctx = _aggregate_history(history, metrics)
            target = _label_recommendation(metrics["sps"], ctx)

            row = {
                "PatientID": patient_id,
                "session_index": t,
                "phenotype": phenotype,
                **{
                    k: raw[k]
                    for k in (
                        "comprehension_score",
                        "correct_key_objects",
                        "correct_secondary_objects",
                        "incorrect_objects",
                        "total_key_objects",
                        "total_secondary_objects",
                        "correct_events",
                        "total_events",
                        "expected_interactions",
                        "interaction_events",
                        "total_questions",
                        "incorrect_answers",
                    )
                },
                "response_times": raw["response_times"],
                "ORS": metrics["ors"],
                "ERS": metrics["ers"],
                "SCS": metrics["scs"],
                "RTA": metrics["rta"],
                "ATS": metrics["ats"],
                "ER": metrics["er"],
                "SPS": metrics["sps"],
                "baseline_sps": ctx["baseline_sps"],
                "slope_sps": ctx["slope_sps"],
                "delta_sps": ctx["delta_sps"],
                "mean_ors": ctx["mean_ors"],
                "mean_ers": ctx["mean_ers"],
                "mean_er": ctx["mean_er"],
                "mean_rta": ctx["mean_rta"],
                "std_sps": ctx["std_sps"],
                "session_count": ctx["session_count"],
                "cold_start": ctx["cold_start"],
                "Target_Recommendation": target,
            }
            rows.append(row)
            per_session_metrics.append(metrics)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} sessions for {df['PatientID'].nunique()} patients")
    print(f"Saved to {OUTPUT_PATH}")
    print("\nTarget distribution:")
    print(df["Target_Recommendation"].value_counts(normalize=True).round(3))
    print("\nPhenotype distribution:")
    print(df["phenotype"].value_counts(normalize=True).round(3))
    print(f"\nORS range: [{df['ORS'].min():.3f}, {df['ORS'].max():.3f}]")
    print(f"SPS range: [{df['SPS'].min():.3f}, {df['SPS'].max():.3f}]")
