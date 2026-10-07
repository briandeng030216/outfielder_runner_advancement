"""Reusable features and evaluation helpers for runner-advancement analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


MPH_TO_FPS = 1.46667
BASE_COORDINATES = {
    2: (-63.64, 63.64),  # Runner on second advances toward third.
    3: (0.0, 0.0),      # Runner on third advances toward home.
}

MODEL_FEATURES = [
    "hit_distance",
    "hangtime",
    "fielder_throw_speed",
    "runner_sprint_speed",
    "exit_speed",
    "speed_ratio",
    "launch_angle",
    "caught_pos_x",
    "caught_pos_y",
    "runner_time",
    "throw_time",
    "time_difference",
    "throw_distance",
    "fielder_run_distance",
]

CONTEXT_FEATURES = [
    "hit_distance",
    "hangtime",
    "runner_sprint_speed",
    "exit_speed",
    "launch_angle",
    "caught_pos_x",
    "caught_pos_y",
    "runner_time",
    "throw_distance",
]


def throw_distance(caught_x: float, caught_y: float, runner_base: int) -> float:
    """Calculate the direct distance from the catch point to the target base."""
    if runner_base not in BASE_COORDINATES:
        raise ValueError("runner_base must be 2 or 3")
    target_x, target_y = BASE_COORDINATES[runner_base]
    return float(np.hypot(caught_x - target_x, caught_y - target_y))


def travel_time(distance_feet: float, speed_mph: float) -> float:
    """Convert a distance and speed into a theoretical travel time in seconds."""
    if speed_mph <= 0:
        raise ValueError("speed_mph must be positive")
    return float(distance_feet / (speed_mph * MPH_TO_FPS))


def add_engineered_features(plays: pd.DataFrame) -> pd.DataFrame:
    """Add the physical features used by the public demonstration workflow."""
    required = {
        "runner_base",
        "caught_pos_x",
        "caught_pos_y",
        "start_pos_x",
        "start_pos_y",
        "fielder_throw_speed",
        "runner_sprint_speed",
        "exit_speed",
        "launch_angle",
        "hit_distance",
        "hangtime",
    }
    missing = required.difference(plays.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = plays.copy()
    result["throw_distance"] = result.apply(
        lambda row: throw_distance(
            row["caught_pos_x"], row["caught_pos_y"], int(row["runner_base"])
        ),
        axis=1,
    )
    result["runner_time"] = 90 / (result["runner_sprint_speed"] * MPH_TO_FPS)
    result["throw_time"] = result["throw_distance"] / (
        result["fielder_throw_speed"] * MPH_TO_FPS
    )
    result["time_difference"] = result["runner_time"] - result["throw_time"]
    result["speed_ratio"] = (
        result["runner_sprint_speed"] / result["fielder_throw_speed"]
    )
    result["fielder_run_distance"] = np.hypot(
        result["start_pos_x"] - result["caught_pos_x"],
        result["start_pos_y"] - result["caught_pos_y"],
    )
    result["launch_angle_squared"] = result["launch_angle"] ** 2
    result["exit_speed_squared"] = result["exit_speed"] ** 2
    result["log_hit_distance"] = np.log1p(result["hit_distance"])
    result["runner_window"] = result["runner_sprint_speed"] * result["hangtime"]
    return result


def classification_metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    """Return probability-focused metrics for an advancement model."""
    return {
        "log_loss": float(log_loss(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def estimate_prevention_difficulty(
    reference: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    features: list[str] | None = None,
    neighbors: int = 20,
    outcome_column: str = "runner_advance",
) -> pd.DataFrame:
    """Estimate expected prevention probability from similar historical plays."""
    features = features or CONTEXT_FEATURES
    if neighbors < 1 or neighbors > len(reference):
        raise ValueError("neighbors must be between 1 and the reference sample size")

    scaler = StandardScaler()
    reference_scaled = scaler.fit_transform(reference[features])
    evaluation_scaled = scaler.transform(evaluation[features])
    nearest = NearestNeighbors(n_neighbors=neighbors).fit(reference_scaled)
    _, indices = nearest.kneighbors(evaluation_scaled)

    outcomes = reference[outcome_column].to_numpy()
    result = evaluation.copy()
    result["expected_advance_probability"] = outcomes[indices].mean(axis=1)
    result["expected_prevention_probability"] = (
        1 - result["expected_advance_probability"]
    )
    if outcome_column in result:
        result["actual_prevention"] = 1 - result[outcome_column]
        result["prevention_above_expected"] = (
            result["actual_prevention"]
            - result["expected_prevention_probability"]
        )
    return result


def summarize_fielders(
    scored_plays: pd.DataFrame,
    *,
    fielder_column: str = "fielder_id",
    minimum_plays: int = 1,
) -> pd.DataFrame:
    """Aggregate context-adjusted prevention performance by fielder."""
    required = {
        fielder_column,
        "runner_advance",
        "expected_advance_probability",
        "prevention_above_expected",
    }
    missing = required.difference(scored_plays.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    summary = (
        scored_plays.groupby(fielder_column)
        .agg(
            plays=(fielder_column, "size"),
            actual_advance_rate=("runner_advance", "mean"),
            expected_advance_rate=("expected_advance_probability", "mean"),
            prevention_above_expected=("prevention_above_expected", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["plays"] >= minimum_plays].copy()
    summary["context_adjusted_rank"] = summary[
        "prevention_above_expected"
    ].rank(ascending=False, method="min")
    return summary.sort_values("context_adjusted_rank")
