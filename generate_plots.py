"""Generate public demonstration plots from fully synthetic baseball plays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from runner_advance import (
    MODEL_FEATURES,
    add_engineered_features,
    estimate_prevention_difficulty,
    summarize_fielders,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
RANDOM_SEED = 42


def simulate_plays(rows: int = 2500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create plausible but non-real advancement opportunities."""
    rng = np.random.default_rng(seed)
    fielder_ids = np.array([f"SYN_F{i:02d}" for i in range(1, 21)])
    fielder_skill = dict(zip(fielder_ids, rng.normal(0, 0.38, len(fielder_ids))))

    fielder_id = rng.choice(fielder_ids, rows)
    position = rng.choice(["LF", "CF", "RF"], rows, p=[0.3, 0.4, 0.3])
    runner_base = rng.choice([2, 3], rows, p=[0.58, 0.42])
    exit_speed = np.clip(rng.normal(92, 9, rows), 65, 115)
    launch_angle = np.clip(rng.normal(31, 11, rows), 5, 60)
    hit_distance = np.clip(
        185 + 2.5 * launch_angle + 0.55 * (exit_speed - 85) + rng.normal(0, 24, rows),
        150,
        390,
    )
    hangtime = np.clip(1.9 + hit_distance / 105 + rng.normal(0, 0.55, rows), 2.1, 7.2)
    angle = rng.uniform(-0.75, 0.75, rows)
    caught_pos_x = np.sin(angle) * hit_distance
    caught_pos_y = np.cos(angle) * hit_distance
    start_pos_x = caught_pos_x + rng.normal(0, 22, rows)
    start_pos_y = caught_pos_y + rng.normal(-8, 18, rows)
    throw_speed = np.clip(rng.normal(91.5, 5.0, rows), 75, 105)
    sprint_speed = np.clip(rng.normal(28.2, 1.6, rows), 23, 32)

    plays = pd.DataFrame(
        {
            "play_id": [f"SYN_{i:05d}" for i in range(rows)],
            "fielder_id": fielder_id,
            "fielder_position": position,
            "runner_base": runner_base,
            "exit_speed": exit_speed,
            "launch_angle": launch_angle,
            "hit_distance": hit_distance,
            "hangtime": hangtime,
            "start_pos_x": start_pos_x,
            "start_pos_y": start_pos_y,
            "caught_pos_x": caught_pos_x,
            "caught_pos_y": caught_pos_y,
            "fielder_throw_speed": throw_speed,
            "runner_sprint_speed": sprint_speed,
        }
    )
    plays = add_engineered_features(plays)

    # Synthetic data-generating process: longer throws, faster runners, and
    # shorter hang times increase advancement; fielder skill reduces it.
    linear_score = (
        -0.45
        + 0.022 * (plays["throw_distance"] - 250)
        - 0.85 * (plays["hangtime"] - 4.5)
        + 0.42 * (plays["runner_sprint_speed"] - 28)
        - 0.18 * (plays["fielder_throw_speed"] - 92)
        + 0.38 * (plays["runner_base"] == 3).astype(float)
        - np.array([fielder_skill[value] for value in fielder_id])
    )
    advance_probability = 1 / (1 + np.exp(-linear_score))
    plays["runner_advance"] = rng.binomial(1, advance_probability)
    return plays


def save_figure(filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plays = simulate_plays()
    training, evaluation = train_test_split(
        plays,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=plays["runner_advance"],
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    model.fit(training[MODEL_FEATURES], training["runner_advance"])
    probability = model.predict_proba(evaluation[MODEL_FEATURES])[:, 1]
    evaluation = evaluation.copy()
    evaluation["predicted_advance_probability"] = probability

    observed, predicted = calibration_curve(
        evaluation["runner_advance"], probability, n_bins=10, strategy="quantile"
    )
    plt.figure(figsize=(7.2, 5.2))
    plt.plot([0, 1], [0, 1], "--", color="#777777", label="Perfect calibration")
    plt.plot(predicted, observed, marker="o", color="#005F73", linewidth=2, label="Model")
    plt.xlabel("Predicted advancement probability")
    plt.ylabel("Observed advancement rate")
    plt.title("Synthetic Model Calibration")
    plt.legend(frameon=False)
    save_figure("model_calibration.png")

    false_positive, true_positive, _ = roc_curve(evaluation["runner_advance"], probability)
    roc_auc = auc(false_positive, true_positive)
    plt.figure(figsize=(7.2, 5.2))
    plt.plot(false_positive, true_positive, color="#0A9396", linewidth=2, label=f"Model AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="#777777", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Synthetic Runner Advancement ROC Curve")
    plt.legend(frameon=False)
    save_figure("model_roc_curve.png")

    plt.figure(figsize=(7.2, 5.2))
    sns.histplot(
        data=evaluation,
        x="predicted_advance_probability",
        hue="runner_advance",
        bins=24,
        stat="density",
        common_norm=False,
        element="step",
        alpha=0.28,
        palette={0: "#005F73", 1: "#EE9B00"},
    )
    plt.xlabel("Predicted advancement probability")
    plt.ylabel("Density")
    plt.title("Synthetic Advancement Probability Distribution")
    save_figure("advance_probability_distribution.png")

    plt.figure(figsize=(7.2, 5.2))
    sns.regplot(
        data=evaluation.sample(min(500, len(evaluation)), random_state=RANDOM_SEED),
        x="time_difference",
        y="predicted_advance_probability",
        scatter_kws={"alpha": 0.35, "s": 24, "color": "#005F73"},
        line_kws={"color": "#AE2012", "linewidth": 2},
        lowess=True,
    )
    plt.xlabel("Runner time minus throw time (seconds)")
    plt.ylabel("Predicted advancement probability")
    plt.title("Synthetic Timing Advantage and Advancement Probability")
    save_figure("timing_advantage.png")

    scored = estimate_prevention_difficulty(
        training,
        evaluation,
        neighbors=30,
    )
    fielder_summary = summarize_fielders(scored, minimum_plays=20).sort_values(
        "prevention_above_expected"
    )
    plt.figure(figsize=(8.0, 6.2))
    colors = ["#AE2012" if value < 0 else "#0A9396" for value in fielder_summary["prevention_above_expected"]]
    plt.barh(
        fielder_summary["fielder_id"],
        fielder_summary["prevention_above_expected"],
        color=colors,
    )
    plt.axvline(0, color="#444444", linewidth=1)
    plt.xlabel("Prevention above expected per opportunity")
    plt.ylabel("Synthetic fielder")
    plt.title("Synthetic Context-Adjusted Fielder Performance")
    save_figure("fielder_performance.png")

    print(f"Created five synthetic demonstration plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
