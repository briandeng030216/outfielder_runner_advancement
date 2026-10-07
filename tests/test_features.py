import math

import pandas as pd

from runner_advance import add_engineered_features, throw_distance, travel_time


def test_throw_to_home_uses_origin() -> None:
    assert math.isclose(throw_distance(30, 40, 3), 50.0)


def test_travel_time_converts_mph_to_feet_per_second() -> None:
    assert math.isclose(travel_time(88, 60), 1.0, rel_tol=1e-4)


def test_feature_engineering_preserves_rows() -> None:
    source = pd.DataFrame(
        {
            "runner_base": [3],
            "caught_pos_x": [30.0],
            "caught_pos_y": [40.0],
            "start_pos_x": [20.0],
            "start_pos_y": [30.0],
            "fielder_throw_speed": [90.0],
            "runner_sprint_speed": [30.0],
            "exit_speed": [100.0],
            "launch_angle": [25.0],
            "hit_distance": [300.0],
            "hangtime": [4.0],
        }
    )
    result = add_engineered_features(source)
    assert len(result) == 1
    assert result.loc[0, "throw_distance"] == 50.0
    assert "time_difference" in result
