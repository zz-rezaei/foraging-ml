"""
Simple tests for the cleaning steps in src/data_processing.py.

Each test builds a tiny hand-made table where we know the correct answer,
runs one function on it, and checks the result.

Run from the project root with:   python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_processing as dp


def make_two_sessions_on_one_day():
    """One mouse, one date, 8 trials = two sessions with different pellet orders."""
    return pd.DataFrame({
        "Id": [1] * 8,
        "Trial": ["2021_01_01"] * 8,
        "PelletOrder": ["1234"] * 4 + ["4321"] * 4,
        "TrialNumber": [1, 2, 3, 4, 1, 2, 3, 4],
        "PelletSize": [1, 2, 3, 4, 4, 3, 2, 1],
        "Behavior": ["Eat", "NoApproach", "Carry", "Eat", "NoLeave", "Eat", "PartialEat", "Eat"],
        "RawRow": range(8),
    })


def test_two_sessions_on_the_same_day_are_separated():
    df = dp.add_session_columns(make_two_sessions_on_one_day())
    assert df["SessionId"].nunique() == 2
    assert list(df["TrialInSession"]) == [1, 2, 3, 4, 1, 2, 3, 4]
    assert list(df["SessionNumber"]) == [1, 1, 1, 1, 2, 2, 2, 2]


def test_previous_behavior_stays_within_session_and_ignores_later_removal():
    df = dp.add_session_columns(make_two_sessions_on_one_day())
    df = dp.pool_eat_behaviors(df)
    df = dp.add_previous_behavior(df)

    # first trial of each session has no history
    assert df.loc[0, "PrevBehavior"] == "SessionStart"
    assert df.loc[4, "PrevBehavior"] == "SessionStart"   # not "Eat" from the other session
    # trial 3 followed a NoApproach trial
    assert df.loc[2, "PrevBehavior"] == "NoApproach"

    # after removing NoApproach, trial 3 must STILL remember it followed NoApproach
    cleaned = dp.drop_unused_trials(df)
    assert "NoApproach" not in set(cleaned["Behavior"])
    assert cleaned.loc[2, "PrevBehavior"] == "NoApproach"


def test_partial_eat_is_pooled_into_eat():
    df = dp.pool_eat_behaviors(make_two_sessions_on_one_day())
    assert "PartialEat" not in set(df["Behavior"])


def test_measurement_artifacts_are_fixed():
    df = pd.DataFrame({
        "Behavior": ["Eat", "Carry", "NoLeave"],
        "AtFoodArea": [300.0, 60.0, 90.0],
        "BackAndForth": [0.0, 1.0, 0.0],
        "DelayToStart": [-0.5, 3.0, 200.0],
        "InboundVel": [200.0, 20.0, 15.0],
        "InitialVelIn": [10.0, 10.0, 10.0],
        "InitialVelOut": [10.0, 10.0, 10.0],
        "OutboundVel": [25.0, 30.0, 12.0],
    })
    fixed = dp.fix_measurement_artifacts(df)
    assert fixed.loc[0, "DelayToStart"] == 0                 # negative -> 0
    assert np.isnan(fixed.loc[0, "InboundVel"])               # ceiling -> missing
    assert fixed.loc[0, "AtFoodArea_s"] == 10                 # 300 frames / 30 fps


def test_invalid_noleave_trials_are_removed_and_next_trial_gets_unknown_history():
    df = pd.DataFrame({
        "Id": [1, 1, 1],
        "Trial": ["2021_01_01"] * 3,
        "PelletOrder": ["123"] * 3,
        "TrialNumber": [1, 2, 3],
        "PelletSize": [1, 2, 3],
        "Behavior": ["Eat", "NoLeave", "Carry"],
        "RawRow": [0, 1, 2],
    })
    for column in dp.MOVEMENT_COLUMNS:
        df[column] = 5.0                     # the NoLeave trial has movement data -> invalid
    df = dp.add_session_columns(df)
    df = dp.remove_invalid_noleave_labels(df)
    df = dp.add_previous_behavior(df)
    cleaned = dp.drop_unused_trials(df)

    assert list(cleaned["Behavior"]) == ["Eat", "Carry"]
    assert cleaned.loc[2, "PrevBehavior"] == "Unknown"


def test_mouse_5028_block_is_dropped():
    df = pd.DataFrame({
        "Id": [5028, 5028, 5028, 4988],
        "Date": pd.to_datetime(["2021-03-12", "2021-03-16", "2021-04-12", "2021-03-20"]),
    })
    kept = dp.drop_mouse_5028_block(df)
    assert list(kept.index) == [0, 3]       # other dates and other mice are kept


def test_pellet_sizes_are_pooled():
    df = pd.DataFrame({"PelletSize": [1, 2, 3, 4, np.nan]})
    pooled = dp.add_pellet_size_category(df)
    assert list(pooled["SizeCategory"][:4]) == ["small", "small", "large", "large"]
    assert pd.isna(pooled["SizeCategory"][4])
