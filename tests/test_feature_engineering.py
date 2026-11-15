"""
Tests for the mouse-level features in src/feature_engineering.py.

Run from the project root with:   python -m pytest tests
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import feature_engineering as fe


def one_mouse():
    # 4 small-pellet trials (all Eat) and 4 large-pellet trials (2 Eat, 2 Carry)
    return pd.DataFrame({
        "Id": [1] * 8,
        "SizeCategory": ["small"] * 4 + ["large"] * 4,
        "Behavior": ["Eat"] * 4 + ["Eat", "Eat", "Carry", "Carry"],
        "PrevBehavior": ["SessionStart", "Eat", "Eat", "Eat", "SessionStart", "Eat", "Eat", "Carry"],
    })


def test_proportions_add_up_to_one():
    proportions = fe.behavior_proportions(one_mouse(), "ss")
    assert proportions.loc[1].sum() == 1
    assert proportions.loc[1, "ss_Carry"] == 0.25


def test_size_effect_is_large_minus_small():
    effects = fe.size_effects(one_mouse(), "ss")
    assert effects.loc[1, "ss_size_effect_Carry"] == 0.5    # 2/4 - 0/4
    assert effects.loc[1, "ss_size_effect_Eat"] == -0.5     # 2/4 - 4/4


def test_switch_rate_skips_session_starts():
    # known previous behaviors: Eat->Eat, Eat->Eat, Eat->Eat, Eat->Eat, Eat->Carry, Carry->Carry
    rate = fe.switch_rate(one_mouse(), "ss")
    assert abs(rate.loc[1, "ss_switch_rate"] - 1 / 6) < 1e-12
