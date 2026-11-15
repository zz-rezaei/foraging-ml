"""
Tests for src/evaluation.py.

Run from the project root with:   python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluation

CLASSES = ["Eat", "Carry", "NosePoke", "NoLeave"]   # NOT alphabetical on purpose


def test_log_loss_does_not_depend_on_column_order():
    # A perfect, confident prediction must give a log loss near 0,
    # whatever order the probability columns are stored in.
    predictions = pd.DataFrame({
        "Id": [1, 1, 2, 2],
        "true": ["Eat", "Carry", "NosePoke", "NoLeave"],
        "predicted": ["Eat", "Carry", "NosePoke", "NoLeave"],
    })
    for behavior in CLASSES:
        predictions[behavior] = (predictions["true"] == behavior).astype(float)
    assert evaluation.summary_metrics(predictions, CLASSES)["log_loss"] < 1e-6


class RememberTrainingMice(BaseEstimator, ClassifierMixin):
    """A fake model that records which mice it was trained on."""
    seen = []

    def fit(self, X, y):
        RememberTrainingMice.seen.append(set(X["Id"]))
        self.classes_ = np.array(sorted(set(y)))
        return self

    def predict(self, X):
        return np.array([self.classes_[0]] * len(X))

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


def test_leave_one_mouse_out_never_trains_on_the_test_mouse():
    data = pd.DataFrame({
        "Id": [1, 1, 2, 2, 3, 3],
        "Behavior": ["Eat", "Carry", "Eat", "NosePoke", "NoLeave", "Eat"],
    })
    RememberTrainingMice.seen = []
    predictions = evaluation.leave_one_mouse_out(
        RememberTrainingMice(), {}, data, ["Id"], "Behavior", CLASSES)

    # one model per mouse, and each model saw exactly the other mice
    assert RememberTrainingMice.seen == [{2, 3}, {1, 3}, {1, 2}]
    # every trial got exactly one out-of-fold prediction
    assert len(predictions) == len(data)
