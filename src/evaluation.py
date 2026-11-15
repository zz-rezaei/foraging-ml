"""
Leakage-safe evaluation with mice as the unit of validation.

Main idea: LEAVE-ONE-MOUSE-OUT.
For every mouse in turn:
  1. remove that mouse from the data  -> it becomes the test mouse
  2. tune and train the model on the remaining mice only
  3. predict every trial of the test mouse
After the loop, every trial has exactly one prediction, made by a model
that never saw ANY trial of that mouse. These are called
"out-of-fold" predictions, and all metrics are computed from them.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, GroupKFold


def leave_one_mouse_out(model, grid, data, features, target, classes, inner_splits=5):
    """Out-of-fold predictions with leave-one-mouse-out (outer loop) and
    grouped cross-validation for hyperparameter tuning (inner loop).

    This is NESTED cross-validation:
      - the outer loop estimates performance on unseen mice;
      - the inner loop (GroupKFold over the training mice only) chooses
        hyperparameters. The test mouse is never used for tuning.

    Returns a DataFrame with one row per trial: Id, true label, predicted
    label, one probability column per class, and the chosen hyperparameters.
    """
    results = []
    for test_mouse in data["Id"].unique():
        is_test = data["Id"] == test_mouse
        train = data[~is_test]
        test = data[is_test]

        if grid:
            search = GridSearchCV(
                clone(model), grid,
                cv=GroupKFold(n_splits=inner_splits),
                scoring="balanced_accuracy",
                n_jobs=-1,
            )
            search.fit(train[features], train[target], groups=train["Id"])
            fitted = search.best_estimator_
            chosen = str(search.best_params_)
        else:
            fitted = clone(model).fit(train[features], train[target])
            chosen = "-"

        # predict_proba columns follow fitted.classes_; reorder to `classes`
        probabilities = pd.DataFrame(fitted.predict_proba(test[features]),
                                     columns=fitted.classes_, index=test.index)
        probabilities = probabilities.reindex(columns=classes, fill_value=0.0)

        fold = pd.DataFrame({
            "Id": test["Id"],
            "true": test[target],
            "predicted": fitted.predict(test[features]),
            "hyperparameters": chosen,
        }, index=test.index)
        results.append(pd.concat([fold, probabilities], axis=1))

    return pd.concat(results).sort_index()


def train_on_one_set_test_on_another(model, train_data, test_data, features, target, classes):
    """Fit on all of `train_data` and predict `test_data` (used for
    cross-group transfer: train on Control mice, test on AD mice)."""
    fitted = clone(model).fit(train_data[features], train_data[target])
    probabilities = pd.DataFrame(fitted.predict_proba(test_data[features]),
                                 columns=fitted.classes_, index=test_data.index)
    probabilities = probabilities.reindex(columns=classes, fill_value=0.0)
    out = pd.DataFrame({"Id": test_data["Id"], "true": test_data[target],
                        "predicted": fitted.predict(test_data[features])},
                       index=test_data.index)
    return pd.concat([out, probabilities], axis=1)


def summary_metrics(predictions, classes):
    """Balanced accuracy, macro-F1 and multiclass log loss.

    PITFALL: sklearn's log_loss expects the probability columns in
    ALPHABETICAL order of the class names (Carry, Eat, NoLeave, NosePoke),
    whatever order is given in `labels`. Passing them in another order
    gives a wrong number without any error. We therefore sort explicitly.
    """
    alphabetical = sorted(classes)
    probabilities = predictions[alphabetical].to_numpy()
    probabilities = np.clip(probabilities, 1e-15, 1)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return {
        "balanced_accuracy": balanced_accuracy_score(predictions["true"], predictions["predicted"]),
        "macro_F1": f1_score(predictions["true"], predictions["predicted"],
                             labels=classes, average="macro", zero_division=0),
        "log_loss": log_loss(predictions["true"], probabilities, labels=alphabetical),
    }


def bootstrap_over_mice(predictions, classes, n_bootstrap=1000, seed=0):
    """Confidence intervals that respect the mouse as the unit.

    We resample MICE (not trials) with replacement, recompute the metrics
    on all trials of the resampled mice, and repeat. The spread of the
    results shows how much the score depends on which mice happened to be
    in the study. Returns one row per bootstrap sample.
    """
    rng = np.random.default_rng(seed)
    mice = predictions["Id"].unique()
    trials_by_mouse = {mouse: predictions[predictions["Id"] == mouse] for mouse in mice}

    rows = []
    for _ in range(n_bootstrap):
        sampled_mice = rng.choice(mice, size=len(mice), replace=True)
        sample = pd.concat([trials_by_mouse[m] for m in sampled_mice])
        rows.append(summary_metrics(sample, classes))
    return pd.DataFrame(rows)


def paired_bootstrap_difference(predictions_a, predictions_b, classes, metric="macro_F1",
                                n_bootstrap=1000, seed=0):
    """Bootstrap the DIFFERENCE in a metric between two models evaluated on
    the same trials (model B minus model A). The same mice are drawn for
    both models in every repetition, so the comparison is paired."""
    rng = np.random.default_rng(seed)
    mice = predictions_a["Id"].unique()
    a_by_mouse = {m: predictions_a[predictions_a["Id"] == m] for m in mice}
    b_by_mouse = {m: predictions_b[predictions_b["Id"] == m] for m in mice}

    differences = []
    for _ in range(n_bootstrap):
        sampled_mice = rng.choice(mice, size=len(mice), replace=True)
        sample_a = pd.concat([a_by_mouse[m] for m in sampled_mice])
        sample_b = pd.concat([b_by_mouse[m] for m in sampled_mice])
        differences.append(summary_metrics(sample_b, classes)[metric]
                           - summary_metrics(sample_a, classes)[metric])
    return np.array(differences)


# ---------------------------------------------------------------------------
# Mouse-level (Control vs AD) evaluation, used in notebook 03
# ---------------------------------------------------------------------------

def repeated_cv_mouse_level(model, grid, X, y, n_repeats=20, n_splits=5, seed=0):
    """Repeated stratified k-fold cross-validation with one row per MOUSE.

    In each repetition the mice are split into `n_splits` folds in a new
    random way (stratified: every fold keeps the AD/Control ratio). Every
    mouse gets one out-of-fold probability of being AD, and we compute the
    ROC-AUC and balanced accuracy over all mice. Repeating the split shows
    how much the result depends on the particular random split.

    If `grid` is not empty, the hyperparameters are tuned INSIDE each
    training set with an inner stratified 4-fold CV (nested CV).

    Returns (scores, mean_probability):
      scores           - one row per repetition (auc, balanced_accuracy)
      mean_probability - each mouse's out-of-fold P(AD), averaged over repetitions
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    score_rows = []
    all_probabilities = []
    for repetition in range(n_repeats):
        outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repetition)
        probability = pd.Series(np.nan, index=X.index)
        for train_index, test_index in outer.split(X, y):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            if grid:
                inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + repetition)
                search = GridSearchCV(clone(model), grid, cv=inner, scoring="roc_auc")
                search.fit(X_train, y_train)
                fitted = search.best_estimator_
            else:
                fitted = clone(model).fit(X_train, y_train)
            probability.iloc[test_index] = fitted.predict_proba(X.iloc[test_index])[:, 1]
        score_rows.append({
            "auc": roc_auc_score(y, probability),
            "balanced_accuracy": balanced_accuracy_score(y, (probability >= 0.5).astype(int)),
        })
        all_probabilities.append(probability)
    mean_probability = pd.concat(all_probabilities, axis=1).mean(axis=1)
    return pd.DataFrame(score_rows), mean_probability


def permutation_test_auc(model, X, y, n_permutations=1000, n_repeats=5, seed=0):
    """Is the AUC better than what we would get by chance?

    We shuffle the AD/Control labels across mice (destroying any real
    link between behavior and group), run the same cross-validation, and
    record the AUC. Repeating this many times gives the distribution of
    AUC values under "no real difference" (the null distribution).
    The p-value is the fraction of shuffled AUCs that are at least as
    high as the real one.

    To keep this fast, `model` should have fixed hyperparameters (no grid).
    Returns (observed_auc, null_aucs, p_value).
    """
    rng = np.random.default_rng(seed)
    observed = repeated_cv_mouse_level(model, {}, X, y, n_repeats=n_repeats, seed=seed)[0]["auc"].mean()
    null = []
    for _ in range(n_permutations):
        shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        null.append(repeated_cv_mouse_level(model, {}, X, shuffled, n_repeats=n_repeats, seed=seed)[0]["auc"].mean())
    null = np.array(null)
    p_value = (np.sum(null >= observed) + 1) / (n_permutations + 1)
    return observed, null, p_value


def per_class_table(predictions, classes):
    """Precision, recall and F1 for each class."""
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["true"], predictions["predicted"], labels=classes, zero_division=0)
    return pd.DataFrame({"precision": precision, "recall": recall, "F1": f1,
                         "n_trials": support}, index=classes)


def normalized_confusion_matrix(predictions, classes):
    """Confusion matrix where each ROW (true class) sums to 1, i.e. the
    diagonal shows the recall of each class."""
    matrix = confusion_matrix(predictions["true"], predictions["predicted"],
                              labels=classes, normalize="true")
    return pd.DataFrame(matrix, index=classes, columns=classes)
