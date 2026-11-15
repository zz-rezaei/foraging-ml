"""
Model definitions for the behavior-classification task.

Every model is a scikit-learn Pipeline:   preprocessing  ->  classifier.
Putting preprocessing INSIDE the pipeline is what prevents data leakage:
when the pipeline is fitted on the training mice, the encoder and scaler
learn only from the training mice, never from the held-out mouse.
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np

RANDOM_STATE = 42


def make_preprocessor(categorical_features, numeric_features, log_transform=False):
    """One-hot encode categorical columns, impute + scale numeric columns.

    - OneHotEncoder turns e.g. Length = short/long into 0/1 columns.
      handle_unknown="ignore" means a category never seen in training
      (possible for rare PrevBehavior values) becomes all zeros instead
      of crashing.
    - SimpleImputer(median) fills ordinary missing numeric values with the
      median of the TRAINING data (only used in the retrospective analysis).
    - log1p (optional) compresses the long right tails of times and speeds.
    - StandardScaler gives every numeric column mean 0 and SD 1, so that
      logistic-regression coefficients are comparable and regularization
      treats all features equally.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if log_transform:
        numeric_steps.append(("log", FunctionTransformer(np.log1p)))
    numeric_steps.append(("scale", StandardScaler()))

    transformers = []
    if categorical_features:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))
    return ColumnTransformer(transformers)


def make_model(model_name, categorical_features, numeric_features, log_transform=False):
    """Return (pipeline, parameter_grid) for one of the supported models.

    The parameter grid lists the hyperparameter values tried during tuning.
    Grids are deliberately small: with 13-28 mice per group, a large search
    would mostly fit noise in the validation scores.
    """
    preprocessor = make_preprocessor(categorical_features, numeric_features, log_transform)

    if model_name == "Baseline (class frequencies)":
        # Always predicts the class proportions seen in training.
        classifier = DummyClassifier(strategy="prior")
        grid = {}

    elif model_name == "Logistic regression":
        classifier = LogisticRegression(class_weight="balanced", max_iter=5000)
        # C = inverse regularization strength: small C = simpler model
        grid = {"classifier__C": [0.01, 0.1, 1, 10]}

    elif model_name == "Logistic regression (unweighted)":
        # Same model WITHOUT class weights: worse at finding rare classes,
        # but its probabilities are honest (used for log loss / calibration).
        classifier = LogisticRegression(max_iter=5000)
        grid = {"classifier__C": [0.01, 0.1, 1, 10]}

    elif model_name == "Decision tree":
        classifier = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        grid = {"classifier__max_depth": [2, 4, 6],
                "classifier__min_samples_leaf": [20, 50]}

    elif model_name == "Random forest":
        # n_jobs=1: the grid search already runs several fits in parallel
        classifier = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=1)
        grid = {"classifier__max_depth": [4, 8, None],
                "classifier__min_samples_leaf": [10, 30]}

    elif model_name == "Gradient boosting":
        classifier = HistGradientBoostingClassifier(class_weight="balanced",
                                                    random_state=RANDOM_STATE)
        grid = {"classifier__learning_rate": [0.03, 0.1],
                "classifier__max_depth": [2, 4]}

    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
    return pipeline, grid


def make_group_classifier(model_name):
    """Models for Control vs AD classification with one row per mouse
    (notebook 03). All features are numeric mouse-level summaries.

    - median imputation: a safety net for the rare mouse without a value
    - scaling: needed for regularized logistic regression
    - class_weight="balanced": 13 AD vs 28 Control mice
    """
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]

    if model_name == "Logistic regression":
        steps.append(("classifier", LogisticRegression(class_weight="balanced", max_iter=5000)))
        grid = {"classifier__C": [0.01, 0.1, 1, 10]}
    elif model_name == "Random forest":
        steps.append(("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1)))
        grid = {}   # no tuning: too few mice for a reliable inner search
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(steps), grid
