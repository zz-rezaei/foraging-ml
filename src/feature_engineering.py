"""
Mouse-level features for Control vs AD classification (notebook 03).

The mouse is the independent unit, so every feature is ONE number per
mouse, summarizing that mouse's trials. Each function below builds one
family of features and returns a DataFrame with one row per mouse
(index = mouse Id).

Rule used throughout: compare like with like. Behavior proportions depend
strongly on the condition (notebook 01), and not every mouse experienced
every condition. So most features are computed within ONE condition that
the mice share, instead of over "all trials".
"""

import numpy as np
import pandas as pd

BEHAVIORS = ["Eat", "Carry", "NosePoke", "NoLeave"]


def behavior_proportions(trials, prefix):
    """Proportion of each behavior per mouse, e.g. prefix='ss' gives
    columns ss_Eat, ss_Carry, ss_NosePoke, ss_NoLeave. The four values of
    a mouse add up to 1."""
    table = pd.crosstab(trials["Id"], trials["Behavior"], normalize="index")
    table = table.reindex(columns=BEHAVIORS, fill_value=0.0)
    table.columns = [f"{prefix}_{behavior}" for behavior in BEHAVIORS]
    return table


def size_effects(trials, prefix):
    """How much the choice changes from small to large pellets, per mouse:
    P(behavior | large) - P(behavior | small).
    A positive Carry value means the mouse carries more when the pellet is large."""
    small = behavior_proportions(trials[trials["SizeCategory"] == "small"], "p")
    large = behavior_proportions(trials[trials["SizeCategory"] == "large"], "p")
    effects = pd.DataFrame(index=small.index)
    for behavior in ["Eat", "Carry"]:
        effects[f"{prefix}_size_effect_{behavior}"] = large[f"p_{behavior}"] - small[f"p_{behavior}"]
    return effects


def switch_rate(trials, prefix):
    """Fraction of trials on which the mouse did something different from
    the previous trial of the same session. A simple measure of behavioral
    variability / flexibility. Trials without a known previous behavior
    (first trial of a session, previous trial unknown) are skipped."""
    known = trials[trials["PrevBehavior"].isin(BEHAVIORS)]
    switched = known["Behavior"] != known["PrevBehavior"]
    return switched.groupby(known["Id"]).mean().to_frame(f"{prefix}_switch_rate")


def distance_effects(short_trials, long_trials, prefix):
    """Change in choice from the short to the long distance (soft food),
    per mouse: P(behavior | long) - P(behavior | short).
    Only defined for mice that experienced both distances."""
    short = behavior_proportions(short_trials, "s")
    long = behavior_proportions(long_trials, "l")
    mice = short.index.intersection(long.index)
    effects = pd.DataFrame(index=mice)
    for behavior in ["Eat", "Carry", "NoLeave"]:
        effects[f"{prefix}_distance_effect_{behavior}"] = (
            long.loc[mice, f"l_{behavior}"] - short.loc[mice, f"s_{behavior}"])
    return effects


def movement_on_eat_trials(trials, prefix):
    """Movement summaries computed on EAT trials only.

    Restricting to one behavior is essential: movement differs strongly
    between behaviors (notebook 01, section 10.2). If AD mice simply ate
    more often, an all-trial average would differ between groups even if
    each behavior looked identical. Within Eat trials, a group difference
    reflects HOW the mice perform the same behavior.

    Medians are used because the variables are strongly skewed.
    """
    eat = trials[trials["Behavior"] == "Eat"]
    features = pd.DataFrame({
        f"{prefix}_delay_median": eat.groupby("Id")["DelayToStart"].median(),
        f"{prefix}_initial_speed_out_median": eat.groupby("Id")["InitialVelOut"].median(),
        f"{prefix}_time_at_food_median": eat.groupby("Id")["AtFoodArea_s"].median(),
        f"{prefix}_speed_in_median": eat.groupby("Id")["InboundVel"].median(),
        f"{prefix}_back_and_forth_fraction": (eat["BackAndForth"] > 0).groupby(eat["Id"]).mean(),
    })
    return features
