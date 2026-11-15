"""
Data loading and cleaning for the mouse foraging dataset.

Every function here does ONE small step and returns a new DataFrame.
The notebooks call them one at a time so that the effect of each step
can be inspected.

IMPORTANT: the order in which these steps are applied matters.
Session numbers and "previous behavior" must be computed on the FULL,
correctly ordered data, *before* any rows are removed. Otherwise a
dropped trial would silently disappear from the history of the next one.
"""

import numpy as np
import pandas as pd


# Video frame rate used to convert frame counts into seconds.
# Evidence: most DelayToStart values are exact multiples of 1/30 s.
FRAMES_PER_SECOND = 30

# Velocity values of exactly 200 are a ceiling produced by the tracking
# software: dozens of trials sit at exactly 200.0, while values just
# below it occur only once each. 200 therefore means ">= 200", not a speed.
VELOCITY_CEILING = 200

MOVEMENT_COLUMNS = [
    "AtFoodArea", "BackAndForth", "DelayToStart",
    "InboundVel", "InitialVelIn", "InitialVelOut", "OutboundVel",
]
VELOCITY_COLUMNS = ["InboundVel", "InitialVelIn", "InitialVelOut", "OutboundVel"]


def load_raw_data(path):
    """Read the raw CSV and keep the original row order in a column.

    The file is already ordered chronologically within each mouse, and
    that order is the only reliable record of which trial came first.
    We store it as `RawRow` so it can always be restored.
    """
    df = pd.read_csv(path)
    df["RawRow"] = np.arange(len(df))
    df["Date"] = pd.to_datetime(df["Trial"], format="%Y_%m_%d")
    # Some PelletOrder strings contain invisible control characters
    # (e.g. '\x03null\x02\x01'). Keep only the visible characters.
    df["PelletOrder"] = df["PelletOrder"].str.replace(r"[^0-9a-z]", "", regex=True)
    return df


def merge_control_cohorts(df):
    """The two control cohorts ('Cont_beh', 'Cont_img') form one 'Control' group."""
    df = df.copy()
    df["Status"] = df["Status"].replace({"Cont_beh": "Control", "Cont_img": "Control"})
    return df


def drop_mouse_5028_block(df):
    """Remove the 16 March - 12 April 2021 block of mouse 5028.

    In this block the mouse received one pellet size per day and the
    recorded pellet order and trial numbers do not match the sizes.
    The lab confirmed that the task information for these days is not
    reliable, so the block is excluded.
    """
    in_block = (
        (df["Id"] == 5028)
        & (df["Date"] >= "2021-03-16")
        & (df["Date"] <= "2021-04-12")
    )
    return df[~in_block].copy()


def add_session_columns(df):
    """Give every trial a session id, a session number and its position.

    Why this is needed: the column `Trial` is a DATE, not a session.
    Some mice ran two or three sessions on the same day.

    Rule: a new session starts when
      - the mouse changes, or
      - the date changes, or
      - the current session already has >= 4 trials AND either the
        PelletOrder changes or TrialNumber goes back down (a restart).

    New columns:
      SessionId       unique id across the whole dataset
      SessionNumber   1, 2, 3, ... counted separately for each mouse
      TrialInSession  1, 2, 3, ... position of the trial in its session
    """
    df = df.sort_values("RawRow").copy()

    session_ids = []
    session_id = 0
    trials_in_current_session = 0
    previous = None

    for row in df.itertuples():
        start_new = False
        if previous is None:
            start_new = True
        elif row.Id != previous.Id or row.Trial != previous.Trial:
            start_new = True
        elif trials_in_current_session >= 4:
            order_changed = row.PelletOrder != previous.PelletOrder
            number_restarted = row.TrialNumber <= previous.TrialNumber  # False if either is NaN
            if order_changed or number_restarted:
                start_new = True

        if start_new:
            session_id += 1
            trials_in_current_session = 0

        trials_in_current_session += 1
        session_ids.append(session_id)
        previous = row

    df["SessionId"] = session_ids
    df["TrialInSession"] = df.groupby("SessionId").cumcount() + 1

    # Number the sessions 1, 2, 3, ... within each mouse.
    first_row_of_session = df.drop_duplicates("SessionId")[["Id", "SessionId"]].copy()
    first_row_of_session["SessionNumber"] = first_row_of_session.groupby("Id").cumcount() + 1
    df = df.merge(first_row_of_session[["SessionId", "SessionNumber"]], on="SessionId")
    return df


def remove_invalid_noleave_labels(df):
    """NoLeave trials that contain movement data are invalid
    (the lab confirmed these labels are wrong).

    We do NOT delete the rows here. Instead we erase their label
    (Behavior -> missing). Two things then happen automatically:
      - add_previous_behavior() gives the NEXT trial the history
        'Unknown', which is honest: we do not know what happened before it.
      - drop_unused_trials() removes these rows, like any unlabeled trial.
    """
    df = df.copy()
    has_movement_data = df[MOVEMENT_COLUMNS].notna().any(axis=1)
    is_invalid = (df["Behavior"] == "NoLeave") & has_movement_data
    df.loc[is_invalid, "Behavior"] = np.nan
    return df


def pool_eat_behaviors(df):
    """PartialEat is a form of eating at the food location -> 'Eat'."""
    df = df.copy()
    df["Behavior"] = df["Behavior"].replace({"PartialEat": "Eat"})
    return df


def add_previous_behavior(df):
    """Add the behavior of the previous trial in the SAME session.

    - First trial of a session -> 'SessionStart'
    - Previous trial had no label -> 'Unknown'
    - NoApproach is kept here as a real previous outcome, even though
      NoApproach trials themselves will later be removed as targets.

    This only uses information from the past, so it is a valid
    predictor for a prospective model.
    """
    df = df.sort_values("RawRow").copy()
    previous = df.groupby("SessionId")["Behavior"].shift(1)
    previous = previous.fillna("Unknown")
    is_first_trial = df["TrialInSession"] == 1
    previous[is_first_trial] = "SessionStart"
    df["PrevBehavior"] = previous
    return df


def drop_unused_trials(df):
    """Remove trials that cannot be used for classification:
    rows without a behavior label, rows without a pellet size
    (the task condition is unknown), and NoApproach trials."""
    has_label = df["Behavior"].notna()
    has_size = df["PelletSize"].notna()
    not_noapproach = df["Behavior"] != "NoApproach"
    return df[has_label & has_size & not_noapproach].copy()


def fix_measurement_artifacts(df):
    """Correct known measurement problems in the movement columns.

    1. Negative DelayToStart (21 trials, at most -0.83 s): the mouse
       started moving a few frames before the gate signal was
       registered. Set to 0 = "started immediately".
    2. Velocities (cm/s) of exactly 200 are a tracking ceiling -> set to NaN.
    3. AtFoodArea is recorded in video frames -> add a column in seconds.

    Note: NoLeave trials have no movement values at this point, because
    the invalid NoLeave trials were removed earlier.
    """
    df = df.copy()

    df.loc[df["DelayToStart"] < 0, "DelayToStart"] = 0

    for column in VELOCITY_COLUMNS:
        df.loc[df[column] >= VELOCITY_CEILING, column] = np.nan

    df["AtFoodArea_s"] = df["AtFoodArea"] / FRAMES_PER_SECOND
    return df


def add_pellet_size_category(df):
    """Pool pellet sizes 1-2 -> 'small' and 3-4 -> 'large'."""
    df = df.copy()
    df["SizeCategory"] = np.where(df["PelletSize"] <= 2, "small", "large")
    df.loc[df["PelletSize"].isna(), "SizeCategory"] = np.nan
    return df
