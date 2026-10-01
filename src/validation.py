"""Retrospective checks of model selection and uncertainty assumptions."""

import numpy as np
import pandas as pd

from src.analysis import paired_accuracy_bootstrap
from src.modeling import TEST_SEASON, select_candidate, week_bootstrap


def replay_selection(folds, candidates):
    """Choose before each season using only scores from earlier validation seasons.

    The original candidate library is held fixed. This audits the decision rule;
    it does not make the historically designed library a new independent test.
    """
    if folds.empty or folds.season.ge(TEST_SEASON).any():
        raise ValueError("Selection replay requires development seasons only")
    if folds.duplicated(["season", "name"]).any():
        raise ValueError("A candidate must have one score per validation season")
    meta = candidates[["name", "kind", "n_features"]].copy()
    if meta.name.duplicated().any():
        raise ValueError("Candidate names must be unique")
    expected = set(meta.name)
    if any(set(group.name) != expected for _, group in folds.groupby("season")):
        raise ValueError("Every season must score the same candidate library")
    if not np.isfinite(folds.log_loss).all():
        raise ValueError("Every candidate needs a finite validation score")
    rows = []
    for season in sorted(folds.season.unique())[1:]:
        past = folds[folds.season.lt(season)]
        summary = (
            past.groupby("name")
            .log_loss.mean()
            .reset_index()
            .merge(meta, on="name", validate="one_to_one")
        )
        choice = select_candidate(summary)
        scored = folds[folds.season.eq(season)].set_index("name").loc[choice]
        rows.append(
            dict(
                season=int(season),
                last_tuning_season=int(past.season.max()),
                prior_validation_seasons=int(past.season.nunique()),
                candidates=len(meta),
                selected_name=choice,
                prior_mean_log_loss=float(
                    summary.set_index("name").loc[choice, "log_loss"]
                ),
                **{c: scored[c] for c in ["log_loss", "brier", "auc", "accuracy"]},
                n=int(scored["n"]),
            )
        )
    return pd.DataFrame(rows)


def uncertainty_sensitivity(predictions, block_lengths=(1, 2, 4), repetitions=2000):
    """Report every specified block length; do not choose the most favorable one."""
    rows = []
    for weeks in block_lengths:
        accuracy = paired_accuracy_bootstrap(predictions, repetitions, weeks)
        probability = week_bootstrap(
            predictions, repetitions=repetitions, block_weeks=weeks
        )
        for frame, metric in [
            (accuracy, "accuracy_difference"),
            (probability, "delta_log_loss"),
        ]:
            row = frame.set_index("metric").loc[metric].to_dict()
            row["blocks"] = int(row["blocks"])
            row["repetitions"] = int(row["repetitions"])
            rows.append(dict(block_weeks=weeks, metric=metric, **row))
    return pd.DataFrame(rows)
