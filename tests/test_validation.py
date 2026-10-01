import numpy as np
import pandas as pd
import pytest

from src.data import ROOT
from src.modeling import calendar_block_ids
from src.validation import replay_selection, uncertainty_sensitivity


def toy_candidates():
    meta = pd.DataFrame(
        [dict(name=n, kind="logistic", n_features=2) for n in ["A", "B"]]
    )
    folds = pd.DataFrame(
        [
            dict(
                season=s,
                name=n,
                log_loss=loss,
                brier=0.2,
                auc=0.7,
                accuracy=0.65,
                n=100,
            )
            for s, losses in [
                (2017, (0.4, 0.7)),
                (2018, (0.9, 0.3)),
                (2019, (0.6, 0.5)),
            ]
            for n, loss in zip(["A", "B"], losses)
        ]
    )
    return folds, meta


def test_selection_replay_uses_only_preceding_validation_scores():
    folds, meta = toy_candidates()
    original = replay_selection(folds, meta).set_index("season")
    assert original.selected_name.tolist() == ["A", "B"]
    assert (original.last_tuning_season < original.index).all()
    # Giving B perfect current/future scores cannot change the earlier choice.
    changed = folds.copy()
    changed.loc[changed.season.ge(2018) & changed.name.eq("B"), "log_loss"] = 0.001
    assert replay_selection(changed, meta).iloc[0].selected_name == "A"
    # An earlier score can change that choice: it is a real selection replay.
    changed.loc[changed.season.eq(2017) & changed.name.eq("B"), "log_loss"] = 0.1
    assert replay_selection(changed, meta).iloc[0].selected_name == "B"
    # Final aggregate scores are metadata, never a source for choosing a model.
    pd.testing.assert_frame_equal(
        replay_selection(folds, meta.assign(log_loss=[99, 0])),
        replay_selection(folds, meta.assign(log_loss=[0, 99])),
    )


def test_replay_rejects_test_season_and_incomplete_searches():
    folds, meta = toy_candidates()
    with pytest.raises(ValueError, match="same candidate library"):
        replay_selection(folds.iloc[:-1], meta)
    with pytest.raises(ValueError, match="development seasons"):
        replay_selection(folds.assign(season=2023), meta)


def test_calendar_blocks_and_original_intervals_are_preserved():
    dates = pd.Series(
        pd.to_datetime(["2024-01-07", "2024-01-08", "2024-01-14", "2024-01-15"])
    )
    assert calendar_block_ids(dates, 1).tolist() == [0, 1, 1, 2]
    assert calendar_block_ids(dates, 2).tolist() == [0, 0, 0, 1]
    with pytest.raises(ValueError, match="positive integer"):
        calendar_block_ids(dates, 0)
    tables = ROOT / "artifacts/tables"
    predictions = pd.read_csv(tables / "record_comparison_games.csv")
    reproduced = uncertainty_sensitivity(predictions, block_lengths=(1,)).set_index(
        "metric"
    )
    for filename, metric in [
        ("test_bootstrap.csv", "delta_log_loss"),
        ("record_accuracy_bootstrap.csv", "accuracy_difference"),
    ]:
        original = pd.read_csv(tables / filename).set_index("metric")
        np.testing.assert_allclose(
            reproduced.loc[metric, ["estimate", "lower", "upper"]],
            original.loc[metric, ["estimate", "lower", "upper"]],
            atol=1e-10,
            rtol=0,
        )


def test_all_block_lengths_keep_predictions_paired():
    predictions = pd.read_csv(ROOT / "artifacts/tables/record_comparison_games.csv")
    predictions["net_rating_probability"] = predictions.probability
    predictions["record_pick_home"] = predictions.probability.ge(0.5).astype(int)
    intervals = uncertainty_sensitivity(predictions, repetitions=100)
    np.testing.assert_array_equal(intervals[["estimate", "lower", "upper"]], 0)
    assert intervals.groupby("block_weeks").blocks.first().tolist() == [22, 11, 6]
