import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import better_record_picks, paired_accuracy_bootstrap
from src.data import ROOT


def test_record_rule_uses_percentage_and_exact_ties():
    games = pd.DataFrame(
        {
            "home_prior_games": [10, 12, 6, 10],
            "away_prior_games": [10, 10, 9, 12],
            "home_win_pct": [0.6, 7 / 12, round(2 / 6, 10), 0.4],
            "away_win_pct": [0.5, 0.6, round(3 / 9, 10), 0.5],
            # These fields must not affect a pregame record decision.
            "y": [0, 1, 0, 1],
            "home_points": [10, 200, 10, 200],
        }
    )
    result = better_record_picks(games)
    assert result.record_pick_home.tolist() == [1, 0, 1, 0]
    assert result.record_tie.tolist() == [False, False, True, False]
    altered = games.assign(y=1 - games.y, home_points=0)
    pd.testing.assert_frame_equal(result, better_record_picks(altered))
    with pytest.raises(ValueError, match="prior games"):
        better_record_picks(games.assign(home_prior_games=0))


def test_record_report_uses_same_games_and_consistent_counts():
    tables = ROOT / "artifacts/tables"
    report = pd.read_csv(tables / "record_comparison_games.csv")
    original = pd.read_csv(tables / "test_predictions.csv")
    assert report.nbagameid.tolist() == original.nbagameid.tolist()
    np.testing.assert_array_equal(report.probability, original.probability)
    assert report.record_correct.sum() == 688
    assert report.model_correct.sum() == 723
    assert report.record_tie.sum() == 18
    matrix = pd.read_csv(tables / "record_disagreements.csv")
    assert matrix.games.sum() == len(original)


def test_paired_accuracy_interval_preserves_pairing():
    frame = pd.read_csv(ROOT / "artifacts/tables/record_comparison_games.csv")
    frame["record_pick_home"] = frame.probability.ge(0.5).astype(int)
    interval = paired_accuracy_bootstrap(frame, repetitions=100).set_index("metric")
    np.testing.assert_array_equal(
        interval.loc["accuracy_difference", ["estimate", "lower", "upper"]], 0
    )


def test_explanations_reconstruct_predictions_and_worked_features():
    tables = ROOT / "artifacts/tables"
    contributions = pd.read_csv(tables / "prediction_contributions.csv.gz")
    predictions = pd.read_csv(tables / "test_predictions.csv").set_index("nbagameid")
    guide = pd.read_csv(tables / "feature_guide.csv")
    assert not contributions.duplicated(["nbagameid", "feature"]).any()
    assert contributions.groupby("nbagameid").size().eq(22).all()
    assert set(contributions.feature) == set(guide.feature)
    assert set(contributions.nbagameid) == set(predictions.index)
    intercept = json.loads((tables / "intercept.json").read_text())["intercept"]
    logits = contributions.groupby("nbagameid").contribution.sum() + intercept
    np.testing.assert_allclose(
        1 / (1 + np.exp(-logits)),
        predictions.loc[logits.index, "probability"],
        atol=1e-9,
        rtol=0,
    )

    example = json.loads((tables / "worked_example.json").read_text())
    roles = pd.read_csv(tables / "worked_player_roles.csv")
    matchups = pd.read_csv(tables / "matchups.csv.gz").set_index("nbagameid")
    match = matchups.loc[example["game_id"]]
    assert pd.to_datetime(roles.history_end).lt(pd.Timestamp(example["gamedate"])).all()
    np.testing.assert_allclose([roles.role_weight.sum(), roles.scoring_weight.sum()], 1)
    np.testing.assert_allclose(
        [
            roles.ppa_contribution.sum(),
            roles.absence_contribution.sum(),
            roles.participation_contribution.sum(),
        ],
        [
            match.home_weighted_ppa,
            match.home_weighted_absence,
            match.home_weighted_participation,
        ],
        atol=1e-9,
        rtol=0,
    )
