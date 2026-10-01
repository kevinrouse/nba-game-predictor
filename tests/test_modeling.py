import json

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT
from src.modeling import select_candidate, week_bootstrap

TABLES = ROOT / "artifacts/tables"


def test_published_predictions_match_frozen_model():
    data = pd.read_csv(TABLES / "matchups.csv.gz")
    selection = json.loads((ROOT / "artifacts/models/selection.json").read_text())
    assert max(selection["train_seasons"]) < selection["test_season"]
    assert max(selection["validation_seasons"]) < selection["test_season"]
    model = joblib.load(ROOT / "artifacts/models/frozen_models.joblib")[
        selection["config"]["name"]
    ]
    train = data[data.eligible & data.season.lt(selection["test_season"])]
    training_values = train[model["features"]]
    np.testing.assert_allclose(
        model["model"][0].statistics_, training_values.median(), atol=1e-9
    )
    np.testing.assert_allclose(
        model["model"][1].mean_,
        model["model"][0].transform(training_values).mean(axis=0),
        atol=1e-9,
    )
    test = data[data.eligible & data.season.eq(selection["test_season"])]
    saved = pd.read_csv(TABLES / "test_predictions.csv")
    assert saved.nbagameid.tolist() == test.nbagameid.tolist()
    p = model["model"].predict_proba(test[model["features"]])[:, 1]
    np.testing.assert_allclose(saved.probability, p, atol=1e-9)
    np.testing.assert_allclose(saved.probability + saved.away_probability, 1, atol=1e-9)
    assert saved.probability.between(0, 1).all()
    assert not saved.nbagameid.duplicated().any()


def test_selection_follows_prespecified_rule():
    summary = pd.read_csv(TABLES / "cv_summary.csv")
    selected = json.loads((ROOT / "artifacts/models/selection.json").read_text())
    assert select_candidate(summary) == selected["config"]["name"]
    mock = pd.DataFrame(
        [
            dict(name="complex", kind="forest", log_loss=0.6000, n_features=20),
            dict(name="simple", kind="logistic", log_loss=0.6008, n_features=5),
            dict(name="too weak", kind="logistic", log_loss=0.6020, n_features=1),
        ]
    )
    assert select_candidate(mock) == "simple"


def test_paired_bootstrap_has_zero_difference_for_identical_models():
    saved = pd.read_csv(TABLES / "test_predictions.csv")
    saved["net_rating_probability"] = saved.probability
    interval = week_bootstrap(saved, repetitions=100).set_index("metric")
    np.testing.assert_allclose(
        interval.loc[["delta_log_loss", "delta_brier"], ["estimate", "lower", "upper"]],
        0,
    )
