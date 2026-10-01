"""Evaluate frozen models once and save basketball-oriented summaries."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from scripts.build_data import TABLES, save_table
from src.data import ROOT
from src.modeling import SEED, TEST_SEASON, calibration_table, metrics, week_bootstrap


def main():
    data = pd.read_csv(TABLES / "matchups.csv.gz", parse_dates=["gamedate"])
    selection = json.loads((ROOT / "artifacts/models/selection.json").read_text())
    models = joblib.load(ROOT / "artifacts/models/frozen_models.joblib")
    name = selection["config"]["name"]
    test = data[data.eligible & data.season.eq(TEST_SEASON)].copy()
    predictions = test[
        [
            "nbagameid",
            "gamedate",
            "season",
            "home_off_team",
            "away_off_team",
            "home_off_team_name",
            "away_off_team_name",
            "home_points",
            "away_points",
            "y",
        ]
    ].copy()
    scores = []
    aliases = {
        "Home win rate": "home_probability",
        "Baseline win_pct": "win_pct_probability",
        "Baseline net_rating": "net_rating_probability",
    }
    for candidate, bundle in models.items():
        p = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        predictions[aliases.get(candidate, "probability")] = p
        if candidate == name:
            predictions["probability"] = p
        scores.append(
            dict(model=candidate, selected=candidate == name, **metrics(test.y, p))
        )
    selected = models[name]
    estimator = selected["model"][-1]
    p = predictions.probability
    predictions["away_probability"] = 1 - p
    predictions["confidence"] = np.maximum(p, 1 - p)
    predictions["correct"] = (p >= 0.5).eq(predictions.y)
    predictions["result_surprise"] = np.where(predictions.y.eq(1), 1 - p, p)
    save_table(pd.DataFrame(scores), "test_metrics.csv")
    save_table(predictions, "test_predictions.csv")
    save_table(calibration_table(test.y, p), "test_calibration.csv")
    save_table(week_bootstrap(predictions), "test_bootstrap.csv")
    save_table(
        predictions[~predictions.correct]
        .sort_values("confidence", ascending=False)
        .head(25),
        "biggest_misses.csv",
    )
    home = pd.DataFrame(
        dict(team=predictions.home_off_team, probability=p, actual=predictions.y)
    )
    away = pd.DataFrame(
        dict(
            team=predictions.away_off_team, probability=1 - p, actual=1 - predictions.y
        )
    )
    team_errors = (
        pd.concat([home, away])
        .groupby("team")
        .agg(
            predicted_win_rate=("probability", "mean"),
            actual_win_rate=("actual", "mean"),
            games=("actual", "size"),
        )
        .reset_index()
    )
    team_errors["actual_minus_predicted"] = (
        team_errors.actual_win_rate - team_errors.predicted_win_rate
    )
    save_table(team_errors.sort_values("actual_minus_predicted"), "team_errors.csv")
    if hasattr(estimator, "coef_"):
        importance = pd.DataFrame(
            dict(feature=selected["features"], coefficient=estimator.coef_[0])
        )
        importance["absolute_importance"] = importance.coefficient.abs()
        importance["odds_multiplier_per_sd"] = np.exp(importance.coefficient)
        importance["training_sd"] = selected["model"][1].scale_
        importance["coefficient_per_unit"] = (
            importance.coefficient / importance.training_sd
        )
        save_table(
            importance.sort_values("absolute_importance", ascending=False),
            "importance.csv",
        )
        (TABLES / "intercept.json").write_text(
            json.dumps(
                {
                    "intercept": float(estimator.intercept_[0]),
                    "probability_at_training_means": float(
                        1 / (1 + np.exp(-estimator.intercept_[0]))
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    else:
        permutation = permutation_importance(
            selected["model"],
            test[selected["features"]],
            test.y,
            scoring="neg_log_loss",
            n_repeats=20,
            random_state=SEED,
            n_jobs=-1,
        )
        importance = pd.DataFrame(
            dict(
                feature=selected["features"],
                absolute_importance=permutation.importances_mean,
                std=permutation.importances_std,
            )
        )
        save_table(
            importance.sort_values("absolute_importance", ascending=False),
            "importance.csv",
        )
    # Calibration slope/intercept are diagnostics, never used to recalibrate the test.
    from sklearn.linear_model import LogisticRegression

    logits = np.log(p / (1 - p)).to_numpy().reshape(-1, 1)
    diagnostic = LogisticRegression(C=1e8, max_iter=1000).fit(logits, test.y)
    bins = calibration_table(test.y, p)
    result = dict(
        selected_model=name,
        test_season=TEST_SEASON,
        **metrics(test.y, p),
        observed_home_win_rate=float(test.y.mean()),
        predicted_home_win_rate=float(p.mean()),
        calibration_intercept=float(diagnostic.intercept_[0]),
        calibration_slope=float(diagnostic.coef_[0, 0]),
        calibration_ece=float(
            np.average(abs(bins.mean_probability - bins.win_rate), weights=bins.n)
        ),
        train_games=int((data.eligible & data.season.lt(TEST_SEASON)).sum()),
        validation_log_loss=selection["validation_log_loss"],
    )
    (TABLES / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
