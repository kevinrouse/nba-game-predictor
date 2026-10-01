"""Compare development seasons and freeze the selection before test evaluation."""

import json
import joblib
import numpy as np
import pandas as pd
from src.data import ROOT
from src.features import feature_sets
from src.modeling import VALIDATION_SEASONS, TEST_SEASON, candidates, make_model, metrics, select_candidate
from scripts.build_data import TABLES, save_table


def main():
    data = pd.read_csv(TABLES / "matchups.csv.gz", parse_dates=["gamedate"])
    development = data[data.eligible & data.season.lt(TEST_SEASON)].copy()
    configs, sets = candidates(), feature_sets()
    folds, predictions, coefficients = [], [], []
    for number, config in enumerate(configs, 1):
        features = sets[config["features"]]
        for season in VALIDATION_SEASONS:
            train = development[development.season.lt(season)]
            valid = development[development.season.eq(season)]
            assert train.gamedate.max() < valid.gamedate.min()
            model = make_model(config).fit(train[features], train.y)
            p = model.predict_proba(valid[features])[:, 1]
            train_score = metrics(train.y, model.predict_proba(train[features])[:, 1])
            folds.append(dict(name=config["name"], season=season, train_n=len(train),
                              train_log_loss=train_score["log_loss"], **metrics(valid.y, p)))
            pred = valid[["nbagameid", "gamedate", "season", "y"]].copy()
            pred["probability"], pred["name"] = p, config["name"]
            predictions.append(pred)
            if config["kind"] == "logistic":
                coefficients.extend(dict(name=config["name"], season=season, feature=f, coefficient=float(c))
                                    for f, c in zip(features, model[-1].coef_[0]))
        print(f"{number}/{len(configs)} {config['name']}", flush=True)
    folds = pd.DataFrame(folds)
    summary = folds.groupby("name").agg(log_loss=("log_loss", "mean"), brier=("brier", "mean"),
                         auc=("auc", "mean"), accuracy=("accuracy", "mean"),
                         season_sd=("log_loss", "std"), train_log_loss=("train_log_loss", "mean")).reset_index()
    meta = pd.DataFrame([dict(name=c["name"], kind=c["kind"], features=c["features"], n_features=len(sets[c["features"]])) for c in configs])
    summary = summary.merge(meta, on="name").sort_values("log_loss")
    chosen = select_candidate(summary)
    config = next(c for c in configs if c["name"] == chosen)
    selection = dict(config=config, features=sets[config["features"]],
                     train_seasons=sorted(development.season.unique().astype(int).tolist()),
                     validation_seasons=VALIDATION_SEASONS, test_season=TEST_SEASON,
                     rule="Within 0.001 of minimum mean validation log loss: fewer features, then logistic, then score",
                     validation_log_loss=float(summary.set_index("name").loc[chosen, "log_loss"]))
    save_table(folds, "cv_folds.csv")
    save_table(summary, "cv_summary.csv")
    save_table(pd.concat(predictions), "cv_predictions.csv.gz")
    save_table(pd.DataFrame(coefficients), "cv_coefficients.csv.gz")
    (TABLES / "candidate_configs.json").write_text(json.dumps(configs, indent=2) + "\n")
    models = {}
    for conf in [config, *configs[:3]]:
        models[conf["name"]] = dict(model=make_model(conf).fit(development[sets[conf["features"]]], development.y),
                                     features=sets[conf["features"]])
    joblib.dump(models, ROOT / "artifacts/models/frozen_models.joblib", compress=3)
    (ROOT / "artifacts/models/selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    # The correlations and stability checks use development rows only.
    numeric = development[sets["combined10"]]
    correlation = numeric.corr()
    pairs = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "correlation"]
    save_table(pairs.reindex(pairs.correlation.abs().sort_values(ascending=False).index), "feature_correlations.csv")
    print(summary.head(12).to_string(index=False), flush=True)
    print("Frozen selection:", chosen, flush=True)


if __name__ == "__main__":
    main()
