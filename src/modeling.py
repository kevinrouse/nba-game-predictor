"""Small model candidates and chronological evaluation helpers."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import feature_sets

SEED = 78
VALIDATION_SEASONS = list(range(2017, 2023))
TEST_SEASON = 2023


def candidates():
    configs = [dict(name="Home win rate", kind="home", features="net_rating")]
    for features in ["win_pct", "net_rating"]:
        configs.append(
            dict(
                name=f"Baseline {features}",
                kind="logistic",
                features=features,
                C=1e8,
                l1_ratio=0,
            )
        )
    for features in [s for s in feature_sets() if s not in ["win_pct", "net_rating"]]:
        configs.append(
            dict(
                name=f"Logistic {features}",
                kind="logistic",
                features=features,
                C=1e8,
                l1_ratio=0,
            )
        )
        for ratio, label in [(0, "L2"), (1, "L1")]:
            for strength in [0.01, 0.1, 1.0]:
                configs.append(
                    dict(
                        name=f"{label} {features} C={strength:g}",
                        kind="logistic",
                        features=features,
                        C=strength,
                        l1_ratio=ratio,
                    )
                )
    for features in ["team", "combined10"]:
        for leaf in [20, 60]:
            configs.append(
                dict(
                    name=f"Forest {features} leaf={leaf}",
                    kind="forest",
                    features=features,
                    min_samples_leaf=leaf,
                )
            )
        for leaves in [3, 7]:
            configs.append(
                dict(
                    name=f"Boosting {features} leaves={leaves}",
                    kind="boosting",
                    features=features,
                    max_leaf_nodes=leaves,
                )
            )
    return configs


def make_model(config):
    if config["kind"] == "home":
        model = DummyClassifier(strategy="prior")
    elif config["kind"] == "logistic":
        model = LogisticRegression(
            C=config["C"],
            l1_ratio=config["l1_ratio"],
            solver="liblinear" if config["l1_ratio"] else "lbfgs",
            max_iter=3000,
            random_state=SEED,
        )
    elif config["kind"] == "forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=config["min_samples_leaf"],
            max_features=0.8,
            random_state=SEED,
            n_jobs=-1,
        )
    else:
        # Random early-stopping splits would violate the chronological design.
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=config["max_leaf_nodes"],
            min_samples_leaf=40,
            l2_regularization=5,
            early_stopping=False,
            random_state=SEED,
        )
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def metrics(y, p):
    return dict(
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
        auc=float(roc_auc_score(y, p)),
        accuracy=float(accuracy_score(y, np.asarray(p) >= 0.5)),
        n=len(y),
    )


def select_candidate(summary):
    close = summary[summary.log_loss <= summary.log_loss.min() + 0.001].copy()
    close["nonlinear"] = close.kind.isin(["forest", "boosting"])
    return close.sort_values(["n_features", "nonlinear", "log_loss", "name"]).iloc[0][
        "name"
    ]


def calibration_table(y, p):
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.cut(df.p, np.linspace(0, 1, 11), include_lowest=True)
    result = (
        df.groupby("bin", observed=True)
        .agg(mean_probability=("p", "mean"), win_rate=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
    # Wilson intervals display the uncertainty within each reliability bin.
    z = 1.96
    n, rate = result.n, result.win_rate
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    result["lower"], result["upper"] = center - half, center + half
    return result


def week_bootstrap(predictions, baseline="net_rating_probability", repetitions=2000):
    """Paired resampling of calendar-week blocks within one test season."""
    frame = predictions.copy()
    frame["week"] = pd.to_datetime(frame.gamedate).dt.to_period("W")
    y, p, b = (
        frame.y.to_numpy(),
        frame.probability.to_numpy(),
        frame[baseline].to_numpy(),
    )
    def loss(prob):
        return -(
            y * np.log(np.clip(prob, 1e-12, 1 - 1e-12))
            + (1 - y) * np.log(np.clip(1 - prob, 1e-12, 1 - 1e-12))
        )

    frame["loss"] = loss(p)
    frame["delta_log_loss"] = loss(p) - loss(b)
    frame["brier"] = (p - y) ** 2
    frame["delta_brier"] = (p - y) ** 2 - (b - y) ** 2
    grouped = frame.groupby("week").agg(
        n=("y", "size"),
        log_loss=("loss", "sum"),
        delta_log_loss=("delta_log_loss", "sum"),
        brier=("brier", "sum"),
        delta_brier=("delta_brier", "sum"),
    )
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
    totals = grouped.to_numpy()[draws].sum(axis=1)
    rows = []
    for j, name in enumerate(grouped.columns[1:], start=1):
        values = totals[:, j] / totals[:, 0]
        rows.append(
            dict(
                metric=name,
                estimate=grouped[name].sum() / grouped.n.sum(),
                lower=np.quantile(values, 0.025),
                upper=np.quantile(values, 0.975),
                blocks=len(grouped),
                repetitions=repetitions,
            )
        )
    return pd.DataFrame(rows)
