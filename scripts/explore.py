"""Describe basketball patterns and the development choices behind the model."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.build_data import TABLES, save_table
from src.charts import CLAY, TEAL, feature_label
from src.data import ROOT

FIGURES = ROOT / "artifacts/figures"


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160, bbox_inches="tight", facecolor="#faf9f6")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#faf9f6",
            "figure.facecolor": "#faf9f6",
            "axes.labelcolor": "#24343b",
            "text.color": "#24343b",
        }
    )
    history = pd.read_csv(TABLES / "team_history.csv.gz", parse_dates=["gamedate"])
    matchups = pd.read_csv(TABLES / "matchups.csv.gz")
    development = matchups[matchups.season.lt(2023)]
    home = (
        development.groupby("season")
        .agg(home_win_rate=("y", "mean"), games=("y", "size"))
        .reset_index()
    )
    save_table(home, "eda_home_court.csv")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(home.season, home.home_win_rate, color=CLAY, marker="o")
    ax.axhline(0.5, color=TEAL, linestyle="--", linewidth=1)
    ax.set(
        xlabel="Season starting year",
        ylabel="Designated-home win rate",
        title="Home teams usually won more often · development seasons",
        ylim=(0.48, 0.64),
    )
    save_figure(fig, "home_court.png")
    eligible = history[history.prior_games.ge(10) & history.season.lt(2023)]
    teams = (
        eligible.groupby(["season", "off_team"])
        .agg(
            top5_share=("top5_share", "mean"),
            continuity=("minute_continuity", "mean"),
            volatility=("rotation_volatility", "mean"),
            participation=("weighted_participation", "mean"),
        )
        .reset_index()
    )
    save_table(teams, "eda_rotations.csv")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(teams.top5_share, bins=22, color=TEAL, edgecolor="#faf9f6")
    ax.set(
        xlabel="Mean prior-ten-game minute share",
        ylabel="Team-seasons",
        title="How much of the rotation belongs to the top five?",
    )
    save_figure(fig, "rotation_concentration.png")
    # These two contrasting histories are selected from development data only.
    stable = teams.sort_values("volatility").iloc[0]
    shifting = teams.sort_values("volatility").iloc[-1]
    stories = []
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharey=True)
    for ax, row, color in zip(axes, [stable, shifting], [TEAL, CLAY]):
        h = eligible[
            eligible.season.eq(row.season) & eligible.off_team.eq(row.off_team)
        ]
        ax.plot(
            h.gamedate,
            h.minute_continuity,
            color=color,
            label="Minute-share continuity",
        )
        ax.set(
            title=f"{row.off_team} · {int(row.season)}–{str(int(row.season) + 1)[-2:]}",
            ylabel="Share retained",
            ylim=(0, 1),
        )
        stories.append(
            dict(
                team=row.off_team,
                season=int(row.season),
                volatility=float(row.volatility),
                continuity=float(row.continuity),
                top5_share=float(row.top5_share),
            )
        )
    save_figure(fig, "rotation_stories.png")
    summary = pd.read_csv(TABLES / "cv_summary.csv")
    best = summary.sort_values("log_loss").groupby("features", sort=False).head(1)
    save_table(best, "feature_comparison.csv")
    folds = pd.read_csv(TABLES / "cv_folds.csv")
    comparison = folds[
        folds.name.isin(
            ["L1 team C=0.1", "L1 team_players C=0.1", "Baseline net_rating"]
        )
    ].pivot(index="season", columns="name", values="log_loss")
    comparison["players_minus_team"] = (
        comparison["L1 team_players C=0.1"] - comparison["L1 team C=0.1"]
    )
    save_table(comparison.reset_index(), "player_comparison_by_season.csv")
    cv = pd.read_csv(TABLES / "cv_predictions.csv.gz")
    player_change = (
        cv[cv.name.isin(["L1 team C=0.1", "L1 team_players C=0.1"])]
        .pivot(index="nbagameid", columns="name", values="probability")
        .reset_index()
    )
    player_change["probability_change"] = (
        player_change["L1 team_players C=0.1"] - player_change["L1 team C=0.1"]
    )
    player_change = player_change.merge(
        development[
            [
                "nbagameid",
                "gamedate",
                "season",
                "home_off_team",
                "away_off_team",
                "y",
                "weighted_absence_diff",
            ]
        ],
        on="nbagameid",
        validate="one_to_one",
    )
    save_table(
        player_change.loc[
            player_change.probability_change.abs().sort_values(ascending=False).index
        ].head(30),
        "player_prediction_changes.csv",
    )
    importance = pd.read_csv(TABLES / "importance.csv")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    top = importance.head(8).sort_values("coefficient")
    ax.barh(top.feature.map(feature_label), top.coefficient, color=CLAY)
    ax.axvline(0, color=TEAL, linewidth=1)
    ax.set(
        xlabel="Coefficient per training standard deviation",
        title="What moves the home-win prediction?",
    )
    save_figure(fig, "model_inputs.png")
    calibration = pd.read_csv(TABLES / "test_calibration.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color=TEAL, linewidth=1)
    ax.errorbar(
        calibration.mean_probability,
        calibration.win_rate,
        yerr=[
            calibration.win_rate - calibration.lower,
            calibration.upper - calibration.win_rate,
        ],
        fmt="o-",
        color=CLAY,
        capsize=3,
    )
    ax.set(
        xlabel="Predicted home-win probability",
        ylabel="Observed home-win rate",
        title="The reserved 2023–24 season",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    save_figure(fig, "calibration.png")
    eda = dict(
        development_home_win_rate=float(development.y.mean()),
        mean_top5_share=float(teams.top5_share.mean()),
        mean_continuity=float(teams.continuity.mean()),
        recent_net_correlation=float(
            development.loc[
                development.eligible, ["net_rating_diff", "last10_net_rating_diff"]
            ]
            .corr()
            .iloc[0, 1]
        ),
        player_improvement_seasons=int(comparison.players_minus_team.lt(0).sum()),
        rotation_stories=stories,
    )
    (TABLES / "eda.json").write_text(json.dumps(eda, indent=2) + "\n")
    print(json.dumps(eda, indent=2))


if __name__ == "__main__":
    main()
