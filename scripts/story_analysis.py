"""Build shared evidence for the expanded analytical story without refitting models."""

import json

import joblib
import numpy as np
import pandas as pd

from scripts.build_data import TABLES, save_table
from src.analysis import (
    better_record_picks,
    binned_outcomes,
    linear_contributions,
    model_family,
    paired_accuracy_bootstrap,
    player_role_trace,
)
from src.charts import feature_label
from src.data import ROOT, load_data, validate_data
from src.modeling import VALIDATION_SEASONS


def feature_guide(importance):
    rows = [
        (
            "net_rating",
            "Season strength",
            "Does outscoring opponents per possession describe strength beyond wins?",
            "100 × prior points / possessions, minus opponent equivalent",
            "Opponent strength and early-season noise are not fully adjusted",
        ),
        (
            "win_pct",
            "Season strength",
            "How much does the simplest record tell us?",
            "Prior wins / prior games",
            "Ignores margin and can duplicate net rating",
        ),
        (
            "efg",
            "Shooting",
            "Does the team's shot-making profile add detail?",
            "(Prior FG made + 0.5 × threes made) / FG attempted",
            "Shooting fluctuates; strong overlap with net rating",
        ),
        (
            "opp_efg",
            "Shooting",
            "Has the defense been limiting efficient shots?",
            "Opponent version of effective FG percentage",
            "Does not separate shot quality from opponent shooting luck",
        ),
        (
            "tov_rate",
            "Possession control",
            "Is the offense losing scoring chances?",
            "Prior turnovers / (shot attempts + turnovers)",
            "A conditional coefficient is not a causal effect of turnovers",
        ),
        (
            "oreb_rate",
            "Possession control",
            "Does the offense create second chances?",
            "Offensive rebounds / offensive rebound chances",
            "Team rebounds and opponent rows need correct interpretation",
        ),
        (
            "dreb_rate",
            "Possession control",
            "Does the defense finish possessions?",
            "Defensive rebounds from opponent rows / opponent rebound chances",
            "Must join the opponent offense rather than use the same row",
        ),
        (
            "ft_rate",
            "Shooting",
            "Does getting to the line add a distinct signal?",
            "Prior FT attempts / FG attempts",
            "Correlated with other scoring measures; not player foul-drawing talent",
        ),
        (
            "rest_days",
            "Schedule",
            "Are rested teams favored after accounting for strength?",
            "Days between games minus one, capped from zero to seven",
            "Does not measure travel or actual fatigue",
        ),
        (
            "back_to_back",
            "Schedule",
            "Is playing yesterday different from ordinary rest?",
            "Previous team game was one calendar day earlier",
            "Related to rest and schedule density",
        ),
        (
            "games_last7",
            "Schedule",
            "Does a crowded recent schedule add context?",
            "Earlier team games within the previous seven calendar days",
            "No mileage, time zones or neutral-site indicator",
        ),
        (
            "top5_share",
            "Rotation",
            "How dependent is the team on a small core?",
            "Top five prior-ten-game player minute totals / all minutes",
            "Concentration is not automatically good or bad",
        ),
        (
            "top7_share",
            "Rotation",
            "Does a deeper core describe the rotation better?",
            "Top seven prior-ten-game player minute totals / all minutes",
            "Highly correlated with top-five share",
        ),
        (
            "rotation_size",
            "Rotation",
            "How many players are routinely getting meaningful minutes?",
            "Average prior-ten-game count playing at least ten minutes",
            "Does not measure the quality of those players",
        ),
        (
            "minute_continuity",
            "Rotation",
            "Is the distribution of playing time stable?",
            "Overlap of minute shares in prior five games versus five before them",
            "Can reflect both voluntary rotation changes and nonparticipation",
        ),
        (
            "starter_continuity",
            "Rotation",
            "Are the starting groups staying together?",
            "Mean fraction of five starting slots retained across prior transitions",
            "Previous starters are not the current starting lineup",
        ),
        (
            "rotation_volatility",
            "Rotation",
            "How abruptly has the allocation of minutes shifted?",
            "Mean half-L1 distance of consecutive prior minute distributions",
            "Cannot identify why a rotation changed",
        ),
        (
            "scoring_concentration",
            "Player roles",
            "How concentrated is scoring responsibility?",
            "Sum of squared player shares of prior-ten-game points",
            "Overlaps strongly with shooting-role concentration",
        ),
        (
            "usage_concentration",
            "Player roles",
            "Does concentration of scoring opportunities add information?",
            "Sum of squared shares of shot attempts + turnovers in prior ten games",
            "Uses individual opportunity counts, not summed on-court team totals",
        ),
        (
            "weighted_ppa",
            "Player form",
            "Are established rotation players scoring efficiently lately?",
            "Prior-five-game points per shot attempt, weighted by earlier roles",
            "Short windows are noisy; players without attempts are omitted and weights renormalized",
        ),
        (
            "weighted_absence",
            "Player form",
            "Have players with established roles recently missed games?",
            "Prior-five-game missed rate among rostered records, weighted by prior role",
            "Historical absence only, not a current injury report",
        ),
        (
            "weighted_participation",
            "Player form",
            "How consistently have normal rotation players been participating?",
            "Prior-five-game positive-seconds fraction, weighted by prior role",
            "Includes DNPs and overlaps with absence history",
        ),
    ]
    guide = pd.DataFrame(
        rows, columns=["metric", "group", "question", "construction", "caution"]
    )
    guide["feature"] = guide.metric + "_diff"
    guide["label"] = guide.metric.map(feature_label)
    guide = guide.merge(
        importance[["feature", "coefficient"]], on="feature", validate="one_to_one"
    )
    guide["decision"] = np.where(
        guide.coefficient.eq(0),
        "Tested; shrunk to zero",
        "Retained with a nonzero coefficient",
    )
    return guide


def main():
    games = pd.read_csv(TABLES / "matchups.csv.gz", parse_dates=["gamedate"])
    history = pd.read_csv(
        TABLES / "team_history.csv.gz",
        parse_dates=["gamedate", "history_end", "player_history_end"],
    )
    predictions = pd.read_csv(TABLES / "test_predictions.csv", parse_dates=["gamedate"])
    selected = json.loads((ROOT / "artifacts/models/selection.json").read_text())
    summary = pd.read_csv(TABLES / "cv_summary.csv")
    cv = pd.read_csv(TABLES / "cv_predictions.csv.gz", parse_dates=["gamedate"])
    configs = json.loads((TABLES / "candidate_configs.json").read_text())
    eligible = games[games.eligible].copy()
    records = better_record_picks(eligible)
    eligible = pd.concat([eligible, records], axis=1)
    record_columns = [
        "nbagameid",
        "home_prior_games",
        "away_prior_games",
        "home_prior_wins",
        "away_prior_wins",
        "record_tie",
        "record_pick_home",
    ]
    test = predictions.merge(
        eligible[record_columns], on="nbagameid", validate="one_to_one"
    )
    test["record_correct"] = test.record_pick_home.eq(test.y)
    test["model_correct"] = test.probability.ge(0.5).eq(test.y)
    test["agree"] = test.probability.ge(0.5).eq(test.record_pick_home)
    save_table(test, "record_comparison_games.csv")
    decisions = []
    choices = {
        "Always home": pd.Series(1, index=test.index),
        "Better pregame record": test.record_pick_home,
        "Win-percentage logistic": test.win_pct_probability.ge(0.5),
        "Net-rating logistic": test.net_rating_probability.ge(0.5),
        "Selected team + player model": test.probability.ge(0.5),
    }
    for name, pick in choices.items():
        decisions.append(
            dict(
                strategy=name,
                correct=int(pick.eq(test.y).sum()),
                games=len(test),
                accuracy=pick.eq(test.y).mean(),
            )
        )
    save_table(pd.DataFrame(decisions), "decision_baselines.csv")
    intervals = paired_accuracy_bootstrap(test)
    save_table(intervals, "record_accuracy_bootstrap.csv")
    season_rows = []
    for season, group in eligible.groupby("season"):
        row = dict(
            season=season,
            games=len(group),
            ties=int(group.record_tie.sum()),
            record_accuracy=group.record_pick_home.eq(group.y).mean(),
            stage="Initial training"
            if season < 2017
            else "Validation"
            if season < 2023
            else "Final test",
        )
        p = (
            cv[cv.season.eq(season) & cv.name.eq(selected["config"]["name"])]
            if season < 2023
            else test
        )
        row["model_accuracy"] = (
            p.probability.ge(0.5).eq(p.y).mean() if len(p) else np.nan
        )
        season_rows.append(row)
    save_table(pd.DataFrame(season_rows), "decision_by_season.csv")
    disagreement = (
        test.groupby(["model_correct", "record_correct"])
        .size()
        .reset_index(name="games")
    )
    save_table(disagreement, "record_disagreements.csv")

    summary["family"] = summary.name.map(model_family)
    summary["generalization_gap"] = summary.log_loss - summary.train_log_loss
    best = summary.sort_values("log_loss").groupby("family", sort=False).head(1)
    # Keep both strength baselines visible rather than collapsing their family.
    best = pd.concat(
        [
            best[~best.family.eq("One-variable logistic baseline")],
            summary[summary.name.str.startswith("Baseline")],
        ]
    )
    if selected["config"]["name"] not in set(best.name):
        best = pd.concat([best, summary[summary.name.eq(selected["config"]["name"])]])
    best["selected"] = best.name.eq(selected["config"]["name"])
    # DummyClassifier accepts a placeholder column but uses no matchup inputs.
    best.loc[best.kind.eq("home"), "n_features"] = 0
    save_table(best.sort_values("log_loss"), "model_family_comparison.csv")
    common = (
        summary[summary.features.eq("combined10")]
        .sort_values("log_loss")
        .groupby("family", sort=False)
        .head(1)
    )
    save_table(common, "matched_input_comparison.csv")
    candidate_details = pd.DataFrame(
        [
            dict(
                name=c["name"],
                family=model_family(c["name"]),
                configuration=json.dumps(
                    {k: v for k, v in c.items() if k not in ["name", "kind"]},
                    sort_keys=True,
                ),
            )
            for c in configs
        ]
    )
    save_table(candidate_details, "candidate_details.csv")
    all_coefficients = pd.read_csv(TABLES / "cv_coefficients.csv.gz")
    coefs = all_coefficients[all_coefficients.name.eq(selected["config"]["name"])]
    stability = (
        coefs.groupby("feature")
        .coefficient.agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    save_table(stability, "coefficient_stability.csv")
    timeline = []
    for season in [*VALIDATION_SEASONS, 2023]:
        train = eligible[eligible.season.lt(season)]
        valid = eligible[eligible.season.eq(season)]
        timeline.append(
            dict(
                evaluation_season=season,
                stage="Final test" if season == 2023 else "Validation",
                first_train_season=int(train.season.min()),
                last_train_season=int(train.season.max()),
                train_games=len(train),
                evaluation_games=len(valid),
                last_train_date=train.gamedate.max(),
                first_evaluation_date=valid.gamedate.min(),
            )
        )
    save_table(pd.DataFrame(timeline), "split_timeline.csv")

    development = eligible[eligible.season.lt(2023)].copy()
    measures = [
        "win_pct_diff",
        "net_rating_diff",
        "efg_diff",
        "opp_efg_diff",
        "tov_rate_diff",
        "oreb_rate_diff",
        "dreb_rate_diff",
        "ft_rate_diff",
        "weighted_ppa_diff",
        "weighted_absence_diff",
        "minute_continuity_diff",
        "scoring_concentration_diff",
    ]
    save_table(
        pd.concat(
            [binned_outcomes(development, col) for col in measures], ignore_index=True
        ),
        "eda_feature_relationships.csv",
    )
    development["recent_departure"] = (
        development.last10_net_rating_diff - development.net_rating_diff
    )
    development["strength_band"] = pd.qcut(
        development.net_rating_diff,
        3,
        labels=["Away stronger", "Similar strength", "Home stronger"],
    )
    development["recent_direction"] = np.where(
        development.recent_departure.gt(0),
        "Recent form favors home more",
        "Recent form favors home less",
    )
    recent = (
        development.groupby(["strength_band", "recent_direction"], observed=True)
        .agg(home_win_rate=("y", "mean"), games=("y", "size"))
        .reset_index()
    )
    save_table(recent, "eda_recent_conditional.csv")
    development["schedule"] = np.select(
        [
            development.home_back_to_back.eq(1) & development.away_back_to_back.eq(1),
            development.home_back_to_back.eq(1),
            development.away_back_to_back.eq(1),
        ],
        [
            "Both on back-to-backs",
            "Only home on back-to-back",
            "Only away on back-to-back",
        ],
        default="Neither on back-to-back",
    )
    save_table(
        development.groupby("schedule")
        .agg(
            home_win_rate=("y", "mean"),
            games=("y", "size"),
            mean_strength_diff=("net_rating_diff", "mean"),
        )
        .reset_index(),
        "eda_rest.csv",
    )
    raw_team, raw_players = load_data()
    paired, _, _, _ = validate_data(raw_team, raw_players)
    regular = paired[paired.gametype.eq(2) & paired.season.lt(2023)]
    seasons = (
        regular.groupby(["season", "off_team", "off_team_name"])
        .agg(
            games=("nbagameid", "size"),
            wins=("off_win", "sum"),
            points=("points", "sum"),
            opp_points=("opp_points", "sum"),
            possessions=("possessions", "sum"),
            opp_possessions=("opp_possessions", "sum"),
        )
        .reset_index()
    )
    seasons["win_pct"] = seasons.wins / seasons.games
    seasons["ortg"] = 100 * seasons.points / seasons.possessions
    seasons["drtg"] = 100 * seasons.opp_points / seasons.opp_possessions
    seasons["net_rating"] = seasons.ortg - seasons.drtg
    save_table(seasons, "eda_team_seasons.csv")
    final_importance = pd.read_csv(TABLES / "importance.csv")
    save_table(feature_guide(final_importance), "feature_guide.csv")

    example = (
        development[
            development.season.eq(2018)
            & development.home_off_team.eq("OKC")
            & development.home_prior_games.ge(25)
        ]
        .sort_values("gamedate")
        .iloc[0]
    )
    trace = player_role_trace(raw_players, history, example.nbagameid, "OKC")
    save_table(trace, "worked_player_roles.csv")
    np.testing.assert_allclose(
        [
            trace.ppa_contribution.sum(),
            trace.absence_contribution.sum(),
            trace.participation_contribution.sum(),
        ],
        [
            example.home_weighted_ppa,
            example.home_weighted_absence,
            example.home_weighted_participation,
        ],
        atol=1e-8,
    )
    totals = []
    for side in ["home", "away"]:
        team = example[f"{side}_off_team"]
        prior = regular[
            regular.season.eq(example.season)
            & regular.off_team.eq(team)
            & regular.gamedate.lt(example.gamedate)
        ]
        totals.append(
            dict(
                team=team,
                side=side,
                prior_games=len(prior),
                prior_wins=int(prior.off_win.sum()),
                points=int(prior.points.sum()),
                possessions=int(prior.possessions.sum()),
                opponent_points=int(prior.opp_points.sum()),
                opponent_possessions=int(prior.opp_possessions.sum()),
                ortg=100 * prior.points.sum() / prior.possessions.sum(),
                drtg=100 * prior.opp_points.sum() / prior.opp_possessions.sum(),
                history_end=prior.gamedate.max(),
            )
        )
    save_table(pd.DataFrame(totals), "worked_team_totals.csv")
    info = dict(
        game_id=int(example.nbagameid),
        gamedate=str(example.gamedate.date()),
        season=int(example.season),
        home=example.home_off_team,
        away=example.away_off_team,
        prior_home_games=int(example.home_prior_games),
        prior_away_games=int(example.away_prior_games),
    )
    (TABLES / "worked_example.json").write_text(json.dumps(info, indent=2) + "\n")

    bundle = joblib.load(ROOT / "artifacts/models/frozen_models.joblib")[
        selected["config"]["name"]
    ]
    test_inputs = eligible.set_index("nbagameid").loc[test.nbagameid].reset_index()
    contribution = linear_contributions(bundle, test_inputs)
    save_table(contribution, "prediction_contributions.csv.gz")
    error_data = test.merge(
        eligible[
            ["nbagameid", "net_rating_diff", "home_back_to_back", "away_back_to_back"]
        ],
        on="nbagameid",
        validate="one_to_one",
    )
    error_data["stage_of_season"] = pd.cut(
        np.minimum(error_data.home_prior_games, error_data.away_prior_games),
        [9, 24, 49, 82],
        labels=["10–24 prior games", "25–49 prior games", "50+ prior games"],
    )
    subgroup = (
        error_data.groupby("stage_of_season", observed=True)
        .agg(
            games=("y", "size"),
            model_accuracy=("model_correct", "mean"),
            record_accuracy=("record_correct", "mean"),
        )
        .reset_index()
    )
    save_table(subgroup, "test_season_stages.csv")
    confidence = test.copy()
    confidence["confidence_band"] = pd.cut(
        confidence.confidence,
        [0.499999, 0.6, 0.7, 0.8, 0.9, 1],
        labels=["50–60%", "60–70%", "70–80%", "80–90%", "90–100%"],
    )
    save_table(
        confidence.groupby("confidence_band", observed=True)
        .agg(
            games=("y", "size"),
            predicted_favorite_rate=("confidence", "mean"),
            actual_favorite_rate=("model_correct", "mean"),
        )
        .reset_index(),
        "test_favorites.csv",
    )
    story = dict(
        test_games=len(test),
        model_correct=int(test.model_correct.sum()),
        record_correct=int(test.record_correct.sum()),
        model_accuracy=float(test.model_correct.mean()),
        record_accuracy=float(test.record_correct.mean()),
        ties=int(test.record_tie.sum()),
        extra_correct=int(test.model_correct.sum() - test.record_correct.sum()),
        disagreements=int((~test.agree).sum()),
        model_only_correct=int((test.model_correct & ~test.record_correct).sum()),
        record_only_correct=int((~test.model_correct & test.record_correct).sum()),
        accuracy_interval=intervals[intervals.metric.eq("accuracy_difference")]
        .iloc[0]
        .to_dict(),
        rule="Higher pregame win percentage; exact ties favor home",
        comparison_status="Added after the original test evaluation; frozen predictions unchanged",
    )
    (TABLES / "story.json").write_text(json.dumps(story, indent=2) + "\n")
    print(json.dumps(story, indent=2), flush=True)
    print("Worked example:", info, flush=True)


if __name__ == "__main__":
    main()
