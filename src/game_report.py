"""Portable audit records for the historical forecasts shown in the app."""

import pandas as pd


def resolve_game_id(value, available_ids):
    """Accept links only to games covered by the published test predictions."""
    try:
        game_id = int(value)
    except (ValueError, TypeError):
        return None
    return game_id if game_id in set(available_ids) else None


def forecast_record(game, matchup, contributions, release):
    """Keep prediction inputs and subsequently observed results in separate fields."""
    game_id = int(game.name)
    if int(matchup.name) != game_id:
        raise ValueError("Prediction and input rows must refer to the same game")
    terms = contributions[contributions.nbagameid.eq(game_id)].set_index("feature")
    if not terms.index.is_unique or set(terms.index) != set(release["features"]):
        raise ValueError("The forecast needs exactly one contribution per model input")
    date = pd.Timestamp(game.gamedate)
    cutoffs = {}
    for side in ["home", "away"]:
        for source in ["history_end", "player_history_end"]:
            cutoff = pd.Timestamp(matchup[f"{side}_{source}"])
            if pd.isna(cutoff) or cutoff >= date:
                raise ValueError("A forecast record must use strictly earlier history")
            cutoffs[f"{side}_{source}"] = cutoff.date().isoformat()

    def number(value):
        return None if pd.isna(value) else float(value)

    inputs = []
    for feature in release["features"]:
        metric = feature.removesuffix("_diff")
        inputs.append(
            dict(
                feature=feature,
                home_value=number(matchup[f"home_{metric}"]),
                away_value=number(matchup[f"away_{metric}"]),
                home_minus_away=number(matchup[feature]),
                log_odds_contribution=number(terms.loc[feature, "contribution"]),
            )
        )
    p = float(game.probability)
    return dict(
        schema_version=1,
        model_id=release["model_id"],
        model_name=release["selected_model"],
        fitted_through_season_start_year=max(release["train_seasons"]),
        game=dict(
            id=game_id,
            date=date.date().isoformat(),
            home=game.home_off_team,
            away=game.away_off_team,
        ),
        pregame_forecast=dict(
            intercept_log_odds=release["intercept_log_odds"],
            home_probability=p,
            away_probability=1 - p,
            favorite=game.home_off_team if p >= 0.5 else game.away_off_team,
            better_record_favorite=game.home_off_team
            if game.record_pick_home
            else game.away_off_team,
            record_rule="Higher pregame winning percentage; exact ties favor home",
            records={
                side: dict(
                    wins=int(game[f"{side}_prior_wins"]),
                    games=int(game[f"{side}_prior_games"]),
                )
                for side in ["home", "away"]
            },
            history_cutoffs=cutoffs,
            inputs=inputs,
        ),
        observed_result=dict(
            availability="Postgame; excluded from the prediction inputs",
            home_points=int(game.home_points),
            away_points=int(game.away_points),
            home_win=bool(game.y),
        ),
    )
