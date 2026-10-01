"""Small analysis helpers shared by notebooks, reports and the dashboard."""

import numpy as np
import pandas as pd

from src.modeling import SEED, calendar_block_ids


def better_record_picks(games):
    """Choose the higher pregame win percentage; exact ties favor home.

    Wins are recovered from saved pregame rates and integer games played. Cross
    multiplication treats 2/6 and 3/9 as the same record despite CSV rounding.
    """
    wins, counts = {}, {}
    for side in ["home", "away"]:
        counts[side] = games[f"{side}_prior_games"]
        rates = games[f"{side}_win_pct"]
        if (counts[side].isna() | counts[side].le(0) | rates.isna()).any():
            raise ValueError(
                "A record pick requires prior games and a win percentage for both teams"
            )
        inferred = counts[side] * rates
        if not np.allclose(inferred, np.rint(inferred), atol=1e-6, rtol=0):
            raise ValueError(
                "Pregame win percentage does not reconcile to integer wins"
            )
        wins[side] = np.rint(inferred).astype(int)
    difference = wins["home"] * counts["away"] - wins["away"] * counts["home"]
    return pd.DataFrame(
        {
            "home_prior_wins": wins["home"],
            "away_prior_wins": wins["away"],
            "record_tie": difference.eq(0),
            "record_pick_home": difference.ge(0).astype(int),
        },
        index=games.index,
    )


def paired_accuracy_bootstrap(games, repetitions=2000, block_weeks=1):
    """Resample complete calendar weeks, preserving the paired winner picks."""
    frame = games.copy()
    frame["week"] = calendar_block_ids(frame.gamedate, block_weeks)
    frame["model_correct"] = frame.probability.ge(0.5).eq(frame.y).astype(int)
    frame["record_correct"] = frame.record_pick_home.eq(frame.y).astype(int)
    blocks = frame.groupby("week").agg(
        n=("y", "size"),
        model_correct=("model_correct", "sum"),
        record_correct=("record_correct", "sum"),
    )
    draws = np.random.default_rng(SEED).integers(
        0, len(blocks), size=(repetitions, len(blocks))
    )
    totals = blocks.to_numpy()[draws].sum(axis=1)
    values = {
        "model_accuracy": totals[:, 1] / totals[:, 0],
        "record_accuracy": totals[:, 2] / totals[:, 0],
        "accuracy_difference": (totals[:, 1] - totals[:, 2]) / totals[:, 0],
    }
    observed = {
        "model_accuracy": frame.model_correct.mean(),
        "record_accuracy": frame.record_correct.mean(),
        "accuracy_difference": (frame.model_correct - frame.record_correct).mean(),
    }
    return pd.DataFrame(
        [
            dict(
                metric=k,
                estimate=observed[k],
                lower=np.quantile(v, 0.025),
                upper=np.quantile(v, 0.975),
                blocks=len(blocks),
                repetitions=repetitions,
            )
            for k, v in values.items()
        ]
    )


def model_family(name):
    if name == "Home win rate":
        return "Home-rate baseline"
    if name.startswith("Baseline"):
        return "One-variable logistic baseline"
    if name.startswith("Logistic"):
        return "Unregularized logistic"
    if name.startswith("L1"):
        return "L1 logistic"
    if name.startswith("L2"):
        return "L2 logistic"
    if name.startswith("Forest"):
        return "Random forest"
    if name.startswith("Boosting"):
        return "Gradient boosting"
    raise ValueError(f"Unknown model family: {name}")


def binned_outcomes(frame, feature, bins=5):
    """Descriptive development association, not an adjusted or causal effect."""
    values = frame[[feature, "y"]].dropna().copy()
    values["bin"] = pd.qcut(values[feature], bins, duplicates="drop")
    summary = (
        values.groupby("bin", observed=True)
        .agg(
            value=(feature, "mean"),
            home_win_rate=("y", "mean"),
            wins=("y", "sum"),
            n=("y", "size"),
        )
        .reset_index()
    )
    rate, n, z = summary.home_win_rate, summary.n, 1.96
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    summary["lower"], summary["upper"] = center - half, center + half
    summary.insert(0, "feature", feature)
    return summary


def player_role_trace(players, team_history, game_id, team):
    """Expose the actual prior rows, roles and weighted contributions for one game."""
    current = team_history[
        (team_history.nbagameid == game_id) & (team_history.off_team == team)
    ].iloc[0]
    past = team_history[
        (team_history.season == current.season)
        & (team_history.off_team == team)
        & (team_history.gamedate < current.gamedate)
    ].sort_values("gamedate")
    if past.empty:
        raise ValueError("A player trace needs at least one earlier game")
    previous_ids = past.nbagameid.tail(20)
    prior = players[
        (players.team == team) & players.nbagameid.isin(previous_ids)
    ].copy()
    assert prior.gamedate.max() < current.gamedate
    known = prior[prior.nbagameid.eq(past.iloc[-1].nbagameid)].nbapersonid
    prior["appeared"] = prior.seconds.gt(0)
    normal = prior.groupby("nbapersonid").agg(
        seconds=("seconds", "sum"), appearances=("appeared", "sum")
    )
    normal = normal.reindex(known).fillna(0)
    normal["role_minutes"] = normal.seconds / normal.appearances.clip(lower=1) / 60
    normal["role_weight"] = normal.role_minutes / normal.role_minutes.sum()
    recent = prior[prior.nbagameid.isin(past.nbagameid.tail(5))].copy()
    recent["historical_absence"] = recent.missed.eq(1) & recent.seconds.eq(0)
    recent = recent.groupby("nbapersonid").agg(
        attempts=("shotattempts", "sum"),
        attempt_points=("shotattemptpoints", "sum"),
        participation=("appeared", "mean"),
        absence=("historical_absence", "mean"),
        rostered_games=("nbagameid", "size"),
    )
    trace = normal.join(recent).fillna(0)
    names = (
        prior.sort_values("gamedate")
        .drop_duplicates("nbapersonid", keep="last")
        .set_index("nbapersonid")
        .player_name
    )
    trace.insert(0, "player_name", names.reindex(trace.index))
    trace["ppa"] = trace.attempt_points / trace.attempts.replace(0, np.nan)
    trace["scoring_weight"] = trace.role_weight.where(trace.attempts.gt(0), 0)
    trace["scoring_weight"] /= trace.scoring_weight.sum()
    trace["ppa_contribution"] = trace.scoring_weight * trace.ppa.fillna(0)
    trace["absence_contribution"] = trace.role_weight * trace.absence
    trace["participation_contribution"] = trace.role_weight * trace.participation
    trace["history_end"] = past.gamedate.max()
    return (
        trace[trace.role_weight.gt(0)]
        .sort_values("role_weight", ascending=False)
        .reset_index()
    )


def linear_contributions(bundle, games):
    """Return exact additive log-odds contributions for the frozen logistic model."""
    features, pipeline = bundle["features"], bundle["model"]
    standardized = pipeline[1].transform(pipeline[0].transform(games[features]))
    values = standardized * pipeline[-1].coef_[0]
    rows = pd.DataFrame(values, columns=features, index=games.nbagameid)
    rows.index.name = "nbagameid"
    logits = values.sum(axis=1) + pipeline[-1].intercept_[0]
    np.testing.assert_allclose(
        1 / (1 + np.exp(-logits)),
        pipeline.predict_proba(games[features])[:, 1],
        atol=1e-12,
    )
    return rows.reset_index().melt(
        id_vars="nbagameid", var_name="feature", value_name="contribution"
    )
