"""Test the information boundary, not just the resulting table shape."""

import numpy as np
import pandas as pd
import pytest
from src.data import load_data, validate_data
from src.features import build_team_history, build_roster_history, build_matchups, feature_sets, ROSTER


@pytest.fixture(scope="module")
def records():
    team, player = load_data()
    games, audit, _, mismatch = validate_data(team, player)
    assert audit["duplicates"] == 0 and mismatch.empty
    keep = games.season.eq(2014) & games.gamedate.le("2014-12-15") & games.gametype.eq(2)
    games = games[keep].copy()
    player = player[player.nbagameid.isin(games.nbagameid)].copy()
    return games, player


def construct(games, players):
    history = build_team_history(games)
    roster, _ = build_roster_history(history, players)
    return build_matchups(history.merge(roster, on=["nbagameid", "off_team"], validate="one_to_one"))


def test_matchup_keys_and_strict_history(records):
    games, player = records
    matchups = construct(games, player)
    assert not matchups.nbagameid.duplicated().any()
    assert len(matchups) == games.nbagameid.nunique()
    for side in ["home", "away"]:
        for source in ["history_end", "player_history_end"]:
            column = matchups[f"{side}_{source}"]
            assert (column.isna() | (column < matchups.gamedate)).all()
    assert matchups.y.eq(matchups.home_points > matchups.away_points).all()


def test_current_and_future_box_scores_cannot_change_pregame_features(records):
    games, player = records
    cutoff = pd.Timestamp("2014-12-01")
    reference = construct(games, player)
    altered_games, altered_player = games.copy(), player.copy()
    future = altered_games.gamedate.ge(cutoff)
    for c in ["points", "opp_points", "fgmade", "fg3made", "turnovers", "reboffensive"]:
        altered_games.loc[future, c] *= 3
    altered_games.loc[future, "off_win"] = 1 - altered_games.loc[future, "off_win"]
    future = altered_player.gamedate.ge(cutoff)
    for c in ["seconds", "points", "shotattempts", "shotattemptpoints"]:
        altered_player.loc[future, c] *= 5
    altered_player.loc[future, "starter"] = 1 - altered_player.loc[future, "starter"]
    altered_player.loc[future, "missed"] = 1 - altered_player.loc[future, "missed"]
    altered = construct(altered_games, altered_player)
    columns = sorted(set(sum(feature_sets().values(), [])))
    mask = reference.gamedate.le(cutoff)
    pd.testing.assert_frame_equal(reference.loc[mask, columns], altered.loc[mask, columns])


def test_deleting_future_records_preserves_history(records):
    games, player = records
    cutoff = pd.Timestamp("2014-12-01")
    full = construct(games, player)
    short = construct(games[games.gamedate.le(cutoff)], player[player.gamedate.le(cutoff)])
    columns = ["nbagameid", *sorted(set(sum(feature_sets().values(), [])))]
    pd.testing.assert_frame_equal(full.loc[full.gamedate.le(cutoff), columns].reset_index(drop=True), short[columns])


def test_roles_weight_prior_regulars_and_ignore_current_roster():
    dates = pd.date_range("2020-01-01", periods=11, freq="2D")
    history = pd.DataFrame(dict(season=2019, off_team="T", nbagameid=np.arange(11), gamedate=dates))
    rows = []
    for game in range(11):
        for person, minutes in [(1, 30), (2, 3)]:
            rows.append(dict(season=2019, team="T", nbagameid=game, gametype=2,
                             nbapersonid=person, player_name=str(person), seconds=minutes*60,
                             points=10, shotattempts=5, shotattemptpoints=10, turnovers=1, starter=person==1, missed=0))
    players = pd.DataFrame(rows)
    snapshots, roles = build_roster_history(history, players)
    prior = roles[roles.nbagameid.eq(10)].set_index("player_id")
    assert prior.loc[1, "role_weight"] == pytest.approx(10 / 11)
    # Remove the entire current-game player file: the forecast must be identical.
    reduced, _ = build_roster_history(history, players[players.nbagameid.ne(10)])
    pd.testing.assert_series_equal(snapshots.loc[10, ROSTER], reduced.loc[10, ROSTER])
