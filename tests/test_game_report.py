import json

import numpy as np
import pandas as pd
import pytest

from src.data import ROOT
from src.game_report import forecast_record, resolve_game_id


def test_links_only_accept_published_game_ids():
    assert resolve_game_id("22300401", [22300401, 22300402]) == 22300401
    for value in [None, "invalid", "2024-01-01", "22300403"]:
        assert resolve_game_id(value, [22300401, 22300402]) is None


def test_export_reconstructs_forecast_and_separates_the_outcome():
    tables = ROOT / "artifacts/tables"
    games = pd.read_csv(tables / "record_comparison_games.csv").set_index("nbagameid")
    inputs = pd.read_csv(tables / "matchups.csv.gz").set_index("nbagameid")
    terms = pd.read_csv(tables / "prediction_contributions.csv.gz")
    release = json.loads((ROOT / "artifacts/models/release.json").read_text())
    game = games.iloc[0]
    matchup = inputs.loc[game.name]
    report = forecast_record(game, matchup, terms, release)
    json.dumps(report, allow_nan=False)
    pregame = report["pregame_forecast"]
    assert "observed_result" in report and "observed_result" not in pregame
    assert report["game"]["id"] == game.name
    assert pregame["better_record_favorite"] in [game.home_off_team, game.away_off_team]
    assert len(pregame["inputs"]) == 22
    assert all(
        pd.Timestamp(d) < pd.Timestamp(game.gamedate)
        for d in pregame["history_cutoffs"].values()
    )
    score = pregame["intercept_log_odds"] + sum(
        x["log_odds_contribution"] for x in pregame["inputs"]
    )
    assert 1 / (1 + np.exp(-score)) == pytest.approx(
        pregame["home_probability"], abs=1e-9
    )
    changed = matchup.copy()
    changed["home_history_end"] = game.gamedate
    with pytest.raises(ValueError, match="strictly earlier"):
        forecast_record(game, changed, terms, release)
    with pytest.raises(ValueError, match="same game"):
        forecast_record(game, matchup.rename(-1), terms, release)
