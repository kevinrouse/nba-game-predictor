"""Read supplied game records and validate their relationships."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_data(root=ROOT):
    team = pd.read_csv(Path(root) / "Data/team_game_data.csv.gz", parse_dates=["gamedate"])
    player = pd.read_csv(
        Path(root) / "Data/player_game_data.csv.gz", parse_dates=["gamedate"]
    )
    return team, player


def validate_data(team, player):
    assert not team.isna().any().any()
    assert not player.drop(columns="player_name").isna().any().any()
    assert not team.duplicated(["nbagameid", "off_team"]).any()
    assert not player.duplicated(["nbagameid", "nbapersonid"]).any()
    assert team.groupby("nbagameid").size().eq(2).all()
    assert team.possessions.gt(0).all()
    assert team.reboffensive.le(team.reboundchance).all()
    assert team.off_home.eq(1 - team.def_home).all()
    assert team.off_win.eq(1 - team.def_win).all()
    for col in ["season", "gametype", "gamedate"]:
        assert team.groupby("nbagameid")[col].nunique().eq(1).all()
    assert team.groupby("nbagameid").off_win.sum().eq(1).all()
    assert team.groupby("nbagameid").off_home.sum().eq(1).all()
    for frame in [team, player]:
        assert not frame.select_dtypes("number").lt(0).any().any()
        for prefix in ["fg", "fg2", "fg3", "ft"]:
            assert (
                (frame[prefix + "made"] + frame[prefix + "missed"])
                .eq(frame[prefix + "attempted"])
                .all()
            )
        assert frame.points.eq(
            2 * frame.fg2made + 3 * frame.fg3made + frame.ftmade
        ).all()
    opponent = team[
        [
            "nbagameid",
            "off_team",
            "points",
            "possessions",
            "fgmade",
            "fg3made",
            "fgattempted",
        ]
    ].rename(
        columns={
            "off_team": "def_team",
            **{
                c: "opp_" + c
                for c in ["points", "possessions", "fgmade", "fg3made", "fgattempted"]
            },
        }
    )
    games = team.merge(opponent, on=["nbagameid", "def_team"], validate="one_to_one")
    assert len(games) == len(team)
    assert games.off_win.eq(games.points.gt(games.opp_points)).all()
    linked = player.merge(
        team[["nbagameid", "off_team", "points", "season", "gametype", "gamedate"]],
        left_on=["nbagameid", "team"],
        right_on=["nbagameid", "off_team"],
        suffixes=("", "_team"),
        how="left",
        validate="many_to_one",
    )
    assert linked.off_team.notna().all()
    for col in ["season", "gametype", "gamedate"]:
        assert linked[col].eq(linked[col + "_team"]).all()
    totals = player.groupby(["nbagameid", "team"]).agg(
        player_points=("points", "sum"), seconds=("seconds", "sum")
    )
    totals = totals.join(
        team.rename(columns={"off_team": "team"}).set_index(["nbagameid", "team"])[
            ["points"]
        ],
        validate="one_to_one",
    )
    point_mismatch = totals[totals.player_points.ne(totals.points)]
    contradictions = player[(player.missed == 1) & (player.seconds > 0)]
    report = {
        "team_shape": list(team.shape),
        "player_shape": list(player.shape),
        "date_min": str(team.gamedate.min().date()),
        "date_max": str(team.gamedate.max().date()),
        "team_missing": team.isna().sum()[lambda s: s > 0].to_dict(),
        "player_missing": player.isna().sum()[lambda s: s > 0].to_dict(),
        "duplicates": 0,
        "player_team_point_mismatches": len(point_mismatch),
        "missed_with_positive_seconds": len(contradictions),
        "minute_total_range": [
            float(totals.seconds.min() / 60),
            float(totals.seconds.max() / 60),
        ],
    }
    report["zero_seconds_with_points"] = int(
        ((player.seconds == 0) & (player.points > 0)).sum()
    )
    report["team_games_below_239_minutes"] = int((totals.seconds < 239 * 60).sum())
    report["team_ids"] = int(team.offensivenbateamid.nunique())
    report["player_ids"] = int(player.nbapersonid.nunique())
    return (
        games.sort_values(["gamedate", "nbagameid", "off_team"]),
        report,
        contradictions,
        point_mismatch,
    )

