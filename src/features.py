"""Construct team and rotation histories before matching home and road teams."""

import numpy as np
import pandas as pd

TEAM = ["net_rating", "win_pct", "efg", "opp_efg", "tov_rate", "oreb_rate", "dreb_rate", "ft_rate"]
CONTEXT = ["rest_days", "back_to_back", "games_last7"]
ROSTER = ["top5_share", "top7_share", "rotation_size", "minute_continuity",
          "starter_continuity", "rotation_volatility", "scoring_concentration",
          "usage_concentration", "weighted_ppa", "weighted_absence", "weighted_participation"]


def divide(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def build_team_history(games):
    """Shift counting statistics before cumulative or rolling aggregation."""
    regular = games[games.gametype.eq(2)].copy()
    extra = ["rebdefensive", "reboundchance", "turnovers", "shotattempts"]
    opponents = regular[["nbagameid", "off_team", *extra]].rename(
        columns={"off_team": "def_team", **{c: f"opp_{c}" for c in extra}}
    )
    regular = regular.merge(opponents, on=["nbagameid", "def_team"], validate="one_to_one")
    counts = ["points", "possessions", "fgmade", "fg3made", "fgattempted", "turnovers",
              "shotattempts", "reboffensive", "reboundchance", "ftattempted", "off_win",
              "opp_points", "opp_possessions", "opp_fgmade", "opp_fg3made", "opp_fgattempted",
              "opp_rebdefensive", "opp_reboundchance"]
    result = []
    for (_, _), group in regular.groupby(["season", "off_team"], sort=True):
        group = group.sort_values(["gamedate", "nbagameid"]).reset_index(drop=True)
        if group.gamedate.duplicated().any():
            raise ValueError("Cannot order two games for a team on the same date without tipoff times")
        out = group[["nbagameid", "gamedate", "season", "off_team", "off_team_name",
                     "def_team", "off_home", "off_win", "points", "opp_points"]].copy()
        out["prior_games"] = np.arange(len(group))
        out["history_end"] = group.gamedate.shift(1)
        lagged = group[counts].shift(1)
        for window in [None, 5, 10, 20]:
            total = lagged.expanding(min_periods=1).sum() if window is None else lagged.rolling(window, min_periods=1).sum()
            prefix = "" if window is None else f"last{window}_"
            n = out.prior_games if window is None else out.prior_games.clip(upper=window)
            ortg = 100 * divide(total.points, total.possessions)
            drtg = 100 * divide(total.opp_points, total.opp_possessions)
            out[prefix + "net_rating"] = ortg - drtg
            out[prefix + "efg"] = divide(total.fgmade + 0.5 * total.fg3made, total.fgattempted)
            out[prefix + "win_pct"] = divide(total.off_win, n)
            if window is None:
                out["ortg"] = ortg
                out["drtg"] = drtg
                out["point_diff"] = divide(total.points - total.opp_points, n)
                out["opp_efg"] = divide(total.opp_fgmade + 0.5 * total.opp_fg3made, total.opp_fgattempted)
                out["tov_rate"] = divide(total.turnovers, total.shotattempts + total.turnovers)
                out["oreb_rate"] = divide(total.reboffensive, total.reboundchance)
                out["dreb_rate"] = divide(total.opp_rebdefensive, total.opp_reboundchance)
                out["ft_rate"] = divide(total.ftattempted, total.fgattempted)
        gap = group.gamedate.diff().dt.days
        out["rest_days"] = (gap - 1).clip(0, 7)
        out["back_to_back"] = gap.eq(1).astype(int)
        days = group.gamedate.to_numpy(dtype="datetime64[D]")
        out["games_last7"] = np.arange(len(days)) - np.searchsorted(days, days - np.timedelta64(7, "D"))
        result.append(out)
    return pd.concat(result, ignore_index=True)


def build_roster_history(team_history, player):
    """Use past team games and past roster membership, never the predicted roster.

    Normal role is mean playing time in appearances over the prior twenty team
    games. Players on the last observed roster get weights proportional to that
    role. Ratios use prior counting totals; on-court team counts are not summed.
    """
    player = player[player.gametype.eq(2)]
    groups = dict(tuple(player.groupby(["season", "team"], sort=False)))
    result, roles = [], []
    columns = ["seconds", "points", "shotattempts", "shotattemptpoints", "turnovers", "starter", "missed"]
    for key, history in team_history.groupby(["season", "off_team"], sort=True):
        history = history.sort_values(["gamedate", "nbagameid"])
        pg = groups[key]
        game_ids = history.nbagameid.to_numpy()
        ids = sorted(pg.nbapersonid.unique())
        arrays = {}
        for col in columns:
            arrays[col] = pg.pivot(index="nbagameid", columns="nbapersonid", values=col).reindex(index=game_ids, columns=ids).to_numpy(float)
        names = pg.dropna(subset=["player_name"]).drop_duplicates("nbapersonid").set_index("nbapersonid").player_name
        for i, game_id in enumerate(game_ids):
            row = {"nbagameid": game_id, "off_team": key[1], "player_history_end": history.gamedate.iloc[i-1] if i else pd.NaT}
            if not i:
                row.update({name: np.nan for name in ROSTER})
                result.append(row)
                continue
            # Only these slices are available before the current tipoff.
            prior = {col: np.nan_to_num(a[max(0, i-20):i], nan=0.0) for col, a in arrays.items()}
            recorded = ~np.isnan(arrays["seconds"][max(0, i-20):i])
            seconds = prior["seconds"]
            normal_minutes = seconds.sum(axis=0) / np.maximum((seconds > 0).sum(axis=0), 1) / 60
            known_roster = recorded[-1]
            normal_minutes *= known_roster
            weights = normal_minutes / max(normal_minutes.sum(), 1)
            recent_seconds = seconds[-10:].sum(axis=0)
            shares = recent_seconds / max(recent_seconds.sum(), 1)
            recent5 = seconds[-5:].sum(axis=0)
            recent5 /= max(recent5.sum(), 1)
            previous5 = seconds[-10:-5].sum(axis=0)
            previous5 /= max(previous5.sum(), 1)
            daily_shares = seconds[-10:] / np.maximum(seconds[-10:].sum(axis=1, keepdims=True), 1)
            recent_recorded = recorded[-5:].sum(axis=0)
            absence = ((prior["missed"][-5:] > 0) & (seconds[-5:] == 0)).sum(axis=0) / np.maximum(recent_recorded, 1)
            participation = (seconds[-5:] > 0).sum(axis=0) / np.maximum(recent_recorded, 1)
            attempts = prior["shotattempts"][-5:].sum(axis=0)
            ppa = prior["shotattemptpoints"][-5:].sum(axis=0) / np.maximum(attempts, 1)
            eligible_weights = weights * (attempts > 0)
            starters = prior["starter"][-6:] > 0
            starter_overlap = np.minimum(starters[1:], starters[:-1]).sum(axis=1) / 5 if len(starters) > 1 else np.array([np.nan])
            points = prior["points"][-10:].sum(axis=0)
            usage = (prior["shotattempts"] + prior["turnovers"])[-10:].sum(axis=0)
            row.update(
                top5_share=np.sort(shares)[-5:].sum(), top7_share=np.sort(shares)[-7:].sum(),
                rotation_size=(seconds[-10:] / 60 >= 10).sum(axis=1).mean(),
                minute_continuity=np.minimum(recent5, previous5).sum() if i > 5 else np.nan,
                starter_continuity=np.mean(starter_overlap),
                rotation_volatility=np.abs(np.diff(daily_shares, axis=0)).sum(axis=1).mean() / 2 if i > 1 else np.nan,
                scoring_concentration=np.square(points / max(points.sum(), 1)).sum(),
                usage_concentration=np.square(usage / max(usage.sum(), 1)).sum(),
                weighted_ppa=np.dot(eligible_weights, ppa) / max(eligible_weights.sum(), 1e-12),
                weighted_absence=np.dot(weights, absence), weighted_participation=np.dot(weights, participation),
            )
            result.append(row)
            # A small per-game table makes the historical roster explorer transparent.
            for j in np.flatnonzero(weights > 0):
                roles.append(dict(nbagameid=game_id, team=key[1], player_id=ids[j],
                                  player_name=names.get(ids[j], f"Player {ids[j]}"),
                                  role_minutes=normal_minutes[j], role_weight=weights[j],
                                  recent_minute_share=shares[j], prior5_participation=participation[j],
                                  prior5_ppa=ppa[j] if attempts[j] else np.nan))
    return pd.DataFrame(result), pd.DataFrame(roles)


def build_matchups(history):
    metadata = ["nbagameid", "gamedate", "season"]
    home = history[history.off_home.eq(1)].copy()
    away = history[history.off_home.eq(0)].copy()
    home = home.rename(columns={c: "home_" + c for c in home if c not in metadata})
    away = away.rename(columns={c: "away_" + c for c in away if c not in metadata})
    paired = home.merge(away, on=metadata, validate="one_to_one")
    assert paired.home_def_team.eq(paired.away_off_team).all()
    metrics = [*TEAM, *CONTEXT, *ROSTER, "ortg", "drtg", "point_diff"]
    metrics += [f"last{n}_{metric}" for n in [5, 10, 20] for metric in ["net_rating", "efg", "win_pct"]]
    differences = {m + "_diff": paired["home_" + m] - paired["away_" + m] for m in metrics}
    paired = pd.concat([paired, pd.DataFrame(differences)], axis=1)
    paired["y"] = paired.home_off_win
    paired["eligible"] = paired.home_prior_games.ge(10) & paired.away_prior_games.ge(10)
    for side in ["home", "away"]:
        for source in ["history_end", "player_history_end"]:
            end = paired[f"{side}_{source}"]
            assert (end.isna() | end.lt(paired.gamedate)).all()
    return paired.sort_values(["gamedate", "nbagameid"]).reset_index(drop=True)


def feature_sets():
    team = [m + "_diff" for m in [*TEAM, *CONTEXT]]
    roster = [m + "_diff" for m in ROSTER]
    sets = {"win_pct": ["win_pct_diff"], "net_rating": ["net_rating_diff"], "team": team, "team_players": team + roster}
    for n in [5, 10, 20]:
        recent = [f"last{n}_net_rating_diff", f"last{n}_efg_diff"]
        sets[f"recent{n}"] = team + recent
        sets[f"combined{n}"] = team + recent + roster
    sets["compact"] = ["net_rating_diff", "rest_days_diff", "last10_net_rating_diff",
                        "minute_continuity_diff", "weighted_ppa_diff", "weighted_absence_diff"]
    return sets
