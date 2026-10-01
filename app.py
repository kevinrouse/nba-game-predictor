"""Explore historical NBA games and their pregame win probabilities."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.charts import CLAY, TEAL, feature_label, offense_defense, style
from src.data import ROOT

st.set_page_config(page_title="NBA Game Predictor", page_icon="🏀", layout="wide")
TABLES = ROOT / "artifacts/tables"


@st.cache_data
def table(name):
    frame = pd.read_csv(TABLES / name)
    for column in ["gamedate", "history_end", "player_history_end"]:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


@st.cache_data
def metadata():
    return (
        json.loads((TABLES / "results.json").read_text()),
        json.loads((ROOT / "artifacts/models/selection.json").read_text()),
    )


def season_label(year):
    return f"{year}–{str(year + 1)[-2:]}"


def choose_team(history, prefix):
    a, b = st.columns([1, 2])
    season = a.selectbox(
        "Season",
        sorted(history.season.unique(), reverse=True),
        format_func=season_label,
        key=f"{prefix}_season",
    )
    teams = (
        history[history.season.eq(season)]
        .drop_duplicates("off_team")
        .set_index("off_team")
        .off_team_name.to_dict()
    )
    codes = sorted(teams)
    code = b.selectbox(
        "Team",
        codes,
        index=codes.index("OKC") if "OKC" in codes else 0,
        format_func=lambda t: teams[t],
        key=f"{prefix}_team",
    )
    return (
        history[history.season.eq(season) & history.off_team.eq(code)].sort_values(
            "gamedate"
        ),
        code,
        season,
    )


def probability_cards(game):
    away, home = st.columns(2)
    away.metric(str(game.away_off_team_name), f"{1 - game.probability:.1%}")
    home.metric(str(game.home_off_team_name) + " · home", f"{game.probability:.1%}")
    fig = go.Figure()
    for label, value, color in [
        (game.away_off_team, 1 - game.probability, TEAL),
        (game.home_off_team, game.probability, CLAY),
    ]:
        fig.add_bar(
            y=["Pregame"],
            x=[value],
            name=label,
            orientation="h",
            marker_color=color,
            text=[f"{label} {value:.0%}"],
            textposition="inside",
            hovertemplate=f"{label}: %{{x:.1%}}<extra></extra>",
        )
    fig.update_layout(
        barmode="stack",
        showlegend=False,
        height=85,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


results, selection = metadata()
history = table("team_history.csv.gz")
predictions = table("test_predictions.csv")
with st.sidebar:
    st.markdown("### NBA Game Predictor")
    st.caption("A basketball project by Kevin Rouse")
    page = st.radio(
        "Explore",
        [
            "Home",
            "Team explorer",
            "Roster explorer",
            "Historical game predictor",
            "What the model uses",
            "Performance",
            "Biggest misses",
            "About the model",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("2014–15 through 2023–24\n\nHistorical data. Pregame probabilities.")
    st.link_button(
        "View the project on GitHub", "https://github.com/kevinrouse/nba-game-predictor"
    )


if page == "Home":
    st.caption("BEFORE TIPOFF")
    st.title("What makes a team the favorite?")
    st.markdown(
        "I wanted to see how well NBA games could be predicted from what teams and players had already done. Season records are a starting point, but rotations shift, shooting comes and goes, and a team's last few weeks can look different from its whole season."
    )
    st.markdown(
        "This project turns those histories into a probability that the home team wins, then checks what happened."
    )
    a, b, c = st.columns(3)
    a.metric("Games in the final test", f"{results['n']:,}")
    b.metric("Winner picked correctly", f"{results['accuracy']:.1%}")
    c.metric("Test log loss · lower is better", f"{results['log_loss']:.3f}")
    st.caption(
        "2023–24 regular season. The model was selected and fitted using earlier seasons; both teams need ten prior games."
    )
    st.subheader("Thirty teams, two sides of the court")
    st.write(
        "Good offense moves a team up. Good defense moves it right. Where did each team stand before its final regular-season game?"
    )
    st.plotly_chart(offense_defense(history, 2023), width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("Follow a team's season")
        st.write(
            "Explore how season strength, recent form and player roles changed from one game to the next."
        )
    with right:
        st.subheader("Put a prediction under the microscope")
        st.write(
            "Choose a held-out game, compare its pregame histories and see the result. Then look at the games the model got most wrong."
        )

elif page == "Team explorer":
    st.title("A season, one game at a time")
    st.write("Every point describes the team before that game's tipoff.")
    team, code, season = choose_team(history, "team")
    metric = st.selectbox(
        "What to follow",
        ["net_rating", "ortg", "drtg", "win_pct", "efg", "point_diff"],
        format_func=feature_label,
    )
    columns = [metric]
    if metric in ["net_rating", "win_pct", "efg"]:
        columns += [f"last10_{metric}"]
    chart = team.melt(
        id_vars=["gamedate", "def_team"],
        value_vars=columns,
        var_name="view",
        value_name="value",
    )
    chart["view"] = chart.view.map(
        lambda x: "Last 10 games" if x.startswith("last10") else "Season so far"
    )
    fig = px.line(
        chart,
        x="gamedate",
        y="value",
        color="view",
        hover_data=["def_team"],
        labels={"gamedate": "Game date", "value": feature_label(metric)},
    )
    st.plotly_chart(style(fig), width="stretch")
    a, b, c = st.columns(3)
    a.metric("Regular-season games", len(team))
    b.metric(
        "Final record",
        f"{int(team.off_win.sum())}–{len(team) - int(team.off_win.sum())}",
    )
    c.metric("Net rating before final game", f"{team.iloc[-1].net_rating:+.1f}")
    st.caption(
        "The final record uses all results. Chart values and the net-rating card use only earlier games."
    )
    with st.expander("Game-by-game history"):
        show = team[
            [
                "gamedate",
                "def_team",
                "prior_games",
                "win_pct",
                "net_rating",
                "last10_net_rating",
                "rest_days",
            ]
        ].copy()
        show.columns = [
            "Game date",
            "Opponent",
            "Earlier games",
            "Win percentage",
            "Net rating",
            "Last 10 net rating",
            "Rest days",
        ]
        show = show.round({c: 3 for c in show.select_dtypes("number")})
        st.dataframe(show, hide_index=True, width="stretch")

elif page == "Roster explorer":
    st.title("Who's carrying the minutes?")
    st.write(
        "A regular playing thirty minutes has more influence on these summaries than someone playing four. Roles come from past appearances, and roster membership comes from the previous game."
    )
    team, code, season = choose_team(history, "roster")
    available = team[team.prior_games.ge(10)]
    game_id = st.selectbox(
        "Before this game",
        available.nbagameid.tolist(),
        index=len(available) - 1,
        format_func=lambda gid: (
            f"{available.set_index('nbagameid').loc[gid, 'gamedate']:%b %d, %Y} · vs {available.set_index('nbagameid').loc[gid, 'def_team']}"
        ),
    )
    snapshot = available.set_index("nbagameid").loc[game_id]
    a, b, c = st.columns(3)
    a.metric("Top-five minute share · prior 10", f"{snapshot.top5_share:.1%}")
    b.metric("Minute-share continuity", f"{snapshot.minute_continuity:.1%}")
    c.metric(
        "Role-weighted participation · prior 5",
        f"{snapshot.weighted_participation:.1%}",
    )
    roles = table("player_roles.csv.gz")
    roles = roles[roles.nbagameid.eq(game_id) & roles.team.eq(code)].sort_values(
        "role_minutes"
    )
    fig = px.bar(
        roles,
        x="role_minutes",
        y="player_name",
        orientation="h",
        color_discrete_sequence=[TEAL],
        labels={
            "role_minutes": "Minutes per appearance in the prior 20 team games",
            "player_name": "",
        },
        hover_data={"prior5_participation": ":.0%", "recent_minute_share": ":.0%"},
    )
    st.plotly_chart(style(fig, max(360, len(roles) * 29)), width="stretch")
    st.caption(
        "This is the last observed roster, not a confirmed lineup or injury report. New arrivals may be unknown, and recently departed players can remain until the next game is observed."
    )
    form = available.melt(
        id_vars="gamedate",
        value_vars=["top5_share", "minute_continuity", "weighted_participation"],
        var_name="metric",
        value_name="value",
    )
    form["metric"] = form.metric.map(feature_label)
    fig = px.line(
        form,
        x="gamedate",
        y="value",
        color="metric",
        labels={"gamedate": "Game date", "value": "Share"},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig), width="stretch")
    with st.expander("How these histories are calculated"):
        st.write(
            "Minute continuity is the overlap between the last five games' minute shares and the five before them. Participation is the share of prior rostered games in which each player played, weighted by their usual minutes. A missed-game flag is used only after that game is over; positive playing time takes precedence over a contradictory missed flag."
        )

elif page == "Historical game predictor":
    st.title("Before the result was known")
    st.write(
        "Choose a game from the reserved 2023–24 test season. These are stored predictions from a model fitted through 2022–23, with team and player histories updated before each tipoff."
    )
    teams = (
        history[history.season.eq(2023)]
        .drop_duplicates("off_team")
        .set_index("off_team")
        .off_team_name.to_dict()
    )
    code = st.selectbox(
        "Team", ["All teams", *sorted(teams)], format_func=lambda x: teams.get(x, x)
    )
    available = (
        predictions
        if code == "All teams"
        else predictions[
            predictions.home_off_team.eq(code) | predictions.away_off_team.eq(code)
        ]
    )
    available = available.sort_values("gamedate", ascending=False).set_index(
        "nbagameid"
    )
    game_id = st.selectbox(
        "Game",
        available.index.tolist(),
        format_func=lambda gid: (
            f"{available.loc[gid, 'gamedate']:%b %d, %Y} · {available.loc[gid, 'away_off_team']} at {available.loc[gid, 'home_off_team']}"
        ),
    )
    game = available.loc[game_id]
    st.subheader("Pregame win probability")
    probability_cards(game)
    match = table("matchups.csv.gz").set_index("nbagameid").loc[game_id]
    st.subheader("What was known about the matchup")
    metrics = [
        "net_rating",
        "win_pct",
        "last10_net_rating",
        "rest_days",
        "minute_continuity",
        "weighted_ppa",
        "weighted_absence",
    ]
    comparison = pd.DataFrame(
        {
            "Basketball measure": [feature_label(m) for m in metrics],
            str(game.away_off_team): [match[f"away_{m}"] for m in metrics],
            str(game.home_off_team): [match[f"home_{m}"] for m in metrics],
            "Nonzero final coefficient": [
                m + "_diff"
                in table("importance.csv")
                .query("absolute_importance > 0")
                .feature.tolist()
                for m in metrics
            ],
        }
    )
    st.dataframe(comparison.round(3), hide_index=True, width="stretch")
    st.caption(
        "Ratings are points per 100 possessions. Win percentage, continuity and absence history are fractions. Histories end before this date; current-game minutes and participation are excluded."
    )
    st.divider()
    st.subheader("Actual result")
    st.write(
        f"**{game.away_off_team} {int(game.away_points)} · {game.home_off_team} {int(game.home_points)}**"
    )
    st.caption(
        "The result is shown for comparison and was not used to create this game's probability."
    )

elif page == "What the model uses":
    st.title("What moves the prediction?")
    importance = table("importance.csv").head(10).copy()
    importance["Basketball measure"] = importance.feature.map(feature_label)
    value = "coefficient" if "coefficient" in importance else "absolute_importance"
    fig = px.bar(
        importance.sort_values(value),
        x=value,
        y="Basketball measure",
        orientation="h",
        color_discrete_sequence=[CLAY],
    )
    st.plotly_chart(style(fig, 430), width="stretch")
    if value == "coefficient":
        st.write(
            "Each input compares the home team with its opponent. Positive bars favor the home team when that difference increases; negative bars favor the visiting team. Bars use a one-standard-deviation change measured in the training data."
        )
        st.caption(
            "These describe associations used by the model, not isolated causal effects. Correlated measures can share credit."
        )
    else:
        st.write(
            "Bars show how much prediction quality falls when a measure is shuffled. Correlated measures can share credit."
        )
    st.subheader("The basketball ideas")
    st.markdown(
        "**Team strength:** scoring and conceding points efficiently over the season.\n\n**Recent form:** whether the past five, ten or twenty games add something beyond the season average.\n\n**Rotations:** how minutes are distributed, how stable the lineup has been, and how players have performed in their previous roles.\n\n**Game context:** rest and recent schedule density. Designated home court is represented by the model's intercept."
    )
    st.caption(
        f"Selected model: {results['selected_model']}. See Performance for the comparisons that informed this choice."
    )
    with st.expander("All retained inputs"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Input": selection["features"],
                    "Basketball measure": [
                        feature_label(f) for f in selection["features"]
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )

elif page == "Performance":
    st.title("How good were the predictions?")
    st.write(
        "The final check is 2023–24: an entire later season that did not participate in model selection. Earlier games within that season update histories; they never refit the model."
    )
    score = table("test_metrics.csv").copy()
    score["model"] = score.model.replace(
        {
            "Baseline net_rating": "Season net rating",
            "Baseline win_pct": "Season win percentage",
        }
    )
    st.dataframe(
        score[["model", "log_loss", "brier", "auc", "accuracy", "n"]].round(4),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Lower log loss and Brier score are better. Higher AUC and accuracy are better. All rows use the same 1,074 games."
    )
    st.subheader("Did 70% favorites win about 70% of the time?")
    calibration = table("test_calibration.csv")
    fig = go.Figure(
        go.Scatter(
            x=calibration.mean_probability,
            y=calibration.win_rate,
            mode="markers+lines",
            name="Held-out games",
            marker=dict(size=np.sqrt(calibration.n) * 1.3, color=CLAY),
            error_y=dict(
                type="data",
                symmetric=False,
                array=calibration.upper - calibration.win_rate,
                arrayminus=calibration.win_rate - calibration.lower,
            ),
            customdata=calibration.n,
            hovertemplate="Predicted %{x:.0%}<br>Won %{y:.0%}<br>%{customdata} games<extra></extra>",
        )
    )
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color=TEAL, dash="dash"),
        name="Perfect calibration",
    )
    fig.update_xaxes(
        title="Predicted home win probability", tickformat=".0%", range=[0, 1]
    )
    fig.update_yaxes(title="Observed home win rate", tickformat=".0%", range=[0, 1])
    st.plotly_chart(style(fig), width="stretch")
    st.caption(
        "Circle size reflects the number of games; bars show within-bin 95% Wilson intervals, which treat games as independent. They do not capture all schedule dependence."
    )
    intervals = table("test_bootstrap.csv")
    row = intervals[intervals.metric.eq("delta_log_loss")].iloc[0]
    st.write(
        f"Against the net-rating baseline, the log-loss difference is **{row.estimate:+.4f}** (95% week-block bootstrap interval: **{row.lower:+.4f} to {row.upper:+.4f}**). Negative favors the selected model."
    )
    st.caption(
        "The bootstrap resamples calendar weeks together. It measures uncertainty within this test season, not across future seasons."
    )
    st.subheader("Earlier seasons informed the choice")
    folds = table("cv_folds.csv")
    subset = folds[
        folds.name.isin(
            [
                results["selected_model"],
                "Baseline net_rating",
                "Baseline win_pct",
                "Home win rate",
            ]
        )
    ].copy()
    subset["Season"] = subset.season.map(season_label)
    fig = px.line(
        subset,
        x="Season",
        y="log_loss",
        color="name",
        markers=True,
        labels={"log_loss": "Log loss"},
    )
    st.plotly_chart(style(fig), width="stretch")
    st.caption(
        "Each validation season is predicted from earlier training seasons. These results were used for selection, so they are not an independent final performance estimate."
    )
    with st.expander("Model comparisons and training fit"):
        summary = table("cv_summary.csv")
        st.dataframe(summary.round(4), hide_index=True, width="stretch")
        st.write(
            "Models were ranked by average season log loss. Within 0.001 of the best score, I preferred fewer inputs, then logistic regression. Training performance helps identify overfitting; the held-out season did not decide the winner."
        )

elif page == "Biggest misses":
    st.title("The games that went the other way")
    st.write(
        "Even a sensible favorite loses sometimes. These were the most confident wrong predictions in the held-out season."
    )
    misses = table("biggest_misses.csv").copy()
    misses["Favorite"] = np.where(
        misses.probability.ge(0.5), misses.home_off_team, misses.away_off_team
    )
    misses["Matchup"] = misses.away_off_team + " at " + misses.home_off_team
    misses["Score · away–home"] = (
        misses.away_points.astype(int).astype(str)
        + "–"
        + misses.home_points.astype(int).astype(str)
    )
    show = misses[
        ["gamedate", "Matchup", "Favorite", "confidence", "Score · away–home"]
    ].rename(columns={"gamedate": "Date", "confidence": "Favorite probability"})
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "Favorite probability": st.column_config.NumberColumn(format="percent")
        },
    )
    st.subheader("Which teams surprised the model?")
    errors = table("team_errors.csv")
    fig = px.bar(
        errors,
        x="actual_minus_predicted",
        y="team",
        orientation="h",
        color_discrete_sequence=[TEAL],
        labels={
            "actual_minus_predicted": "Actual win rate minus predicted win rate",
            "team": "",
        },
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 710), width="stretch")
    st.caption(
        "Positive means a team won more than its probabilities suggested. This includes only eligible test games. The supplied data cannot establish explanations involving injuries, trades or tactical changes."
    )

else:
    st.title("About the model")
    st.markdown(
        "I used the supplied team and player game records to build a pregame home-win model. The question is simple: **how much can past basketball performance tell us about the next game?**"
    )
    st.subheader("Data and coverage")
    st.write(
        "The files contain 27,144 team rows and 434,797 player rows, including 11,979 regular-season games from 2014–15 through 2023–24. One offensive row per team is paired with its opponent. After a ten-game warmup for each team, 10,418 games are eligible for modeling."
    )
    st.subheader("What is available before tipoff")
    st.write(
        "Season totals and rolling windows stop at the previous game. Player roles use minutes in prior appearances; role-weighted scoring uses previous shot-attempt totals. Historical absence flags describe earlier participation only. The most recent observed roster supplies membership, so there is no assumption that the current lineup was known."
    )
    st.subheader("Choosing and checking the model")
    st.write(
        "Training starts with 2014–15 through 2016–17. Six expanding windows validate on 2017–18 through 2022–23. I compared home-win-rate, win-percentage and net-rating baselines with ordinary and regularized logistic regression, random forest and gradient boosting. Preprocessing is fitted only on each training window. The selected model was then frozen for 2023–24."
    )
    st.write(
        "Log loss penalizes confident mistakes; Brier score measures squared probability error; AUC measures ranking; accuracy checks the favorite. Calibration asks whether probabilities match observed frequencies. No final-test outcomes choose inputs or tune the model."
    )
    st.subheader("Limits of the information")
    st.markdown(
        "- No live predictions, confirmed injury reports, current starting lineups, travel data or betting prices.\n- Early-season games need more history and are outside prediction coverage.\n- Historical rosters can lag trades or arrivals.\n- Designated home teams are known; neutral-site games are not separately identified.\n- Unusual source-clock totals make pace unreliable, so it is omitted.\n- One held-out season is useful evidence, not a guarantee of future performance."
    )
    st.link_button(
        "Read the notebooks and reproduce the analysis",
        "https://github.com/kevinrouse/nba-game-predictor/tree/main/notebooks",
    )
