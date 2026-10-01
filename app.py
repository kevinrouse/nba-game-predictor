"""Explore historical NBA games and their pregame win probabilities."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.charts import CLAY, TEAL, feature_label, offense_defense, style
from src.data import ROOT
from src.game_report import forecast_record, resolve_game_id
from src.story_views import basketball_questions, feature_workshop, model_comparison

st.set_page_config(page_title="NBA Game Predictor", page_icon="🏀", layout="wide")
TABLES = ROOT / "artifacts/tables"


@st.cache_data(max_entries=60)
def read_table(name, modified_ns):
    # The modification time participates in the cache key after a local rebuild.
    frame = pd.read_csv(TABLES / name)
    for column in ["gamedate", "history_end", "player_history_end"]:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def table(name):
    return read_table(name, (TABLES / name).stat().st_mtime_ns)


@st.cache_data
def metadata():
    return (
        json.loads((TABLES / "results.json").read_text()),
        json.loads((ROOT / "artifacts/models/selection.json").read_text()),
    )


@st.cache_data
def story_metadata():
    return json.loads((TABLES / "story.json").read_text())


def go_to(section):
    st.session_state["section"] = section
    sync_navigation()


def sync_navigation():
    if st.session_state["section"] != "Historical game predictor":
        st.query_params.pop("game", None)


def open_game(game_id):
    st.session_state["section"] = "Historical game predictor"
    st.session_state["predictor_team"] = "All teams"
    st.session_state["prediction_game"] = int(game_id)
    st.query_params["game"] = str(game_id)


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
story = story_metadata()
history = table("team_history.csv.gz")
predictions = table("record_comparison_games.csv")
linked_game = resolve_game_id(st.query_params.get("game"), predictions.nbagameid)
if "section" not in st.session_state:
    st.session_state["section"] = (
        "Historical game predictor" if linked_game is not None else "Home"
    )
with st.sidebar:
    st.markdown("### NBA Game Predictor")
    st.caption("A basketball project by Kevin Rouse")
    st.caption("Follow the analytical story, then explore the teams and games.")
    page = st.radio(
        "Explore",
        [
            "Home",
            "Basketball questions",
            "Feature workshop",
            "Model comparison",
            "Performance",
            "Historical game predictor",
            "Team explorer",
            "Roster explorer",
            "What the model uses",
            "Biggest misses",
            "About the model",
        ],
        label_visibility="collapsed",
        key="section",
        on_change=sync_navigation,
    )
    st.divider()
    st.caption("2014–15 through 2023–24\n\nHistorical data. Pregame probabilities.")
    st.link_button(
        "View the project on GitHub", "https://github.com/kevinrouse/nba-game-predictor"
    )


if st.query_params.get("game") is not None and linked_game is None:
    st.info(
        "That game link is outside the eligible historical test games. Choose a game in the explorer."
    )

if page == "Home":
    st.caption("BEFORE TIPOFF")
    st.title("What makes a team the favorite?")
    st.markdown(
        "I wanted to see how well NBA games could be predicted from what teams and players had already done. Season records are a starting point, but rotations shift, shooting comes and goes, and a team's last few weeks can look different from its whole season."
    )
    st.markdown(
        "A natural starting point is to pick the team with the better record. I compare that rule with a model that also considers how teams have played, who's been carrying the minutes, and their recent schedule. The aim is a useful pregame probability, with enough detail to understand its strengths and mistakes."
    )
    a, b, c, d = st.columns(4)
    a.metric("Model picks correct", f"{story['model_accuracy']:.1%}")
    b.metric("Better-record picks correct", f"{story['record_accuracy']:.1%}")
    c.metric("Additional correct picks", f"{story['extra_correct']:+d}")
    d.metric("Same held-out games", f"{story['test_games']:,}")
    st.caption(
        "2023–24 regular season. The model was selected and fitted using earlier seasons; both teams need ten prior games."
    )
    st.write(
        "The comparison is a modest gain over a strong basketball rule. The methods disagree on 163 games: the model wins that comparison 99–64. Their uncertainty and individual mistakes are part of the story too."
    )
    st.caption(
        "The record rule uses pregame winning percentage and breaks exact ties toward home. It was added after the original test evaluation; the model's stored predictions are unchanged."
    )
    st.subheader("Follow the reasoning")
    steps = [
        (
            "Basketball questions",
            "1 · Look at the basketball",
            "What does record leave out, and which signals are worth investigating?",
        ),
        (
            "Feature workshop",
            "2 · Rebuild the history",
            "Follow one game from earlier box scores to team and player inputs.",
        ),
        (
            "Model comparison",
            "3 · Make complexity earn its place",
            "Compare baselines, logistic models, forests and boosting on later seasons.",
        ),
        (
            "Performance",
            "4 · Check the predictions",
            "See the record comparison, probability quality, uncertainty and misses.",
        ),
    ]
    for column, (destination, heading, description) in zip(st.columns(4), steps):
        with column:
            st.markdown(f"**{heading}**")
            st.write(description)
            st.button(
                "Read this chapter",
                key=f"home_{destination}",
                on_click=go_to,
                args=(destination,),
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

elif page == "Basketball questions":
    basketball_questions(
        table("eda_feature_relationships.csv"),
        table("eda_rest.csv"),
        table("eda_recent_conditional.csv"),
        table("feature_guide.csv"),
    )

elif page == "Feature workshop":
    example = json.loads((TABLES / "worked_example.json").read_text())
    feature_workshop(
        table("worked_team_totals.csv"),
        table("worked_player_roles.csv"),
        table("feature_guide.csv"),
        example,
    )

elif page == "Model comparison":
    configs = json.loads((TABLES / "candidate_configs.json").read_text())
    model_comparison(
        table("model_family_comparison.csv"),
        table("matched_input_comparison.csv"),
        table("cv_folds.csv"),
        configs,
        table("cv_summary.csv"),
        selection,
        table("feature_comparison.csv"),
        table("selection_replay.csv"),
        table("selection_replay_comparison.csv"),
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
    reasons = {
        "net_rating": "Net rating measures how much a team outscores opponents per 100 possessions. I use it as the main summary of strength; comparing windows shows how quickly the picture can change.",
        "ortg": "Offensive rating describes points scored per 100 possessions. It helps explain the basketball, while net rating combines offense and defense in the model.",
        "drtg": "Defensive rating uses the opponent's offense to measure points allowed per 100 possessions. Lower is better; full-season defensive totals are never assigned backward to earlier games.",
        "win_pct": "Winning percentage is the simplest strength comparison. The record rule picks the higher pregame percentage; the model asks what additional histories contribute.",
        "efg": "Effective field goal percentage credits the extra value of threes. Short windows react quickly to shooting changes, but can also be noisy and overlap with net rating.",
        "point_diff": "Average point differential distinguishes narrow wins from larger margins. It is shown descriptively; the model uses possession-adjusted net rating instead of adding another closely related strength summary.",
    }
    st.write(reasons[metric])
    columns = [metric]
    if metric in ["net_rating", "win_pct", "efg"]:
        windows = st.multiselect("Recent windows to compare", [5, 10, 20], default=[10])
        columns += [f"last{window}_{metric}" for window in windows]
    chart = team.melt(
        id_vars=["gamedate", "def_team"],
        value_vars=columns,
        var_name="view",
        value_name="value",
    )
    chart["view"] = chart.view.map(
        lambda x: (
            f"Last {x.split('_')[0][4:]} games"
            if x.startswith("last")
            else "Season so far"
        )
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
    st.subheader("Why use these roles in a forecast?")
    st.write(
        "Team records can lag a changing rotation. I tested whether the players who normally carry the minutes had been scoring efficiently and participating consistently. Adding those histories helped the same L1 team model in five of six development seasons, so they earned a place in the final model."
    )
    explanation = roles.copy()
    explanation["scoring_weight"] = explanation.role_weight.where(
        explanation.prior5_ppa.notna(), 0
    )
    explanation["scoring_weight"] /= explanation.scoring_weight.sum()
    explanation["efficiency_contribution"] = (
        explanation.scoring_weight * explanation.prior5_ppa.fillna(0)
    )
    a, b = st.columns(2)
    a.metric("Role-weighted recent scoring efficiency", f"{snapshot.weighted_ppa:.3f}")
    b.metric("Role-weighted historical absence", f"{snapshot.weighted_absence:.1%}")
    with st.expander("See how the players add up"):
        st.dataframe(
            explanation[
                [
                    "player_name",
                    "role_minutes",
                    "role_weight",
                    "prior5_participation",
                    "prior5_ppa",
                    "scoring_weight",
                    "efficiency_contribution",
                ]
            ].round(4),
            hide_index=True,
            width="stretch",
        )
        st.write(
            "Scoring efficiency is shot-attempt points per shot attempt over prior games. Players without recent attempts have no efficiency estimate; remaining scoring weights are renormalized. The weighted contributions sum to the scoring-efficiency card."
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
        "Team",
        ["All teams", *sorted(teams)],
        format_func=lambda x: teams.get(x, x),
        key="predictor_team",
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
    if st.session_state.get("prediction_game") not in available.index:
        st.session_state["prediction_game"] = (
            linked_game if linked_game in available.index else int(available.index[0])
        )
    game_id = st.selectbox(
        "Game",
        available.index.tolist(),
        format_func=lambda gid: (
            f"{available.loc[gid, 'gamedate']:%b %d, %Y} · {available.loc[gid, 'away_off_team']} at {available.loc[gid, 'home_off_team']}"
        ),
        key="prediction_game",
    )
    if st.query_params.get("game") != str(game_id):
        st.query_params["game"] = str(game_id)
    game = available.loc[game_id]
    st.caption(
        "The address bar now links to this game. Copy its URL to share the same forecast."
    )
    st.subheader("Pregame win probability")
    probability_cards(game)
    record_favorite = (
        game.home_off_team if game.record_pick_home else game.away_off_team
    )
    model_favorite = (
        game.home_off_team if game.probability >= 0.5 else game.away_off_team
    )
    st.subheader("What would the record rule have picked?")
    st.write(
        f"Before this game: **{game.away_off_team} {int(game.away_prior_wins)}–{int(game.away_prior_games - game.away_prior_wins)}**, "
        f"**{game.home_off_team} {int(game.home_prior_wins)}–{int(game.home_prior_games - game.home_prior_wins)}**. "
        f"The better-record rule picks **{record_favorite}**; the model favors **{model_favorite}**."
    )
    if game.record_tie:
        st.caption(
            "The pregame winning percentages are exactly tied, so the record rule uses its fixed home-team tie-break."
        )
    else:
        st.caption(
            "The rule compares winning percentage because the teams may have played different numbers of games. It chooses a winner without claiming a probability."
        )
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
    st.subheader("How the model weighed this game")
    contributions = table("prediction_contributions.csv.gz")
    contributions = contributions[contributions.nbagameid.eq(game_id)].merge(
        table("feature_guide.csv")[["feature", "group"]],
        on="feature",
        validate="one_to_one",
    )
    grouped = contributions.groupby("group", sort=False).contribution.sum()
    intercept = json.loads((TABLES / "intercept.json").read_text())["intercept"]
    fig = go.Figure(
        go.Waterfall(
            x=["Starting point", *grouped.index, "Final score"],
            y=[intercept, *grouped.values, 0],
            measure=["absolute", *(["relative"] * len(grouped)), "total"],
            increasing={"marker": {"color": CLAY}},
            decreasing={"marker": {"color": TEAL}},
            totals={"marker": {"color": "#24343b"}},
        )
    )
    fig.update_yaxes(title="Home-win log odds")
    st.plotly_chart(style(fig, 360), width="stretch")
    st.caption(
        "The model adds these contributions in log-odds space, then converts the total to the probability above. The starting point is its training-mean intercept. These are conditional model calculations, not causal explanations of the result."
    )
    st.divider()
    st.subheader("Actual result")
    st.write(
        f"**{game.away_off_team} {int(game.away_points)} · {game.home_off_team} {int(game.home_points)}**"
    )
    st.caption(
        "The result is shown for comparison and was not used to create this game's probability."
    )
    release = json.loads((ROOT / "artifacts/models/release.json").read_text())
    report = forecast_record(
        game, match, table("prediction_contributions.csv.gz"), release
    )
    st.download_button(
        "Download this game's audit record",
        json.dumps(report, indent=2, allow_nan=False),
        file_name=f"nba-pregame-{game_id}.json",
        mime="application/json",
    )
    st.caption(
        "Includes the model identifier, history cutoffs, all input contributions and the observed result in a separate field."
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
    with st.expander("All candidate inputs and final decisions"):
        st.dataframe(
            table("feature_guide.csv")[
                ["label", "question", "coefficient", "decision", "caution"]
            ],
            hide_index=True,
            width="stretch",
        )

elif page == "Performance":
    st.caption("THE STORY · 4 / 4")
    st.title("How good were the predictions?")
    st.write(
        "The final check is 2023–24: an entire later season that did not participate in model selection. Earlier games within that season update histories; they never refit the model."
    )
    st.subheader("First: beat picking the better record")
    a, b, c = st.columns(3)
    a.metric(
        "Better-record rule",
        f"{story['record_accuracy']:.1%}",
        help="Higher pregame winning percentage; exact ties favor home.",
    )
    b.metric("Selected model", f"{story['model_accuracy']:.1%}")
    c.metric("Additional correct picks", f"{story['extra_correct']:+d}")
    st.dataframe(
        table("decision_baselines.csv"),
        hide_index=True,
        width="stretch",
        column_config={"accuracy": st.column_config.NumberColumn(format="percent")},
    )
    interval = story["accuracy_interval"]
    st.write(
        f"The model adds **{100 * interval['estimate']:.2f} percentage points** in accuracy. "
        f"The paired week-block 95% interval is **{100 * interval['lower']:.2f} to {100 * interval['upper']:.2f} points**. "
        f"On their {story['disagreements']} disagreements, the model is right {story['model_only_correct']} times and the rule {story['record_only_correct']} times."
    )
    st.caption(
        "This record-rule comparison was added after the original test evaluation; all model predictions remain frozen. The interval describes this season, not performance across future seasons."
    )
    with st.expander("Does the uncertainty depend on the length of a block?"):
        sensitivity = table("block_sensitivity.csv")
        shown = sensitivity[sensitivity.metric.eq("accuracy_difference")][
            ["block_weeks", "blocks", "estimate", "lower", "upper"]
        ].copy()
        for column in ["estimate", "lower", "upper"]:
            shown[column] *= 100
        shown.columns = [
            "Weeks per block",
            "Blocks",
            "Gain (percentage points)",
            "95% lower",
            "95% upper",
        ]
        st.dataframe(shown.round(2), hide_index=True, width="stretch")
        st.write(
            "The gain remains positive for all three choices. Longer blocks preserve more nearby games together, but leave fewer blocks to resample—only six for four weeks. This sensitivity check does not capture between-season uncertainty or turn the retrospective comparison into a new independent test."
        )
    st.subheader("Then: check the quality of the probabilities")
    st.write(
        "The record rule gives a winner, not a calibrated probability. For log loss and Brier score, compare against fitted win-percentage and net-rating baselines instead of assigning the rule artificial 100% confidence."
    )
    score = table("test_metrics.csv").copy()
    score["model"] = score.model.replace(
        {
            "Baseline net_rating": "Season net rating",
            "Baseline win_pct": "Season win percentage",
            results["selected_model"]: "Team + player histories (L1)",
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
        "Even a sensible favorite loses sometimes. I inspect confident errors and both sides of the model-versus-record disagreement, rather than explaining the project through only its successful picks."
    )
    view = st.selectbox(
        "Games to examine",
        [
            "Most confident model misses",
            "Model right, record rule wrong",
            "Record rule right, model wrong",
        ],
    )
    if view == "Most confident model misses":
        misses = (
            predictions[~predictions.model_correct]
            .sort_values("confidence", ascending=False)
            .head(25)
            .copy()
        )
    elif view == "Model right, record rule wrong":
        misses = (
            predictions[predictions.model_correct & ~predictions.record_correct]
            .sort_values("confidence", ascending=False)
            .head(25)
            .copy()
        )
    else:
        misses = (
            predictions[~predictions.model_correct & predictions.record_correct]
            .sort_values("confidence", ascending=False)
            .head(25)
            .copy()
        )
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
    inspect_id = st.selectbox(
        "Inspect a game from this comparison",
        misses.nbagameid.tolist(),
        format_func=lambda gid: (
            f"{misses.set_index('nbagameid').loc[gid, 'gamedate']:%b %d, %Y} · "
            f"{misses.set_index('nbagameid').loc[gid, 'Matchup']}"
        ),
    )
    st.button(
        "Inspect this forecast",
        on_click=open_game,
        args=(inspect_id,),
        key="inspect_forecast",
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
    st.subheader("Does the gain depend on the part of the season?")
    st.dataframe(
        table("test_season_stages.csv"),
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="percent")
            for c in ["model_accuracy", "record_accuracy"]
        },
    )
    st.caption(
        "Groups use the smaller of the teams' pregame game counts. These are diagnostics after evaluation, not new groups used to tune the model."
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
    st.dataframe(
        table("split_timeline.csv")[
            [
                "evaluation_season",
                "stage",
                "first_train_season",
                "last_train_season",
                "train_games",
                "evaluation_games",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.write(
        "The better-record winner rule was added during a later reporting revision. It compares pregame win percentages, breaks exact ties in favor of home, and is scored on the same games. That retrospective addition did not change the original model, predictions or favorite threshold."
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


chapters = {
    "Home": ("Basketball questions", "01_data_and_eda.ipynb"),
    "Basketball questions": ("Feature workshop", "01_data_and_eda.ipynb"),
    "Feature workshop": ("Model comparison", "02_pregame_features.ipynb"),
    "Model comparison": ("Performance", "03_modeling.ipynb"),
    "Performance": ("Historical game predictor", "04_model_interpretation.ipynb"),
    "Historical game predictor": ("Biggest misses", "04_model_interpretation.ipynb"),
    "Team explorer": ("Roster explorer", "01_data_and_eda.ipynb"),
    "Roster explorer": ("What the model uses", "02_pregame_features.ipynb"),
    "What the model uses": ("Model comparison", "03_modeling.ipynb"),
    "Biggest misses": ("About the model", "04_model_interpretation.ipynb"),
}
if page in chapters:
    destination, notebook = chapters[page]
    st.divider()
    left, right = st.columns(2)
    left.button(
        f"Continue: {destination}",
        on_click=go_to,
        args=(destination,),
        key="chapter_next",
    )
    right.link_button(
        "Read the full notebook",
        f"https://github.com/kevinrouse/nba-game-predictor/blob/main/notebooks/{notebook}",
    )
