"""The guided analytical chapters of the basketball dashboard."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.charts import CLAY, TEAL, feature_label, style


def basketball_questions(relationships, rest, recent, guide):
    st.caption("THE STORY · 1 / 4")
    st.title("What could the record be missing?")
    st.write(
        "A team's record is a strong starting point. I looked for basketball information it might compress away: how the team scores and defends, whether its recent form differs from the season, who's carrying the minutes, and how crowded the schedule has been."
    )
    st.info(
        "These explorations use earlier development seasons, 2014–15 through 2022–23. They motivate candidates; later-season validation decides whether those candidates help."
    )
    st.subheader("A basketball reason for each candidate")
    available = relationships.feature.unique().tolist()
    feature = st.selectbox(
        "Explore a pregame signal", available, format_func=feature_label
    )
    row = guide[guide.feature.eq(feature)].iloc[0]
    st.markdown(
        f"**The question:** {row.question}\n\n**The measure:** {row.construction}."
    )
    data = relationships[relationships.feature.eq(feature)]
    fig = go.Figure(
        go.Scatter(
            x=data.value,
            y=data.home_win_rate,
            mode="markers+lines",
            marker_color=TEAL,
            customdata=data.n,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": data.upper - data.home_win_rate,
                "arrayminus": data.home_win_rate - data.lower,
            },
            hovertemplate="Difference: %{x:.3f}<br>Home wins: %{y:.1%}<br>Games: %{customdata}<extra></extra>",
        )
    )
    fig.update_xaxes(title=feature_label(feature) + " · home minus away")
    fig.update_yaxes(title="Observed home-win rate", tickformat=".0%", range=[0, 1])
    st.plotly_chart(style(fig), width="stretch")
    st.write(
        f"**What to be careful about:** {row.caution}. This is an unadjusted relationship, so related team characteristics can explain part of the pattern."
    )
    st.caption(
        "Each point groups roughly one fifth of eligible development games. Error bars are descriptive Wilson intervals under independence; they do not capture all repeated-team or schedule dependence."
    )
    st.subheader("Does recent form tell us something different?")
    st.write(
        "Five-game windows react quickly, but can chase noise. Ten or twenty games may be steadier. As a first check, I compare recent-form direction within broad season-strength groups."
    )
    fig = px.bar(
        recent,
        x="strength_band",
        y="home_win_rate",
        color="recent_direction",
        barmode="group",
        hover_data=["games"],
        labels={
            "strength_band": "Season net-rating difference",
            "home_win_rate": "Home-win rate",
        },
        category_orders={
            "strength_band": ["Away stronger", "Similar strength", "Home stronger"]
        },
        color_discrete_sequence=[TEAL, CLAY],
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(legend={"orientation": "h", "y": 1.2, "font": {"size": 10}})
    st.plotly_chart(style(fig), width="stretch")
    st.write(
        "The descriptive separation is a reason to test recent windows. It is not proof of a separate hot-hand effect: teams still differ within each broad group. Model comparison later shows how little extra team-form information survives once other histories are included."
    )
    st.subheader("What can the schedule add?")
    shown = rest.rename(
        columns={
            "schedule": "Pregame schedule",
            "games": "Games",
            "home_win_rate": "Home-win rate",
            "mean_strength_diff": "Average net-rating difference",
        }
    )
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "Home-win rate": st.column_config.NumberColumn(format="percent"),
            "Average net-rating difference": st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
    )
    st.write(
        "Only the home team on a back-to-back corresponds to about 50.4% home wins, compared with 61.3% when only the visitor is on one. I test rest and schedule density, while keeping travel and neutral-site claims out of the story."
    )
    st.subheader("From curiosity to a candidate set")
    st.markdown(
        "**Season strength** gives the baseline. **Shooting and possession control** describe how teams create advantages. **Recent form** asks whether the season average is slow to catch up. **Player roles and participation** ask who has been producing that performance. **Rest** supplies game context. Each still has to earn its place in validation."
    )


def feature_workshop(totals, roles, guide, example):
    st.caption("THE STORY · 2 / 4")
    st.title("Build one forecast before tipoff")
    st.write(
        f"A worked example: **{example['away']} at {example['home']} · {example['gamedate']}**. This game was chosen by its date and available history, not by its result. Follow the records available before it into a team comparison."
    )
    st.subheader("1. Stop the team history at the previous game")
    display = totals.copy()
    display["Net rating"] = display.ortg - display.drtg
    display["Record"] = (
        display.prior_wins.astype(int).astype(str)
        + "–"
        + (display.prior_games - display.prior_wins).astype(int).astype(str)
    )
    shown = display[
        [
            "team",
            "Record",
            "prior_games",
            "points",
            "possessions",
            "ortg",
            "drtg",
            "Net rating",
            "history_end",
        ]
    ].rename(
        columns={
            "team": "Team",
            "prior_games": "Earlier games",
            "points": "Earlier points",
            "possessions": "Earlier possessions",
            "ortg": "Offensive rating",
            "drtg": "Defensive rating",
            "history_end": "Last source game",
        }
    )
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.2f")
            for c in ["Offensive rating", "Defensive rating", "Net rating"]
        },
    )
    st.write(
        "Offensive rating is 100 × earlier points / earlier possessions. Defensive rating comes from the opponent's offensive rows. The counts are shifted before accumulation; the current box score is absent. Rates compare teams with different numbers of games and possessions."
    )
    st.subheader("2. Estimate roles from earlier appearances")
    st.write(
        "The last observed roster supplies membership. Minutes per appearance over the prior twenty team games estimate each player's normal role. Normalize those estimates to get weights; do not use the minutes they actually play tonight."
    )
    fig = px.bar(
        roles.sort_values("role_weight"),
        x="role_weight",
        y="player_name",
        orientation="h",
        color_discrete_sequence=[TEAL],
        labels={"role_weight": "Weight from prior role", "player_name": ""},
        hover_data={"role_minutes": ":.1f", "appearances": True},
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, max(380, len(roles) * 28)), width="stretch")
    st.subheader("3. Add up weighted recent player information")
    a, b, c = st.columns(3)
    a.metric("Role-weighted scoring efficiency", f"{roles.ppa_contribution.sum():.3f}")
    b.metric(
        "Role-weighted participation", f"{roles.participation_contribution.sum():.1%}"
    )
    c.metric(
        "Role-weighted historical absence", f"{roles.absence_contribution.sum():.1%}"
    )
    st.write(
        "Recent scoring uses each player's prior-five-game points per shot attempt. Players with no recent attempts are excluded from that average and the scoring weights are renormalized. Participation and historical absence retain the original role weights."
    )
    with st.expander("Follow each player's contribution"):
        shown = roles[
            [
                "player_name",
                "role_minutes",
                "role_weight",
                "attempts",
                "attempt_points",
                "ppa",
                "scoring_weight",
                "ppa_contribution",
                "absence_contribution",
            ]
        ].copy()
        st.dataframe(shown.round(4), hide_index=True, width="stretch")
    st.caption(
        "Historical absence is not a current injury report. Previous rosters can lag arrivals and departures; the files do not establish when a same-game missed flag became known."
    )
    st.subheader("4. Compare home with away, then test what helps")
    metric = st.selectbox(
        "Inspect a candidate feature", guide.metric.tolist(), format_func=feature_label
    )
    row = guide[guide.metric.eq(metric)].iloc[0]
    st.markdown(
        f"**Why look at it?** {row.question}\n\n**How it is built:** {row.construction}. The matchup input is home minus away.\n\n**What happened in the final fit:** {row.decision}.\n\n**Limitation:** {row.caution}."
    )
    st.write(
        "Both teams need ten earlier games. Tests change current/future statistics, remove future rows and even remove the current player file; the earlier pregame inputs must stay identical. The full notebook shows those checks and reconciles the worked calculations to the saved matchup row."
    )


def model_comparison(
    families, matched, folds, configs, summary, selection, feature_comparison
):
    st.caption("THE STORY · 3 / 4")
    st.title("Which model earned the extra detail?")
    st.write(
        "I start with home win rate, record and net rating, then compare ordinary logistic regression, L1 and L2 regularization, random forest and gradient boosting. Six later-season validation windows decide the model; the final test does not."
    )
    st.markdown(
        "**Why these families?** Logistic regression gives an additive, inspectable forecast. L1/L2 control overlapping basketball inputs. Forests and boosting test whether nonlinear thresholds and interactions do better. More complex models must improve later-season predictions to earn their place."
    )
    st.subheader("The model-family comparison")
    shown = families[
        [
            "family",
            "name",
            "n_features",
            "log_loss",
            "brier",
            "auc",
            "accuracy",
            "selected",
        ]
    ].copy()
    shown.columns = [
        "Family",
        "Configuration",
        "Inputs",
        "Validation log loss",
        "Brier",
        "AUC",
        "Accuracy",
        "Selected",
    ]
    st.dataframe(shown.round(4), hide_index=True, width="stretch")
    ordered = families.sort_values("log_loss", ascending=False)
    fig = px.scatter(
        ordered,
        x="log_loss",
        y="name",
        color="selected",
        color_discrete_map={True: CLAY, False: TEAL},
        labels={
            "log_loss": "Mean validation-season log loss · lower is better",
            "name": "",
        },
    )
    fig.update_traces(marker_size=11)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(categoryorder="array", categoryarray=ordered.name.tolist())
    fig.update_xaxes(range=[0.59, 0.70])
    st.plotly_chart(style(fig, 420), width="stretch")
    st.caption(
        "Best searched configuration from each family, both strength baselines, plus the selected smaller L1 version. These scores were used for selection; they are not the final independent result."
    )
    st.subheader("Keep the inputs the same")
    st.write(
        "A family can look better simply because it uses different features. This comparison holds the same 24 combined-ten-game inputs fixed. Each family's setting is still selected from its original development candidates."
    )
    st.dataframe(
        matched[["family", "log_loss", "brier", "accuracy", "train_log_loss"]].round(4),
        hide_index=True,
        width="stretch",
    )
    gap = matched.melt(
        id_vars="family",
        value_vars=["train_log_loss", "log_loss"],
        var_name="Evaluation",
        value_name="Loss",
    )
    gap["Evaluation"] = gap.Evaluation.map(
        {"train_log_loss": "Training", "log_loss": "Validation"}
    )
    fig = px.scatter(
        gap,
        x="Loss",
        y="family",
        color="Evaluation",
        color_discrete_sequence=[TEAL, CLAY],
        labels={"family": ""},
    )
    fig.update_traces(marker_size=12)
    st.plotly_chart(style(fig, 310), width="stretch")
    st.write(
        "The forest fits training games more closely but falls behind on later seasons. Shallow boosting is closer, yet the regularized linear models still lead on the common inputs. This is evidence about the settings tried, not a claim that every possible tree model would lose."
    )
    st.subheader("Inspect a candidate across seasons")
    names = families.name.tolist()
    choice = st.selectbox(
        "Candidate model", names, index=names.index(selection["config"]["name"])
    )
    chosen = folds[folds.name.eq(choice)]
    fig = px.line(
        chosen,
        x="season",
        y="log_loss",
        markers=True,
        labels={"season": "Validation season starting year", "log_loss": "Log loss"},
    )
    st.plotly_chart(style(fig, 280), width="stretch")
    config = next(c for c in configs if c["name"] == choice)
    with st.expander("Exact configuration and fold scores"):
        st.json(config)
        st.dataframe(chosen.round(4), hide_index=True, width="stretch")
    st.subheader("What did the feature comparisons change?")
    st.dataframe(
        feature_comparison[["features", "name", "n_features", "log_loss"]].round(5),
        hide_index=True,
        width="stretch",
    )
    st.write(
        "Adding player histories to the same L1 team model helps five of six validation seasons. Recent team form helps a little too. The best twenty-game combined model reaches 0.62438 loss, but removing its two extra team-form inputs costs only 0.00060."
    )
    st.subheader("Why this final model?")
    best = summary.log_loss.min()
    close = summary[summary.log_loss.le(best + 0.001)].sort_values(
        ["n_features", "log_loss"]
    )
    st.write(
        "The prespecified rule prefers fewer inputs within 0.001 of the lowest mean validation loss, then logistic regression. It selects L1 team + player histories with C=0.1: 22 input columns, 19 nonzero coefficients after the final fit."
    )
    with st.expander("Candidates inside the selection tolerance"):
        st.dataframe(
            close[["name", "n_features", "log_loss"]].round(6),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "The selected model was frozen before the original 2023–24 evaluation. This presentation revision adds explanations and a record-rule comparison; it does not retune the model using that test."
    )
