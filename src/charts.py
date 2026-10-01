"""Consistent chart labels and a restrained basketball palette."""

import plotly.express as px

INK = "#24343b"
CLAY = "#b55832"
TEAL = "#427b83"
LABELS = {
    "net_rating": "Net rating",
    "ortg": "Offensive rating",
    "drtg": "Defensive rating",
    "win_pct": "Win percentage",
    "efg": "Effective field goal percentage",
    "opp_efg": "Opponent shooting efficiency",
    "tov_rate": "Turnover rate",
    "oreb_rate": "Offensive rebounding",
    "dreb_rate": "Defensive rebounding",
    "ft_rate": "Free-throw rate",
    "rest_days": "Days of rest",
    "back_to_back": "Back-to-back games",
    "games_last7": "Games in the past week",
    "top5_share": "Top-five minute share",
    "top7_share": "Top-seven minute share",
    "rotation_size": "Average count playing 10+ minutes",
    "minute_continuity": "Minute-share continuity",
    "starter_continuity": "Starter continuity",
    "rotation_volatility": "Rotation volatility",
    "scoring_concentration": "Scoring concentration",
    "usage_concentration": "Shooting-role concentration",
    "weighted_ppa": "Role-weighted scoring efficiency",
    "weighted_absence": "Role-weighted absence history",
    "weighted_participation": "Role-weighted participation",
    "point_diff": "Point differential",
    "last5_net_rating": "Net rating · last 5",
    "last10_net_rating": "Net rating · last 10",
    "last20_net_rating": "Net rating · last 20",
    "last5_efg": "Shooting · last 5",
    "last10_efg": "Shooting · last 10",
    "last20_efg": "Shooting · last 20",
}


def feature_label(name):
    return LABELS.get(name.removesuffix("_diff"), name.replace("_", " "))


def style(fig, height=390):
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Arial, sans-serif", color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=40, b=20),
        colorway=[CLAY, TEAL, "#929b83"],
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e7e4dc", zerolinecolor="#c9c5b9")
    return fig


def offense_defense(history, season):
    last = (
        history[history.season.eq(season)]
        .sort_values("gamedate")
        .groupby("off_team")
        .tail(1)
    )
    fig = px.scatter(
        last,
        x="drtg",
        y="ortg",
        text="off_team",
        color="net_rating",
        color_continuous_scale=[TEAL, "#ded8c9", CLAY],
        hover_name="off_team_name",
        labels=LABELS,
    )
    fig.update_traces(marker_size=13, textposition="top center")
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(coloraxis_showscale=False)
    return style(fig, 440)
