import streamlit as st
import plotly.express as px
import pandas as pd

from src.analytics.football_analytics import (
    load_data,
    team_summary,
    recent_form,
    home_away_summary,
)


st.title("⚽ Team Analytics")

st.markdown(
    """
    Explore historical performance, recent form, goals and
    home/away performance for any Premier League team.
    """
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def get_data():
    return load_data()


df = get_data()


# =========================================================
# TEAM + SEASON FILTERS
# =========================================================

teams = sorted(
    set(df["home_team"])
    | set(df["away_team"])
)

col1, col2 = st.columns(2)

with col1:
    selected_team = st.selectbox(
        "Select Team",
        teams
    )

with col2:
    seasons = ["All Seasons"] + sorted(
        df["season"].dropna().unique().tolist()
    )

    selected_season = st.selectbox(
        "Select Season",
        seasons
    )


# =========================================================
# FILTER DATA
# =========================================================

if selected_season != "All Seasons":

    team_df = df[
        df["season"] == selected_season
    ].copy()

else:

    team_df = df.copy()


# =========================================================
# TEAM SUMMARY
# =========================================================

stats = team_summary(
    team_df,
    selected_team
)


st.subheader(
    f"{selected_team} — {selected_season}"
)


# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Matches",
        stats["matches"]
    )

with col2:
    st.metric(
        "Wins",
        stats["wins"]
    )

with col3:
    st.metric(
        "Draws",
        stats["draws"]
    )

with col4:
    st.metric(
        "Losses",
        stats["losses"]
    )

with col5:
    st.metric(
        "Points",
        stats["points"]
    )


st.divider()


# =========================================================
# PERFORMANCE METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Points / Game",
        f"{stats['points_per_game']:.2f}"
    )

with col2:
    st.metric(
        "Goals Scored",
        stats["goals_scored"]
    )

with col3:
    st.metric(
        "Goals Conceded",
        stats["goals_conceded"]
    )

with col4:
    st.metric(
        "Goal Difference",
        stats["goal_difference"]
    )


st.divider()


# =========================================================
# HOME VS AWAY
# =========================================================

st.subheader("🏠 Home vs Away Performance")

venue_df = home_away_summary(
    team_df,
    selected_team
)

col1, col2 = st.columns(2)

with col1:

    fig_points = px.bar(
        venue_df,
        x="venue",
        y="points_per_game",
        text="points_per_game",
        title="Points per Game"
    )

    fig_points.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig_points.update_layout(
        xaxis_title="",
        yaxis_title="Points per Game"
    )

    st.plotly_chart(
        fig_points,
        use_container_width=True
    )


with col2:

    goals_venue = venue_df.melt(
        id_vars="venue",
        value_vars=[
            "goals_scored",
            "goals_conceded"
        ],
        var_name="Metric",
        value_name="Goals"
    )

    fig_goals = px.bar(
        goals_venue,
        x="venue",
        y="Goals",
        color="Metric",
        barmode="group",
        title="Goals: Scored vs Conceded"
    )

    fig_goals.update_layout(
        xaxis_title="",
        yaxis_title="Goals"
    )

    st.plotly_chart(
        fig_goals,
        use_container_width=True
    )


st.dataframe(
    venue_df,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# RECENT FORM
# =========================================================

st.subheader("📈 Recent Form")

# For "All Seasons", use the complete historical data.
# For a selected season, use that season's matches.

form_df = recent_form(
    team_df,
    selected_team,
    n=5
)


if form_df.empty:

    st.info(
        "No completed matches are available for this team "
        "in the selected period."
    )

else:

    # Show newest match first
    display_form = form_df.copy()

    display_form["date"] = (
        pd.to_datetime(
            display_form["date"]
        ).dt.strftime("%d %b %Y")
    )

    st.dataframe(
        display_form,
        use_container_width=True,
        hide_index=True
    )

    # Form sequence
    form_string = "  ".join(
        form_df["result"].tolist()
    )

    st.markdown(
        f"### Last {len(form_df)} Matches"
    )

    st.code(
        form_string,
        language=None
    )


    # Recent points
    recent_points = (
        form_df["points"].sum()
    )

    recent_ppg = (
        recent_points / len(form_df)
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Recent Points",
            int(recent_points)
        )

    with col2:
        st.metric(
            "Recent PPG",
            f"{recent_ppg:.2f}"
        )

    with col3:
        st.metric(
            "Recent Goals",
            int(form_df["goals_for"].sum())
        )


st.divider()


# =========================================================
# GOALS
# =========================================================

st.subheader("⚽ Goals Analysis")

goal_data = pd.DataFrame({
    "Metric": [
        "Goals Scored",
        "Goals Conceded"
    ],
    "Goals": [
        stats["goals_scored"],
        stats["goals_conceded"]
    ]
})


fig_goal_total = px.bar(
    goal_data,
    x="Metric",
    y="Goals",
    text="Goals",
    title=f"{selected_team} Goals"
)

fig_goal_total.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_goal_total,
    use_container_width=True
)


# =========================================================
# INTERPRETATION
# =========================================================

st.subheader("📌 Performance Summary")

goal_difference = stats["goal_difference"]

if stats["points_per_game"] >= 2.0:

    performance_text = (
        f"**{selected_team}** has a strong points-per-game "
        f"record of **{stats['points_per_game']:.2f}**."
    )

elif stats["points_per_game"] >= 1.0:

    performance_text = (
        f"**{selected_team}** has a moderate points-per-game "
        f"record of **{stats['points_per_game']:.2f}**."
    )

else:

    performance_text = (
        f"**{selected_team}** has a low points-per-game "
        f"record of **{stats['points_per_game']:.2f}**."
    )


st.write(performance_text)

st.write(
    f"The team has scored **{stats['goals_scored']} goals** "
    f"and conceded **{stats['goals_conceded']} goals**, "
    f"giving a goal difference of **{goal_difference:+d}**."
)