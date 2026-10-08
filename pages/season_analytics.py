import streamlit as st
import plotly.express as px

from src.analytics.football_analytics import (
    load_data,
    season_summary,
)


st.title("📅 Season Analytics")

st.markdown(
    """
    Analyse Premier League performance across seasons using the
    historical ESPN dataset.
    """
)


@st.cache_data
def get_data():
    return load_data()


df = get_data()
season_df = season_summary(df)


# =========================================================
# SEASON SELECTOR
# =========================================================

seasons = sorted(
    season_df["season"].dropna().unique()
)

selected_season = st.selectbox(
    "Select Season",
    seasons,
)


selected = season_df[
    season_df["season"] == selected_season
].iloc[0]


# =========================================================
# KPI SECTION
# =========================================================

st.subheader(f"{selected_season} Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Matches",
        f"{int(selected['matches']):,}"
    )

with col2:
    st.metric(
        "Total Goals",
        f"{int(selected['total_goals']):,}"
    )

with col3:
    st.metric(
        "Avg Goals / Match",
        f"{selected['average_goals']:.2f}"
    )

with col4:
    st.metric(
        "Home Win %",
        f"{selected['home_wins'] / selected['matches'] * 100:.1f}%"
    )


st.divider()


# =========================================================
# OUTCOME DISTRIBUTION
# =========================================================

st.subheader("Match Outcome Distribution")

outcome_data = {
    "Outcome": [
        "Home Win",
        "Draw",
        "Away Win",
    ],
    "Matches": [
        int(selected["home_wins"]),
        int(selected["draws"]),
        int(selected["away_wins"]),
    ],
}

fig_outcomes = px.bar(
    outcome_data,
    x="Outcome",
    y="Matches",
    text="Matches",
    title=f"Match Outcomes — {selected_season}",
)

fig_outcomes.update_traces(
    textposition="outside"
)

fig_outcomes.update_layout(
    yaxis_title="Number of Matches",
    xaxis_title="",
)

st.plotly_chart(
    fig_outcomes,
    use_container_width=True,
)


# =========================================================
# GOAL TREND ACROSS SEASONS
# =========================================================

st.subheader("Goals Across Seasons")

fig_goals = px.line(
    season_df,
    x="season",
    y="total_goals",
    markers=True,
    title="Total Goals by Season",
)

fig_goals.update_layout(
    xaxis_title="Season",
    yaxis_title="Total Goals",
)

st.plotly_chart(
    fig_goals,
    use_container_width=True,
)


# =========================================================
# AVERAGE GOALS TREND
# =========================================================

fig_average = px.line(
    season_df,
    x="season",
    y="average_goals",
    markers=True,
    title="Average Goals per Match",
)

fig_average.update_layout(
    xaxis_title="Season",
    yaxis_title="Average Goals",
)

st.plotly_chart(
    fig_average,
    use_container_width=True,
)


# =========================================================
# COMPLETE SEASON TABLE
# =========================================================

st.subheader("Season Statistics")

display_df = season_df.copy()

display_df["average_goals"] = (
    display_df["average_goals"].round(2)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)