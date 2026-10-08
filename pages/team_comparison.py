import streamlit as st
import plotly.express as px
import pandas as pd

from src.analytics.football_analytics import (
    load_data,
    compare_teams,
    head_to_head,
    recent_form,
)

st.title("⚔️ Team Comparison")

st.markdown(
    """
    Compare two Premier League teams using historical performance,
    recent form and head-to-head results from ESPN data.
    """
)


@st.cache_data
def get_data():
    return load_data()


df = get_data()

# ---------------------------------------------------------
# Filters
# ---------------------------------------------------------

seasons = ["All Seasons"] + sorted(
    df["season"].dropna().unique().tolist()
)

selected_season = st.selectbox(
    "Select Season",
    seasons
)

if selected_season != "All Seasons":
    comparison_df = df[
        df["season"] == selected_season
    ].copy()
else:
    comparison_df = df.copy()

teams = sorted(
    set(comparison_df["home_team"])
    | set(comparison_df["away_team"])
)

if len(teams) < 2:
    st.warning("Not enough teams available for comparison.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox(
        "Team 1",
        teams,
        index=0
    )

with col2:
    default_team2 = 1 if len(teams) > 1 else 0

    team2 = st.selectbox(
        "Team 2",
        teams,
        index=default_team2
    )


if team1 == team2:
    st.warning("Please select two different teams.")
    st.stop()


# ---------------------------------------------------------
# Overall Comparison
# ---------------------------------------------------------

st.divider()

st.subheader(
    f"📊 {team1} vs {team2}"
)

comparison = compare_teams(
    comparison_df,
    team1,
    team2
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Performance Comparison Chart
# ---------------------------------------------------------

st.subheader("📈 Performance Comparison")

chart_df = comparison[
    comparison["metric"].isin(
        [
            "Wins",
            "Draws",
            "Losses",
            "Goals Scored",
            "Goals Conceded",
            "Goal Difference",
        ]
    )
].copy()

long_df = chart_df.melt(
    id_vars="metric",
    var_name="Team",
    value_name="Value"
)

fig = px.bar(
    long_df,
    x="metric",
    y="Value",
    color="Team",
    barmode="group",
    text="Value",
    title=f"{team1} vs {team2} — Key Performance Metrics"
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Metric",
    yaxis_title="Value"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# Points Per Game
# ---------------------------------------------------------

st.subheader("🏆 Points Per Game")

ppg_df = comparison[
    comparison["metric"] == "Points Per Game"
].melt(
    id_vars="metric",
    var_name="Team",
    value_name="Points Per Game"
)

ppg_df["Points Per Game"] = ppg_df[
    "Points Per Game"
].round(2)

fig_ppg = px.bar(
    ppg_df,
    x="Team",
    y="Points Per Game",
    text="Points Per Game",
    title="Points Per Game Comparison"
)

fig_ppg.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

fig_ppg.update_layout(
    yaxis_title="Points Per Game",
    xaxis_title=""
)

st.plotly_chart(
    fig_ppg,
    use_container_width=True
)


# ---------------------------------------------------------
# Head-to-Head
# ---------------------------------------------------------

st.divider()

st.subheader("🤝 Head-to-Head History")

h2h_df = head_to_head(
    df,
    team1,
    team2
)

if h2h_df.empty:

    st.info(
        f"No previous head-to-head matches found between "
        f"{team1} and {team2}."
    )

else:

    # If a specific season is selected, still show
    # historical H2H across all seasons.
    st.write(
        f"Total historical meetings: **{len(h2h_df)}**"
    )

    h2h_display = h2h_df.copy()

    h2h_display["start_date"] = pd.to_datetime(
        h2h_display["start_date"]
    ).dt.strftime("%d %b %Y")

    st.dataframe(
        h2h_display,
        use_container_width=True,
        hide_index=True
    )

    # H2H outcome summary
    h2h_counts = (
        h2h_df["result_type"]
        .value_counts()
        .reindex(
            ["Home Win", "Draw", "Away Win"],
            fill_value=0
        )
        .reset_index()
    )

    h2h_counts.columns = [
        "Result",
        "Matches"
    ]

    fig_h2h = px.bar(
        h2h_counts,
        x="Result",
        y="Matches",
        text="Matches",
        title=f"Head-to-Head Match Outcomes"
    )

    fig_h2h.update_traces(
        textposition="outside"
    )

    fig_h2h.update_layout(
        xaxis_title="Result",
        yaxis_title="Number of Matches"
    )

    st.plotly_chart(
        fig_h2h,
        use_container_width=True
    )


# ---------------------------------------------------------
# Recent Form Comparison
# ---------------------------------------------------------

st.divider()

st.subheader("🔥 Recent Form")

col1, col2 = st.columns(2)

with col1:

    st.markdown(f"### {team1}")

    form1 = recent_form(
        df,
        team1,
        n=5
    )

    if form1.empty:
        st.info("No recent matches available.")
    else:

        form1_display = form1.copy()

        form1_display["date"] = pd.to_datetime(
            form1_display["date"]
        ).dt.strftime("%d %b %Y")

        st.dataframe(
            form1_display,
            use_container_width=True,
            hide_index=True
        )

        form1_points = form1["points"].sum()

        st.metric(
            "Last 5 Points",
            int(form1_points)
        )


with col2:

    st.markdown(f"### {team2}")

    form2 = recent_form(
        df,
        team2,
        n=5
    )

    if form2.empty:
        st.info("No recent matches available.")
    else:

        form2_display = form2.copy()

        form2_display["date"] = pd.to_datetime(
            form2_display["date"]
        ).dt.strftime("%d %b %Y")

        st.dataframe(
            form2_display,
            use_container_width=True,
            hide_index=True
        )

        form2_points = form2["points"].sum()

        st.metric(
            "Last 5 Points",
            int(form2_points)
        )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

st.divider()

st.subheader("📌 Comparison Summary")

stats1 = comparison[
    comparison["metric"] == "Points Per Game"
][team1].iloc[0]

stats2 = comparison[
    comparison["metric"] == "Points Per Game"
][team2].iloc[0]

if stats1 > stats2:

    st.success(
        f"**{team1}** has the higher historical "
        f"points-per-game record ({stats1:.2f} vs {stats2:.2f})."
    )

elif stats2 > stats1:

    st.success(
        f"**{team2}** has the higher historical "
        f"points-per-game record ({stats2:.2f} vs {stats1:.2f})."
    )

else:

    st.info(
        f"Both teams have the same historical "
        f"points-per-game record of {stats1:.2f}."
    )