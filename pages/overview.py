import streamlit as st
import plotly.express as px

from src.analytics.football_analytics import (
    load_data,
    league_overview,
    season_summary,
    league_table,
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("⚽ Football Big Data Analytics")

st.markdown(
    """
    ### English Premier League Analytics & Match Outcome Prediction

    An end-to-end football analytics platform built using historical
    ESPN match data, statistical analysis, feature engineering and
    an XGBoost-based match outcome prediction system.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_data():

    return load_data()


df = get_data()

overview = league_overview(df)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.divider()

st.subheader("📊 Dataset Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Matches",
        f"{overview['matches']:,}",
    )


with col2:

    st.metric(
        "Total Goals",
        f"{overview['goals']:,}",
    )


with col3:

    st.metric(
        "Average Goals / Match",
        f"{overview['average_goals_per_match']:.2f}",
    )


with col4:

    st.metric(
        "Seasons",
        df["season"].nunique(),
    )


# ============================================================
# OUTCOME SUMMARY
# ============================================================

st.subheader("🏆 Match Outcome Distribution")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🏠 Home Wins",
        f"{overview['home_wins']:,}",
    )


with col2:

    st.metric(
        "🤝 Draws",
        f"{overview['draws']:,}",
    )


with col3:

    st.metric(
        "✈️ Away Wins",
        f"{overview['away_wins']:,}",
    )


# ============================================================
# OUTCOME CHART
# ============================================================

outcome_df = {
    "Outcome": [
        "Home Win",
        "Draw",
        "Away Win",
    ],
    "Matches": [
        overview["home_wins"],
        overview["draws"],
        overview["away_wins"],
    ],
}

import pandas as pd

outcome_df = pd.DataFrame(
    outcome_df
)


fig_outcomes = px.pie(
    outcome_df,
    names="Outcome",
    values="Matches",
    title="Overall Match Outcomes",
    hole=0.45,
)


st.plotly_chart(
    fig_outcomes,
    use_container_width=True,
)


# ============================================================
# SEASON TREND
# ============================================================

st.divider()

st.subheader("📅 Season Performance")


season_df = season_summary(df)


fig_season = px.line(
    season_df,
    x="season",
    y="average_goals",
    markers=True,
    title="Average Goals per Match by Season",
)


fig_season.update_layout(
    xaxis_title="Season",
    yaxis_title="Average Goals",
)


st.plotly_chart(
    fig_season,
    use_container_width=True,
)


# ============================================================
# SEASON MATCH OUTCOMES
# ============================================================

season_outcomes = season_df[
    [
        "season",
        "home_wins",
        "draws",
        "away_wins",
    ]
].copy()


season_outcomes = season_outcomes.melt(
    id_vars="season",
    var_name="Outcome",
    value_name="Matches",
)


season_outcomes["Outcome"] = (
    season_outcomes["Outcome"]
    .replace(
        {
            "home_wins": "Home Wins",
            "draws": "Draws",
            "away_wins": "Away Wins",
        }
    )
)


fig_season_outcomes = px.bar(
    season_outcomes,
    x="season",
    y="Matches",
    color="Outcome",
    barmode="group",
    title="Match Outcomes Across Seasons",
)


fig_season_outcomes.update_layout(
    xaxis_title="Season",
    yaxis_title="Matches",
)


st.plotly_chart(
    fig_season_outcomes,
    use_container_width=True,
)


# ============================================================
# CURRENT / LATEST SEASON TABLE
# ============================================================

st.divider()

st.subheader("🏆 Latest Completed Season")


completed_seasons = sorted(
    df["season"]
    .dropna()
    .astype(str)
    .unique()
)


latest_season = completed_seasons[-1]


latest_table = league_table(
    df,
    season=latest_season,
)


if not latest_table.empty:

    st.dataframe(
        latest_table.head(10),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PROJECT ARCHITECTURE
# ============================================================

st.divider()

st.subheader("🏗️ System Architecture")


st.markdown(
    """
    **ESPN Football Data**

    ↓

    **Data Cleaning & Validation**

    ↓

    **Processed Football Dataset**

    ↓

    **Historical Feature Engineering**

    ↓

    ┌───────────────────────────┬───────────────────────────┐

    **Football Analytics**       **XGBoost Prediction**

    ↓                            ↓

    Season / Team / H2H          Match Probabilities

    Analytics                    ↓

    ↓                            Predicted Score

    **Interactive Dashboard**

    ↓

    **Insights & What-If Analysis**
    """
)


# ============================================================
# PROJECT MODULES
# ============================================================

st.subheader("🚀 Platform Modules")


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        ### 📈 Football Analytics

        - League overview
        - Season analysis
        - Team statistics
        - Home/away performance
        - Head-to-head analysis
        - Recent form
        """
    )


with col2:

    st.markdown(
        """
        ### 🤖 Machine Learning

        - XGBoost classification
        - Home / Draw / Away prediction
        - Prediction probabilities
        - Scoreline estimation
        - Model evaluation
        - Feature importance
        """
    )


with col3:

    st.markdown(
        """
        ### 🎛️ Decision Support

        - Exact fixture prediction
        - Pre-match feature analysis
        - What-if scenarios
        - Probability comparison
        - Scenario interpretation
        """
    )


# ============================================================
# DATA SCIENCE PIPELINE
# ============================================================

st.divider()

st.subheader("🔬 Data Science Pipeline")


pipeline_cols = st.columns(6)


pipeline_steps = [
    ("1️⃣", "ESPN Data"),
    ("2️⃣", "Cleaning"),
    ("3️⃣", "EDA"),
    ("4️⃣", "Features"),
    ("5️⃣", "XGBoost"),
    ("6️⃣", "Dashboard"),
]


for column, (number, label) in zip(
    pipeline_cols,
    pipeline_steps,
):

    with column:

        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding:12px;
                border-radius:10px;
                background-color:#f5f7fb;
            ">

            <div style="font-size:25px;">
                {number}
            </div>

            <div style="
                font-weight:600;
                margin-top:5px;
            ">
                {label}
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Data source: ESPN | Machine Learning: XGBoost | "
    "Dashboard: Streamlit | Visualization: Plotly"
)