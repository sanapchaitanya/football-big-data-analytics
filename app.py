import streamlit as st


st.set_page_config(
    page_title="Football Big Data Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


overview = st.Page(
    "pages/overview.py",
    title="Overview",
    icon=":material/dashboard:",
    default=True,
)

season = st.Page(
    "pages/season_analytics.py",
    title="Season Analytics",
    icon=":material/calendar_month:",
)

team = st.Page(
    "pages/team_analytics.py",
    title="Team Analytics",
    icon=":material/groups:",
)

comparison = st.Page(
    "pages/team_comparison.py",
    title="Team Comparison",
    icon=":material/compare_arrows:",
)

prediction = st.Page(
    "pages/match_prediction.py",
    title="Match Prediction",
    icon=":material/psychology:",
)

model = st.Page(
    "pages/model_performance.py",
    title="Model Performance",
    icon=":material/analytics:",
)

hadoop_analytics = st.Page(
    "pages/8_Hadoop_Team_Analytics.py",
    title="Hadoop Team Analytics",
    icon=":material/bar_chart:",
)


pg = st.navigation(
    {
        "Football Analytics": [
            overview,
            season,
            team,
            comparison,
        ],
        "Machine Learning": [
            prediction,
            model,
            hadoop_analytics,
        ],
    }
)


pg.run()