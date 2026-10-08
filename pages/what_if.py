import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from xgboost import XGBClassifier

from src.features.football_features import (
    FEATURE_COLUMNS,
    build_prediction_features,
    predict_score_from_features,
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🎛️ What-If Match Analysis")

st.markdown(
    """
    Explore how changes in pre-match team strength and recent form
    can influence the XGBoost match-outcome prediction.
    """
)

st.caption(
    "This is a scenario analysis tool. It does not change the "
    "trained XGBoost model or retrain the model."
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "football_espn_clean.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_match_outcome_model.json"
)


CLASS_NAMES = [
    "Home Win",
    "Draw",
    "Away Win",
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_parquet(
        DATA_PATH
    )

    df["start_date"] = pd.to_datetime(
        df["start_date"],
        utc=True,
    )

    return df


df = load_data()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# ============================================================
# SELECT SEASON
# ============================================================

st.subheader("⚽ Select Match Scenario")

seasons = sorted(
    df["season"]
    .dropna()
    .astype(str)
    .unique()
)

selected_season = st.selectbox(
    "Prediction Season",
    seasons,
    index=len(seasons) - 1,
)


# ============================================================
# FIXTURES
# ============================================================

season_fixtures = df[
    df["season"].astype(str)
    == selected_season
].copy()

season_fixtures = season_fixtures[
    season_fixtures["home_team"].notna()
    & season_fixtures["away_team"].notna()
    & season_fixtures["start_date"].notna()
].copy()

season_fixtures = season_fixtures.sort_values(
    "start_date"
).reset_index(drop=True)


def fixture_label(row):

    return (
        f"{row['start_date'].strftime('%d %b %Y')} | "
        f"{row['start_date'].strftime('%H:%M UTC')} | "
        f"{row['home_team']} vs {row['away_team']}"
    )


fixture_options = [
    fixture_label(row)
    for _, row in season_fixtures.iterrows()
]


if not fixture_options:

    st.warning(
        "No fixtures are available for this season."
    )

    st.stop()


# Prefer a scheduled fixture
scheduled_mask = (
    season_fixtures["event_status"]
    .astype(str)
    .str.lower()
    == "scheduled"
)

scheduled_indices = (
    season_fixtures.index[
        scheduled_mask
    ].tolist()
)

if scheduled_indices:

    default_index = scheduled_indices[0]

else:

    default_index = len(
        fixture_options
    ) - 1


selected_fixture_label = st.selectbox(
    "Exact Fixture",
    fixture_options,
    index=default_index,
)


selected_index = fixture_options.index(
    selected_fixture_label
)

selected_fixture = season_fixtures.iloc[
    selected_index
]


fixture_date = selected_fixture[
    "start_date"
]

home_team = selected_fixture[
    "home_team"
]

away_team = selected_fixture[
    "away_team"
]


# ============================================================
# BASELINE FEATURES
# ============================================================

try:

    baseline_features = build_prediction_features(
        df,
        home_team,
        away_team,
        selected_season,
        prediction_date=fixture_date,
    )

except Exception as e:

    st.error(
        "Unable to build the baseline scenario."
    )

    st.exception(e)

    st.stop()


X_baseline = baseline_features[
    FEATURE_COLUMNS
].copy()


baseline_probabilities = model.predict_proba(
    X_baseline
)[0]


baseline_index = (
    baseline_probabilities.argmax()
)

baseline_result = CLASS_NAMES[
    baseline_index
]


baseline_home_goals, baseline_away_goals = (
    predict_score_from_features(
        baseline_features
    )
)


# ============================================================
# BASELINE RESULT
# ============================================================

st.divider()

st.subheader("📌 Baseline Prediction")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Fixture",
        f"{home_team} vs {away_team}",
    )

with col2:

    st.metric(
        "Date",
        fixture_date.strftime(
            "%d %b %Y"
        ),
    )

with col3:

    st.metric(
        "Baseline Result",
        baseline_result,
    )


st.markdown(
    f"""
    ### Baseline Score

    ## {home_team} {baseline_home_goals}
    – {baseline_away_goals} {away_team}
    """
)


# ============================================================
# BASELINE PROBABILITIES
# ============================================================

baseline_probability_df = pd.DataFrame(
    {
        "Outcome": [
            f"{home_team} Win",
            "Draw",
            f"{away_team} Win",
        ],
        "Probability": [
            baseline_probabilities[0] * 100,
            baseline_probabilities[1] * 100,
            baseline_probabilities[2] * 100,
        ],
    }
)


fig_baseline = px.bar(
    baseline_probability_df,
    x="Outcome",
    y="Probability",
    text="Probability",
    title="Baseline XGBoost Prediction",
)

fig_baseline.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)

fig_baseline.update_layout(
    yaxis_title="Probability (%)",
    xaxis_title="",
    yaxis_range=[
        0,
        max(
            100,
            baseline_probability_df[
                "Probability"
            ].max() + 15,
        ),
    ],
)

st.plotly_chart(
    fig_baseline,
    use_container_width=True,
)


# ============================================================
# WHAT-IF CONTROLS
# ============================================================

st.divider()

st.subheader("🎚️ Modify Match Scenario")

st.write(
    """
    Adjust selected pre-match performance indicators and observe
    how the model prediction changes.
    """
)


# ============================================================
# CURRENT VALUES
# ============================================================

home_ppg_current = float(
    X_baseline.iloc[0][
        "home_points_per_game_before"
    ]
)

away_ppg_current = float(
    X_baseline.iloc[0][
        "away_points_per_game_before"
    ]
)

home_recent_current = float(
    X_baseline.iloc[0][
        "home_recent_points_per_game_5"
    ]
)

away_recent_current = float(
    X_baseline.iloc[0][
        "away_recent_points_per_game_5"
    ]
)

home_gd_current = float(
    X_baseline.iloc[0][
        "home_goal_difference_per_game_before"
    ]
)

away_gd_current = float(
    X_baseline.iloc[0][
        "away_goal_difference_per_game_before"
    ]
)


# ============================================================
# HOME TEAM CONTROLS
# ============================================================

st.markdown(
    f"### 🏠 {home_team}"
)

col1, col2, col3 = st.columns(3)

with col1:

    home_ppg = st.slider(
        "Points Per Game",
        min_value=0.0,
        max_value=3.0,
        value=float(
            min(
                3.0,
                max(
                    0.0,
                    home_ppg_current,
                ),
            )
        ),
        step=0.1,
        key="home_ppg",
    )

with col2:

    home_recent_ppg = st.slider(
        "Recent PPG (Last 5)",
        min_value=0.0,
        max_value=3.0,
        value=float(
            min(
                3.0,
                max(
                    0.0,
                    home_recent_current,
                ),
            )
        ),
        step=0.1,
        key="home_recent_ppg",
    )

with col3:

    home_goal_diff = st.slider(
        "Goal Difference / Game",
        min_value=-3.0,
        max_value=3.0,
        value=float(
            min(
                3.0,
                max(
                    -3.0,
                    home_gd_current,
                ),
            )
        ),
        step=0.1,
        key="home_goal_diff",
    )


# ============================================================
# AWAY TEAM CONTROLS
# ============================================================

st.markdown(
    f"### ✈️ {away_team}"
)

col1, col2, col3 = st.columns(3)

with col1:

    away_ppg = st.slider(
        "Points Per Game",
        min_value=0.0,
        max_value=3.0,
        value=float(
            min(
                3.0,
                max(
                    0.0,
                    away_ppg_current,
                ),
            )
        ),
        step=0.1,
        key="away_ppg",
    )

with col2:

    away_recent_ppg = st.slider(
        "Recent PPG (Last 5)",
        min_value=0.0,
        max_value=3.0,
        value=float(
            min(
                3.0,
                max(
                    0.0,
                    away_recent_current,
                ),
            )
        ),
        step=0.1,
        key="away_recent_ppg",
    )

with col3:

    away_goal_diff = st.slider(
        "Goal Difference / Game",
        min_value=-3.0,
        max_value=3.0,
        value=float(
            min(
                3.0,
                max(
                    -3.0,
                    away_gd_current,
                ),
            )
        ),
        step=0.1,
        key="away_goal_diff",
    )


# ============================================================
# RUN WHAT-IF
# ============================================================

st.divider()

if st.button(
    "🎛️ Run What-If Scenario",
    type="primary",
    use_container_width=True,
):

    scenario_features = X_baseline.copy()

    # --------------------------------------------------------
    # MODIFY SELECTED FEATURES
    # --------------------------------------------------------

    scenario_features.loc[
        0,
        "home_points_per_game_before"
    ] = home_ppg

    scenario_features.loc[
        0,
        "away_points_per_game_before"
    ] = away_ppg

    scenario_features.loc[
        0,
        "home_recent_points_per_game_5"
    ] = home_recent_ppg

    scenario_features.loc[
        0,
        "away_recent_points_per_game_5"
    ] = away_recent_ppg

    scenario_features.loc[
        0,
        "home_goal_difference_per_game_before"
    ] = home_goal_diff

    scenario_features.loc[
        0,
        "away_goal_difference_per_game_before"
    ] = away_goal_diff


    # --------------------------------------------------------
    # RECALCULATE DIFFERENCE FEATURES
    # --------------------------------------------------------

    scenario_features.loc[
        0,
        "points_per_game_diff"
    ] = (
        home_ppg
        - away_ppg
    )

    scenario_features.loc[
        0,
        "recent_points_per_game_5_diff"
    ] = (
        home_recent_ppg
        - away_recent_ppg
    )

    scenario_features.loc[
        0,
        "goal_difference_per_game_diff"
    ] = (
        home_goal_diff
        - away_goal_diff
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    scenario_probabilities = model.predict_proba(
        scenario_features
    )[0]

    scenario_index = (
        scenario_probabilities.argmax()
    )

    scenario_result = CLASS_NAMES[
        scenario_index
    ]


    # --------------------------------------------------------
    # SCORE PREDICTION
    # --------------------------------------------------------

    scenario_score_features = (
        baseline_features.copy()
    )

    scenario_score_features.loc[
        0,
        "home_points_per_game_before"
    ] = home_ppg

    scenario_score_features.loc[
        0,
        "away_points_per_game_before"
    ] = away_ppg

    scenario_score_features.loc[
        0,
        "home_recent_points_per_game_5"
    ] = home_recent_ppg

    scenario_score_features.loc[
        0,
        "away_recent_points_per_game_5"
    ] = away_recent_ppg

    scenario_score_features.loc[
        0,
        "home_goal_difference_per_game_before"
    ] = home_goal_diff

    scenario_score_features.loc[
        0,
        "away_goal_difference_per_game_before"
    ] = away_goal_diff


    scenario_home_goals, scenario_away_goals = (
        predict_score_from_features(
            scenario_score_features
        )
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("📊 What-If Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Baseline Prediction",
            baseline_result,
        )

    with col2:

        st.metric(
            "Scenario Prediction",
            scenario_result,
        )

    with col3:

        result_changed = (
            baseline_result
            != scenario_result
        )

        st.metric(
            "Prediction Changed?",
            "YES" if result_changed else "NO",
        )


    # ========================================================
    # SCORE COMPARISON
    # ========================================================

    st.markdown(
        f"""
        ### ⚽ Predicted Score

        **Baseline**

        {home_team} **{baseline_home_goals}–{baseline_away_goals}**
        {away_team}

        **What-If Scenario**

        {home_team} **{scenario_home_goals}–{scenario_away_goals}**
        {away_team}
        """
    )


    # ========================================================
    # PROBABILITY COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        {
            "Outcome": [
                f"{home_team} Win",
                "Draw",
                f"{away_team} Win",
            ],
            "Baseline": [
                baseline_probabilities[0] * 100,
                baseline_probabilities[1] * 100,
                baseline_probabilities[2] * 100,
            ],
            "What-If Scenario": [
                scenario_probabilities[0] * 100,
                scenario_probabilities[1] * 100,
                scenario_probabilities[2] * 100,
            ],
        }
    )


    comparison_long = comparison_df.melt(
        id_vars="Outcome",
        var_name="Scenario",
        value_name="Probability",
    )


    fig_comparison = px.bar(
        comparison_long,
        x="Outcome",
        y="Probability",
        color="Scenario",
        barmode="group",
        text="Probability",
        title="Baseline vs What-If Probabilities",
    )


    fig_comparison.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )


    fig_comparison.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="",
        yaxis_range=[
            0,
            max(
                100,
                comparison_long[
                    "Probability"
                ].max() + 15,
            ),
        ],
    )


    st.plotly_chart(
        fig_comparison,
        use_container_width=True,
    )


    # ========================================================
    # PROBABILITY CHANGE
    # ========================================================

    change_df = pd.DataFrame(
        {
            "Outcome": [
                f"{home_team} Win",
                "Draw",
                f"{away_team} Win",
            ],
            "Change": [
                (
                    scenario_probabilities[0]
                    - baseline_probabilities[0]
                ) * 100,

                (
                    scenario_probabilities[1]
                    - baseline_probabilities[1]
                ) * 100,

                (
                    scenario_probabilities[2]
                    - baseline_probabilities[2]
                ) * 100,
            ],
        }
    )


    change_df["Change"] = (
        change_df["Change"].round(2)
    )


    st.subheader(
        "📈 Change in Prediction Probability"
    )


    st.dataframe(
        change_df,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader(
        "🧠 Scenario Interpretation"
    )


    largest_change_index = (
        abs(
            change_df["Change"]
        ).idxmax()
    )


    largest_change_outcome = (
        change_df.loc[
            largest_change_index,
            "Outcome",
        ]
    )


    largest_change_value = (
        change_df.loc[
            largest_change_index,
            "Change",
        ]
    )


    if result_changed:

        st.success(
            f"""
            The hypothetical changes alter the model's predicted
            outcome from **{baseline_result}** to
            **{scenario_result}**.
            """
        )

    else:

        st.info(
            f"""
            The hypothetical changes do not change the model's
            most likely outcome. The prediction remains
            **{scenario_result}**.
            """
        )


    st.write(
        f"""
        The largest probability movement was for
        **{largest_change_outcome}**, changing by
        **{largest_change_value:+.2f} percentage points**.
        """
    )


    # ========================================================
    # SCENARIO VALUES
    # ========================================================

    with st.expander(
        "🔍 View Scenario Feature Values"
    ):

        scenario_display = pd.DataFrame(
            {
                "Feature": [
                    "Home Points Per Game",
                    "Away Points Per Game",
                    "Home Recent PPG",
                    "Away Recent PPG",
                    "Home Goal Difference / Game",
                    "Away Goal Difference / Game",
                    "Home PPG Difference",
                    "Recent PPG Difference",
                    "Goal Difference / Game Difference",
                ],
                "Baseline": [
                    home_ppg_current,
                    away_ppg_current,
                    home_recent_current,
                    away_recent_current,
                    home_gd_current,
                    away_gd_current,
                    X_baseline.iloc[0][
                        "points_per_game_diff"
                    ],
                    X_baseline.iloc[0][
                        "recent_points_per_game_5_diff"
                    ],
                    X_baseline.iloc[0][
                        "goal_difference_per_game_diff"
                    ],
                ],
                "Scenario": [
                    home_ppg,
                    away_ppg,
                    home_recent_ppg,
                    away_recent_ppg,
                    home_goal_diff,
                    away_goal_diff,
                    home_ppg - away_ppg,
                    home_recent_ppg - away_recent_ppg,
                    home_goal_diff - away_goal_diff,
                ],
            }
        )

        st.dataframe(
            scenario_display.round(3),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# METHODOLOGY
# ============================================================

st.divider()

with st.expander(
    "ℹ️ How does What-If Analysis work?"
):

    st.markdown(
        """
        The What-If Analysis starts with the actual pre-match
        feature vector for the selected fixture.

        The user can then modify selected team-performance
        indicators such as:

        - Points per game
        - Recent points per game
        - Goal difference per game

        The modified feature vector is passed through the same
        trained XGBoost model.

        This allows the user to study how hypothetical changes
        in team strength or form affect the predicted match
        outcome and its probability distribution.

        The trained model itself is not changed or retrained.
        """
    )