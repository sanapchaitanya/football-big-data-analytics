import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from xgboost import XGBClassifier

from src.features.football_features import (
    FEATURE_COLUMNS,
    build_prediction_features,
    predict_score_from_features,
)


st.title("🔮 Match Prediction")

st.markdown(
    """
    Predict the outcome of an exact Premier League fixture using
    the trained XGBoost match-outcome prediction model.
    """
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
# LOAD ESPN DATA
# ============================================================

@st.cache_data
def load_fixture_data():

    df = pd.read_parquet(DATA_PATH)

    df["start_date"] = pd.to_datetime(
        df["start_date"],
        utc=True,
    )

    return df


df = load_fixture_data()


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
# HEADER
# ============================================================

st.subheader("⚽ Select Premier League Fixture")

st.write(
    "Select an exact fixture from the ESPN match schedule. "
    "The prediction uses only information available before kickoff."
)


# ============================================================
# SEASON SELECTION
# ============================================================

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
# FIXTURE DATA
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


# ============================================================
# FIXTURE LABEL
# ============================================================

def fixture_label(row):

    date_string = row["start_date"].strftime(
        "%d %b %Y"
    )

    time_string = row["start_date"].strftime(
        "%H:%M UTC"
    )

    return (
        f"{date_string} | "
        f"{time_string} | "
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


# ============================================================
# DEFAULT TO UPCOMING/SCHEDULED FIXTURE
# ============================================================

scheduled_mask = (
    season_fixtures["event_status"]
    .astype(str)
    .str.lower()
    == "scheduled"
)

scheduled_indices = (
    season_fixtures.index[scheduled_mask]
    .tolist()
)

if scheduled_indices:

    default_index = (
        fixture_options.index(
            fixture_label(
                season_fixtures.loc[
                    scheduled_indices[0]
                ]
            )
        )
    )

else:

    default_index = len(
        fixture_options
    ) - 1


selected_fixture_label = st.selectbox(
    "Exact Fixture",
    fixture_options,
    index=default_index,
)


# ============================================================
# GET SELECTED FIXTURE
# ============================================================

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

event_status = selected_fixture[
    "event_status"
]


# ============================================================
# FIXTURE CARD
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "📅 Date",
        fixture_date.strftime(
            "%d %B %Y"
        ),
    )

with col2:

    st.metric(
        "🕐 Kickoff",
        fixture_date.strftime(
            "%H:%M UTC"
        ),
    )

with col3:

    st.metric(
        "📌 Status",
        str(event_status),
    )


st.markdown(
    f"""
    ## 🏟️ {home_team} vs {away_team}

    **{fixture_date.strftime("%A, %d %B %Y")}**
    """
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔮 Predict This Exact Match",
    type="primary",
    use_container_width=True,
):

    try:

        # ----------------------------------------------------
        # BUILD LEAKAGE-SAFE FEATURES
        # ----------------------------------------------------

        feature_row = build_prediction_features(
            df,
            home_team,
            away_team,
            selected_season,
            prediction_date=fixture_date,
        )

        X_prediction = feature_row[
            FEATURE_COLUMNS
        ]


        # ----------------------------------------------------
        # XGBOOST PREDICTION
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X_prediction
        )[0]

        predicted_index = probabilities.argmax()

        predicted_result = CLASS_NAMES[
            predicted_index
        ]

        home_probability = probabilities[0]
        draw_probability = probabilities[1]
        away_probability = probabilities[2]


        # ----------------------------------------------------
        # SCORE PREDICTION
        # ----------------------------------------------------

        predicted_home_goals, predicted_away_goals = (
            predict_score_from_features(
                feature_row
            )
        )


        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        if predicted_result == "Home Win":

            result_text = (
                f"{home_team} {predicted_home_goals}"
                f"–{predicted_away_goals} {away_team}"
            )

        elif predicted_result == "Away Win":

            result_text = (
                f"{home_team} {predicted_home_goals}"
                f"–{predicted_away_goals} {away_team}"
            )

        else:

            result_text = (
                f"{home_team} {predicted_home_goals}"
                f"–{predicted_away_goals} {away_team}"
            )


        # ====================================================
        # MAIN PREDICTION CARD
        # ====================================================

        st.divider()

        st.subheader("🏆 Model Prediction")

        st.success(
            f"### Predicted Result: {predicted_result}"
        )


        # ====================================================
        # SCORE
        # ====================================================

        st.markdown(
            f"""
            <div style="
                padding: 25px;
                border-radius: 15px;
                background-color: #f5f7fb;
                text-align: center;
                margin: 20px 0;
            ">

            <div style="
                font-size: 20px;
                font-weight: 600;
            ">
                Predicted Score
            </div>

            <div style="
                font-size: 42px;
                font-weight: 700;
                margin-top: 10px;
            ">
                {home_team}
                &nbsp;
                {predicted_home_goals}
                –
                {predicted_away_goals}
                &nbsp;
                {away_team}
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Match Outcome Probabilities"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                f"🏠 {home_team}",
                f"{home_probability * 100:.1f}%",
            )

        with col2:

            st.metric(
                "🤝 Draw",
                f"{draw_probability * 100:.1f}%",
            )

        with col3:

            st.metric(
                f"✈️ {away_team}",
                f"{away_probability * 100:.1f}%",
            )


        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        probability_df = pd.DataFrame(
            {
                "Outcome": [
                    f"{home_team} Win",
                    "Draw",
                    f"{away_team} Win",
                ],
                "Probability": [
                    home_probability * 100,
                    draw_probability * 100,
                    away_probability * 100,
                ],
            }
        )


        fig = px.bar(
            probability_df,
            x="Outcome",
            y="Probability",
            text="Probability",
            title=(
                f"XGBoost Prediction — "
                f"{home_team} vs {away_team}"
            ),
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="",
            yaxis_range=[
                0,
                max(
                    100,
                    probability_df[
                        "Probability"
                    ].max() + 15,
                ),
            ],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ====================================================
        # MATCH SUMMARY
        # ====================================================

        st.subheader(
            "📝 Prediction Summary"
        )

        st.markdown(
            f"""
            **Fixture:** {home_team} vs {away_team}

            **Date:** {fixture_date.strftime("%d %B %Y")}

            **Kickoff:** {fixture_date.strftime("%H:%M UTC")}

            **Predicted Result:** {predicted_result}

            **Predicted Score:**  
            **{home_team} {predicted_home_goals}–{predicted_away_goals} {away_team}**

            **Home Win Probability:** {home_probability * 100:.1f}%

            **Draw Probability:** {draw_probability * 100:.1f}%

            **Away Win Probability:** {away_probability * 100:.1f}%
            """
        )


        # ====================================================
        # SCORE METHODOLOGY
        # ====================================================

        with st.expander(
            "ℹ️ How is the predicted score calculated?"
        ):

            st.write(
                """
                The match outcome probabilities are generated
                by the trained XGBoost classifier.

                The predicted scoreline is a statistical
                estimate derived from the teams' pre-match
                attacking and defensive goal rates.

                A Poisson scoreline calculation evaluates
                possible score combinations and selects the
                most probable score.

                No second machine-learning model is used.
                """
            )


        # ====================================================
        # FEATURE SNAPSHOT
        # ====================================================

        with st.expander(
            "🔍 View Pre-Match Feature Snapshot"
        ):

            feature_snapshot = (
                X_prediction
                .T
                .reset_index()
            )

            feature_snapshot.columns = [
                "Feature",
                "Value",
            ]

            feature_snapshot["Value"] = (
                feature_snapshot["Value"]
                .round(4)
            )

            st.dataframe(
                feature_snapshot,
                use_container_width=True,
                hide_index=True,
            )


    except Exception as e:

        st.error(
            "Unable to generate the prediction."
        )

        st.exception(e)