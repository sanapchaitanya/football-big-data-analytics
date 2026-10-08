import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.title("📊 XGBoost Model Performance")

st.markdown(
    """
    Evaluate the performance of the XGBoost football match-outcome
    prediction model using chronological test-set results.
    """
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

METRICS_PATH = (
    REPORT_DIR
    / "xgboost_metrics.json"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "xgboost_test_predictions.csv"
)

IMPORTANCE_PATH = (
    REPORT_DIR
    / "xgboost_feature_importance.csv"
)


# ============================================================
# LOAD REPORT
# ============================================================

@st.cache_data
def load_metrics():

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


@st.cache_data
def load_predictions():

    return pd.read_csv(
        PREDICTIONS_PATH
    )


@st.cache_data
def load_importance():

    return pd.read_csv(
        IMPORTANCE_PATH
    )


metrics_report = load_metrics()
predictions_df = load_predictions()
importance_df = load_importance()


# ============================================================
# EXTRACT METRICS
# ============================================================

test_metrics = metrics_report[
    "test_metrics"
]

classification_report = test_metrics[
    "classification_report"
]

confusion_matrix = test_metrics[
    "confusion_matrix"
]

class_names = [
    "Home Win",
    "Draw",
    "Away Win",
]


# ============================================================
# DATASET SUMMARY
# ============================================================

st.subheader("📚 Evaluation Dataset")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Training Matches",
        f"{metrics_report['train_rows']:,}",
    )

with col2:

    st.metric(
        "Validation Matches",
        f"{metrics_report['validation_rows']:,}",
    )

with col3:

    st.metric(
        "Test Matches",
        f"{metrics_report['test_rows']:,}",
    )


st.caption(
    "The model uses chronological train, validation and test splits "
    "to preserve the temporal nature of football prediction."
)


# ============================================================
# MAIN METRICS
# ============================================================

st.divider()

st.subheader("🎯 Overall Model Performance")

col1, col2, col3, col4, col5, col6 = st.columns(6)


with col1:

    st.metric(
        "Accuracy",
        f"{test_metrics['accuracy'] * 100:.1f}%",
    )


with col2:

    st.metric(
        "Precision",
        f"{test_metrics['precision_macro'] * 100:.1f}%",
    )


with col3:

    st.metric(
        "Recall",
        f"{test_metrics['recall_macro'] * 100:.1f}%",
    )


with col4:

    st.metric(
        "F1 Score",
        f"{test_metrics['f1_macro'] * 100:.1f}%",
    )


with col5:

    st.metric(
        "ROC-AUC",
        (
            f"{test_metrics['roc_auc_ovr_macro']:.3f}"
            if test_metrics["roc_auc_ovr_macro"] is not None
            else "N/A"
        ),
    )


with col6:

    st.metric(
        "Log Loss",
        f"{test_metrics['log_loss']:.3f}",
    )


# ============================================================
# METRIC EXPLANATION
# ============================================================

with st.expander("ℹ️ What do these metrics mean?"):

    st.markdown(
        """
        **Accuracy**  
        Percentage of matches where the predicted outcome exactly
        matches the actual result.

        **Precision**  
        Measures how often predictions for each outcome are correct.

        **Recall**  
        Measures how effectively the model identifies matches belonging
        to each outcome class.

        **F1 Score**  
        Harmonic mean of precision and recall.

        **ROC-AUC**  
        Measures the model's ability to distinguish between the
        three possible match outcomes.

        **Log Loss**  
        Evaluates the quality of the predicted probabilities.
        Lower values indicate better probabilistic predictions.
        """
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.divider()

st.subheader("🔲 Confusion Matrix")

st.markdown(
    """
    The confusion matrix compares the actual match outcome with
    the outcome predicted by XGBoost.
    """
)


confusion_df = pd.DataFrame(
    confusion_matrix,
    index=class_names,
    columns=class_names,
)


fig_cm = px.imshow(
    confusion_df,
    text_auto=True,
    labels={
        "x": "Predicted Outcome",
        "y": "Actual Outcome",
        "color": "Number of Matches",
    },
    x=class_names,
    y=class_names,
    title="Actual vs Predicted Match Outcomes",
)


fig_cm.update_layout(
    xaxis_title="Predicted Outcome",
    yaxis_title="Actual Outcome",
)


st.plotly_chart(
    fig_cm,
    use_container_width=True,
)


# ============================================================
# CLASS-WISE PERFORMANCE
# ============================================================

st.subheader("📋 Class-wise Performance")


class_rows = []

for class_name in class_names:

    class_metrics = classification_report[
        class_name
    ]

    class_rows.append(
        {
            "Outcome": class_name,
            "Precision": class_metrics["precision"],
            "Recall": class_metrics["recall"],
            "F1 Score": class_metrics["f1-score"],
            "Support": int(
                class_metrics["support"]
            ),
        }
    )


class_df = pd.DataFrame(
    class_rows
)


display_class_df = class_df.copy()

display_class_df[
    [
        "Precision",
        "Recall",
        "F1 Score",
    ]
] = (
    display_class_df[
        [
            "Precision",
            "Recall",
            "F1 Score",
        ]
    ]
    * 100
)


display_class_df[
    [
        "Precision",
        "Recall",
        "F1 Score",
    ]
] = display_class_df[
    [
        "Precision",
        "Recall",
        "F1 Score",
    ]
].round(1)


st.dataframe(
    display_class_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CLASS PERFORMANCE CHART
# ============================================================

class_chart_df = class_df.melt(
    id_vars="Outcome",
    value_vars=[
        "Precision",
        "Recall",
        "F1 Score",
    ],
    var_name="Metric",
    value_name="Score",
)


class_chart_df["Score"] *= 100


fig_class = px.bar(
    class_chart_df,
    x="Outcome",
    y="Score",
    color="Metric",
    barmode="group",
    text="Score",
    title="Performance by Match Outcome",
)


fig_class.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)


fig_class.update_layout(
    yaxis_title="Score (%)",
    xaxis_title="",
    yaxis_range=[
        0,
        100,
    ],
)


st.plotly_chart(
    fig_class,
    use_container_width=True,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.subheader("🧠 XGBoost Feature Importance")

st.markdown(
    """
    Feature importance shows which pre-match characteristics
    contributed most strongly to the XGBoost model's decisions.
    """
)


top_n = st.slider(
    "Number of features to display",
    min_value=5,
    max_value=min(20, len(importance_df)),
    value=min(15, len(importance_df)),
)


top_features = (
    importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
    .head(top_n)
    .sort_values(
        "importance",
        ascending=True,
    )
)


fig_importance = px.bar(
    top_features,
    x="importance",
    y="feature",
    orientation="h",
    title=f"Top {top_n} XGBoost Features",
    text="importance",
)


fig_importance.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
)


fig_importance.update_layout(
    xaxis_title="Feature Importance",
    yaxis_title="Feature",
)


st.plotly_chart(
    fig_importance,
    use_container_width=True,
)


# ============================================================
# TEST SET PREDICTIONS
# ============================================================

st.divider()

st.subheader("🔮 Test Set Predictions")

st.markdown(
    """
    The table below shows the actual outcome, predicted outcome,
    and probability assigned to each possible result.
    """
)


prediction_display = predictions_df.copy()


prediction_display["start_date"] = pd.to_datetime(
    prediction_display["start_date"],
    errors="coerce",
)


prediction_display["Date"] = (
    prediction_display["start_date"]
    .dt.strftime("%d %b %Y")
)


prediction_display = prediction_display[
    [
        "Date",
        "season",
        "home_team",
        "away_team",
        "target",
        "predicted_target",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
    ]
].copy()


prediction_display.columns = [
    "Date",
    "Season",
    "Home Team",
    "Away Team",
    "Actual Result",
    "Predicted Result",
    "Home Win %",
    "Draw %",
    "Away Win %",
]


prediction_display[
    [
        "Home Win %",
        "Draw %",
        "Away Win %",
    ]
] *= 100


prediction_display[
    [
        "Home Win %",
        "Draw %",
        "Away Win %",
    ]
] = prediction_display[
    [
        "Home Win %",
        "Draw %",
        "Away Win %",
    ]
].round(1)


st.dataframe(
    prediction_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

st.subheader("📈 Actual vs Predicted Outcomes")


actual_counts = (
    predictions_df["target"]
    .value_counts()
    .reindex(
        class_names,
        fill_value=0,
    )
)


predicted_counts = (
    predictions_df["predicted_target"]
    .value_counts()
    .reindex(
        class_names,
        fill_value=0,
    )
)


distribution_df = pd.DataFrame(
    {
        "Outcome": class_names,
        "Actual": actual_counts.values,
        "Predicted": predicted_counts.values,
    }
)


distribution_long = distribution_df.melt(
    id_vars="Outcome",
    var_name="Type",
    value_name="Matches",
)


fig_distribution = px.bar(
    distribution_long,
    x="Outcome",
    y="Matches",
    color="Type",
    barmode="group",
    text="Matches",
    title="Actual vs Predicted Outcome Distribution",
)


fig_distribution.update_layout(
    xaxis_title="Match Outcome",
    yaxis_title="Number of Matches",
)


st.plotly_chart(
    fig_distribution,
    use_container_width=True,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

st.divider()

st.subheader("⚙️ Model Configuration")

best_params = metrics_report.get(
    "best_params",
    {},
)


config_df = pd.DataFrame(
    [
        {
            "Parameter": key,
            "Value": value,
        }
        for key, value in best_params.items()
    ]
)


st.dataframe(
    config_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# PROJECT INTERPRETATION
# ============================================================

st.divider()

st.subheader("💡 Model Interpretation")

st.markdown(
    """
    The XGBoost model predicts three possible football match outcomes:

    - 🏠 **Home Win**
    - 🤝 **Draw**
    - ✈️ **Away Win**

    The model uses historical team performance, recent form,
    home/away performance, goal-scoring and defensive statistics,
    historical cross-season information and head-to-head statistics.

    All predictive features are constructed from information available
    before the corresponding match, preserving the chronological
    nature of the football prediction problem.

    Feature importance provides additional insight into which
    pre-match characteristics contribute most to the model's
    predictions.
    """
)