# ⚽ Football Big Data Analytics and Match Outcome Prediction

An end-to-end football analytics and match outcome prediction system built using historical ESPN football data, Python, Pandas, XGBoost, Plotly, and Streamlit.

The project combines football data processing, statistical analysis, historical feature engineering, machine learning, model evaluation, and interactive prediction into a single dashboard.

---

## 📌 Project Overview

The system analyses historical Premier League football matches and provides insights into:

- League and season performance
- Team performance
- Home and away performance
- Recent form
- Head-to-head statistics
- Match outcome prediction
- Model probabilities
- Predicted scoreline
- Model performance
- Feature importance
- What-if analysis

The machine learning component predicts three possible match outcomes:

- 🏠 Home Win
- 🤝 Draw
- ✈️ Away Win

---

## 🏗️ System Architecture

```text
ESPN Football Data
        ↓
Raw Football Dataset
        ↓
Data Cleaning & Validation
        ↓
Processed Football Dataset
        ↓
Historical Feature Engineering
        ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Football Analytics    XGBoost Model
 │                       │
 ├── Season Analysis     ▼
 ├── Team Analysis   Match Probabilities
 └── H2H Analysis        │
                         ▼
                  Predicted Result
                         │
                         ▼
                  Score Estimation
                         │
                         ▼
                Streamlit Dashboard
                  │      │      │
                  ▼      ▼      ▼
              Analytics  ML   What-If
```

---

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Data Source | ESPN |
| Data Processing | Pandas |
| Data Storage | Parquet |
| Machine Learning | XGBoost |
| Evaluation | Scikit-learn |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Development | PyCharm |
| Version Control | Git / GitHub |

---

## 📊 Data

The project uses football match data obtained from ESPN.

The dataset contains multiple Premier League seasons and includes information such as:

- Match date
- Home team
- Away team
- Match result
- Season
- Historical team performance
- Home/away performance
- Recent form
- Head-to-head information

The data is cleaned, validated, and stored in Parquet format for efficient processing.

---

## 🧠 Feature Engineering

Pre-match features are created using historical information available **before the match begins**.

Features include:

- Points per game
- Goals scored per game
- Goals conceded per game
- Goal difference
- Recent form
- Home performance
- Away performance
- Head-to-head statistics
- Matches played
- Season match number
- Historical performance across seasons

The feature engineering process preserves chronological ordering to reduce the risk of data leakage.

---

## 🤖 Machine Learning

The project uses **XGBoost as the single machine learning model**.

### Prediction Classes

| Class | Outcome |
|---|---|
| 0 | Home Win |
| 1 | Draw |
| 2 | Away Win |

The model produces probabilities for all three outcomes.

### Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Log Loss
- Confusion Matrix
- Feature Importance

The dataset is divided chronologically into training, validation, and testing periods.

---

## ⚽ Match Prediction

The prediction dashboard allows users to select an exact fixture and view:

- Home team
- Away team
- Match date
- Predicted outcome
- Home win probability
- Draw probability
- Away win probability
- Predicted scoreline
- Important pre-match features

The scoreline is estimated using a statistical Poisson-based calculation from pre-match team performance rates.

---

## 🔬 What-If Analysis

The What-If module allows users to modify selected pre-match performance features and observe how the XGBoost prediction changes.

Users can experiment with factors such as:

- Points per game
- Recent form
- Goal difference

The model is **not retrained** during what-if analysis. The existing trained XGBoost model is used to evaluate the hypothetical scenario.

---

## 📈 Football Analytics Dashboard

The Streamlit dashboard provides interactive analytics including:

### Overview
- Dataset summary
- League statistics
- Outcome distribution
- Season trends
- League table

### Season Analytics
- Season performance
- Goals
- Results
- Team statistics

### Team Analytics
- Team performance
- Recent form
- Home/away records
- Goals and points

### Team Comparison
- Compare two teams
- Performance statistics
- Historical comparison

### Machine Learning
- Match prediction
- Model performance
- Feature importance
- What-if analysis

---

## 📂 Project Structure

```text
sports_calendar/
│
├── app.py
│
├── pages/
│   ├── overview.py
│   ├── season_analytics.py
│   ├── team_analytics.py
│   ├── team_comparison.py
│   ├── match_prediction.py
│   ├── model_performance.py
│   └── what_if.py
│
├── src/
│   ├── analytics/
│   │   └── football_analytics.py
│   │
│   ├── features/
│   │   └── football_features.py
│   │
│   ├── espn_client.py
│   ├── espn_premier_league.py
│   └── train_xgboost_model.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│   ├── xgboost_match_outcome_model.json
│   ├── feature_columns.json
│   └── target_mapping.json
│
├── reports/
│   ├── xgboost_metrics.json
│   ├── xgboost_test_predictions.csv
│   └── xgboost_feature_importance.csv
│
├── README.md
└── .gitignore
```

---

## 🚀 Running the Project

Clone the repository:

```bash
git clone https://github.com/sanapchaitanya/football-big-data-analytics.git
```

Navigate to the project:

```bash
cd football-big-data-analytics
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

## 🔄 Project Workflow

```text
ESPN Data
   ↓
Cleaning & Validation
   ↓
Feature Engineering
   ↓
Football Analytics
   ↓
XGBoost Training
   ↓
Model Evaluation
   ↓
Match Prediction
   ↓
Streamlit Dashboard
```

---

## ⚠️ Limitations

- Football match outcomes are inherently difficult to predict.
- The model only uses information available before the match.
- The latest season may contain fewer completed matches than previous seasons.
- Predicted scorelines are statistical estimates and should not be interpreted as guaranteed results.
- Model performance may change as additional seasons and matches become available.

---

## 🔮 Future Improvements

Possible future extensions include:

- Additional football leagues
- More advanced player-level features
- Live match integration
- Automated model retraining
- Additional historical seasons
- More advanced feature engineering
- Deployment as a public web application

---

## 👩‍💻 Author

**CHaitanya Sanap**

Football Big Data Analytics and Match Outcome Prediction System

**Technologies:** Python · Pandas · XGBoost · Plotly · Streamlit · ESPN · GitHub
