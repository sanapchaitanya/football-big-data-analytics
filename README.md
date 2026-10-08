⚽ Football Big Data Analytics and Match Outcome Prediction System
An end-to-end Football Big Data Analytics and Match Outcome Prediction
System built using historical ESPN football data, Python, Pandas,
Plotly, Streamlit, and XGBoost.
The project combines large-scale football data processing, statistical
analytics, historical feature engineering, machine-learning-based match
prediction, model evaluation, and interactive what-if analysis into a
single dashboard.
📌 Project Overview
Football produces large volumes of structured match data containing team
performance, results, goals, home/away behaviour, recent form, and
head-to-head information.
This project uses that data to answer two main questions:
1. How have Premier League teams and matches performed
   historically?
2. Can pre-match information be used to predict the outcome of a
   future football match?
The system processes historical ESPN match data and provides:
- League-level analytics
- Season-wise analysis
- Team performance analysis
- Head-to-head analysis
- Recent-form analysis
- Home/away performance analysis
- XGBoost match-outcome prediction
- Home Win / Draw / Away Win probabilities
- Statistical predicted scoreline
- Model performance evaluation
- Feature-importance analysis
- What-if scenario analysis
🎯 Objectives
The main objectives of the project are:
- Collect and process football match data from ESPN.
- Clean and validate the historical football dataset.
- Perform exploratory and descriptive football analytics.
- Engineer leakage-safe pre-match features.
- Preserve chronological ordering during model development.
- Train a single XGBoost classification model for match outcomes.
- Predict three possible outcomes:
  - Home Win
  - Draw
  - Away Win
- Display model probabilities for each outcome.
- Estimate a likely scoreline using a statistical Poisson score
  calculation.
- Evaluate the trained model using standard classification metrics.
- Provide an interactive Streamlit dashboard.
- Allow users to explore hypothetical changes through what-if
  analysis.
🏗️ System Architecture
                         ESPN Football Data
                                │
                                ▼
                       Raw Football Dataset
                                │
                                ▼
                     Data Cleaning & Validation
                                │
                                ▼
                  Processed Football Dataset
                                │
                                ▼
                  Historical Feature Engineering
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
         Football Analytics              XGBoost Model
                  │                           │
        ┌─────────┼─────────┐                 │
        ▼         ▼         ▼                 ▼
      Season    Team       H2H         Match Outcome
     Analysis  Analysis  Analysis       Probabilities
        │         │         │                 │
        └─────────┼─────────┘                 ▼
                  │                    Predicted Result
                  │                           │
                  │                           ▼
                  │                    Score Estimation
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                       Streamlit Dashboard
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
        Analytics          ML Prediction       What-If
        Dashboard            Dashboard          Analysis
🧰 Technology Stack
  Component                 Technology
  Programming Language      Python
  Data Source               ESPN
  Data Processing           Pandas
  Data Storage              Parquet
  Machine Learning          XGBoost
  ML Evaluation             Scikit-learn
  Visualization             Plotly
  Dashboard                 Streamlit
  Development Environment   PyCharm
  Version Control           Git / GitHub
📂 Project Structure
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
│   │   ├── __init__.py
│   │   └── football_analytics.py
│   │
│   ├── features/
│   │   └── football_features.py
│   │
│   ├── validation/
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
└── README.md
📊 Data Source
The project uses ESPN football data as its football data source.
The current dataset covers multiple Premier League seasons and contains
completed as well as scheduled fixtures where available.
The processed dataset is stored in Parquet format for efficient loading
and analysis.
Important
The prediction system only uses information that would be available
before the match being predicted.
Match results and post-match information are not used as predictive
features for that same fixture.
🧹 Data Processing
The football data passes through a preprocessing and validation pipeline
before being used by the analytics and machine-learning components.
The processing stage includes:
1. Loading ESPN football data.
2. Converting match dates into datetime format.
3. Identifying completed matches.
4. Extracting home and away goals from match results.
5. Validating match records.
6. Removing invalid or unusable records.
7. Sorting matches chronologically.
8. Storing the processed dataset in Parquet format.
🧠 Feature Engineering
The prediction system uses pre-match historical features.
The feature engineering layer maintains historical state for teams and
head-to-head relationships while processing matches chronologically.
Team performance features
Examples include:
- Matches played
- Points per game
- Goals scored per game
- Goals conceded per game
- Goal difference per game
- Historical points per game
- Historical goals scored
- Historical goals conceded
- Historical goal difference
Recent-form features
The system calculates recent performance using the team's previous five
matches, including:
- Recent points per game
- Recent goals scored
- Recent goals conceded
Venue-specific features
The system also considers:
- Home points per game
- Away points per game
- Home goals scored
- Away goals scored
- Home goals conceded
- Away goals conceded
Head-to-head features
Historical encounters between the two teams are used to calculate:
- Number of previous meetings
- Head-to-head points per game
- Head-to-head goals scored
- Team-specific head-to-head performance
Difference features
Differences between the home and away teams are also calculated,
including:
- Points-per-game difference
- Recent-form difference
- Goals-scored difference
- Goals-conceded difference
- Goal-difference difference
- Historical performance differences
🚨 Data Leakage Prevention
Preventing data leakage is an important part of the project.
The model must not use information from the future to predict a match.
The feature-generation process therefore:
- Processes matches chronologically.
- Calculates features before updating team states with the current
  match.
- Uses previous matches to construct current features.
- Maintains historical information across seasons.
- Uses only completed matches before the prediction fixture.
- Uses chronological train, validation and test splits.
For an exact fixture prediction, the system reconstructs the available
historical information before that fixture's kickoff.
🤖 Machine Learning Model
The project uses one machine-learning algorithm: XGBoost.
The prediction task is a three-class classification problem:
0 → Home Win
1 → Draw
2 → Away Win
XGBoost configuration
The model uses:
- multi:softprob objective
- 3 output classes
- Multiclass log-loss evaluation
- Histogram-based tree construction
- Hyperparameter selection using a chronological validation set
The final model is trained using the training and validation datasets
after hyperparameter selection.
🕒 Chronological Model Split
Because football prediction is a temporal problem, the data is not
randomly shuffled into train and test sets.
The project uses:
Training:
2022-23
2023-24
2024-25

Validation:
2025-26

Testing:
2026-27
This design ensures that the model is evaluated on a later period than
the data used for training.
The final test season is incomplete in the current dataset, so the
test-set size represents only the completed matches available at the
time of model generation.
📈 Model Evaluation
The model is evaluated using:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- ROC-AUC
- Log Loss
- Confusion Matrix
- Class-wise Precision / Recall / F1
- Feature Importance
The current model evaluation reported approximately:
  Metric       Result
  Accuracy      46.9%
  Macro F1      36.7%
  ROC-AUC       0.686
  Log Loss      1.016
These values are based on the current test-set run and can change if the
underlying dataset or model configuration is regenerated.
🔮 Match Prediction
The Match Prediction page allows the user to select an exact Premier
League fixture.
The system displays:
- Fixture date
- Kickoff time
- Home team
- Away team
- Match status
- Predicted result
- Home-win probability
- Draw probability
- Away-win probability
- Predicted scoreline
- Pre-match feature snapshot
Example presentation:
10 October 2026
16:30 UTC

Manchester United vs Tottenham Hotspur

Predicted Result:
Manchester United Win

Predicted Score:
Manchester United 2–1 Tottenham Hotspur

Home Win: 52.4%
Draw: 24.1%
Away Win: 23.5%
The probabilities are generated by the trained XGBoost model.
⚽ Predicted Scoreline
XGBoost is used specifically for the three-class match outcome
prediction.
It does not directly predict an exact football score.
Therefore, the dashboard includes a separate statistical scoreline layer
based on pre-match attacking and defensive goal rates.
A Poisson-based scoreline calculation evaluates possible score
combinations and selects the most probable score.
This does not introduce a second machine-learning model.
📊 Model Performance Dashboard
The Model Performance page provides an interactive evaluation of the
trained XGBoost model.
It contains:
Overall metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Log Loss
Confusion matrix
Shows actual outcomes against predicted outcomes:
                 Predicted
              H       D       A

Actual H      ✓       ?       ?
Actual D      ?       ✓       ?
Actual A      ?       ?       ✓
Class-wise evaluation
The dashboard reports precision, recall, F1 score and support for:
- Home Win
- Draw
- Away Win
Feature importance
The system visualizes the most important features used by XGBoost.
Test predictions
The dashboard also displays:
- Date
- Season
- Home team
- Away team
- Actual result
- Predicted result
- Home-win probability
- Draw probability
- Away-win probability
🎛️ What-If Analysis
The What-If Analysis page provides scenario-based analysis using the
trained XGBoost model.
The user can modify selected pre-match indicators such as:
- Home points per game
- Away points per game
- Home recent points per game
- Away recent points per game
- Home goal difference per game
- Away goal difference per game
The modified feature vector is passed through the same trained XGBoost
model.
The dashboard compares:
Baseline Prediction
        vs
What-If Prediction
It also shows:
- Baseline probabilities
- Scenario probabilities
- Probability changes
- Prediction changes
- Baseline score
- Scenario score
The original trained model is not retrained during what-if analysis.
📈 Football Analytics
The analytics section contains four major pages.
Overview
Provides:
- Total matches
- Total goals
- Average goals per match
- Home wins
- Draws
- Away wins
- Season trends
- Outcome distribution
- League table
- Project architecture
Season Analytics
Provides season-level analysis including:
- Matches
- Goals
- Average goals
- Home wins
- Draws
- Away wins
- Season comparisons
Team Analytics
Provides team-level analysis including:
- Matches
- Wins
- Draws
- Losses
- Points
- Points per game
- Goals scored
- Goals conceded
- Goal difference
- Home/away performance
- Recent form
- Head-to-head history
Team Comparison
Allows two teams to be compared using:
- Overall statistics
- Points per game
- Goals
- Goal difference
- Recent form
- Head-to-head results
🚀 Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd sports_calendar
2. Create a virtual environment
Windows:
python -m venv .venv
Activate it:
.venv\Scripts\activate
3. Install dependencies
If a requirements.txt file is available:
pip install -r requirements.txt
Otherwise install the main packages:
pip install pandas pyarrow streamlit plotly xgboost scikit-learn requests
4. Run the application
streamlit run app.py
The dashboard will open in the browser.
▶️ Running the Project
The main application is:
app.py
Run:
streamlit run app.py
The Streamlit navigation contains:
Football Analytics
├── Overview
├── Season Analytics
├── Team Analytics
└── Team Comparison

Machine Learning
├── Match Prediction
├── Model Performance
└── What-If Analysis
🔄 Rebuilding the Model
The XGBoost training pipeline is implemented in:
src/train_xgboost_model.py
The model training pipeline:
1. Loads chronological feature datasets.
2. Tunes XGBoost hyperparameters using the validation set.
3. Combines training and validation data.
4. Trains the final XGBoost model.
5. Evaluates the model on the test set.
6. Saves the trained model.
7. Saves model metrics.
8. Saves test predictions.
9. Saves feature importance.
Generated model artifacts include:
models/xgboost_match_outcome_model.json
models/feature_columns.json
models/target_mapping.json

reports/xgboost_metrics.json
reports/xgboost_test_predictions.csv
reports/xgboost_feature_importance.csv
💾 Data Formats
The project uses:
Parquet
Used for efficient storage of:
- Processed football data
- Feature datasets
- Train/validation/test datasets
CSV
Used for:
- Test predictions
- Feature importance reports
JSON
Used for:
- Model metrics
- Feature configuration
- Target encoding
- Model metadata
🔐 Prediction Design Principles
The project follows several principles:
1. No future information
Only pre-match information is used for prediction.
2. Chronological evaluation
Future seasons are kept separate from historical training data.
3. Single ML model
XGBoost is the only machine-learning model used for match-outcome
prediction.
4. Reproducibility
The model uses a fixed random state and saves its trained artifacts and
evaluation reports.
5. Explainability
Feature importance and probability outputs are provided to help
interpret predictions.
📌 Limitations
The current system has several limitations:
- Football outcomes are inherently uncertain.
- Match outcomes depend on factors that may not be represented in
  historical match data.
- Injuries, suspensions, lineups, tactical changes and other real-time
  information are not fully represented in the current feature set.
- The exact scoreline is a statistical estimate rather than an XGBoost
  prediction.
- The current test season is incomplete.
- Model probabilities should not be interpreted as guaranteed
  outcomes.
The system is intended as a data analytics and prediction research
project, not as a guaranteed match-result forecasting system.
🔮 Future Enhancements
Possible future improvements include:
- Live ESPN fixture updates
- Automated scheduled data refresh
- Player-level statistics
- Injury and suspension information
- Starting-lineup information
- Weather information
- More advanced probability calibration
- Automated model retraining
- Additional explainability methods
- Deployment to a cloud platform
- Automated reporting
- Historical prediction tracking
These are future enhancements and are not required for the current core
implementation.
🧪 Example Use Case
A user selects:
Season:
2026-27

Fixture:
Manchester United vs Tottenham Hotspur

Date:
10 October 2026
The system reconstructs the information available before kickoff and
generates:
Predicted Outcome:
Manchester United Win

Predicted Score:
Manchester United 2–1 Tottenham Hotspur

Home Win Probability:
XX.X%

Draw Probability:
XX.X%

Away Win Probability:
XX.X%
The user can then move to What-If Analysis to examine how
hypothetical changes in team performance could affect the prediction.
📚 Project Workflow
1. Data Collection
        ↓
2. Data Cleaning
        ↓
3. Data Validation
        ↓
4. Exploratory Data Analysis
        ↓
5. Historical Feature Engineering
        ↓
6. Chronological Dataset Split
        ↓
7. XGBoost Training
        ↓
8. Model Evaluation
        ↓
9. Match Prediction
        ↓
10. Scoreline Estimation
        ↓
11. What-If Analysis
        ↓
12. Interactive Streamlit Dashboard
👩‍💻 Author
Chaitanya Sanap
Football Big Data Analytics and Match Outcome Prediction System
Built using Python, ESPN data, XGBoost, Streamlit, Pandas and Plotly.
