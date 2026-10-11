# Football Big Data Analytics and Match Outcome Prediction

## Project Overview

**Football Data Analytics and Match Outcome Prediction** is a Big Data Analytics project that analyses English Premier League match data across multiple seasons. It combines Hadoop-based data processing, statistical analysis, machine learning, and interactive visualisation to understand team performance and predict match outcomes.

The project uses **Hadoop HDFS, Hadoop MapReduce, Hadoop Streaming, and YARN** to process football match records and generate team-wise and season-wise statistics. An XGBoost classification model predicts whether a match will result in a Home Win, Draw, or Away Win.

The results are presented through an interactive Streamlit dashboard.

## Objectives

- Analyse English Premier League match data across multiple seasons.
- Demonstrate Big Data processing using Hadoop.
- Calculate team-wise and season-wise performance statistics using MapReduce.
- Identify trends in wins, draws, losses, goals, and points.
- Predict football match outcomes using machine learning.
- Visualise analytical results through an interactive dashboard.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing, analytics, and model implementation |
| ESPN | Source of football match data |
| Pandas | Data manipulation and preparation |
| Parquet | Columnar storage of processed match data |
| Hadoop HDFS | Storage of input data for Hadoop processing |
| Hadoop MapReduce | Distributed processing and aggregation of match records |
| Hadoop Streaming | Execution of Python Mapper and Reducer scripts |
| Hadoop YARN | Management and execution of Hadoop jobs |
| XGBoost | Three-class match outcome prediction |
| Scikit-learn | Machine learning utilities and evaluation |
| Streamlit | Interactive web dashboard |
| Plotly | Data visualisation |
| Git and GitHub | Version control and project management |

## System Architecture

```text
        ESPN Football Match Data
                  |
                  v
       Data Collection and Cleaning
                  |
                  v
        Processed Parquet Dataset
                  |
                  v
       CSV Input for Hadoop Processing
                  |
                  v
              Hadoop HDFS
                  |
                  v
          Hadoop MapReduce
          /             \
         v               v
      Mapper           Reducer
         \               /
          v             v
       Team and Season Statistics
                  |
                  v
          MapReduce Output
                  |
                  v
       Streamlit Analytics Dashboard
```

The match prediction component uses the prepared historical match features and a trained XGBoost model to generate outcome predictions.

## Big Data Implementation

### 1. Hadoop Distributed File System (HDFS)

HDFS stores the football match dataset used as input to the Hadoop processing pipeline. The input CSV is uploaded to HDFS before the MapReduce job is executed.

### 2. Hadoop MapReduce

MapReduce is used to aggregate match records and generate team-season statistics.

**Mapper (`mapper.py`)**

The Mapper processes individual match records and emits intermediate key-value pairs for the relevant season and team, along with the corresponding match statistics.

**Reducer (`reducer.py`)**

The Reducer aggregates the intermediate values to calculate:

- Matches played
- Wins
- Draws
- Losses
- Goals scored
- Goals conceded
- Total points

### 3. Hadoop Streaming

Hadoop Streaming allows the Python Mapper and Reducer scripts to execute as a Hadoop MapReduce job without requiring the processing logic to be written in Java.

### 4. Hadoop YARN

YARN manages the execution of the MapReduce application and its processing tasks.

### 5. MapReduce Output

The aggregated results are saved in a tab-separated output file:

`reports/team_season_stats.tsv`

The output is used by the Hadoop Team Analytics page in the Streamlit dashboard.

## Machine Learning

### XGBoost Classification

The project uses **XGBoost Classifier** to predict football match outcomes.

The model classifies a match into one of three categories:

1. Home Win
2. Draw
3. Away Win

The model uses historical match features to estimate the probabilities of the possible outcomes.

The machine learning component is separate from the Hadoop MapReduce analytics component. Hadoop processes and aggregates the data, while XGBoost performs the match outcome prediction.

### Model Evaluation

The previously recorded model evaluation results are:

| Metric | Recorded Result |
|---|---:|
| Accuracy | 46.9% |
| Macro F1-score | 36.7% |
| ROC-AUC | 0.686 |
| Log Loss | 1.016 |

These figures represent the recorded evaluation of the existing model and should be interpreted in the context of its test dataset.

## Dashboard Features

The Streamlit dashboard provides analytical views of football matches and team performance.

### Football Analytics
- Overview of the available match dataset
- Season-level performance analysis
- Team-level statistics
- Team comparison
- Interactive charts and visualisations

### Match Prediction
- Select home and away teams
- Generate predictions for Home Win, Draw, and Away Win
- Display the model's predicted outcome probabilities

### Model Performance
- View recorded machine learning evaluation metrics
- Understand the performance of the match outcome classifier

### Hadoop Team Analytics
- View MapReduce-generated team-season statistics
- Examine matches played, wins, draws, and losses
- Compare goals scored and conceded
- Analyse team points using interactive charts

## Dataset and Processing

The project uses historical and scheduled English Premier League match records collected from ESPN.

The processed dataset is stored in Parquet format. A CSV representation is prepared for Hadoop Streaming, and the resulting MapReduce statistics are saved separately for dashboard visualisation.

The current Hadoop workflow uses **batch processing**. New match data is not automatically collected or processed in real time; the dataset and MapReduce output must be refreshed by running the relevant processing steps.

## Project Structure

```text
football-big-data-analytics/
│
├── app.py
│
├── pages/
│   ├── match_prediction.py
│   └── 8_Hadoop_Team_Analytics.py
│
├── data/
│   ├── raw/
│   └── processed/
│       └── football_matches_mapreduce.csv
│
├── src/
│   ├── hadoop_mapreduce/
│   │   ├── mapper.py
│   │   └── reducer.py
│   └──
│
├── models/
│
├── reports/
│   └── team_season_stats.tsv
│
├── README.md
└── .gitignore
```

Other existing project files, including model artefacts, data preparation scripts, and reports, may also be present in their respective directories.

## Installation and Setup

### Prerequisites

- Python
- Git
- Hadoop 3.x
- Java compatible with the installed Hadoop version
- A Python virtual environment

### 1. Clone the repository

```bash
git clone https://github.com/sanapchaitanya/football-big-data-analytics.git
cd football-big-data-analytics
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

Install the packages required by the project:

```powershell
python -m pip install streamlit pandas pyarrow xgboost scikit-learn plotly numpy scipy
```

### 4. Configure Hadoop

Configure Java and Hadoop, including HDFS and YARN, according to the Hadoop installation environment.

Verify the installation:

```bash
hadoop version
```

### 5. Run the dashboard

From the project root, run:

```powershell
python -m streamlit run app.py
```

The dashboard will open in your browser.

## Running the Hadoop MapReduce Job

The MapReduce job is executed separately from the Streamlit dashboard.

1. Ensure Hadoop HDFS and YARN are running.
2. Upload the prepared CSV dataset to HDFS.
3. Execute the Hadoop Streaming job using `mapper.py` and `reducer.py`.
4. Save the output to an HDFS output directory.
5. Retrieve the generated output and save it as `reports/team_season_stats.tsv`.
6. Open the Hadoop Team Analytics page to explore the generated statistics.

The Hadoop Streaming JAR path depends on the installed Hadoop version. For Hadoop 3.4.3, the JAR used during development was:

```text
/opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.4.3.jar
```

## Limitations

- The data pipeline currently operates in batch mode rather than automatically updating in real time.
- The accuracy of analytics depends on the completeness and correctness of the collected match data.
- Match outcome prediction is probabilistic; actual match results may differ from predictions.
- Hadoop is configured as a single-node, pseudo-distributed environment for development and demonstration rather than a production multi-node cluster.
- The recorded machine learning evaluation results may vary when the model or dataset is updated.

## Conclusion

This project demonstrates the application of Big Data technologies and machine learning to football analytics. Hadoop HDFS provides data storage, MapReduce performs team and season aggregation, Hadoop Streaming enables Python-based processing, and YARN manages job execution. XGBoost provides match outcome predictions, while Streamlit and Plotly make the analytical results accessible through an interactive dashboard.

The project demonstrates a practical batch-processing workflow that combines Big Data processing, machine learning, and visual analytics in a single application.

## Author

**GitHub Repository:** https://github.com/sanapchaitanya/football-big-data-analytics
