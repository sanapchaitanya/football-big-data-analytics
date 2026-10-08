from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_ESPN_COLUMNS = [
    "event_id",
    "start_date",
    "sport_type",
    "country",
    "country_code",
    "tournament",
    "season",
    "match",
    "home_team",
    "away_team",
    "event_status",
    "result",
    "source",
]


def load_espn_raw(input_path: Path) -> pd.DataFrame:
    """Load the existing ESPN raw parquet without changing it."""
    df = pd.read_parquet(input_path)
    missing = [column for column in REQUIRED_ESPN_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"ESPN raw dataset is missing required columns: {missing}")
    return df


def clean_espn_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Create an ESPN-only clean football dataset for analytics/modeling."""
    clean = df.copy()

    clean = clean[clean["source"].astype("string").str.upper() == "ESPN"].copy()
    clean["start_date"] = pd.to_datetime(clean["start_date"], utc=True)

    text_columns = [
        "event_id",
        "sport_type",
        "country",
        "country_code",
        "tournament",
        "season",
        "match",
        "home_team",
        "away_team",
        "event_status",
        "result",
        "source",
    ]

    for column in text_columns:
        clean[column] = clean[column].astype("string").str.strip()

    clean = clean.drop_duplicates(subset=["event_id"])
    clean = clean.sort_values(["start_date", "event_id"]).reset_index(drop=True)

    clean["is_completed"] = (
        clean["event_status"].eq("Full Time")
        & clean["result"].notna()
        & clean["result"].str.match(r"^\d+-\d+$", na=False)
    )

    return clean


def save_clean_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
