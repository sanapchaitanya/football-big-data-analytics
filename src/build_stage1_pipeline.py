from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.football_cleaning import clean_espn_matches, load_espn_raw, save_clean_dataset
from features.football_features import FEATURE_COLUMNS, build_modeling_dataset, chronological_split
from validation.football_validation import (
    validate_modeling_dataset,
    validate_no_future_information,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ESPN_PATH = PROJECT_ROOT / "data" / "processed" / "premier_league_2022_2027.parquet"
CLEAN_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "football_espn_clean.parquet"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODELING_OUTPUT_PATH = FEATURE_DIR / "football_modeling_dataset.parquet"
TRAIN_OUTPUT_PATH = FEATURE_DIR / "football_train.parquet"
VALIDATION_OUTPUT_PATH = FEATURE_DIR / "football_validation.parquet"
TEST_OUTPUT_PATH = FEATURE_DIR / "football_test.parquet"


def _date_range(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    return f"{df['start_date'].min()} to {df['start_date'].max()}"


def main() -> None:
    raw_df = load_espn_raw(RAW_ESPN_PATH)
    clean_df = clean_espn_matches(raw_df)
    save_clean_dataset(clean_df, CLEAN_OUTPUT_PATH)

    modeling_df = build_modeling_dataset(clean_df)
    train_df, validation_df, test_df = chronological_split(modeling_df)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    modeling_df.to_parquet(MODELING_OUTPUT_PATH, index=False)
    train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
    validation_df.to_parquet(VALIDATION_OUTPUT_PATH, index=False)
    test_df.to_parquet(TEST_OUTPUT_PATH, index=False)

    validation_errors = []
    validation_errors.extend(validate_modeling_dataset(modeling_df))
    validation_errors.extend(validate_no_future_information(modeling_df))

    print("=" * 80)
    print("STAGE 1 ESPN FOOTBALL DATA + FEATURE ENGINEERING")
    print("=" * 80)
    print(f"Raw ESPN rows: {len(raw_df)}")
    print(f"Clean ESPN rows: {len(clean_df)}")
    print(f"Completed modeling rows: {len(modeling_df)}")
    print()
    print("Output files:")
    print(CLEAN_OUTPUT_PATH)
    print(MODELING_OUTPUT_PATH)
    print(TRAIN_OUTPUT_PATH)
    print(VALIDATION_OUTPUT_PATH)
    print(TEST_OUTPUT_PATH)
    print()
    print("Modeling columns:")
    print(modeling_df.columns.tolist())
    print()
    print("Feature columns:")
    print(FEATURE_COLUMNS)
    print()
    print("Missing values:")
    print(modeling_df.isna().sum().to_string())
    print()
    print("Target distribution:")
    print(modeling_df["target"].value_counts().to_string())
    print()
    print("Split row counts:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test: {len(test_df)}")
    print()
    print("Split date ranges:")
    print(f"Train: {_date_range(train_df)}")
    print(f"Validation: {_date_range(validation_df)}")
    print(f"Test: {_date_range(test_df)}")
    print()
    print("Sample rows:")
    sample_columns = [
        "event_id",
        "start_date",
        "season",
        "home_team",
        "away_team",
        "target",
        "home_points_per_game_before",
        "away_points_per_game_before",
        "home_recent_points_per_game_5",
        "away_recent_points_per_game_5",
        "points_per_game_diff",
    ]
    print(modeling_df[sample_columns].head(10).to_string(index=False))
    print()
    print("Validation:")
    if validation_errors:
        print("FAILED")
        for error in validation_errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
