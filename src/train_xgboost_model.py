from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier

from features.football_features import FEATURE_COLUMNS, TARGET_ENCODING


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = FEATURE_DIR / "football_train.parquet"
VALIDATION_PATH = FEATURE_DIR / "football_validation.parquet"
TEST_PATH = FEATURE_DIR / "football_test.parquet"

MODEL_PATH = MODEL_DIR / "xgboost_match_outcome_model.json"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
TARGET_MAPPING_PATH = MODEL_DIR / "target_mapping.json"
METRICS_PATH = REPORT_DIR / "xgboost_metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "xgboost_test_predictions.csv"
IMPORTANCE_PATH = REPORT_DIR / "xgboost_feature_importance.csv"

CLASS_NAMES = ["Home Win", "Draw", "Away Win"]
RANDOM_STATE = 42


PARAMETER_GRID = [
    {
        "n_estimators": 120,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 180,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 160,
        "max_depth": 3,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "gamma": 0.2,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 220,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.5,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
    },
    {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.75,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.5,
        "reg_alpha": 0.2,
        "reg_lambda": 4.0,
    },
]

INTEGER_PARAMS = {
    "n_estimators",
    "max_depth",
    "min_child_weight",
}


def _load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing_features = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        raise ValueError(f"{path.name} is missing features: {missing_features}")
    if "target_encoded" not in df.columns:
        raise ValueError(f"{path.name} is missing target_encoded")
    return df


def _make_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        **params,
    )


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS], df["target_encoded"].astype(int)


def tune_hyperparameters(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    x_train, y_train = _xy(train_df)
    x_validation, y_validation = _xy(validation_df)
    results = []

    for index, params in enumerate(PARAMETER_GRID, start=1):
        model = _make_model(params)
        model.fit(x_train, y_train)
        validation_probabilities = model.predict_proba(x_validation)
        validation_predictions = validation_probabilities.argmax(axis=1)
        row = {
            "candidate": index,
            "validation_log_loss": log_loss(
                y_validation, validation_probabilities, labels=[0, 1, 2]
            ),
            "validation_accuracy": accuracy_score(
                y_validation, validation_predictions
            ),
            **params,
        }
        results.append(row)

    tuning_df = pd.DataFrame(results).sort_values(
        ["validation_log_loss", "validation_accuracy"],
        ascending=[True, False],
    )
    best_params = {}
    for key in PARAMETER_GRID[0].keys():
        value = tuning_df.iloc[0][key].item()
        if key in INTEGER_PARAMS:
            value = int(value)
        else:
            value = float(value)
        best_params[key] = value
    return best_params, tuning_df.reset_index(drop=True)


def evaluate_model(
    model: XGBClassifier,
    test_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    x_test, y_test = _xy(test_df)
    probabilities = model.predict_proba(x_test)
    predictions = probabilities.argmax(axis=1)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            labels=[0, 1, 2],
            average="weighted",
            zero_division=0,
        )
    )

    try:
        roc_auc = roc_auc_score(
            y_test,
            probabilities,
            labels=[0, 1, 2],
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        roc_auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1, 2]),
        "roc_auc_ovr_macro": roc_auc,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1, 2]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }

    predictions_df = test_df[
        ["event_id", "start_date", "season", "home_team", "away_team", "target"]
    ].copy()
    predictions_df["predicted_target"] = [CLASS_NAMES[index] for index in predictions]
    predictions_df["prob_home_win"] = probabilities[:, 0]
    predictions_df["prob_draw"] = probabilities[:, 1]
    predictions_df["prob_away_win"] = probabilities[:, 2]

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return metrics, predictions_df, importance_df


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    train_df = _load_split(TRAIN_PATH)
    validation_df = _load_split(VALIDATION_PATH)
    test_df = _load_split(TEST_PATH)

    best_params, tuning_df = tune_hyperparameters(train_df, validation_df)

    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    x_train_validation, y_train_validation = _xy(train_validation_df)
    final_model = _make_model(best_params)
    final_model.fit(x_train_validation, y_train_validation)

    metrics, predictions_df, importance_df = evaluate_model(final_model, test_df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    final_model.save_model(MODEL_PATH)
    FEATURE_COLUMNS_PATH.write_text(
        json.dumps(FEATURE_COLUMNS, indent=2),
        encoding="utf-8",
    )
    TARGET_MAPPING_PATH.write_text(
        json.dumps(TARGET_ENCODING, indent=2),
        encoding="utf-8",
    )
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    report = {
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "features": FEATURE_COLUMNS,
        "best_params": best_params,
        "tuning_results": tuning_df.to_dict(orient="records"),
        "test_metrics": metrics,
        "model_path": str(MODEL_PATH),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "target_mapping_path": str(TARGET_MAPPING_PATH),
        "predictions_path": str(PREDICTIONS_PATH),
        "feature_importance_path": str(IMPORTANCE_PATH),
    }
    METRICS_PATH.write_text(
        json.dumps(_json_ready(report), indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("STAGE 2 XGBOOST MATCH OUTCOME MODEL")
    print("=" * 80)
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")
    print()
    print("Best XGBoost parameters:")
    print(json.dumps(_json_ready(best_params), indent=2))
    print()
    print("Validation tuning results:")
    print(tuning_df.to_string(index=False))
    print()
    print("Test metrics:")
    printable_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"classification_report", "confusion_matrix"}
    }
    print(json.dumps(_json_ready(printable_metrics), indent=2))
    print()
    print("Confusion matrix rows=true, columns=predicted; class order:")
    print(CLASS_NAMES)
    print(metrics["confusion_matrix"])
    print()
    print("Class-wise performance:")
    class_rows = []
    for class_name in CLASS_NAMES:
        class_metrics = metrics["classification_report"][class_name]
        class_rows.append(
            {
                "class": class_name,
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1-score": class_metrics["f1-score"],
                "support": class_metrics["support"],
            }
        )
    print(pd.DataFrame(class_rows).to_string(index=False))
    print()
    print("Top feature importance:")
    print(importance_df.head(15).to_string(index=False))
    print()
    print("Example predictions:")
    print(predictions_df.head(10).to_string(index=False))
    print()
    print("Saved artifacts:")
    print(MODEL_PATH)
    print(FEATURE_COLUMNS_PATH)
    print(TARGET_MAPPING_PATH)
    print(METRICS_PATH)
    print(PREDICTIONS_PATH)
    print(IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
