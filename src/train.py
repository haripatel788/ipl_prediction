import argparse
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REQUIRED_COLUMNS = [
    "team1",
    "team2",
    "venue",
    "toss_winner",
    "toss_decision",
    "winner",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_pipeline() -> Pipeline:
    categorical_features = ["team1", "team2", "venue", "toss_winner", "toss_decision"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IPL winner prediction model.")
    parser.add_argument("--data", required=True, help="Path to input CSV dataset.")
    parser.add_argument(
        "--model-out",
        default="models/ipl_winner_model.joblib",
        help="Output path for trained model.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    model_out = Path(args.model_out)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    validate_columns(df)

    df = df.dropna(subset=REQUIRED_COLUMNS)

    X = df[["team1", "team2", "venue", "toss_winner", "toss_decision"]]
    y = df["winner"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    metadata = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    metadata_path = model_out.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Model saved to: {model_out}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
