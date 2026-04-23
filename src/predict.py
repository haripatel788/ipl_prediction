import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict IPL match winner.")
    parser.add_argument("--model", required=True, help="Path to trained model file.")
    parser.add_argument("--team1", required=True, help="Team 1 name.")
    parser.add_argument("--team2", required=True, help="Team 2 name.")
    parser.add_argument("--venue", required=True, help="Venue name.")
    parser.add_argument("--toss-winner", required=True, help="Toss winner team.")
    parser.add_argument(
        "--toss-decision",
        required=True,
        choices=["bat", "field"],
        help="Toss decision made by toss winner.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    pipeline = joblib.load(model_path)

    row = pd.DataFrame(
        [
            {
                "team1": args.team1,
                "team2": args.team2,
                "venue": args.venue,
                "toss_winner": args.toss_winner,
                "toss_decision": args.toss_decision,
            }
        ]
    )

    winner = pipeline.predict(row)[0]
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(row)[0]
        classes = pipeline.classes_
        confidence = dict(zip(classes, [float(p) for p in proba]))
        print("Predicted winner:", winner)
        print("Class probabilities:", confidence)
    else:
        print("Predicted winner:", winner)


if __name__ == "__main__":
    main()
