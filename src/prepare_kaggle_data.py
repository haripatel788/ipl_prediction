import argparse
from pathlib import Path

import pandas as pd


POSSIBLE_FILES = [
    "matches.csv",
    "ipl_matches.csv",
    "IPL Matches 2008-2025.csv",
    "IPL_Matches_2008_2025.csv",
]


REQUIRED_OUTPUT_COLUMNS = [
    "team1",
    "team2",
    "venue",
    "toss_winner",
    "toss_decision",
    "winner",
]


def normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def pick_input_file(data_dir: Path) -> Path:
    for file_name in POSSIBLE_FILES:
        candidate = data_dir / file_name
        if candidate.exists():
            return candidate
    csv_files = sorted(data_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    raise FileNotFoundError(
        f"Could not auto-detect a single matches CSV in {data_dir}. "
        f"Found: {[f.name for f in csv_files]}"
    )


def build_column_map(columns: list[str]) -> dict[str, str]:
    normalized = {normalize(col): col for col in columns}
    aliases = {
        "team1": ["team1", "team_1"],
        "team2": ["team2", "team_2"],
        "venue": ["venue", "ground", "stadium"],
        "toss_winner": ["toss_winner", "tosswinner"],
        "toss_decision": ["toss_decision", "tossdecision"],
        "winner": ["winner", "match_winner", "winning_team"],
    }

    mapping = {}
    missing = []
    for out_col, options in aliases.items():
        found = None
        for option in options:
            if option in normalized:
                found = normalized[option]
                break
        if found is None:
            missing.append(out_col)
        else:
            mapping[out_col] = found

    if missing:
        raise ValueError(f"Missing required mapped columns: {missing}")
    return mapping


def build_from_ball_by_ball(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "match_id",
        "batting_team",
        "toss_winner",
        "toss_decision",
        "venue",
        "match_won_by",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Ball-by-ball file missing columns: {missing}")

    temp = df[required].copy()
    temp["batting_team"] = temp["batting_team"].astype(str).str.strip()
    temp["match_won_by"] = temp["match_won_by"].astype(str).str.strip()
    temp["toss_winner"] = temp["toss_winner"].astype(str).str.strip()
    temp["toss_decision"] = temp["toss_decision"].astype(str).str.strip().str.lower()
    temp["venue"] = temp["venue"].astype(str).str.strip()

    team_pairs = (
        temp.groupby("match_id")["batting_team"]
        .apply(lambda s: sorted(set([x for x in s if x and x.lower() != "nan"])))
        .reset_index(name="teams")
    )
    team_pairs = team_pairs[team_pairs["teams"].map(len) == 2]
    team_pairs["team1"] = team_pairs["teams"].map(lambda t: t[0])
    team_pairs["team2"] = team_pairs["teams"].map(lambda t: t[1])

    match_meta = (
        temp.groupby("match_id", as_index=False)
        .agg(
            toss_winner=("toss_winner", "first"),
            toss_decision=("toss_decision", "first"),
            venue=("venue", "first"),
            winner=("match_won_by", "first"),
        )
    )

    out = team_pairs.merge(match_meta, on="match_id", how="inner")
    out = out[
        ["team1", "team2", "venue", "toss_winner", "toss_decision", "winner"]
    ].copy()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    out = out.dropna(subset=REQUIRED_OUTPUT_COLUMNS)
    out = out[(out["winner"] == out["team1"]) | (out["winner"] == out["team2"])]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kaggle IPL dataset for training.")
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing downloaded Kaggle CSV files.",
    )
    parser.add_argument(
        "--out",
        default="data/ipl_matches.csv",
        help="Output path for normalized training CSV.",
    )
    parser.add_argument(
        "--cache-out",
        default="data/cache/ipl_matches.csv.gz",
        help="Output path for compressed cached dataset.",
    )
    parser.add_argument(
        "--cleanup-raw",
        action="store_true",
        help="Delete the source raw CSV after successful preprocessing.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    cache_path = Path(args.cache_out)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    input_csv = pick_input_file(raw_dir)
    df = pd.read_csv(input_csv, low_memory=False)
    normalized_columns = {normalize(c) for c in df.columns}

    if "match_id" in normalized_columns and "batting_team" in normalized_columns:
        normalized_df = build_from_ball_by_ball(df)
    else:
        column_map = build_column_map(df.columns.tolist())
        normalized_df = pd.DataFrame({col: df[source] for col, source in column_map.items()})
        normalized_df = normalized_df.dropna(subset=REQUIRED_OUTPUT_COLUMNS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(out_path, index=False)
    normalized_df.to_csv(cache_path, index=False, compression="gzip")
    print(f"Prepared data saved to: {out_path}")
    print(f"Compressed cache saved to: {cache_path}")
    print(f"Rows: {len(normalized_df)}")

    if args.cleanup_raw:
        input_csv.unlink(missing_ok=True)
        print(f"Deleted raw source file: {input_csv}")


if __name__ == "__main__":
    main()
