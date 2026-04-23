from pathlib import Path
import json
import platform

import joblib
import pandas as pd
import sklearn
import streamlit as st


MODEL_PATH = Path("models/ipl_winner_model.joblib")
META_PATH = MODEL_PATH.with_suffix(".meta.json")
DATA_PATH = Path("data/ipl_matches.csv")


def load_choices() -> tuple[list[str], list[str]]:
    default_teams = [
        "Mumbai Indians",
        "Chennai Super Kings",
        "Royal Challengers Bangalore",
        "Kolkata Knight Riders",
        "Delhi Capitals",
        "Rajasthan Royals",
        "Sunrisers Hyderabad",
        "Punjab Kings",
        "Lucknow Super Giants",
        "Gujarat Titans",
    ]
    default_venues = [
        "Wankhede Stadium",
        "M. Chinnaswamy Stadium",
        "Eden Gardens",
        "Arun Jaitley Stadium",
        "MA Chidambaram Stadium",
        "Narendra Modi Stadium",
    ]
    if not DATA_PATH.exists():
        return sorted(default_teams), sorted(default_venues)
    df = pd.read_csv(DATA_PATH)
    teams = sorted(
        set(df.get("team1", pd.Series(dtype=str)).dropna().unique()).union(
            set(df.get("team2", pd.Series(dtype=str)).dropna().unique())
        )
    )
    venues = sorted(df.get("venue", pd.Series(dtype=str)).dropna().unique())
    if not teams:
        teams = sorted(default_teams)
    if not venues:
        venues = sorted(default_venues)
    return teams, venues


def app_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, #3a165f 0, transparent 35%),
                radial-gradient(circle at 100% 100%, #0f4f5f 0, transparent 35%),
                linear-gradient(135deg, #0d0f1c 0%, #141a2e 45%, #0d0f1c 100%);
            color: #ecedf4;
        }
        .block-container {
            max-width: 1080px;
            padding-top: 2.1rem;
            padding-bottom: 2rem;
        }
        .headline {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.2px;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #b6bdd3;
            margin-bottom: 1.2rem;
            font-size: 0.98rem;
        }
        .panel {
            background: rgba(19, 24, 42, 0.74);
            border: 1px solid rgba(124, 140, 255, 0.3);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            box-shadow: 0 16px 46px rgba(0, 0, 0, 0.36);
            backdrop-filter: blur(4px);
        }
        .match-chip {
            display: inline-block;
            margin-top: 0.35rem;
            margin-bottom: 0.8rem;
            background: linear-gradient(90deg, #6c5ce7, #00c2b8);
            color: #ffffff;
            border-radius: 999px;
            padding: 0.42rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .winner-card {
            border: 1px solid rgba(103, 214, 185, 0.45);
            background: rgba(29, 74, 63, 0.42);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin-top: 0.9rem;
        }
        .winner-title {
            color: #9bf8d7;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }
        .winner-name {
            font-size: 1.55rem;
            font-weight: 800;
            margin-top: 0.15rem;
            margin-bottom: 0.15rem;
        }
        .helper {
            color: #aab2cc;
            font-size: 0.9rem;
        }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stRadio"] label {
            color: #dbe2ff !important;
            font-weight: 600 !important;
        }
        .stButton button {
            width: 100%;
            border-radius: 12px;
            border: none;
            color: #0f1220;
            background: linear-gradient(90deg, #6effc3, #67c2ff);
            font-weight: 800;
            min-height: 2.9rem;
        }
        .stButton button:hover {
            filter: brightness(1.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model():
    if not MODEL_PATH.exists():
        return None, None
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, str(exc)


def read_metadata() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except Exception:
        return {}


def main() -> None:
    st.set_page_config(page_title="IPL Match Pulse", page_icon="🏏", layout="wide")
    app_style()

    st.markdown('<div class="headline">IPL Match Pulse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A sharp, match-day prediction cockpit for winner odds.</div>',
        unsafe_allow_html=True,
    )

    model, model_error = load_model()
    metadata = read_metadata()
    teams, venues = load_choices()

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="match-chip">Match Setup</span>', unsafe_allow_html=True)
        team1 = st.selectbox("Team 1", options=teams, index=0)
        team2_options = [t for t in teams if t != team1]
        team2 = st.selectbox(
            "Team 2",
            options=team2_options,
            index=0 if team2_options else None,
        )
        venue = st.selectbox("Venue", options=venues, index=0)
        toss_winner = st.selectbox("Toss Winner", options=[team1, team2], index=0)
        toss_decision = st.radio(
            "Toss Decision",
            options=["bat", "field"],
            index=1,
            horizontal=True,
        )
        run = st.button("Predict Winner")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="match-chip">Prediction Feed</span>', unsafe_allow_html=True)
        if model is None and model_error is None:
            st.warning("Train the model first to enable predictions.")
            st.markdown(
                '<div class="helper">Run: .venv/bin/python src/train.py --data data/ipl_matches.csv --model-out models/ipl_winner_model.joblib</div>',
                unsafe_allow_html=True,
            )
        elif model is None and model_error is not None:
            st.error("Model file exists but cannot be loaded in this environment.")
            trained_python = metadata.get("python_version", "unknown")
            trained_sklearn = metadata.get("sklearn_version", "unknown")
            current_python = platform.python_version()
            current_sklearn = sklearn.__version__
            st.markdown(
                f'<div class="helper">Trained with Python {trained_python}, scikit-learn {trained_sklearn}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="helper">Current app uses Python {current_python}, scikit-learn {current_sklearn}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="helper">Fix: run .venv/bin/python src/train.py --data data/ipl_matches.csv --model-out models/ipl_winner_model.joblib and refresh the app.</div>',
                unsafe_allow_html=True,
            )
            st.caption(model_error)
        elif run:
            row = pd.DataFrame(
                [
                    {
                        "team1": team1,
                        "team2": team2,
                        "venue": venue,
                        "toss_winner": toss_winner,
                        "toss_decision": toss_decision,
                    }
                ]
            )
            winner = model.predict(row)[0]
            prob_map = {}
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(row)[0]
                for cls, pr in zip(model.classes_, probs):
                    prob_map[str(cls)] = float(pr)
            confidence = prob_map.get(str(winner))
            st.markdown('<div class="winner-card">', unsafe_allow_html=True)
            st.markdown('<div class="winner-title">Projected Winner</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="winner-name">{winner}</div>', unsafe_allow_html=True)
            if confidence is not None:
                st.markdown(
                    f'<div class="helper">Win confidence: {confidence * 100:.2f}%</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
            if prob_map:
                rank = pd.DataFrame(
                    [{"Team": k, "Probability": v} for k, v in prob_map.items()]
                ).sort_values("Probability", ascending=False)
                rank["Probability"] = rank["Probability"].map(lambda x: f"{x * 100:.2f}%")
                st.dataframe(rank, use_container_width=True, hide_index=True)
        else:
            st.info("Set the match context and hit Predict Winner.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
