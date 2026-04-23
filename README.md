# IPL Predictor (Baseline)

This project trains a baseline IPL match winner prediction model using historical match-level data.

## What it does

- Loads historical IPL match data from a CSV file.
- Builds simple pre-match features:
  - `team1`
  - `team2`
  - `venue`
  - `toss_winner`
  - `toss_decision`
- Trains a classification model to predict `winner`.
- Saves the trained model to `models/ipl_winner_model.joblib`.

## Kaggle dataset

Source:

`https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download dataset (Kaggle CLI)

1. Create a Kaggle API token from your Kaggle account and save it as:
   - macOS/Linux: `~/.kaggle/kaggle.json`
2. Set correct permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

3. Download and unzip dataset:

```bash
kaggle datasets download -d chaitu20/ipl-dataset2008-2025 -p data/raw --unzip
```

4. Normalize columns into training file:

```bash
python src/prepare_kaggle_data.py --raw-dir data/raw --out data/ipl_matches.csv
```

## Expected training format

Training script reads:

`data/ipl_matches.csv`

Required columns:

- `team1`
- `team2`
- `venue`
- `toss_winner`
- `toss_decision`
- `winner`

## Train model

```bash
python src/train.py --data data/ipl_matches.csv --model-out models/ipl_winner_model.joblib
```

## Make a prediction

```bash
python src/predict.py \
  --model models/ipl_winner_model.joblib \
  --team1 "Mumbai Indians" \
  --team2 "Chennai Super Kings" \
  --venue "Wankhede Stadium" \
  --toss-winner "Mumbai Indians" \
  --toss-decision "field"
```

## Run the UI

```bash
streamlit run src/ui.py
```

## Next improvements

- Add season/date features and recent form features.
- Add player-level features from playing XI.
- Use probability calibration.
- Compare with XGBoost/LightGBM.
