from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/spotify_dataset.csv")
OUT_PATH = Path("data/processed/spotify_processed.csv")

FEATURE_COLS = [
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

OPTIONAL_COLS = [
    "track_name",
    "artists",
    "track_genre",
]

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    keep_cols = [c for c in OPTIONAL_COLS + FEATURE_COLS if c in df.columns]
    df = df[keep_cols].copy()

    dedupe_cols = [c for c in ["track_name", "artists"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols)
    else:
        df = df.drop_duplicates()

    required = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=required)

    if "popularity" in df.columns:
        df = df[df["popularity"].between(0, 100)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("Saved processed data to:", OUT_PATH)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
