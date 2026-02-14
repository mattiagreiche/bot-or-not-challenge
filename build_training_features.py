import os
import build_user_features

USE_EMBEDDINGS = False

POST_FILES = [
    "raw_data/dataset.posts&users.30.json",
    "raw_data/dataset.posts&users.31.json",
    "raw_data/dataset.posts&users.32.json",
    "raw_data/dataset.posts&users.33.json",
]

BOT_FILES = [
    "raw_data/dataset.bots.30.txt",
    "raw_data/dataset.bots.31.txt",
    "raw_data/dataset.bots.32.txt",
    "raw_data/dataset.bots.33.txt",
]

if USE_EMBEDDINGS:
    OUTPUT_PATH = "training_data/user_features.parquet"
    PREVIEW_PATH = "training_data/user_features_preview.csv"
else:
    OUTPUT_PATH = "training_data/user_features_no_emb.parquet"
    PREVIEW_PATH = "training_data/user_features_no_emb_preview.csv"

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PREVIEW_PATH), exist_ok=True)

    final_df = build_user_features.build_features_df(
        POST_FILES, bot_files=BOT_FILES, use_embeddings=USE_EMBEDDINGS
    )

    print(f"Saving {len(final_df)} rows to {OUTPUT_PATH}")
    final_df.to_parquet(OUTPUT_PATH, index=False)

    final_df.head(10).to_csv(PREVIEW_PATH, index=False)
    print(f"Saved preview to {PREVIEW_PATH}")
    print("Done.")
