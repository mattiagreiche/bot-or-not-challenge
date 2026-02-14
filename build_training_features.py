import os
from botornot.config import (
    USE_EMBEDDINGS,
    TRAINING_POST_FILES,
    TRAINING_BOT_FILES,
    TRAINING_PARQUET_PATH,
    TRAINING_PREVIEW_PATH,
)
from botornot.features import build_features_df

if __name__ == "__main__":
    os.makedirs(os.path.dirname(TRAINING_PARQUET_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TRAINING_PREVIEW_PATH), exist_ok=True)

    final_df = build_features_df(
        TRAINING_POST_FILES, bot_files=TRAINING_BOT_FILES, use_embeddings=USE_EMBEDDINGS
    )

    print(f"Saving {len(final_df)} rows to {TRAINING_PARQUET_PATH}")
    final_df.to_parquet(TRAINING_PARQUET_PATH, index=False)

    final_df.head(10).to_csv(TRAINING_PREVIEW_PATH, index=False)
    print(f"Saved preview to {TRAINING_PREVIEW_PATH}")
    print("Done.")
