import os
from botornot.config import (
    USE_EMBEDDINGS,
    INFERENCE_POST_FILES,
    INFERENCE_PARQUET_PATH,
    INFERENCE_PREVIEW_PATH,
)
from botornot.features import build_features_df

if __name__ == "__main__":
    os.makedirs(os.path.dirname(INFERENCE_PARQUET_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(INFERENCE_PREVIEW_PATH), exist_ok=True)

    final_df = build_features_df(
        INFERENCE_POST_FILES, bot_files=None, use_embeddings=USE_EMBEDDINGS
    )

    print(f"Saving {len(final_df)} rows to {INFERENCE_PARQUET_PATH}")
    final_df.to_parquet(INFERENCE_PARQUET_PATH, index=False)

    final_df.head(10).to_csv(INFERENCE_PREVIEW_PATH, index=False)
    print(f"Saved preview to {INFERENCE_PREVIEW_PATH}")
    print("Done.")
