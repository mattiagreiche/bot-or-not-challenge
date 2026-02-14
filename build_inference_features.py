import os
import build_user_features

USE_EMBEDDINGS = True

POST_FILES = [
    "raw_data/inference.posts&users.json",
]

if USE_EMBEDDINGS:
    OUTPUT_PATH = "inference_data/user_features.parquet"
    PREVIEW_PATH = "inference_data/user_features_preview.csv"
else:
    OUTPUT_PATH = "inference_data/user_features_no_emb.parquet"
    PREVIEW_PATH = "inference_data/user_features_no_emb_preview.csv"

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PREVIEW_PATH), exist_ok=True)

    final_df = build_user_features.build_features_df(
        POST_FILES, bot_files=None, use_embeddings=USE_EMBEDDINGS
    )

    print(f"Saving {len(final_df)} rows to {OUTPUT_PATH}")
    final_df.to_parquet(OUTPUT_PATH, index=False)

    final_df.head(10).to_csv(PREVIEW_PATH, index=False)
    print(f"Saved preview to {PREVIEW_PATH}")
    print("Done.")
