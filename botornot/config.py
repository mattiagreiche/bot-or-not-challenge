import os

# directories
DIR_RAW = "raw_data"
DIR_TRAINING = "training_data"
DIR_INFERENCE = "inference_data"
DIR_MODELS = "models"

# feature building
# note I was in a bit of a rush, so it's entirely possible using USE_EMBEDDINGS=False right now will result
# in all kinds of errors. (just use it, it's the whole interesting part of this model!)
USE_EMBEDDINGS = True

# training data file lists (paths under DIR_RAW)
TRAINING_POST_FILES = [
    os.path.join(DIR_RAW, "dataset.posts&users.30.json"),
    os.path.join(DIR_RAW, "dataset.posts&users.31.json"),
    os.path.join(DIR_RAW, "dataset.posts&users.32.json"),
    os.path.join(DIR_RAW, "dataset.posts&users.33.json"),
]
TRAINING_BOT_FILES = [
    os.path.join(DIR_RAW, "dataset.bots.30.txt"),
    os.path.join(DIR_RAW, "dataset.bots.31.txt"),
    os.path.join(DIR_RAW, "dataset.bots.32.txt"),
    os.path.join(DIR_RAW, "dataset.bots.33.txt"),
]

# inference data file list
INFERENCE_POST_FILES = [
    os.path.join(DIR_RAW, "inference.posts&users.json"), # replace with the inference file at 12pm
]

# derived paths: training outputs
if USE_EMBEDDINGS:
    TRAINING_PARQUET_PATH = os.path.join(DIR_TRAINING, "user_features.parquet")
    TRAINING_PREVIEW_PATH = os.path.join(DIR_TRAINING, "user_features_preview.csv")
else:
    TRAINING_PARQUET_PATH = os.path.join(DIR_TRAINING, "user_features_no_emb.parquet")
    TRAINING_PREVIEW_PATH = os.path.join(DIR_TRAINING, "user_features_no_emb_preview.csv")

# derived paths: inference outputs
if USE_EMBEDDINGS:
    INFERENCE_PARQUET_PATH = os.path.join(DIR_INFERENCE, "user_features.parquet")
    INFERENCE_PREVIEW_PATH = os.path.join(DIR_INFERENCE, "user_features_preview.csv")
else:
    INFERENCE_PARQUET_PATH = os.path.join(DIR_INFERENCE, "user_features_no_emb.parquet")
    INFERENCE_PREVIEW_PATH = os.path.join(DIR_INFERENCE, "user_features_no_emb_preview.csv")

# model artifact and prediction output
ARTIFACT_PATH = os.path.join(DIR_MODELS, "bot_detector.pkl")
PREDICTED_BOT_IDS_PATH = "predicted_bot_ids.txt"
