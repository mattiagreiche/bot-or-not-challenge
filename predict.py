import joblib
import pandas as pd
import numpy as np
from botornot.config import ARTIFACT_PATH, PREDICTED_BOT_IDS_PATH, INFERENCE_PARQUET_PATH

# load artifact once
ARTIFACT = joblib.load(ARTIFACT_PATH)

def predict_bot(new_data_df):
    # unpack
    models = ARTIFACT["models"]
    pca_tweets = ARTIFACT["pca_tweets"]
    pca_desc = ARTIFACT["pca_desc"]

    tweet_cols = ARTIFACT["tweet_emb_cols"]
    desc_cols = ARTIFACT["desc_emb_cols"]
    meta_cols = ARTIFACT["meta_cols"]
    threshold = ARTIFACT["threshold"]

    # transform using saved pca
    X_tweets_pca = pca_tweets.transform(new_data_df[tweet_cols])
    X_desc_pca = pca_desc.transform(new_data_df[desc_cols])

    # reconstruct
    n_tweet = X_tweets_pca.shape[1]
    n_desc = X_desc_pca.shape[1]
    df_tweets = pd.DataFrame(X_tweets_pca, columns=[f"PCA_Tweet_{i}" for i in range(n_tweet)], index=new_data_df.index)
    df_desc = pd.DataFrame(X_desc_pca, columns=[f"PCA_Desc_{i}" for i in range(n_desc)], index=new_data_df.index)

    df_meta = new_data_df[meta_cols].copy()

    X_final = pd.concat([df_meta, df_tweets, df_desc], axis=1)

    # five models
    total_probs = np.zeros(len(X_final))

    for i, model in enumerate(models):
        probs = model.predict_proba(X_final)[:, 1]
        total_probs += probs

    # average of five models
    avg_probs = total_probs / len(models)

    # final decision
    predictions = (avg_probs >= threshold).astype(int)

    # write user ids predicted to be bots
    bot_ids = new_data_df.loc[predictions == 1, "author_id"]
    with open(PREDICTED_BOT_IDS_PATH, "w") as f:
        for uid in bot_ids:
            f.write(f"{uid}\n")
    print(f"Wrote {len(bot_ids)} bot user ids to {PREDICTED_BOT_IDS_PATH}.")

    print(f"Prediction complete. Found {sum(predictions)} bots out of {len(predictions)} users.")
    return predictions, avg_probs


if __name__ == "__main__":
    df = pd.read_parquet(INFERENCE_PARQUET_PATH)
    predict_bot(df)
