import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.decomposition import PCA
from botornot.config import TRAINING_PARQUET_PATH, ARTIFACT_PATH

def train_final_ensemble(X, y):
    # cols
    tweet_emb_cols = [c for c in X.columns if c.startswith('emb_t')]
    desc_emb_cols = [c for c in X.columns if c.startswith('emb_d')]
    meta_cols = [c for c in X.columns if c not in tweet_emb_cols and c not in desc_emb_cols]

    # fit pca
    pca_tweets = PCA(n_components=10, random_state=2)
    pca_desc = PCA(n_components=10, random_state=2)

    X_tweets_pca = pca_tweets.fit_transform(X[tweet_emb_cols])
    X_desc_pca = pca_desc.fit_transform(X[desc_emb_cols])

    # create dfs
    df_tweets = pd.DataFrame(X_tweets_pca, columns=[f"PCA_Tweet_{i}" for i in range(10)], index=X.index)
    df_desc = pd.DataFrame(X_desc_pca, columns=[f"PCA_Desc_{i}" for i in range(10)], index=X.index)

    X_final = pd.concat([X[meta_cols], df_tweets, df_desc], axis=1)

    # train five models (diff seeds)
    ratio = (len(y) - sum(y)) / sum(y)
    seeds = [12, 17, 39, 67, 71]
    models = []

    for i, seed in enumerate(seeds):
        print(f"Training Model {i+1}/5 (Seed {seed})...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            random_state=seed
        )
        model.fit(X_final, y)
        models.append(model)

    THRESHOLD = 0.45 # round about average value I got from multiple CV tests with different random states

    # save everything
    artifact = {
        "models": models,
        "pca_tweets": pca_tweets,
        "pca_desc": pca_desc,
        "tweet_emb_cols": tweet_emb_cols,
        "desc_emb_cols": desc_emb_cols,
        "meta_cols": meta_cols,
        "threshold": THRESHOLD
    }

    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)

    print("Saved ensemble artifact.")


if __name__ == "__main__":
    df = pd.read_parquet(TRAINING_PARQUET_PATH)
    drop_cols = ["author_id", "is_bot"]
    features = [c for c in df.columns if c not in drop_cols]
    X = df[features]
    y = df["is_bot"]
    train_final_ensemble(X, y)
