from collections import Counter
import re
import string
import json
import math
import pandas as pd

# hi! this file extracts the features from the four datasets and puts them into a parquet file for use in the XGBoost model.
# currently planning to create a function in a different file and using an open-source language model (tuned on twitter) to
# get average features of a user's tweets/text, which should prove useful.

# config

POST_FILES = [
    "training_data/dataset.posts&users.30.json",
    "training_data/dataset.posts&users.31.json",
    "training_data/dataset.posts&users.32.json",
    "training_data/dataset.posts&users.33.json",
]

BOT_FILES = [
    "training_data/dataset.bots.30.txt",
    "training_data/dataset.bots.31.txt",
    "training_data/dataset.bots.32.txt",
    "training_data/dataset.bots.33.txt",
]

OUTPUT_PATH = "training_data/user_features.parquet"


# text helpers
URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r'#\w+')
MENTION_RE = re.compile(r"@\w+")

def count_caps(s):
    return sum(1 for c in s if 'A' <= c <= 'Z')

def count_punct(s):
    return sum(1 for c in s if c in string.punctuation)

def count_hashtags(s):
    return len(HASHTAG_RE.findall(s or "")) # avoids empty string errors

def char_entropy(s):
    # shannon entropy
    if not s:
        return 0.0
    cnt = Counter(s)
    L = len(s)
    ent = 0.0
    for c in cnt.values():
        p = c / L
        ent -= p * math.log2(p)
    return ent


# language model integration

def get_text_embeddings(text_series):
    pass

def get_description_embeddings(description_series):
    pass


# core feature extraction

def extract_features_from_posts(posts_list):
    # users_list is the 'users' JSON
    posts = pd.DataFrame(posts_list)
    
    # pre-processing
    posts["created_at"] = pd.to_datetime(posts["created_at"], errors="coerce")
    posts["text"] = posts["text"].fillna("")
    
    # sort by user then time for time-diff calculations
    posts = posts.sort_values(["author_id", "created_at"])
    
    # per-post features
    posts["len"] = posts["text"].str.len()
    posts["num_caps"] = posts["text"].map(count_caps)
    posts["num_punct"] = posts["text"].map(count_punct)
    posts["num_hashtags"] = posts["text"].map(count_hashtags)
    
    # time between posts
    posts["delta_s"] = posts.groupby("author_id")["created_at"].diff().dt.total_seconds()
    
    # INSERT LM POST EMBEDDINGS HERE LATER
    # post_embeddings = get_text_embeddings(posts["text"])
    # posts = pd.concat([posts, post_embeddings], axis=1)
    
    # aggregate per user and get a bunch of metrics
    agg = posts.groupby("author_id").agg({
        "delta_s": ["mean", "min", "std", "median"],
        "len": ["mean", "min", "max", "std"],
        "num_caps": ["mean", "min", "max", "std"],
        "num_punct": ["mean", "min", "max", "std"],
        "num_hashtags": ["mean", "min", "max", "std"],
        "text": "count",
        "created_at": ["min", "max"],
    })
    
    # flatten columns
    agg.columns = ["_".join(col).strip() for col in agg.columns]
    agg = agg.reset_index()
    agg = agg.rename(columns={"text_count": "post_count"})
    
    # activity
    agg["active_span_s"] = (agg["created_at_max"] - agg["created_at_min"]).dt.total_seconds()
    
    agg["posts_per_day"] = agg["post_count"] / (agg["active_span_s"] / 86400 + 1e-9) # avoid division by zero if one post
    
    # fix delta_s to 0 for single posters
    delta_cols = [c for c in agg.columns if c.startswith("delta_s_")]
    for col in delta_cols:
        agg[col] = agg[col].fillna(0)

    # for single posters, posts_per_day = 1/2
    agg["single_post"] = (agg["post_count"] == 1).astype(int)
    agg.loc[agg["single_post"] == 1, "posts_per_day"] = 0.5 # time period is 2 days in the datasets    
    
    agg = agg.drop(columns=["created_at_min", "created_at_max"])
    agg = agg.fillna(0) # fill NaNs from std dev calculations on single posts
    
    return agg

def extract_features_from_users(users_list):
    users = pd.DataFrame(users_list)
    
    # id match
    if "id" in users.columns:
        users = users.rename(columns={"id": "author_id"})
        
    # username
    u = users.get("username", "").fillna("")
    users["username_len"] = u.str.len()
    users["username_digits"] = u.str.count(r"\d")
    users["username_entropy"] = u.apply(lambda s: char_entropy(str(s)) if pd.notna(s) else 0.0)

    # name features
    n = users.get("name", "").fillna("")
    users["name_len"] = n.str.len()
    users["name_nonalpha"] = n.str.count(r"[^A-Za-z0-9_]")
    users["name_word_count"] = n.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)  
    
    # description features
    d = users.get("description", "").fillna("")
    users["description_len"] = d.str.len()
    users["description_has_url"] = d.str.contains(URL_RE).fillna(False).astype(int)
    users["description_has_hashtag"] = d.str.contains(HASHTAG_RE).fillna(False).astype(int)
    users["description_has_mention"] = d.str.contains(MENTION_RE).fillna(False).astype(int)
    users["missing_description"] = (d == "").astype(int)    
    
    # location features
    loc = users.get("location", "").fillna("")
    users["location_present"] = (loc != "").astype(int)
    users["location_len"] = loc.str.len()
    
    # keep only useful features + author_id
    keep_cols = [
        "author_id",
        "username_len", "username_digits", "username_entropy",
        "name_len", "name_nonalpha", "name_word_count", 
        "description_len", "description_has_url", "description_has_hashtag", 
        "description_has_mention", "missing_description",
        "location_present", "location_len",
    ]
    
    users = users[keep_cols]
    
        
    # INSERT LM DESCRIPTION EMBEDDINGS HERE LATER
    # desc_embeddings = get_description_embeddings(users["description"])
    # users = pd.concat([users, desc_embeddings], axis=1)

    return users

def process_single_file(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # extract posts features
    posts_data = data.get('posts', []) 
    posts_df = extract_features_from_posts(posts_data)
    
    # extract users features
    users_data = data.get('users', [])
    users_df = extract_features_from_users(users_data)
    
    # merge (inner join to ensure only keep users we have full data for)
    merged = pd.merge(users_df, posts_df, on="author_id", how="inner")
    return merged
    


# load bot labels
def load_bot_labels(bot_files):
    bot_ids = set()
    for path in bot_files:
        with open(path) as f:
            for line in f:
                bot_ids.add(line.strip())
    return bot_ids

# main

def main():
    all_features = []
    
    # process all data files
    for post_file in POST_FILES:
        df = process_single_file(post_file)
        all_features.append(df)
        
    final_df = pd.concat(all_features, ignore_index=True)
    
    # add Labels
    print("Loading bot labels...")
    bot_ids = load_bot_labels(BOT_FILES)
    
    # save
    print(f"Saving {len(final_df)} rows to {OUTPUT_PATH}")
    final_df.to_parquet(OUTPUT_PATH, index=False)
    
    # prev
    preview_path = "sample_preview.csv"
    final_df.head(100).to_csv(preview_path, index=False)
    print(f"Saved preview to {preview_path}")
    print("Done.")

if __name__ == "__main__":
    main()