import os
import re
import string
import json
import glob
import numpy as np
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

HASHTAG_RE = re.compile(r'#\w+')

def count_caps(s):
    return sum(1 for c in s if 'A' <= c <= 'Z')

def count_punct(s):
    return sum(1 for c in s if c in string.punctuation)

def count_hashtags(s):
    return len(HASHTAG_RE.findall(s or "")) # avoids empty string errors


# CORE feature extraction

def extract_user_features_from_file(path):
    
    with open(path) as f:
        data = json.load(f)
        
    posts = pd.DataFrame(data["posts"])
    
    # should already be sorted but ensure
    posts["created_at"] = pd.to_datetime(posts["created_at"], errors="coerce")
    posts = posts.sort_values(["created_at"])
    
    # per-post features
    posts["len"] = posts["text"].fillna(0).str.len()
    posts["num_caps"] = posts["text"].fillna(0).map(count_caps)
    posts["num_punct"] = posts["text"].fillna(0).map(count_punct)
    posts["num_hashtags"] = posts["text"].fillna(0).map(count_hashtags)
    
    # time between posts
    posts["delta_s"] = (
        posts.groupby("id")["created_at"]
        .diff()
        .dt.total_seconds()
    )
    
    # aggregate per user and get a bunch of metrics
    agg = posts.groupby("id").agg({
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
    
    agg = agg.rename(columns={
        "text_count": "post_count"
    })
    
    # activity
    agg["active_span_s"] = (
        agg["created_at_max"] - agg["created_at_min"]
    ).dt.total_seconds()
    
    agg["posts_per_day"] = (
        agg["post_count"] / (agg["active_span_s"] / 86400 + 1e-9) # avoid division by zero if one post
    )
    
    agg = agg.drop(columns=[
        "created_at_min",
        "created_at_max"
    ])
    
    return agg