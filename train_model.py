import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import shap


# config
INPUT_PATH = "training_data/user_features.parquet"
NUM_FOLDS = 5
SHOW_SHAP = False # set to true to see SHAP graph

# penalties/rewards from the bot-or-not challenge
REWARD_TP = 4   # catch a bot
PENALTY_FP = -2 # accuse a human
PENALTY_FN = -1 # miss a bot
REWARD_TN = 0 # ignore human

# find the best "cutoff" for identifying a human in the challenge
def calculate_profit(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    score = (tp * REWARD_TP) + (fp * PENALTY_FP) + (fn * PENALTY_FN) + (tn * REWARD_TN)
    return score

def find_optimal_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_score = -float("inf")
    
    # check 200 possible thresholds
    thresholds = np.linspace(0.01, 0.99, 200)
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        score = calculate_profit(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
            
    return best_thresh, best_score

def train_and_eval(X, y):
    fold_scores = []
    fold_thresholds = []
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # model
        
        ratio = (len(y_train) - sum(y_train)) / sum(y_train)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05, 
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # pred probs
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # instead of default 0.5, we find the best cutoff for this fold
        best_thresh, best_score = find_optimal_threshold(y_test, y_probs)
        
        y_pred_final = (y_probs >= best_thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
        
        fold_scores.append(best_score)
        fold_thresholds.append(best_thresh)
        
        print(f"Fold {fold}: Best thresh={best_thresh:.2f} | Score={best_score} | (TP={tp}, FP={fp}, FN={fn})")
        fold += 1
    
    print(f"Average profit per fold: {np.mean(fold_scores):.2f}")
    print(f"Optimal threshold (mean): {np.mean(fold_thresholds):.2f}")
    
    avg_thresh = np.mean(fold_thresholds)
    print(f"When running on the test set, use threshold: {avg_thresh:.3f}")
    print(f"i.e., if model.predict_proba() > {avg_thresh:.3f}, call it a BOT.")
    
    if SHOW_SHAP:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("What makes a bot a bot?", fontsize=16)
        plt.show()
    
def main():
    df = pd.read_parquet(INPUT_PATH)
    drop_cols = ["author_id", "is_bot"]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df["is_bot"]
    
    train_and_eval(X, y)

if __name__ == "__main__":
    main()