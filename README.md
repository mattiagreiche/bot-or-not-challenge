# bot-or-not-challenge

Bot detection for the bot-or-not challenge: classify users as bot or human from their posts and profile. All paths and flags are configured in `botornot/config.py` (no CLI args).

## Install

```bash
pip install -r requirements.txt
```

Run from the repo root so that the `botornot` package is on the path.

## Pipeline

1. **Build training features**  
   `python build_training_features.py`  
   Reads raw training data, builds user features (with or without embeddings per `USE_EMBEDDINGS` in config), writes parquet and preview under `training_data/`.

2. **Build inference features**  
   `python build_inference_features.py`  
   Same for inference input; output under `inference_data/`. Set `INFERENCE_POST_FILES` in `botornot/config.py` to your inference JSON(s).

3. **Cross-validation**  
   `python cv_eval.py`  
   Loads training parquet from config, runs 5-fold CV, finds optimal threshold for the challenge reward, optionally shows SHAP.

4. **Train final model**  
   `python train_final.py`  
   Loads training parquet from config, trains 5-model ensemble, saves artifact to `ARTIFACT_PATH` (see config).

5. **Predict**  
   `python predict.py`  
   Loads the inference parquet from config (`INFERENCE_PARQUET_PATH`), runs the model, and writes bot user IDs to `PREDICTED_BOT_IDS_PATH`. Run `build_inference_features.py` first so that parquet exists.

All input/output paths and `USE_EMBEDDINGS` are set in `botornot/config.py`.
