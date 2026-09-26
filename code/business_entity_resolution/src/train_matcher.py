#!/usr/bin/env python3
"""
LightGBM Matcher Training for Business Entity Resolution.
Loads pairwise training and validation sets, trains a binary GBDT classifier,
evaluates on validation AUC with early stopping, reports top 15 features by gain,
and saves the model to data_mini/model_v1.txt.
Strictly country-agnostic.
"""

import os
import sys
import gc
import time
import argparse
from typing import List

import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb


FEATURE_COLS = [
    "token_jaccard_full",
    "token_jaccard_core",
    "char_3gram_cosine",
    "levenshtein_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "initials_match",
    "name_length_diff",
    "legal_suffix_agreement",
    "token_jaccard_addr",
    "is_s1_address_blank",
    "is_cand_address_blank",
    "idf_weighted_token_overlap_addr",
    "house_number_soft_match",
    "city_fuzzy_match",
    "rank_in_candidates",
    "score_gap_to_next_best",
    "num_candidates_for_this_s1",
    "name_token_rarity",
    "distinctive_token_jaccard",
    "shared_distinctive_token_count",
]


def train_model(
    features_path: str,
    out_model_path: str,
    learning_rate: float = 0.1,
    num_leaves: int = 63,
    min_child_samples: int = 50,
    max_depth: int = -1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    n_estimators: int = 300,
    early_stopping_rounds: int = 25,
    n_jobs: int = 6,
    seed: int = 42,
):
    t_start = time.time()
    print("=== Training LightGBM Matcher ===")
    print(f"Loading features from: {features_path}")

    load_cols = ["split", "label"] + FEATURE_COLS

    print("Loading TRAIN split (pushdown filter)...")
    train_tbl = pq.read_table(
        features_path,
        columns=load_cols,
        filters=[("split", "==", "train")],
    )
    print(f"Loaded {len(train_tbl):,} training rows.")

    print("Loading VALIDATION split (pushdown filter)...")
    val_tbl = pq.read_table(
        features_path,
        columns=load_cols,
        filters=[("split", "==", "val")],
    )
    print(f"Loaded {len(val_tbl):,} validation rows.")

    # Convert to numpy arrays
    X_train = train_tbl.select(FEATURE_COLS).to_pandas().to_numpy(dtype=np.float32)
    y_train = train_tbl["label"].to_numpy().astype(np.int8)

    X_val = val_tbl.select(FEATURE_COLS).to_pandas().to_numpy(dtype=np.float32)
    y_val = val_tbl["label"].to_numpy().astype(np.int8)

    # Free pyarrow tables
    del train_tbl, val_tbl
    gc.collect()

    pos_train = int(np.sum(y_train))
    pos_val = int(np.sum(y_val))
    print(f"Train set: {len(y_train):,} rows ({pos_train:,} positives, {len(y_train)-pos_train:,} negatives)")
    print(f"Val set:   {len(y_val):,} rows ({pos_val:,} positives, {len(y_val)-pos_val:,} negatives)")

    print("\nCreating LightGBM Datasets...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=FEATURE_COLS, free_raw_data=True)

    del X_train, y_train, X_val, y_val
    gc.collect()

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "subsample_freq": 1,
        "colsample_bytree": colsample_bytree,
        "n_jobs": n_jobs,
        "verbosity": -1,
        "random_state": seed,
    }

    print("\nTraining Booster...")
    evals_result = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=20),
        lgb.record_evaluation(evals_result),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    best_val_auc = evals_result["val"]["auc"][best_iter - 1] if best_iter > 0 else evals_result["val"]["auc"][-1]
    print(f"\nBest iteration: {best_iter} | Best Validation AUC: {best_val_auc:.5f}")

    # Feature Importance Top 15
    gain_importance = model.feature_importance(importance_type="gain")
    split_importance = model.feature_importance(importance_type="split")

    feat_imp = sorted(
        zip(FEATURE_COLS, gain_importance, split_importance),
        key=lambda x: -x[1],
    )

    print("\n--- Top 15 Features by Gain ---")
    print(f"{'Rank':<5}{'Feature':<35}{'Gain':>15}{'Splits':>10}")
    print("-" * 65)
    for rank, (fname, g_val, s_val) in enumerate(feat_imp[:15], 1):
        print(f"{rank:<5}{fname:<35}{g_val:>15.2f}{s_val:>10}")

    # Save model
    os.makedirs(os.path.dirname(os.path.abspath(out_model_path)), exist_ok=True)
    model.save_model(out_model_path)
    print(f"\nModel saved to: {out_model_path}")
    print(f"Training completed in {time.time() - t_start:.2f}s")
    return model, best_val_auc, feat_imp


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM entity matcher.")
    parser.add_argument(
        "--features-path",
        type=str,
        default="data_mini/pairwise_features.parquet",
    )
    parser.add_argument(
        "--out-model-path",
        type=str,
        default="data_mini/model_v1.txt",
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--early-stopping", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        features_path=args.features_path,
        out_model_path=args.out_model_path,
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
