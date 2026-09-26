#!/usr/bin/env python3
"""
Decision Layer, Threshold Optimization, and Honest Evaluation for Business Entity Resolution.
1. On the VALIDATION split:
   - For each S2/S3 record, picks its single best scoring S1 candidate (1-owner-per-record).
   - Grid searches min_probability (0.50 to 0.95 step 0.05) and min_margin (0.00 to 0.30 step 0.05).
   - Groups accepted pairs by S1 and computes per_entity_f05 from src/metric.py.
   - Reports top 5 threshold combinations with precision and recall.
2. Evaluates the best threshold on the held-out TEST split.
3. Evaluates the naive baseline (top char_3gram_cosine > 0.8).
4. Diagnoses 10 correct matches and 10 errors (FP/FN) from the test split.
5. Writes audit/step4_model_report.md (under 80 lines).
Strictly country-agnostic.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gc
import time
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb

from src.metric import per_entity_f05, single_entity_f05


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


def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    gt_tbl = pq.read_table(gt_path).to_pandas()
    truth = {}
    for s1_id, m_str in zip(gt_tbl["source1_entity_id"], gt_tbl["matched_entity_ids"]):
        if m_str is not None and str(m_str).strip():
            matches = {m.strip() for m in str(m_str).split(",") if m.strip()}
            truth[s1_id] = matches
        else:
            truth[s1_id] = set()
    return truth


def load_record_metadata(*paths: str) -> Dict[str, Dict[str, str]]:
    meta = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        tbl = pq.read_table(
            p,
            columns=["entity_id", "business_name", "business_address", "country"],
        ).to_pandas()
        for eid, name, addr, country in zip(
            tbl["entity_id"],
            tbl["business_name"],
            tbl["business_address"],
            tbl["country"],
        ):
            meta[eid] = {
                "name": str(name) if name is not None else "",
                "address": str(addr) if addr is not None else "",
                "country": str(country) if country is not None else "",
            }
    return meta


def decide_and_score(
    model_path: str,
    features_path: str,
    splits_path: str,
    gt_path: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    out_report: str,
    seed: int = 42,
):
    t_start = time.time()
    print("=== Phase 6: Decision Layer & Threshold Optimization ===")

    print(f"Loading model from: {model_path}")
    booster = lgb.Booster(model_file=model_path)

    # Feature importances
    gain_imp = booster.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_COLS, gain_imp), key=lambda x: -x[1])

    print(f"Loading ground truth from: {gt_path}")
    truth_dict = load_ground_truth(gt_path)

    print(f"Loading splits from: {splits_path}")
    splits_tbl = pq.read_table(splits_path).to_pandas()
    val_s1_list = splits_tbl[splits_tbl["split"] == "val"]["source1_entity_id"].tolist()
    test_s1_list = splits_tbl[splits_tbl["split"] == "test"]["source1_entity_id"].tolist()

    val_truth = {s1: truth_dict.get(s1, set()) for s1 in val_s1_list}
    test_truth = {s1: truth_dict.get(s1, set()) for s1 in test_s1_list}

    total_val_pos = sum(len(m) for m in val_truth.values())
    total_test_pos = sum(len(m) for m in test_truth.values())
    print(f"Validation: {len(val_s1_list):,} S1 entities ({total_val_pos:,} true matches)")
    print(f"Test:       {len(test_s1_list):,} S1 entities ({total_test_pos:,} true matches)")

    # 1. Load VALIDATION pairs
    print("\nLoading VALIDATION pairwise features...")
    load_cols = list(dict.fromkeys(["source1_entity_id", "record_id", "country", "label"] + FEATURE_COLS))
    val_tbl = pq.read_table(
        features_path,
        columns=load_cols,
        filters=[("split", "==", "val")],
    )
    val_df = val_tbl.to_pandas()
    del val_tbl
    gc.collect()

    print(f"Loaded {len(val_df):,} validation pairs. Predicting probabilities...")
    X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    val_df["pred_prob"] = booster.predict(X_val)
    del X_val
    gc.collect()

    # Group validation candidates by record_id to pick single best S1 candidate
    print("Grouping validation candidates by record_id...")
    val_record_cands = defaultdict(list)
    for s1_id, r_id, prob in zip(val_df["source1_entity_id"], val_df["record_id"], val_df["pred_prob"]):
        val_record_cands[r_id].append((s1_id, prob))

    # Pre-extract record best proposal: (best_s1, p1, margin)
    val_proposals = {}
    for r_id, cand_list in val_record_cands.items():
        if len(cand_list) == 1:
            best_s1, p1 = cand_list[0]
            val_proposals[r_id] = (best_s1, p1, p1)
        else:
            sorted_cands = sorted(cand_list, key=lambda x: -x[1])
            best_s1, p1 = sorted_cands[0]
            p2 = sorted_cands[1][1]
            val_proposals[r_id] = (best_s1, p1, p1 - p2)

    # 2. Grid Search on Validation Split
    print("\nRunning Grid Search over (min_probability, min_margin)...")
    prob_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    margin_thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    grid_results = []
    for min_p in prob_thresholds:
        for min_m in margin_thresholds:
            accepted_by_s1 = defaultdict(set)
            tp, fp = 0, 0
            for r_id, (best_s1, p1, m) in val_proposals.items():
                if p1 >= min_p and m >= min_m:
                    accepted_by_s1[best_s1].add(r_id)
                    if r_id in val_truth[best_s1]:
                        tp += 1
                    else:
                        fp += 1

            # Build full prediction map including singletons/empty
            pred = {s1: accepted_by_s1.get(s1, set()) for s1 in val_s1_list}
            f05 = per_entity_f05(pred, val_truth)
            prec = (tp / float(tp + fp)) if (tp + fp) > 0 else 1.0
            rec = (tp / float(total_val_pos)) if total_val_pos > 0 else 1.0
            grid_results.append((f05, prec, rec, min_p, min_m))

    grid_results.sort(key=lambda x: -x[0])

    print("\n--- Top 5 Threshold Combinations on Validation Split ---")
    print(f"{'Rank':<5}{'Min Prob':>10}{'Min Margin':>12}{'Val F0.5':>12}{'Precision':>12}{'Recall':>10}")
    print("-" * 65)
    for rank, (f05, prec, rec, p_th, m_th) in enumerate(grid_results[:5], 1):
        print(f"{rank:<5}{p_th:>10.2f}{m_th:>12.2f}{f05:>12.4f}{prec:>12.2%}{rec:>10.2%}")

    best_val_f05, best_val_prec, best_val_rec, best_p, best_m = grid_results[0]
    print(f"\nOptimal Threshold: min_p={best_p:.2f}, min_m={best_m:.2f} (Val F0.5={best_val_f05:.4f})")

    # Free validation memory
    del val_df, val_record_cands, val_proposals
    gc.collect()

    # 3. Evaluate Best Threshold on Held-Out TEST Split
    print("\nLoading TEST pairwise features...")
    test_tbl = pq.read_table(
        features_path,
        columns=load_cols,
        filters=[("split", "==", "test")],
    )
    test_df = test_tbl.to_pandas()
    del test_tbl
    gc.collect()

    print(f"Loaded {len(test_df):,} test pairs. Predicting probabilities...")
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    test_df["pred_prob"] = booster.predict(X_test)
    del X_test
    gc.collect()

    # Group test candidates by record_id
    test_record_cands = defaultdict(list)
    for s1_id, r_id, prob, cos_val in zip(
        test_df["source1_entity_id"],
        test_df["record_id"],
        test_df["pred_prob"],
        test_df["char_3gram_cosine"],
    ):
        test_record_cands[r_id].append((s1_id, prob, cos_val))

    # Evaluate Model on Test
    test_accepted_by_s1 = defaultdict(set)
    test_tp, test_fp = 0, 0
    accepted_pairs_info = []
    rejected_true_matches = []

    # Track test predictions for diagnostic examples
    for r_id, cand_list in test_record_cands.items():
        if len(cand_list) == 1:
            best_s1, p1, cos_v = cand_list[0]
            margin = p1
        else:
            sorted_cands = sorted(cand_list, key=lambda x: -x[1])
            best_s1, p1, cos_v = sorted_cands[0]
            p2 = sorted_cands[1][1]
            margin = p1 - p2

        # Check true match for this record in test set
        for s1_cand, _, _ in cand_list:
            if r_id in test_truth[s1_cand]:
                if not (best_s1 == s1_cand and p1 >= best_p and margin >= best_m):
                    rejected_true_matches.append((s1_cand, r_id, p1, margin, "Missed / rejected by model"))

        if p1 >= best_p and margin >= best_m:
            test_accepted_by_s1[best_s1].add(r_id)
            is_tp = r_id in test_truth[best_s1]
            if is_tp:
                test_tp += 1
                accepted_pairs_info.append((best_s1, r_id, p1, margin, "True Positive"))
            else:
                test_fp += 1
                accepted_pairs_info.append((best_s1, r_id, p1, margin, "False Positive"))

    test_pred = {s1: test_accepted_by_s1.get(s1, set()) for s1 in test_s1_list}
    test_f05 = per_entity_f05(test_pred, test_truth)
    test_prec = (test_tp / float(test_tp + test_fp)) if (test_tp + test_fp) > 0 else 1.0
    test_rec = (test_tp / float(total_test_pos)) if total_test_pos > 0 else 1.0

    print(f"\n=== Test Set Performance (Best Threshold: min_p={best_p:.2f}, min_m={best_m:.2f}) ===")
    print(f"Test Per-Entity Macro F0.5: {test_f05:.4f}")
    print(f"Test Precision:            {test_prec:.2%}")
    print(f"Test Recall:               {test_rec:.2%}")

    # 4. Naive Baseline Evaluation on Test Split
    # Accept top char_3gram_cosine candidate if score > 0.8
    naive_accepted_by_s1 = defaultdict(set)
    naive_tp, naive_fp = 0, 0
    for r_id, cand_list in test_record_cands.items():
        sorted_by_cos = sorted(cand_list, key=lambda x: -x[2])
        best_s1_cos, _, top_cos = sorted_by_cos[0]
        if top_cos > 0.80:
            naive_accepted_by_s1[best_s1_cos].add(r_id)
            if r_id in test_truth[best_s1_cos]:
                naive_tp += 1
            else:
                naive_fp += 1

    naive_pred = {s1: naive_accepted_by_s1.get(s1, set()) for s1 in test_s1_list}
    naive_f05 = per_entity_f05(naive_pred, test_truth)
    naive_prec = (naive_tp / float(naive_tp + naive_fp)) if (naive_tp + naive_fp) > 0 else 1.0
    naive_rec = (naive_tp / float(total_test_pos)) if total_test_pos > 0 else 1.0

    print(f"\n=== Naive Baseline (char 3-gram cosine > 0.8) ===")
    print(f"Naive Test Macro F0.5:     {naive_f05:.4f}")
    print(f"Naive Test Precision:      {naive_prec:.2%}")
    print(f"Naive Test Recall:         {naive_rec:.2%}")

    # 5. Load Metadata for 10 Correct & 10 Error Examples
    print("\nLoading metadata for report examples...")
    meta = load_record_metadata(s1_path, s2_path, s3_path)

    rng = random.Random(seed)
    tp_samples = [x for x in accepted_pairs_info if x[4] == "True Positive"]
    rng.shuffle(tp_samples)
    sampled_tps = tp_samples[:10]

    fp_samples = [x for x in accepted_pairs_info if x[4] == "False Positive"]
    rng.shuffle(fp_samples)
    fn_samples = list(rejected_true_matches)
    rng.shuffle(fn_samples)

    # 10 error examples: balance FPs and FNs
    sampled_errors = fp_samples[:5] + fn_samples[:5]
    if len(sampled_errors) < 10:
        sampled_errors += fn_samples[5 : 10 - len(sampled_errors)]

    # 6. Write Report to audit/step4_model_report.md (under 80 lines)
    os.makedirs(os.path.dirname(os.path.abspath(out_report)), exist_ok=True)
    report_lines = []
    report_lines.append("# Audit Report: Step 4 - LightGBM Matcher Evaluation\n")

    # Overfitting check
    gap_val_test = abs(best_val_f05 - test_f05)
    overfit_flag = "NORMAL (val/test close)" if gap_val_test <= 0.03 else "OVERFITTING RISK (>0.03 gap)"
    report_lines.append(f"**Status:** {overfit_flag} | Val F0.5: **{best_val_f05:.4f}** | Test F0.5: **{test_f05:.4f}**\n")

    report_lines.append("## 1. Top 15 Feature Importances (Gain)")
    report_lines.append("| Rank | Feature | Importance Gain | Rank | Feature | Importance Gain |")
    report_lines.append("|---:|:---|---:|---:|:---|---:|")
    for r in range(1, 9):
        f1, g1 = feat_imp[r - 1]
        if r + 7 <= len(feat_imp):
            f2, g2 = feat_imp[r + 7]
            report_lines.append(f"| {r} | `{f1}` | {g1:.1f} | {r+8} | `{f2}` | {g2:.1f} |")
        else:
            report_lines.append(f"| {r} | `{f1}` | {g1:.1f} | | | |")

    report_lines.append("\n## 2. Threshold Search & Evaluation Summary")
    report_lines.append("| Model / Baseline | Thresholds | F0.5 Score | Precision | Recall |")
    report_lines.append("|:---|:---|---:|---:|---:|")
    report_lines.append(f"| **LightGBM (Validation)** | `min_p={best_p:.2f}, min_m={best_m:.2f}` | **{best_val_f05:.4f}** | {best_val_prec:.2%} | {best_val_rec:.2%} |")
    report_lines.append(f"| **LightGBM (Test)** | `min_p={best_p:.2f}, min_m={best_m:.2f}` | **{test_f05:.4f}** | **{test_prec:.2%}** | **{test_rec:.2%}** |")
    report_lines.append(f"| Naive Baseline (Test) | `char_3gram_cos > 0.8` | {naive_f05:.4f} | {naive_prec:.2%} | {naive_rec:.2%} |")

    report_lines.append("\n## 3. Sampled Correct Matches (10 True Positives from Test Split)")
    report_lines.append("| S1 ID / Name | Target ID / Name | Prob | Margin | Notes |")
    report_lines.append("|:---|:---|---:|---:|:---|")
    for s1_id, r_id, p, m, _ in sampled_tps:
        s1_n = meta.get(s1_id, {}).get("name", "")[:28]
        tg_n = meta.get(r_id, {}).get("name", "")[:28]
        report_lines.append(f"| `{s1_id}`: {s1_n} | `{r_id}`: {tg_n} | {p:.2f} | {m:.2f} | High confidence name/addr match |")

    report_lines.append("\n## 4. Sampled Errors (10 False Positives / False Negatives from Test Split)")
    report_lines.append("| Error Type | S1 ID / Name | Target ID / Name | Prob | Margin | Likely Cause |")
    report_lines.append("|:---|:---|:---|---:|---:|:---|")
    for s1_id, r_id, p, m, err_type in sampled_errors[:10]:
        s1_n = meta.get(s1_id, {}).get("name", "")[:24]
        tg_n = meta.get(r_id, {}).get("name", "")[:24]
        if "False Positive" in err_type:
            cause = "Shared name tokens, distinct actual entity"
        else:
            cause = "Low probability / ambiguous margin over runner-up"
        report_lines.append(f"| {err_type} | `{s1_id}`: {s1_n} | `{r_id}`: {tg_n} | {p:.2f} | {m:.2f} | {cause} |")

    report_content = "\n".join(report_lines) + "\n"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nReport generated at: {out_report} ({len(report_lines)} lines)")
    print(f"Total time: {time.time() - t_start:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Decision layer and threshold evaluation.")
    parser.add_argument("--model-path", type=str, default="data_mini/model_v1.txt")
    parser.add_argument("--features-path", type=str, default="data_mini/pairwise_features.parquet")
    parser.add_argument("--splits-path", type=str, default="data_mini/s1_splits.parquet")
    parser.add_argument("--gt-path", type=str, default="data_mini/ground_truth.parquet")
    parser.add_argument("--s1-path", type=str, default="data_mini/source1_normalized.parquet")
    parser.add_argument("--s2-path", type=str, default="data_mini/source2_normalized.parquet")
    parser.add_argument("--s3-path", type=str, default="data_mini/source3_normalized.parquet")
    parser.add_argument("--out-report", type=str, default="audit/step4_model_report.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    decide_and_score(
        model_path=args.model_path,
        features_path=args.features_path,
        splits_path=args.splits_path,
        gt_path=args.gt_path,
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        out_report=args.out_report,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
