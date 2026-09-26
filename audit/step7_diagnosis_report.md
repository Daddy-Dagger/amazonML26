# Step 7: Fast Honest Diagnosis & Threshold Retuning Report

## 1. Executive Summary & Old vs New Threshold Performance
Evaluated on realistic stratified sample with full candidate density and true distractor distribution:

| Configuration | min_p | min_m | Precision | Recall | Macro F0.5 | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Old Baseline (Mini-World derived)** | 0.75 | 0.00 | 85.95% | **98.63%** | **0.9012** | Baseline |
| **Top Retuned Threshold #1** | **0.95** | **0.05** | **96.16%** | 95.32% | **0.9592** | **+0.0580** |
| Top Retuned Threshold #2 | 0.96 | 0.05 | 96.66% | 94.25% | 0.9590 | +0.0578 |
| Top Retuned Threshold #3 | 0.95 | 0.15 | 96.25% | 95.09% | 0.9587 | +0.0575 |
| Top Retuned Threshold #4 | 0.95 | 0.10 | 96.19% | 95.13% | 0.9586 | +0.0574 |

- Increasing `min_p` from 0.75 to 0.95 and `min_m` to 0.05 cuts false positives by **76.4%** (from 1,139 down to 269), driving Precision from 85.95% to 96.16% and boosting Macro F0.5 to **0.9592**.

## 2. Root Cause Verdict
- **Verdict:** CONFIRMED scale-driven precision collapse.
- In mini-world, only 5% of distractors were retained, artificially insulating S1 entities from competition.
- At full scale, 2.7M distractors compete against 1.73M S1 entities. At `min_p=0.75, min_m=0.00`, pure distractors suffer a **11.72% (US) and 13.04% (India) False Positive rate**, flooding the output with >320,000 false positive assignments and collapsing leaderboard F0.5 to 0.700.
- Setting `min_p=0.95, min_m=0.05` filters out ~75% of distractor false positives while maintaining >95% recall.

## 3. Reactive Scripts Audit (`group_india.py`, `merge_all.py`, `stream_verify.py`)
- **Row count and integrity:** Exactly 1,732,544 rows preserved, 0 duplicates, 0 multi-assignments, all matches are subsets of candidates. No records silently dropped.
- **Vulnerabilities found:**
  1. `group_india.py:137`: `cand = line[tab+1:-1]` strips only `\n`; under CRLF (`\r\n`) it leaves `\r`. Fixed to `.rstrip('\r\n')`.
  2. `run_full_pipeline.py:524`: DuckDB `COPY ... SELECT` without `ORDER BY` may produce non-deterministic row order under parallel execution (though key-value alignment remains correct).
  3. `code/.../model_v2.txt`: Git CRLF conversion caused LightGBM text parser fatal error. Converted to LF.

## 4. Hardware Utilization Changes Made
- Set LightGBM's `num_threads: 8` explicitly in `src/train_matcher.py` and `code/.../train_matcher.py`.
- Set `PRAGMA threads=8` across all DuckDB connections in `src/run_full_pipeline.py`.
- Fixed model file line endings to LF (`\n`) for cross-platform LightGBM loading.

## 5. Recommendation
- **Rerun Full Test Inference:** YES. We must rerun full test set inference using the corrected thresholds `min_p=0.95, min_m=0.05` to suppress distractor false positives and recover our F0.5 score from 0.700 towards ~0.96+.
