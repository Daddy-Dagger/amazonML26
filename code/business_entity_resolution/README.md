# Business Entity Resolution Pipeline

This directory contains the complete source code and instructions to reproduce the final entity matching results (`output/matching_results.tsv`) and candidate sets (`output/candidate_pairs.tsv`) for the ML Challenge 2026.

---

## 1. System Requirements

- **Operating System:** Linux / macOS / Windows
- **Python Version:** Python 3.9+ (tested on Python 3.9.6)
- **Hardware:** 8 CPU cores, 8.0 GB RAM recommended
- **External Dependencies:** None (no internet access, APIs, or commercial tools required)

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── README.md               # Reproduction guide
├── requirements.txt        # Pinned python dependencies
├── model_v2.txt            # Pre-trained LightGBM model artifact (565 KB)
└── src/
    ├── blocking.py         # Multi-generator candidate generation (3-gram, core token, address)
    ├── decide_and_score.py # Decision layer (1-owner-per-record) & threshold search
    ├── features.py         # 21 pairwise feature extraction engine
    ├── france_readiness_check.py # Cross-country generalization & France sanity checks
    ├── group_india.py      # Low-memory 16-bucket streaming candidate aggregator
    ├── make_miniworld.py   # Stratified 5% mini-world dataset sampler
    ├── measure_blocking_recall.py # Blocking recall auditor
    ├── merge_all.py        # Streaming country partition merger
    ├── metric.py           # Per-entity macro F0.5 metric implementation
    ├── normalize.py        # Text normalization & corporate stopword cleaner
    ├── run_full_pipeline.py # End-to-end full test dataset pipeline
    ├── stream_verify.py    # Fast streaming integrity verification
    └── train_matcher.py    # LightGBM GBDT training script
```

---

## 3. Step-by-Step Reproduction Guide

Execute all commands from the repository root directory (`student_resource/`).

### Step 1: Environment Setup
Create a virtual environment and install the pinned dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r code/business_entity_resolution/requirements.txt
```
*Observed Runtime:* ~1 minute.

---

### Step 2: Mini-World Sampling (from Training Data)
Extract a 5% stratified sample of the training dataset preserving ground truth distributions and singletons:
```bash
python3 src/make_miniworld.py
```
- Reads from `dataset/train/`
- Outputs: `data_mini/source1.parquet`, `source2.parquet`, `source3.parquet`, `ground_truth.parquet`
- *Observed Runtime:* ~20 seconds.

---

### Step 3: Text Normalization & IDF Precomputation
Normalize company names and addresses using Unicode NFKD, Unidecode transliteration, corporate suffix standardization, and compute token IDF weights:
```bash
python3 src/normalize.py
```
- Outputs: `data_mini/source1_normalized.parquet`, `source2_normalized.parquet`, `source3_normalized.parquet`, `data_mini/idf_weights.parquet`
- *Observed Runtime:* ~45 seconds.

---

### Step 4: Candidate Generation (Blocking)
Run the 3-generator inverted index (Character 3-gram TF-IDF, core token inverted index, address token inverted index):
```bash
python3 src/blocking.py
```
- Outputs: `data_mini/candidates_s1_grouped.parquet`, `data_mini/candidate_pairs_by_record.parquet`
- Generates candidates for 516,356 query records against 110,341 $S_1$ reference entities.
- *Observed Runtime:* ~7.7 minutes (464s). Peak RAM: <1.5 GB.

To evaluate recall against training ground truth:
```bash
python3 src/measure_blocking_recall.py
```
*Observed Recall:* **99.46%** overall (99.83% US, 98.90% India).

---

### Step 5: Feature Extraction
Compute 21 pairwise name, address, and candidate-context features across all blocked pairs:
```bash
python3 src/features.py
```
- Generates 12,793,282 feature vectors split into train (70%), validation (15%), and held-out test (15%).
- Outputs: `data_mini/pairwise_features.parquet`
- *Observed Runtime:* ~3.5 minutes (213s).

---

### Step 6: Train LightGBM Matcher (Optional)
Train the binary LightGBM classifier with early stopping on validation AUC:
```bash
python3 src/train_matcher.py \
    --features-path data_mini/pairwise_features.parquet \
    --out-model-path data_mini/model_v2.txt \
    --lr 0.1 \
    --num-leaves 63 \
    --n-estimators 300 \
    --early-stopping 25
```
- *Observed Runtime:* ~3 minutes.
- *Best Iteration:* 82 | *Validation AUC:* **0.99988**.
- *Held-out Test Macro $F_{0.5}$:* **0.9746** (with thresholds `min_p=0.75, min_m=0.00`).
- **Trained Artifact Shipped:** The pre-trained model file `model_v2.txt` is already included directly in this package (`code/business_entity_resolution/model_v2.txt`). Training is 100% deterministic and can be reproduced using the command above, or you may skip training entirely and proceed directly to inference using the shipped model.

---

### Step 7: Full Test Set Pipeline Execution
Run the end-to-end memory-bounded inference pipeline across all test countries (`France`, `United States`, `India`) using the shipped trained model `code/business_entity_resolution/model_v2.txt` (or `data_mini/model_v2.txt` if retrained):
```bash
python3 src/run_full_pipeline.py \
    --test-dir dataset/test \
    --model code/business_entity_resolution/model_v2.txt \
    --out-dir output \
    --min-p 0.75 \
    --min-m 0.00
```
This script executes Pass 1 (normalization and candidate generation), Pass 2 (21-feature extraction and LightGBM scoring), and Pass 3 (1-owner-per-record decision enforcement and partition merging) for each country sequentially.

- **Observed Runtimes on 8 CPU cores / 8.0 GB RAM:**
  - **France** (259,452 $S_1$, 1,434,993 queries): **43 minutes**
  - **United States** (663,106 $S_1$, 3,817,031 queries): **143 minutes** (~2.4 hours)
  - **India** (809,986 $S_1$, 4,717,565 queries): **~8.5 hours**
  - **Total Pipeline Execution Runtime:** **~11.8 hours**
- **Peak Memory:** 3.65 GB (well within 8.0 GB machine capacity).
- **Final Output Files Generated:**
  - `output/matching_results.tsv` (111.81 MB, 1,732,544 $S_1$ rows)
  - `output/candidate_pairs.tsv` (3,055.60 MB, 1,732,544 $S_1$ rows)

---

### Step 8: Submission Validation
Run the official challenge submission validator:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
**Expected Validator Output:**
```
PASS — no blocking issues found. Safe to submit.
```

For ultra-fast streaming validation of ID sets, singleton counts, and 100% candidate subset containment:
```bash
python3 src/stream_verify.py
```
*Verification Runtime:* 78 seconds. Verified 0 duplicate rows, 0 intra-row duplicates, 0 multi-assignments, 0 self-matches, and 100% candidate subset containment.
