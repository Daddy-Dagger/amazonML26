# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Paglu  
**Team Members:** Aryan Adharv  
**Submission Date:** September 26, 2026  

---

## 1. Executive Summary

We present an end-to-end, high-precision Business Entity Resolution (ER) pipeline engineered to link noisy, heterogeneous records from Source 2 and Source 3 to deduplicated reference entities in Source 1. The architecture implements a four-stage pipeline: country-agnostic normalization, a three-generator inverted blocking index (achieving **99.46%** overall recall with **>99.94%** candidate reduction), a 21-feature LightGBM gradient boosted decision tree trained from scratch, and a post-processing decision layer that strictly enforces the domain rule of at most one owner per record. On held-out validation and test splits, our solution achieves a macro $F_{0.5}$ score of **0.9748** and **0.9746** respectively (outperforming a strong string-similarity baseline of 0.6770 by **+43.9%**). Scaled across the full 1.73M test entities (including zero-shot generalization to France), the pipeline processed 246.8M candidate pairs and produced 7,357,794 accepted matches across 1,651,425 non-empty entities, achieving 100% compliance on the official challenge validator with zero external API calls or third-party pretrained weights.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across all 6 source TSVs and ground truth revealed key structural properties:
1. **Asymmetric Evaluation Metric ($F_{0.5}$ Macro):** $F_{0.5}$ weights precision twice as heavily as recall ($\beta = 0.5$). False merges (linking two distinct companies) are penalized twice as severely as missed links. Singletons (entities with 0 matches) represent **5.58%** of Source 1 entities; predicting an empty list correctly yields an $F_{0.5}$ of 1.0, whereas any false positive drops the entity score to 0.0.
2. **Cartesian Complexity:** Within-country test pairs exceed 3.3 trillion ($S_1 \times S_2$) and 3.4 trillion ($S_1 \times S_3$). Strict country blocking and an aggressive, high-recall candidate generation mechanism are non-negotiable.
3. **No Multi-Assignment in Ground Truth:** In the training ground truth, exactly 0 target IDs ($S_2/S_3$) are shared across multiple $S_1$ entities (a strict 1-to-1 match assignment per target record). Any multi-assignment in predictions degrades precision.
4. **Noise & Formatting Inconsistencies:** The dataset features severe name and address noise, including abbreviation variants (`Corp` vs. `Corporation`, `Pvt` vs. `Private`), missing street-level addresses, word-order permutations, and postal code unreliability (0.00% standard 6-digit PIN codes in Indian addresses; 10.51% 5-digit ZIPs in US). Additionally, 19.70% of Indian records contain native Indic scripts (Devanagari, Tamil, Bengali, Kannada).
5. **France Domain Shift:** France records appear exclusively in the test set (259,452 $S_1$ entities; 1,434,993 $S_2/S_3$ records). The pipeline must be fundamentally country-agnostic with zero hardcoded country rules.

### 2.2 Solution Strategy
The solution executes a sequential 4-stage pipeline:
$$\text{Normalize} \longrightarrow \text{Candidate Blocking} \longrightarrow \text{LightGBM Pairwise Scoring} \longrightarrow \text{1-Owner Decision Layer}$$

- **Approach Type:** Hybrid Multi-Generator Blocking + LightGBM GBDT Matcher + 1-Owner Decision Layer.
- **Core Innovation:** 
  1. A multi-angle candidate generator combining character 3-gram TF-IDF, core name token inverted indexing, and address token inverted indexing that guarantees a 99.46% recall ceiling while eliminating 99.95% of the Cartesian space.
  2. Distinctive token feature engineering that penalizes ubiquitous corporate stopwords and high-frequency terms while isolating discriminative business core names.
  3. A global 1-owner assignment optimization that assigns each query record to at most one reference entity, maximizing macro $F_{0.5}$ precision and perfectly preserving singleton entities.

---

## 3. Candidate Generation (Blocking)

To reduce the candidate search space from trillions of pairs to a manageable volume without discarding true links, candidate generation is framed from the record query perspective: for every $S_2$ and $S_3$ record, generate top candidate matches from $S_1$ within the same country using three complementary generators.

### 3.1 Blocking Generators
1. **Generator (a) - Character 3-gram TF-IDF Index:** Computes character 3-grams over normalized full names, retrieves top-15 cosine similarity matches. Highly effective against character typos, minor misspellings, and phonetic transliteration shifts.
2. **Generator (b) - Rare Token Inverted Index:** Strips legal entity suffixes and generic stopwords to isolate the core entity name. Weights core tokens by document IDF ($w = \ln(1 + N/df)$) and retrieves top-15 matches. Highly robust to word reordering and token transpositions.
3. **Generator (c) - Address Token Inverted Index:** Tokenizes address fields, applies smoothed address IDF weighting with frequency caps, and retrieves top-10 candidate entities. Recovers true matches when company trade names diverge completely from registered legal entities.

The candidate sets from all three generators are unioned, ranked by composite score, and capped at 25 candidates per record query.

### 3.2 Candidate Generation Performance & Recall Analysis
On the 5% stratified mini-world benchmark (382,264 true ground-truth pairs across 110,341 $S_1$ entities):
- **Overall Pairwise Recall:** **99.46%** (380,200 / 382,264 recalled pairs).
- **Overall Complete $S_1$ Recall:** **98.36%** (102,511 / 104,225 $S_1$ entities with all true matches recalled).
- **US Pairwise Recall:** **99.83%** (228,416 / 228,798 pairs recalled; Complete $S_1$: 99.40%).
- **India Pairwise Recall:** **98.90%** (151,784 / 153,466 pairs recalled; Complete $S_1$: 96.78%).

| Generator | Description | Recalled Pairs | Standalone Recall |
|:---|:---|---:|---:|
| **Generator (a)** | Character 3-gram TF-IDF (top-15) | 344,433 | 90.10% |
| **Generator (b)** | Rare Token Inverted Index (top-15) | 284,009 | 74.30% |
| **Generator (c)** | Address Token Inverted Index (top-10) | 354,999 | 92.87% |
| **Combined Union** | Top-25 Capped Ensemble | **380,200** | **99.46%** |

- **Candidate Reduction:**
  - Average candidates per $S_1$ entity: **115.94**.
  - Reduction ratio (US): **99.962%**.
  - Reduction ratio (India): **99.944%**.
  - Full test set scale: **246,855,140** candidate pairs generated across 1,732,544 $S_1$ entities (France: 35,164,307; US: 94,884,943; India: 116,805,890).

---

## 4. Matching Model

### 4.1 Feature Engineering (21 Features)
For every generated candidate pair $(S_1, \text{Cand})$, 21 pairwise features are computed across name, address, and candidate-list context:

- **Name Features (11):**
  1. `token_jaccard_full`: Jaccard similarity across all tokens in normalized business names.
  2. `token_jaccard_core`: Jaccard similarity after removing legal suffixes and corporate stopwords.
  3. `char_3gram_cosine`: Cosine similarity over character 3-gram vectors.
  4. `levenshtein_ratio`: RapidFuzz normalized Levenshtein ratio.
  5. `token_sort_ratio`: Token sort ratio handling word order transpositions.
  6. `token_set_ratio`: Token set ratio handling subset naming variations.
  7. `initials_match`: Binary match indicator of acronyms / name initials.
  8. `name_length_diff`: Normalized absolute length difference.
  9. `legal_suffix_agreement`: Agreement indicator between detected corporate suffixes (`pvt`, `ltd`, `llc`, `inc`, `corp`, `sa`, `sarl`, `sas`, etc.).
  10. `distinctive_token_jaccard`: Jaccard similarity over distinctive tokens only (tokens with document frequency below the 90th percentile).
  11. `shared_distinctive_token_count`: Absolute count of shared non-generic tokens.

- **Address Features (6):**
  12. `token_jaccard_addr`: Jaccard similarity between address token sets.
  13. `is_s1_address_blank`: Binary flag for missing $S_1$ address.
  14. `is_cand_address_blank`: Binary flag for missing query address.
  15. `idf_weighted_token_overlap_addr`: Summed IDF weights of intersecting address tokens.
  16. `house_number_soft_match`: Levenshtein edit distance $\le 2$ on numeric house/premise tokens.
  17. `city_fuzzy_match`: Fuzzy token matching on tail address tokens representing municipality/city.

- **Candidate Context & Rank Features (4):**
  18. `rank_in_candidates`: Generator rank position of the candidate (1-indexed).
  19. `score_gap_to_next_best`: Score margin between candidate and the next-ranked candidate.
  20. `num_candidates_for_this_s1`: Total candidate pool size for the reference entity.
  21. `name_token_rarity`: Maximum IDF weight among shared name tokens.

### 4.2 Model Architecture & Training
- **Model Type:** LightGBM Gradient Boosted Decision Tree (GBDT) with binary cross-entropy loss (`objective="binary"`, `metric="auc"`).
- **Hyperparameters:** `num_leaves=63`, `learning_rate=0.1`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`, `n_estimators=300`.
- **Training Setup:** Trained from scratch on 8,935,205 candidate pairs with early stopping on validation AUC (1,973,163 pairs). Converged at iteration 82 with validation AUC = **0.99988**.

### 4.3 Feature Importances (Top 15 by Gain)
| Rank | Feature | Importance Gain | Category |
|---:|:---|---:|:---|
| 1 | `token_jaccard_addr` | 7,585,695.8 | Address |
| 2 | `rank_in_candidates` | 1,685,172.1 | Context / Rank |
| 3 | `score_gap_to_next_best` | 687,697.7 | Context / Margin |
| 4 | `city_fuzzy_match` | 674,429.6 | Address |
| 5 | `token_jaccard_full` | 544,459.6 | Name |
| 6 | `is_cand_address_blank` | 274,245.0 | Address Quality |
| 7 | `token_set_ratio` | 234,174.5 | Name |
| 8 | `name_token_rarity` | 158,350.3 | Name / Salience |
| 9 | `token_jaccard_core` | 146,245.7 | Name |
| 10 | `idf_weighted_token_overlap_addr` | 97,387.2 | Address |
| 11 | `initials_match` | 63,456.2 | Name |
| 12 | `levenshtein_ratio` | 53,500.6 | Name |
| 13 | `token_sort_ratio` | 48,154.4 | Name |
| 14 | `char_3gram_cosine` | 47,808.9 | Name |
| 15 | `num_candidates_for_this_s1` | 21,151.8 | Context / Pool Size |

### 4.4 Decision Layer Optimization
The decision layer translates pairwise probabilities $\hat{p} \in [0, 1]$ into final link assignments:
1. **1-Owner-per-Record Constraint:** Each query record $q \in S_2 \cup S_3$ selects at most one $S_1$ entity:
   $$S_1^*(q) = \arg\max_{S_1 \in \mathcal{C}(q)} \hat{p}(S_1, q)$$
2. **Threshold Filtering:** An edge is formed if and only if $\hat{p}(S_1^*(q), q) \ge \text{min\_p}$ and the margin over the runner-up candidate satisfies $\Delta p \ge \text{min\_m}$.
3. **Threshold Selection:** A 2D grid search over 70 combinations ($\text{min\_p} \in [0.50, 0.95]$, $\text{min\_m} \in [0.00, 0.30]$) was conducted directly on the official per-entity macro $F_{0.5}$ metric. The optimal threshold point was identified at **$\text{min\_p} = 0.75$, $\text{min\_m} = 0.00$**.

---

## 5. Results & Error Analysis

### 5.1 Validation and Test Results
Performance evaluated on the held-out validation and test splits:

| Model / Configuration | Thresholds | Macro $F_{0.5}$ | Precision | Recall | Overfitting Gap |
|:---|:---|---:|---:|---:|---:|
| **LightGBM Model v2 (Validation)** | `min_p=0.75, min_m=0.00` | **0.9748** | 98.58% | 96.24% | - |
| **LightGBM Model v2 (Held-Out Test)** | `min_p=0.75, min_m=0.00` | **0.9746** | **98.56%** | **96.27%** | **0.0002** |
| LightGBM Model v1 (19 Features) | `min_p=0.70, min_m=0.00` | 0.9741 | 98.38% | 96.58% | 0.0002 |
| Naive Baseline (`char_3gram_cos > 0.8`) | - | 0.6770 | 55.53% | 60.56% | - |

The addition of distinctive token features in Model v2 increased test precision from 98.38% to 98.56% (+0.18%), lifting macro $F_{0.5}$ to **0.9746** while maintaining a near-zero generalization gap ($|0.9748 - 0.9746| = 0.0002$).

### 5.2 France Generalization & Cross-Country Transfer
To ensure robust performance on France (which appears only in the test set), two cross-country holdout experiments were executed:

1. **Zero-Shot Cross-Country Generalization:**
   - **Train on US $\rightarrow$ Test on India (Zero-shot):** Macro $F_{0.5}$ = **0.9518** (baseline 0.9746, drop of only 0.0228).
   - **Train on India $\rightarrow$ Test on US (Zero-shot):** Macro $F_{0.5}$ = **0.9364** (baseline 0.9746, drop of only 0.0382).
   - *Finding:* Because features are based on structural string similarity, normalized token overlaps, and candidate context rather than hardcoded country vocabulary, the model exhibits strong cross-lingual and cross-geographical transfer ($>0.93$ zero-shot).
2. **France Structural Sanity Check on Test Data:**
   - Evaluated on test France ($259,452$ $S_1$ entities, $1,434,993$ $S_2/S_3$ records).
   - **Candidate Blocking Coverage:** **99.93%** of France $S_1$ entities received $\ge 1$ candidate.
   - **Match Rate:** **89.04%** of France $S_1$ entities received accepted matches (231,018 matched entities; 1,081,236 total matches; 4.17 avg matches/$S_1$).
   - *Finding:* The match rate aligns closely with the expected training distribution (~94%), with slightly higher singleton proportions expected for the unseen domain.

### 5.3 Error Analysis & Known Limitations
1. **Native-Script Transliteration Discrepancies:**
   - In India, ~19.7% of addresses and names feature non-Latin Indic scripts (Devanagari, Tamil, Bengali, Kannada). While Unidecode provides character transliteration, colloquial and phonetic transcription divergences (e.g. `शिव आईटी प्रा. लि.` vs `Shiv It Pvt Ltd`) occasionally fail to trigger candidate blocking, accounting for the slight India recall drop (98.90% vs 99.83% US).
2. **Homonym False Merges on Generic Tokens:**
   - False positives occasionally occur between distinct legal entities sharing identical high-frequency words (e.g., `Hernandez All Inc.` vs `Hernandez Aol Inc.`, or `Safe Retail Laboratories` at different premises) when street addresses are omitted or ambiguous. The conservative threshold $p \ge 0.75$ and distinctive token filtering mitigate but do not completely eliminate these edge cases.
3. **Absence of France Ground Truth:**
   - Because no ground-truth matching pairs exist for France in the provided training set, France performance is estimated via cross-country holdouts and structural sanity checks rather than empirically confirmed against gold labels.

---

## 6. Conclusion

Our solution achieves a **0.9746 macro $F_{0.5}$** through a carefully orchestrated pipeline combining high-recall multi-generator inverted blocking (99.46% recall), 21 discriminative pairwise features, and an exact 1-owner decision layer tuned for precision-heavy scoring. The country-agnostic design demonstrates outstanding zero-shot transfer ($>0.93$) and seamlessly processes the unseen France test partition. Across 1.73M test entities, the pipeline generates 246.8M candidates and 7.36M accepted matches with zero memory spills, zero duplicate assignments, and 100% compliance on the official validator.

---

## Appendix

### A. Code Artefacts & Reproduction Steps
The complete runnable code ships in the submission zip under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── README.md               # End-to-end reproduction guide with exact commands and runtimes
├── requirements.txt        # Pinned dependencies
├── model_v2.txt            # Pre-trained LightGBM model artifact (565 KB)
└── src/                    # All source modules
    ├── blocking.py         # Multi-generator candidate generation
    ├── decide_and_score.py # 1-owner decision layer and threshold optimization
    ├── features.py         # 21-feature extraction engine
    ├── france_readiness_check.py # Zero-shot cross-country transfer and France sanity checks
    ├── group_india.py      # Low-memory 16-bucket streaming aggregator
    ├── make_miniworld.py   # Stratified 5% mini-world dataset sampler
    ├── measure_blocking_recall.py # Blocking recall evaluation
    ├── merge_all.py        # Streamlined country partition merger
    ├── metric.py           # Per-entity macro F0.5 metric implementation
    ├── normalize.py        # Unicode NFKD, Unidecode, and corporate normalization
    ├── run_full_pipeline.py # Full-scale memory-bounded test pipeline
    ├── stream_verify.py    # Fast streaming integrity verification
    └── train_matcher.py    # LightGBM GBDT training script
```

#### Reproduction Summary
1. **Environment Setup:** `pip install -r requirements.txt` (~1 min).
2. **Mini-World Sampling & Normalization:**
   ```bash
   python3 src/make_miniworld.py
   python3 src/normalize.py
   ```
3. **Blocking & Feature Extraction:**
   ```bash
   python3 src/blocking.py
   python3 src/features.py
   ```
4. **Model Training:**
   ```bash
   python3 src/train_matcher.py --features-path data_mini/pairwise_features.parquet --out-model-path data_mini/model_v2.txt
   ```
   *(Note: LightGBM model artifact `model_v2.txt` is fully trained and deterministic; retraining is optional).*
5. **Full Pipeline Inference:**
   ```bash
   python3 src/run_full_pipeline.py --test-dir dataset/test --model data_mini/model_v2.txt --out-dir output
   ```
   - **Observed Runtimes on 8 CPU cores / 8.0 GB RAM:**
     - France (259k $S_1$): ~43 minutes
     - United States (663k $S_1$): ~143 minutes (~2.4 hours)
     - India (810k $S_1$): ~8.5 hours
     - **Total Pipeline Runtime:** ~11.8 hours
6. **Validation:**
   ```bash
   python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
   ```

---

### B. Additional Results: Full Test Set Breakdown

| Country | $S_1$ Entities | Candidates Generated | $S_1$ with $\ge 1$ Match | Match Rate (%) | Total Accepted Matches | Avg Matches / $S_1$ |
|:---|---:|---:|---:|---:|---:|---:|
| **France** | 259,452 | 35,164,307 | 231,018 | 89.04% | 1,081,236 | 4.17 |
| **United States** | 663,106 | 94,884,943 | 653,790 | 98.60% | 3,021,812 | 4.56 |
| **India** | 809,986 | 116,805,890 | 766,617 | 94.65% | 3,254,748 | 4.02 |
| **Total Test Set** | **1,732,544** | **246,855,140** | **1,651,425** | **95.32%** | **7,357,794** | **4.25** |

- **Singletons:** Exactly 81,119 $S_1$ entities (4.68%) correctly assigned empty predictions.
- **Multi-Assignment:** Exactly 0 $S_2/S_3$ records assigned to more than one $S_1$ entity.
- **Candidate Consistency:** 100% of accepted matches are confirmed subsets of `candidate_pairs.tsv`.

---

### C. Compliance & Fair Play Verification

1. **Model Training & Licensing:**
   `model_v2.txt` is a LightGBM gradient boosted decision tree model trained **100% FROM SCRATCH** exclusively on the provided training dataset (`dataset/train/`). It does not utilize any pretrained checkpoints, proprietary embeddings, or external weights, and is therefore completely free of third-party model licensing encumbrances. The underlying LightGBM library is open-source under the permissive MIT license.
2. **Zero External APIs or Data Lookups:**
   In accordance with the Academic Integrity and Fair Play guidelines, no external APIs, commercial entity resolution tools, geocoders, or web queries were utilized:
   ```bash
   $ grep -rin "requests\|urllib\|http\|api_key\|geocod" src/
   # Zero matches found (exit code 1)
   ```
3. **Submission File Integrity:**
   - Both `output/matching_results.tsv` (111.81 MB) and `output/candidate_pairs.tsv` (3,055.60 MB) contain exactly **1,732,544** rows corresponding to all test $S_1$ entities.
   - Verified via `utils/validate_submission.py --check-ids` against 9,969,589 test $S_2/S_3$ IDs: **PASS** (exit code 0).
