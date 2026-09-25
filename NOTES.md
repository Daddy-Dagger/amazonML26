# Project Notes

## Phase 1: Environment Setup & Data Audit
- Python 3.9.6 virtual environment created in `.venv` with required packages.
- Platform: 8 CPU cores, 8.0 GB RAM.
- Profiling executed across all 6 source TSVs and train ground truth via memory-efficient single-source stream (`audit/audit.py`). Peak memory: 2.93 GB.
- Full audit report generated at `audit/data_report.md` (160 lines).

### Key Audit Findings & Pipeline Decisions:
1. **Strict Country Blocking:** 100% of matched pairs share the same country. Blocking by country reduces search space without recall loss.
2. **Domain Adaptation for France:** France records appear exclusively in the test set. All preprocessors, tokenizers, and matching models must be country-agnostic.
3. **Severe Cartesian Complexity:** Within-country test pairs exceed 3.3 trillion (S1xS2) and 3.4 trillion (S1xS3). A highly selective multi-stage blocking index (e.g. inverted index on standardized name tokens + locality) is essential.
4. **Singletons:** 5.58% of S1 entities have 0 matches in train. Correctly predicting empty match lists is critical for per-entity macro F0.5.
5. **No Multi-Assignment in Ground Truth:** In train ground truth, 0 target IDs are shared across multiple S1 entities (1-to-1 match assignment pattern per target).
6. **No Missing Ground Truth IDs:** All IDs in `train_ground_truth.tsv` exist in the respective source files.

## Phase 2: Repository Setup & Remote Sync
- Initialized local git repository on branch `main`.
- Extended `.gitignore` to protect datasets, mini datasets, logs, virtual environments, parquet/pickle artifacts, __pycache__, and `.DS_Store`.
- Verified 0 large files (>50 MB) and 0 data files staged before commit.
- Linked remote origin `https://github.com/Daddy-Dagger/amazonML26.git` and successfully pushed initial setup to `origin/main`.
