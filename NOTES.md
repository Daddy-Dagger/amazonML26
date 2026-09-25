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

## Phase 3: Mini-World, Evaluation Metric & Country-Agnostic Normalization
1. **Rule Correction & Protocol Updates:**
   - Corrected Surprise #10 in `audit/data_report.md`: target IDs in S2/S3 belong to at most one S1 entity; decision layer MUST assign each S2/S3 record to at most one S1 entity.
   - Enforced rule in `AGENTS.md` and updated `requirements.txt` with `duckdb==1.4.5` and `pyarrow==21.0.0`.
2. **Mini-World Parquet Sampling (`src/make_miniworld.py`):**
   - Sampled 5% stratified train S1 entities (110,341 rows: 44,159 India, 66,182 US).
   - Preserved exact ground truth rows; retained all matched S2/S3 records plus stratified 5% unmatched distractors.
   - Maintained representative 5.54% singleton share and 25.97% distractor share (S2=26.60%, S3=25.36%). Zero missing match references.
3. **Evaluation Metric & Unit Testing (`src/metric.py`, `tests/test_metric.py`):**
   - Implemented per-entity macro F0.5 with exact handling of singleton edge cases (empty/empty=1.0, empty/pred=0.0, truth/empty=0.0).
   - 8 unit tests passed: statement example verified at 5/7 = 0.714286 (0.714).
4. **Country-Agnostic Normalization (`src/normalize.py`):**
   - Standardized NFKD, unidecode, lowercase, ampersand expansion, punctuation stripping, space collapsing, leading 'the' stripping, and consecutive duplicate token removal.
   - Applied global abbreviation dictionary (`rd`->`road`, `st`->`street`, `pvt`->`private`, etc.) and address-only French expansions (`r`->`rue`, `av`->`avenue`, `bd`->`boulevard`).
   - Dynamically discovered top-40 generic tokens per country: India captured synthetic variations `limittedd` (df=24,045) and `praaivett` (df=13,810).
   - Output normalized parquet tables (`source1_normalized.parquet`, `source2_normalized.parquet`, `source3_normalized.parquet`) and `idf_weights.parquet`.
5. **Postal Pattern & Script Audit:**
   - Indian addresses in 200k sample contain 0.00% word-boundary 6-digit postal PINs; blocking cannot rely on PIN codes. US addresses contain 10.51% 5-digit ZIPs.
   - 19.70% of Indian addresses contain native Indic scripts requiring transliteration.

