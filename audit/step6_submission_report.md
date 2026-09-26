# Step 6 Full Test Submission Report

## Pipeline Execution Summary
- **Status**: Completed successfully on local machine (8.0 GB RAM).
- **Total Pipeline Runtime**: ~11.8 hours across all 3 countries (France: 43m, US: 143m, India: 8.5h).
- **Peak Memory**: 3.65 GB during Pass 1/2 inference (well below 7.0 GB hard safety limit).
- **Model & Decision Layer**: LightGBM `model_v2.txt` with thresholds `min_p = 0.75`, `min_m = 0.00`, and strict 1-owner assignment.

## Per-Country Results
| Country | S1 Entities | Candidates Generated | S1 with >= 1 Match | Match Rate (%) | Total Accepted Matches | Avg Matches / S1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **France** | 259,452 | 35,164,307 | 231,018 | 89.04% | 1,081,236 | 4.17 |
| **United States** | 663,106 | 94,884,943 | 653,790 | 98.60% | 3,021,812 | 4.56 |
| **India** | 809,986 | 116,805,890 | 766,617 | 94.65% | 3,254,748 | 4.02 |
| **Total Test Set** | **1,732,544** | **246,855,140** | **1,651,425** | **95.32%** | **7,357,794** | **4.25** |

## Comparison to Training Distribution
- **Training Ground Truth**: India and US match rates are ~94%.
- **Test Set Observation**:
  - India: 94.65% (exact match with training distribution).
  - United States: 98.60% (closely aligned with high density US training entities).
  - France: 89.04% (consistent with unseen country profile; slightly higher singleton rate).
  - All match rates align closely with expected domain distributions; no distributional collapse.

## Submission Validator Status
- **Official Validator (`utils/validate_submission.py`)**: PASS (exit code 0).
  - Checked against 9,969,589 test Source-2/3 entity IDs via `--check-ids`.
  - Exactly 1,732,544 S1 rows (81,119 empty singletons, 1,651,425 non-empty).
  - Zero missing S1 IDs, zero extra S1 IDs.
  - Zero multi-assignment: all 7,357,794 matched query IDs belong to at most one S1 entity.
  - Zero intra-row duplicate IDs, zero `S1-` self-matches, zero invalid ID prefixes.
- **Candidate Cross-Check**: PASS (0 matched IDs missing from candidates; strict subset verified).
- **Issues Found & Fixed**: India candidate aggregation transitioned from in-memory DuckDB `string_agg` to 16-bucket streaming partitioning (`src/group_india.py`), bounding peak grouping RAM to 0.65 GB.

## Submission Output Files
- `output/matching_results.tsv`: 111.81 MB (1,732,545 lines including header).
- `output/candidate_pairs.tsv`: 3,055.60 MB (1,732,545 lines including header).

READY TO SUBMIT
