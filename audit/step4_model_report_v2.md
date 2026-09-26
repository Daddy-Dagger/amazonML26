# Audit Report: Step 4 - LightGBM Matcher Evaluation

**Status:** NORMAL (val/test close) | Val F0.5: **0.9748** | Test F0.5: **0.9746**

## 1. Top 15 Feature Importances (Gain)
| Rank | Feature | Importance Gain | Rank | Feature | Importance Gain |
|---:|:---|---:|---:|:---|---:|
| 1 | `token_jaccard_addr` | 7585695.8 | 9 | `token_jaccard_core` | 146245.7 |
| 2 | `rank_in_candidates` | 1685172.1 | 10 | `idf_weighted_token_overlap_addr` | 97387.2 |
| 3 | `score_gap_to_next_best` | 687697.7 | 11 | `initials_match` | 63456.2 |
| 4 | `city_fuzzy_match` | 674429.6 | 12 | `levenshtein_ratio` | 53500.6 |
| 5 | `token_jaccard_full` | 544459.6 | 13 | `token_sort_ratio` | 48154.4 |
| 6 | `is_cand_address_blank` | 274245.0 | 14 | `char_3gram_cosine` | 47808.9 |
| 7 | `token_set_ratio` | 234174.5 | 15 | `num_candidates_for_this_s1` | 21151.8 |
| 8 | `name_token_rarity` | 158350.3 | 16 | `house_number_soft_match` | 18975.7 |

## 2. Threshold Search & Evaluation Summary
| Model / Baseline | Thresholds | F0.5 Score | Precision | Recall |
|:---|:---|---:|---:|---:|
| **LightGBM (Validation)** | `min_p=0.75, min_m=0.00` | **0.9748** | 98.58% | 96.24% |
| **LightGBM (Test)** | `min_p=0.75, min_m=0.00` | **0.9746** | **98.56%** | **96.27%** |
| Naive Baseline (Test) | `char_3gram_cos > 0.8` | 0.6770 | 55.53% | 60.56% |

## 3. Sampled Correct Matches (10 True Positives from Test Split)
| S1 ID / Name | Target ID / Name | Prob | Margin | Notes |
|:---|:---|---:|---:|:---|
| `S1-871305019`: Mega Traders Private Limited | `S3-314680660`: Shri Mega Traders Private  L | 0.95 | 0.95 | High confidence name/addr match |
| `S1-993334473`: Lilly, Paulsen and Cross | `S3-593015676`: Ariafaye | 0.81 | 0.81 | High confidence name/addr match |
| `S1-716874298`: Sree Shakti Foods Private Li | `S3-902777537`: Sree Limited Foods Private   | 1.00 | 1.00 | High confidence name/addr match |
| `S1-356581080`: Bright Red International Pri | `S2-241961703`: Bright Red Ínternational Pri | 1.00 | 1.00 | High confidence name/addr match |
| `S1-147958297`: Tms Charitable Trust | `S2-739649810`: Tms-Charitable Trust Co | 1.00 | 1.00 | High confidence name/addr match |
| `S1-566066078`: Save Bhoomi Private Limited | `S2-705840291`: Mr Save Bhoomi Private | 0.99 | 0.99 | High confidence name/addr match |
| `S1-647245398`: Kestay Retail | `S2-365674480`: Kestay Services | 0.99 | 0.99 | High confidence name/addr match |
| `S1-776047851`: Maid Auto Body LLC | `S3-133212152`: Maid Auto | 0.99 | 0.97 | High confidence name/addr match |
| `S1-438287507`: Fiva LLC | `S3-728147904`: Fa LLC | 0.99 | 0.99 | High confidence name/addr match |
| `S1-177850970`: Quartz Miluna L.L.C. | `S2-405167746`: QUARTZ MILUNA LLC | 1.00 | 1.00 | High confidence name/addr match |

## 4. Sampled Errors (10 False Positives / False Negatives from Test Split)
| Error Type | S1 ID / Name | Target ID / Name | Prob | Margin | Likely Cause |
|:---|:---|:---|---:|---:|:---|
| False Positive | `S1-707246412`: Hernandez All Inc. | `S3-48830023`: Hernandez Aol Inc. | 0.81 | 0.81 | Shared name tokens, distinct actual entity |
| False Positive | `S1-639080324`: Safe Retail Laboratories | `S2-450468926`: Safe Retail Laboratories | 0.85 | 0.71 | Shared name tokens, distinct actual entity |
| False Positive | `S1-943795914`: Slocum and Kan LLC | `S3-898979408`: Ltd Slocum and Kan | 0.94 | 0.94 | Shared name tokens, distinct actual entity |
| False Positive | `S1-542328999`: WER Continental Mustang  | `S3-101656242`: Holdings WER Continental | 0.94 | 0.94 | Shared name tokens, distinct actual entity |
| False Positive | `S1-851145174`: Jabalpur Solar Private L | `S2-564151578`: jabalpurinvestmentscom | 0.90 | 0.90 | Shared name tokens, distinct actual entity |
| Missed / rejected by model | `S1-141245269`: Dermatology Sterling Spe | `S2-740027736`: ONYXVIO | 0.08 | 0.08 | Low probability / ambiguous margin over runner-up |
| Missed / rejected by model | `S1-248990196`: Anekant Trading Private  | `S3-432805467`: Lumjax | 0.71 | 0.71 | Low probability / ambiguous margin over runner-up |
| Missed / rejected by model | `S1-666100224`: Dream Consultancy Privat | `S3-362615672`: ड्रीम कंसल्टेंसी प्राइवे | 0.01 | 0.01 | Low probability / ambiguous margin over runner-up |
| Missed / rejected by model | `S1-893021988`: Vadodara Ventures Privat | `S3-476966280`: Vadodara Private Limited | 0.01 | 0.01 | Low probability / ambiguous margin over runner-up |
| Missed / rejected by model | `S1-501295946`: Pediatric Care | `S3-542394137`: Pediatric Care Commissio | 0.08 | 0.08 | Low probability / ambiguous margin over runner-up |
