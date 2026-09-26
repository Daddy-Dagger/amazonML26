# Audit Report: Step 4 - LightGBM Matcher Evaluation

**Status:** NORMAL (val/test close) | Val F0.5: **0.9743** | Test F0.5: **0.9741**

## 1. Top 15 Feature Importances (Gain)
| Rank | Feature | Importance Gain | Rank | Feature | Importance Gain |
|---:|:---|---:|---:|:---|---:|
| 1 | `token_jaccard_addr` | 7035167.7 | 9 | `token_jaccard_core` | 38786.0 |
| 2 | `score_gap_to_next_best` | 2265940.8 | 10 | `initials_match` | 28038.3 |
| 3 | `rank_in_candidates` | 391744.4 | 11 | `city_fuzzy_match` | 23394.0 |
| 4 | `is_cand_address_blank` | 191329.6 | 12 | `token_sort_ratio` | 21902.6 |
| 5 | `idf_weighted_token_overlap_addr` | 97757.9 | 13 | `house_number_soft_match` | 17152.0 |
| 6 | `levenshtein_ratio` | 68566.8 | 14 | `token_set_ratio` | 11932.3 |
| 7 | `char_3gram_cosine` | 65261.1 | 15 | `name_token_rarity` | 10327.5 |
| 8 | `token_jaccard_full` | 40393.7 | 16 | `num_candidates_for_this_s1` | 9198.3 |

## 2. Threshold Search & Evaluation Summary
| Model / Baseline | Thresholds | F0.5 Score | Precision | Recall |
|:---|:---|---:|---:|---:|
| **LightGBM (Validation)** | `min_p=0.70, min_m=0.00` | **0.9743** | 98.40% | 96.60% |
| **LightGBM (Test)** | `min_p=0.70, min_m=0.00` | **0.9741** | **98.38%** | **96.58%** |
| Naive Baseline (Test) | `char_3gram_cos > 0.8` | 0.6770 | 55.53% | 60.56% |

## 3. Sampled Correct Matches (10 True Positives from Test Split)
| S1 ID / Name | Target ID / Name | Prob | Margin | Notes |
|:---|:---|---:|---:|:---|
| `S1-55884194`: Navarro, Christal, P.A. LP | `S3-813544443`: NAVARRO, CHRI5TAL, P.A. LP | 0.99 | 0.99 | High confidence name/addr match |
| `S1-78189214`: Lucky Book Store Inc. | `S3-314349064`: Lucky Boko Store Inc. | 1.00 | 1.00 | High confidence name/addr match |
| `S1-713879617`: Dental Center | `S3-836410286`: *** Dental-Center | 1.00 | 1.00 | High confidence name/addr match |
| `S1-741562756`: Jocelyn Donovan Atlantic Hal | `S3-386483765`: Jocelyn Donovn Atlantic Hall | 1.00 | 1.00 | High confidence name/addr match |
| `S1-69095976`: Fairfax Family Services | `S2-713214895`: Fairfax Family Services | 1.00 | 1.00 | High confidence name/addr match |
| `S1-827203155`: Coastal Program | `S2-404692744`: Coastal Program | 1.00 | 1.00 | High confidence name/addr match |
| `S1-219720490`: Varshesh Software Limited | `S3-895139873`: Varshesh Software Ltd | 1.00 | 1.00 | High confidence name/addr match |
| `S1-493828327`: Lai, Perrine & Beydoun LLC | `S3-531276311`: Lyrariza t/a Lai, Perrine &  | 1.00 | 1.00 | High confidence name/addr match |
| `S1-924910601`: Fairdeal Allied Pvt Ltd | `S3-189978315`: Fairdeal Pvt Ltd Services | 0.96 | 0.96 | High confidence name/addr match |
| `S1-487850785`: Human Rights Foundation | `S2-583706777`: Human Rights Foundation LLC | 0.99 | 0.99 | High confidence name/addr match |

## 4. Sampled Errors (10 False Positives / False Negatives from Test Split)
| Error Type | S1 ID / Name | Target ID / Name | Prob | Margin | Likely Cause |
|:---|:---|:---|---:|---:|:---|
| False Positive | `S1-123820928`: G/X Productions | `S2-189394238`: G X & | 0.82 | 0.82 | Shared name tokens, distinct actual entity |
| False Positive | `S1-935333794`: Paramount Agri Pvt Ltd | `S2-714435177`: Paramount Agri Exports P | 0.84 | 0.84 | Shared name tokens, distinct actual entity |
| False Positive | `S1-792955616`: Capital Medical | `S3-994427971`: Medical Capital Valley | 0.93 | 0.93 | Shared name tokens, distinct actual entity |
| False Positive | `S1-133069377`: New Delhi Imperial Priva | `S2-631190492`: New Delhi Ltd Pvt Softwa | 0.81 | 0.81 | Shared name tokens, distinct actual entity |
| False Positive | `S1-625097394`: Premier Enterprises Priv | `S2-162263494`: પ્રીમિયર એન્ટરપ્રાઇઝિસ ફ | 0.91 | 0.91 | Shared name tokens, distinct actual entity |
| Missed / rejected by model | `S1-40849769`: Classic Ventures | `S3-972933308`: Veraonyx | 0.02 | 0.02 | Low probability / ambiguous margin over runner-up |
| Missed / rejected by model | `S1-167840214`: First Constructions Pvt  | `S3-312668117`: ಫಸ್ಟ್ ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ಸ್ ಪ | 0.13 | 0.13 | Low probability / ambiguous margin over runner-up |
| Missed / rejected by model | `S1-697418628`: Frontier Association | `S3-363694881`: Frontier Association Cor | 0.37 | 0.37 | Low probability / ambiguous margin over runner-up |
| Missed / rejected by model | `S1-85177519`: Kai Moore Regional Unite | `S2-580983677`: Kai Moore Regional Unite | 0.33 | 0.33 | Low probability / ambiguous margin over runner-up |
| Missed / rejected by model | `S1-244251926`: Diamond Factory Care | `S2-883386825`: Diamond Factory Center | 0.66 | 0.66 | Low probability / ambiguous margin over runner-up |
