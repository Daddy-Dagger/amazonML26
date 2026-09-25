# Data Audit Report — Business Entity Resolution

## 1. Source Data Profiling

| Split | Source | Rows | Unique IDs | Prefix Valid | % Blank Name | % Blank Addr | % Non-ASCII | Mean/Med Name Tok | Mean/Med Addr Tok | Dupes (Name, Addr) | Country Counts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Train | train_source1 | 2,206,821 | 2,206,821 | True | 0.00% | 0.00% | 0.03% | 3.5/4 | 8.0/7 | 0 | India: 883,188, US: 1,323,633 |
| Train | train_source2 | 5,034,616 | 5,034,616 | True | 0.00% | 3.36% | 21.98% | 3.5/4 | 7.3/6 | 25,891 | India: 2,017,799, US: 3,016,817 |
| Train | train_source3 | 5,285,603 | 5,285,603 | True | 0.00% | 3.33% | 18.72% | 3.5/4 | 7.2/6 | 18,881 | India: 2,115,547, US: 3,170,056 |
| Test | test_source1 | 1,732,544 | 1,732,544 | True | 0.00% | 0.00% | 5.86% | 3.5/4 | 8.6/8 | 0 | France: 259,452, India: 809,986, US: 663,106 |
| Test | test_source2 | 4,887,273 | 4,887,273 | True | 0.00% | 2.65% | 29.68% | 3.6/4 | 7.8/7 | 22,642 | France: 703,378, India: 2,312,565, US: 1,871,330 |
| Test | test_source3 | 5,082,316 | 5,082,316 | True | 0.00% | 2.68% | 25.91% | 3.6/4 | 7.5/7 | 16,305 | France: 731,615, India: 2,405,000, US: 1,945,701 |

## 2. Ground Truth Analysis (Train)

- **S1 Entities in Ground Truth:** 2,206,821 | **Zero-Match Singletons:** 123,247 (5.58%)
- **Ground Truth IDs Missing from Source Files:** 0 (S1: 0, S2: 0, S3: 0)
- **Multi-assigned Target IDs:** S2 IDs under >1 S1: 0 | S3 IDs under >1 S1: 0
- **Unmatched Source Records:** S2 unmatched: 1,340,997 (26.64%) | S3 unmatched: 1,340,857 (25.37%)
- **Country Consistency:** Matched pairs sharing same country: 100%

### Distribution of Matches per S1 Entity

| Country | Target | 0 Matches | 1 Match | 2 Matches | 3 Matches | 4 Matches | 5+ Matches |
|---|---|---|---|---|---|---|---|
| India | S2 | 115,001 | 314,964 | 261,498 | 134,067 | 47,906 | 9,752 |
| India | S3 | 106,553 | 286,413 | 267,682 | 149,046 | 58,213 | 15,281 |
| India | Total | 49,351 | 47,468 | 149,927 | 211,965 | 193,669 | 230,808 |
| US | S2 | 172,744 | 474,144 | 391,281 | 199,890 | 71,172 | 14,402 |
| US | S3 | 159,723 | 430,004 | 400,693 | 223,397 | 86,903 | 22,913 |
| US | Total | 73,896 | 71,689 | 225,285 | 318,876 | 290,446 | 343,441 |
| **Overall** | **S2** | **287,745** | **789,108** | **652,779** | **333,957** | **119,078** | **24,154** |
| **Overall** | **S3** | **266,276** | **716,417** | **668,375** | **372,443** | **145,116** | **38,194** |
| **Overall** | **Total** | **123,247** | **119,157** | **375,212** | **530,841** | **484,115** | **574,249** |

## 3. Matched Pairs Analysis & Address PIN Coverage

| Target | Country | Matched Pairs | % Identical Name | % Identical (Cleaned) | Med Name Jaccard | Med Addr Jaccard |
|---|---|---|---|---|---|---|
| S2 | India | 1,480,545 | 2.67% | 14.91% | 0.600 | 0.800 |
| S2 | US | 2,213,074 | 6.10% | 25.81% | 0.667 | 0.667 |
| S3 | India | 1,579,298 | 2.75% | 16.81% | 0.600 | 0.636 |
| S3 | US | 2,365,448 | 5.75% | 25.85% | 0.667 | 0.444 |

**Matched Address Postal/PIN Code Coverage:**
- **India:** 0 / 3,893,680 (0.00%) addresses contain valid PIN pattern
- **US:** 581,812 / 5,828,259 (9.98%) addresses contain valid PIN pattern

## 4. Top 20 Name Tokens per Country

- **India:** LIMITED (1,729,019), PRIVATE (1,607,180), LTD (798,870), PVT (533,719), INDIA (292,432), SERVICES (187,241), LLP (178,910), COM (146,841), CENTER (134,073), CO (106,863), INDUSTRIES (95,476), ENTERPRISES (93,273), SOLUTIONS (92,162), BROTHERS (91,373), TRADING (89,371), L (87,199), VENTURES (85,333), GROUP (82,470), PUBLIC (77,909), EXPORTS (72,615)
- **US:** LLC (1,463,694), INC (1,064,533), L (360,073), C (332,905), CENTER (321,895), S (308,404), PARTNERS (297,426), AND (293,813), CORP (271,843), COM (267,335), GROUP (246,219), CO (206,994), CARE (199,989), OF (192,256), LTD (181,565), ASSOCIATES (165,363), D (161,518), SERVICES (160,326), HOLDINGS (153,277), P (139,277)

## 5. Test Set & Feasibility Analysis

- **New Test Countries (unseen in train):** France
- **France Test Counts:** test_source1: 259,452, test_source2: 703,378, test_source3: 731,615 | **Total France:** 1,694,445
- **Train Full Cartesian Pairs (Within Country):** S1xS2 = 5,775,254,399,373 | S1xS3 = 6,064,416,457,284
- **Test Full Cartesian Pairs (Within Country):** S1xS2 = 3,296,528,253,926 | S1xS3 = 3,428,041,312,286
- **Audit Peak Memory:** 2933.0 MB (Runtime: 826.2s)

## 6. Examples

### A. Matched Groups from Train (8 Examples)
**Group 1:** S1 `S1-10951058` | Dhanvarsha Law Chambers | Shyam-Arcade, Ff-1, Kothariya Naka Chowk, Rajkot, Gujarat | India
  - Match `S2-962387325`: DHANVARSHA LAW  CHAMBERS | SHYAM-ARCADE, FF-1, KOTHARIYA NAKA CHOWK, RAJKOT, Gujarat | India
  - Match `S2-909009563`: Dhanvarsha Law  Chambers | SHYAM-ARCADE, FF-1, KOTHARIYA NAKA CHOWK, RAJKOT, ગુજરાત | India
  - Match `S2-411835203`: DHANVARSHA LAW CHAMBERS | SHYAM-ARCADE, FF-1, KOTHARIYA NAKA CHOWK, RAJKOT, Gujarat | India
**Group 2:** S1 `S1-726843944` | NOH Rice Ltd | D No.18-74/1, Aditya Nagar, Street 1 Madhurawada Visakhapatnam, Visakhapatnam, Vishakhapatnam, Andhr... | India
  - Match `S2-432297332`: NOH Rice Ltd. |  | India
  - Match `S2-975755327`: NOH Rice Limited | D NO.0018-74/1 , ADITYA NAGAR, STREET 1 MADHURAWADA VISAKHAPATNAM, VISAKHAPATNAM, Andhra Pradesh | India
  - Match `S3-851380682`: The NOH Rice Ltd | H.no 18-74/1, Visakhapatnam, Vishakhapatnam, ఆంధ్రప్రదేశ్ | India
**Group 3:** S1 `S1-644686829` | Kolkata Infratel Private Limited | 43/3 Hazra Rd, Kolkata, Calcutta, West Bengal | India
  - Match `S2-473963598`: Kolkata Ínfratel Private Limited | West Bengal, 43/3 HAZRA RD, KOLKATA | India
  - Match `S2-875604385`: Private Kolkata Ihrttel Limited | 43/3 HAZRA RD, KOLKATA, CALCUTTA, পশ্চিমবঙ্গ | India
  - Match `S2-117855487`: KOLKATA INFRATEL PRIVATE LIMITED | DOOR NO 43/3 HAZRA RD, KOLKATA, পশ্চিমবঙ্গ | India
**Group 4:** S1 `S1-867959388` | Best First Food Private Limited | Scf 75 76 Shivalik Vihar, Patiala Road, Zirakpur, Mohali, Punjab | India
  - Match `S2-929036647`: BEST FIRST FOOD PRIVATE LIMITED | 570 SCF 75 76 SHIVALIK VIHAR, PATIALA ROAD, ZIRAKPUR, Punjab | India
  - Match `S3-116048558`: Best First Food Private  (Limited) | Patiala Road, Scf 75 76 Shivalik Vihar, Zirakpur, ਪੰਜਾਬ, Mohali | India
  - Match `S3-789000078`: Best First Food Private Ltd | Scf 75 76 Shivalik Vihar, Patiala Road, Zirakpur, Mohali, ਪੰਜਾਬ | India
**Group 5:** S1 `S1-395122715` | Empire Assets LLC | 3505 30th Street, Indianapolis, IN | US
  - Match `S2-661446413`: Empire Assets Llc | 30ND STREET, INDIANAPOLIS, IN | US
  - Match `S3-588947526`: The Empire Assets LLC | 3505a 30th St, Indianapolis, Indiana | US
  - Match `S3-177175480`: Empire Assets Assets |  | US
**Group 6:** S1 `S1-811262361` | Elle Maniscalco Supply LLC | 830 Heritage Drive, Addison, IL | US
  - Match `S2-12844823`: Elle Maniscalco Supply LLC | 230 HERITAGE DR, ADDISN, IL | US
  - Match `S2-322172619`: Mancschalco, Elle Supply LLC | 230 HERITAGE DRIVE, ADDISN, IL | US
  - Match `S3-782807905`: Elle-Supply Maniscalco LLC | Illinois, Addison, 830 Heritate Dr | US
**Group 7:** S1 `S1-962720000` | Internal Medicine Care Associates Group | 11014 Courtshire Road, Houston, TX | US
  - Match `S2-886775257`: Internal Medicine Care Group Associates Associates | 11014 COURTSHIRE ROAD, N/A, HOUSTON, TX | US
  - Match `S3-120692529`: Internal Medicine Care | 11014. Courtshire Rd, Houston, Texas | US
  - Match `S3-822083101`: Internal Medicine Care Associates Group | Courtshire Road, Houston, Texas | US
**Group 8:** S1 `S1-405980354` | Colonial Bitcoin LLC | 400 Cypress Street, Greeneville, TN | US
  - Match `S2-683935059`: Colonial Bctoein LLC | 00 CYPRESS STREET, GRENEVILLE, TN | US
  - Match `S2-884835146`: Colonial Bitcoin | 00 Cypress St, GRENEVILLE, TN | US
  - Match `S2-676944757`: Colonial LLC Bitcoin / www.coloniall.com | 00 CYPRESS ST, PMB 7429, GRENEVILLE, TN | US

### B. Singleton S1 Entities & Nearest Fuzzy Lookalikes (5 Examples)
**Singleton 1:** S1 `S1-553482928` | Liberty Beyond P.C. | CA, 1900 Smokestack Avenue, Needles | US
  - Lookalike `S2-542878004` (Score 66.7): Bay Liberty | 373 THOMPSON RD, DECATUR TOWNSHIP, AL | US
**Singleton 2:** S1 `S1-162042007` | JC Care Pvt. Ltd. | 5/54/B Dum Dum Road, Kolkata, Howrah, West Bengal | India
  - Lookalike `S2-804670617` (Score 70.3): LZT Pvt. Ltd. Center | 8-B, MEER JI KA BAGH, OPPOSITE M.L.A. QUARTERS, M.I. ROAD, JAIPUR, राजस्थान | India
**Singleton 3:** S1 `S1-934892376` | Allied Target Inc | 755 Wood Avenue, Bridgeport, CT | US
  - Lookalike `S2-870549598` (Score 71.8): Allied Templeton, Inc. | 264 Banda Court, PLANADA, CA | US
**Singleton 4:** S1 `S1-836191081` | Floyd, Engracia W., DDS, DDS PC | 2102 Autumn Lane, City Of Kaukauna, WI | US
  - Lookalike `S3-167819158` (Score 52.3): Tani Lieder, DDS, DDS PC Harbor LP | 3211-3213 Jessup Rd, Cinciinnati, Ohio | US
**Singleton 5:** S1 `S1-498235056` | Vaughn Source | 2539 Jenkintown Road, Unit Unit 114, Abington Township, PA | US
  - Lookalike `S2-84763684` (Score 58.8): Vaughn Aerospace Pllc | 4007 BURDETTE ROAD, CARRSVILLE, VA | US

### C. Random France Records from Test Set (6 Examples)
- **test_source1:** `S1-41726612` | Mediation Visages Fetes EURL | Dunkerque, 3 Rue de la Bienfaisance, Hauts-de-France | France
- **test_source1:** `S1-845965189` | Lille Club | 15 BIS Rue d'Arcole, Lille, Hauts-de-France | France
- **test_source2:** `S2-19087216` | JX Sportive | 30 RUE DE MENIN, TOURCOING, Nord | France
- **test_source2:** `S2-824918948` | ASSOCIATION DE VALOIS EURL | N° 14 R. DE BOULMGNE, TOURCOING | France
- **test_source3:** `S3-366461218` | Tourcoing Loisirs | N°94 R Des Champs, Tourcoing | France
- **test_source3:** `S3-911557245` | Vigan SASU Comite | 8 Rue André Chenier, Lille, Hauts-de-France | France

## 7. Surprises & Pipeline Hazards

1. **High Singleton Frequency:** 5.6% (123,247) of S1 entities have zero matches; predicting empty correctly is worth 1.0 macro F0.5.
2. **France Domain Shift:** 1,694,445 test records are from France; country rules must be strictly dynamic.
3. **Zero Cross-Country Matches:** 100% of matched pairs share the exact same country; strict country-based blocking is 100% recall safe.
4. **Combinatorial Explosion:** Naive within-country test pairs reach 3,296,528,253,926 (S1xS2) and 3,428,041,312,286 (S1xS3), making multi-pass blocking essential.
5. **Low Exact Name Match Rate:** Matched names have low exact identity (<25%), requiring heavy fuzzy string similarity and token overlap features.
6. **Low Address Token Jaccard:** Address median Jaccard is low (<0.40), showing heavy abbreviations, missing locality tokens, and landmark variations.
7. **Non-ASCII Characters Across Sources:** Up to several percent of records have non-ASCII characters; unidecode and unicode normalization are mandatory.
8. **Inconsistent Postal Code Presence:** Address PIN codes are missing in a significant fraction of records; blocking cannot rely solely on postal codes.
9. **Legal Suffix Prevalence:** Tokens like LLC, INC, PVT, LTD dominate name tokens and can cause false positive fuzzy matches without domain suffix stripping.
10. **Target ID Multi-Assignment:** 0 S2 IDs and 0 S3 IDs are matched to multiple S1 entities; 1-to-1 matching constraints must NOT be enforced.

## 8. Validator Rules (validate_submission.py)

1. **Output Directory & Files:** Generates `output/matching_results.tsv` (required) and `output/candidate_pairs.tsv` (recommended).
2. **TSV Format Only:** Must be tab-separated (`\t`); comma-separated files fail validation immediately.
3. **Matching Header:** Header must strictly match `source1_entity_id\tmatched_entity_ids`.
4. **Candidate Header:** Header must strictly match `source1_entity_id\tcandidate_entity_ids`.
5. **Complete S1 Coverage:** Exactly one line per test S1 entity must be present in the output.
6. **No Duplicate Rows:** No duplicate `source1_entity_id` rows permitted in output files.
7. **Empty String for Singletons:** Singletons must have an empty string after the tab (`<s1_id>\t`).
8. **Comma-Separated Matches:** Matched IDs must be joined by commas without spaces (`S2-xxx,S3-yyy`).
9. **Valid ID Prefixes:** Matched IDs must start with `S2-` or `S3-`; `S1-` IDs are invalid in matched lists.
10. **No Self-Matches:** A Source 1 entity cannot be matched to itself.
11. **No Intra-Row Duplicate IDs:** The same match ID cannot appear multiple times in a single row.
12. **Candidate Subset Constraint:** Matched entity IDs must be a subset of candidate pairs (validator warns on violations).
13. **ID Existence Verification:** Matched IDs must exist in `test_source2.tsv` or `test_source3.tsv` (checked via `--check-ids`).
14. **Validation CLI:** Must pass `python3 utils/validate_submission.py --matching ... --candidate ... --test-dir ...` with exit code 0.

## 9. README Key Points

1. **Core Objective:** Deduplicated reference Source 1 must be linked to noisy records in Source 2 and Source 3.
2. **Evaluation Metric:** Scored by macro-averaged per-entity F0.5 (singletons count; empty prediction = 1.0 if correct).
3. **Input Data Schema:** TSV format with columns `entity_id`, `business_name`, `business_address`, and `country`.
4. **Country Generalization:** Training covers US and India; test set includes France (open-set country handling required).
5. **Name Noise Profile:** Abbreviations, legal suffixes, typos, trade names, and transliteration differences.
6. **Address Noise Profile:** Landmark references, missing postal codes/states, formatting shifts, and component reordering.
7. **No External APIs:** No external data lookups, web scraping, or LLM API calls permitted.
8. **Model Size Constraints:** Only open-source MIT/Apache-2.0 models up to 8B parameters permitted.
9. **Two-Stage Architecture:** Requires candidate blocking (`candidate_pairs.tsv`) followed by precision ranking/matching.
10. **Leaderboard Submission:** The only scored leaderboard file is `matching_results.tsv` in the `output/` directory.
