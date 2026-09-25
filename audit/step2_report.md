# Step 2 Report: Mini-World, Metrics & Normalization

## 1. Mini-World Dataset (5% Stratified Sample, Seed 42)
- **India:** S1: 44,159 | S2: 101,149 | S3: 105,992 | GT: 44,159 | Singletons: 5.50% | Distractors: S2=26.56%, S3=25.30% (Combined: 25.91%)
- **US:** S1: 66,182 | S2: 150,898 | S3: 158,317 | GT: 66,182 | Singletons: 5.57% | Distractors: S2=26.63%, S3=25.41% (Combined: 26.01%)
- **Overall:** S1: 110,341 | S2: 252,047 | S3: 264,309 | GT: 110,341 | Singletons: 5.54% | Distractors: 25.97% | Matches missing: 0

## 2. Test Verification & Metric Validation
- **Metric Tests (`tests/test_metric.py`):** 8/8 tests passed.
  - Statement test (pred 3 vs truth 2): F0.5 = 5/7 = 0.714286 (0.714).
  - Singletons: truth [] + pred [] = 1.0; truth [] + pred [...] = 0.0; truth [...] + pred [] = 0.0.

## 3. Quick Address & Character Set Checks
- **India (200k sample):** 0 (0.000%) contain 6-digit PIN on `\b\d{6}\b` (171 / 0.085% contiguous `\d{6}`).
- **US (200k sample):** 21,023 (10.51%) contain 5-digit ZIP on `\b\d{5}\b` (24,501 / 12.25% contiguous `\d{5}`).
- **Non-Latin Address Share:** India: 19.70% (Indic scripts), US: 0.00%, France (test): 6.44% (25.34% non-ASCII accents).

## 4. Top 15 Data-Driven Generic Tokens per Country
- **India:** limited, private, limittedd, india, praaivett, services, and, llp, company, com, center, corporation, brothers, solutions, industries.
- **US:** llc, inc, and, corporation, center, partners, s, c, com, group, company, care, of, l, limited.

## 5. Before / After Normalization Examples
1. **India (Gujarati Script):**
   - Before: `Dhanvarsha Law Chambers` | `SHYAM-ARCADE, FF-1, KOTHARIYA NAKA CHOWK, RAJKOT, ગુજરાત`
   - After: core=`['dhanvarsha','law','chambers']` | addr=`['shyam','arcade','ff','1','kothariya','naka','chowk','rajkot','gujraat']`
2. **India (Punjabi Script + Generic Suffixes):**
   - Before: `Best First Food Private (Limited)` | `Patiala Road, Scf 75 76 Shivalik Vihar, Zirakpur, ਪੰਜਾਬ, Mohali`
   - After: core=`['best','first','food']` | addr=`['patiala','road','scf','75','76','shivalik','vihar','zirakpur','pnjaab','mohali']`
3. **US (Leading 'The' + Street Abbreviation):**
   - Before: `The Empire Assets LLC` | `3505a 30th St, Indianapolis, Indiana`
   - After: core=`['empire','assets']` | addr=`['3505a','30th','street','indianapolis','indiana']`
4. **US (Repeated Token Glitch):**
   - Before: `Internal Medicine Care Group Associates Associates` | `11014 COURTSHIRE ROAD, N/A, HOUSTON, TX`
   - After: core=`['internal','medicine']` | addr=`['11014','courtshire','road','n','a','houston','tx']`
5. **Typo in Name & Locality:**
   - Before: `Colonial Bctoein LLC` | `00 CYPRESS STREET, GRENEVILLE, TN`
   - After: core=`['colonial','bctoein']` | addr=`['00','cypress','street','greneville','tn']`
6. **India (Bengali Script & Abbreviation):**
   - Before: `Kolkata Ínfratel Private Limited` | `43/3 HAZRA RD, KOLKATA, CALCUTTA, পশ্চিমবঙ্গ`
   - After: core=`['kolkata','infratel']` | addr=`['43','3','hazra','road','kolkata','calcutta','pshcimbngg']`

## 6. Pipeline Hazards & Observations
- **Synthetic Token Noise:** India's top generics capture synthetic noise tokens `limittedd` (rank 3, df=24,045) and `praaivett` (rank 5, df=13,810).
- **Zero PIN Codes in India:** Address blocking must not rely on 6-digit postal PINs for India (0.00% valid PIN presence).
- **Script Transliteration:** 19.70% of Indian records require unidecode/NFKD transliteration for state matching.
