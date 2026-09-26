# Audit Report: Step 3 - Candidate Blocking & Recall Analysis

**Verdict:** RECALL OK (>=95%)

## 1. Recall Metrics Breakdown
| Metric / Dimension | Total | Recalled | Pairwise Recall | Complete S1 Recall |
|:---|---:|---:|---:|---:|
| **Overall** | 382264 | 380200 | **99.46%** | **98.36%** |
| Country: India | 153466 | 151784 | 98.90% | 96.78% |
| Country: US | 228798 | 228416 | 99.83% | 99.40% |

## 2. Generator Contribution & Blocking Efficiency
| Generator | Description | Recalled Pairs | Standalone Recall |
|:---|:---|---:|---:|
| **(a)** | Char 3-gram TF-IDF (top-15) | 344433 | 90.10% |
| **(b)** | Rare Token Inverted Index (top-15) | 284009 | 74.30% |
| **(c)** | Address Token Inverted Index (top-10) | 354999 | 92.87% |

- **Avg Candidates per S1 Entity:** 115.94
- **Reduction Ratio (India):** 99.944080% (avg 24.69 cands/record out of 44,159 S1s)
- **Reduction Ratio (US):** 99.962480% (avg 24.83 cands/record out of 66,182 S1s)

## 3. Analysis of Sampled Missed Pairs (15 Cases)
| # | Country | S1 Entity / Name / Address | Target Record / Name / Address | Likely Cause |
|:--|:---|:---|:---|:---|
| 1 | India | `S1-846065734`: Innovative Healthcare Limited | S.No. 44/A, Shoppers Orbit, Sh | `S3-121130409`: इनोवेटिव हेल्थकेयर Limited | Door No 451 S.no. 44/A, Pune,  | Extreme typo/phonetic distortion + distinct address representation |
| 2 | India | `S1-878469262`: Black Solutions Limited | Plot No.1, 2, 3, Ashok Manor K | `S3-19171666`: பிளாக் சொல்யூஷன்ஸ் லிமிடெட் | Plot No.1, Chennai, Kilkattala | Extreme typo/phonetic distortion + distinct address representation |
| 3 | India | `S1-622687062`: Shiv It Pvt Ltd | 168, 169 Gosavi Wasihappy, Col | `S2-820493525`: शिव आईटी प्रा. लि. | BIOCK G-896. 168, PUNE CITY, P | Extreme typo/phonetic distortion + distinct address representation |
| 4 | India | `S1-812600541`: Best City Healthcare Private L | Maharashtra, Shop No. B-60, Ko | `S3-878078774`: बेस्ट सिटी हेल्थकेयर प्राइवेट  | Shop No. B-60, Raigarh, MH, Na | Extreme typo/phonetic distortion + distinct address representation |
| 5 | India | `S1-567851275`: Innovative Energy Private Limi | 8Th Floor, Stesalit Tower, Blo | `S2-306125684`: ইনোভেটিভ এনার্জি প্রাইভেট লিমি | FLOOR, পশ্চিমবঙ্গ, NORTH 24 PA | Extreme typo/phonetic distortion + distinct address representation |
| 6 | India | `S1-396947832`: Unique Dream Consultancy Limit | 210, Floor -2Nd, 14, Doctor Ho | `S2-918020600`: यूनिक ड्रीम कंसल्टेंसी लिमिटेड | 21, MUMBAI CITY, MUMBAI, Mahar | Extreme typo/phonetic distortion + distinct address representation |
| 7 | India | `S1-770577002`: Sunrise Consulting Private Lim | 2Nd Floor, Flat No 2C, Block-J | `S3-956342702`: सनराइज कंसल्टिंग प्राइवेट लिमि | 2Nd Floor, Delhi, DL, North We | Extreme typo/phonetic distortion + distinct address representation |
| 8 | India | `S1-971217007`: One Technologies Private Limit | Flat No. A-2, Upper Ground Flo | `S2-863489738`: Limited One Private Services | FLAT NO. A-7-2, DELHI, SOUTH D | Extreme typo/phonetic distortion + distinct address representation |
| 9 | India | `S1-834691629`: Jai Management LLP | No.46-A, Heritage, Vigneshapar | `S2-838531813`: ஜெய் மேனேஜ்மெண்ட் எல்எல்பி | NO.046-A, CHENNAI, Tamil Nadu | Extreme typo/phonetic distortion + distinct address representation |
| 10 | India | `S1-977755211`: Life Power Private Limited | T-14, 1405, Blue Ridge Townshi | `S2-370463360`: लाइफ पावर प्राइवेट लिमिटेड | HN 313 T-14, PUNE, Maharashtra | Extreme typo/phonetic distortion + distinct address representation |
| 11 | US | `S1-652643632`: Signature Optics LLC | 9806 Lake Steilacoom Drive, La | `S2-164499597`: Signature LLC  Partners Enterp |  | Blank address; severe name deviation/typo |
| 12 | India | `S1-781064747`: Star Foundation Pvt Ltd | No11, 11Th Main, Binny Layout, | `S2-805610360`: ಸ್ಟಾರ್ ಫೌಂಡೇಶನ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟ | NO #11, BANGALORE, ಕರ್ನಾಟಕ | Extreme typo/phonetic distortion + distinct address representation |
| 13 | India | `S1-832553857`: White Energy Private Limited | 2Nd Floor, 1, Digambar Jain Te | `S2-185474865`: হোয়াইট এনার্জি প্রাইভেট লিমিট | HOWRAH, পশ্চিমবঙ্গ, H.NO 2ND F | Extreme typo/phonetic distortion + distinct address representation |
| 14 | US | `S1-801926875`: Physical Therapy Physicians | 25 Education Drive, Asheville, | `S3-523768870`: Physical Therapy Center Tradin |  | Blank address; severe name deviation/typo |
| 15 | India | `S1-182777890`: Bombay Technology | #334, Khata No 2678/3342Nd Sec | `S2-719468257`: ಬಾಂಬೆ ಟೆಕ್ನಾಲಜಿ | #334, BANGALORE, Karnataka | Extreme typo/phonetic distortion + distinct address representation |
