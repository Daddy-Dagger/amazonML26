#!/usr/bin/env python3
"""
Pairwise Feature Extraction for Business Entity Resolution.
Computes 21 name, address, and candidate-context features for every (S1, candidate)
pair generated during blocking:
  - 19 base features (name jaccards, levenshtein, token ratios, initials, length diff,
    suffix agreement, address jaccards, blank indicators, idf address overlap, house number
    soft match, city fuzzy match, rank in candidates, score gap, candidate count, token rarity)
  - 2 distinctive token features:
      * distinctive_token_jaccard: Jaccard on name tokens after removing top-40 generic tokens
        and any token with document frequency > 90th percentile.
      * shared_distinctive_token_count: Raw count of non-generic tokens shared.
Labels each pair 1 (true match in ground truth) or 0 (candidate negative / hard negative).
Assigns non-leaking S1 entity splits (70% train, 15% val, 15% test).
Saves output to data_mini/pairwise_features.parquet using streaming PyArrow chunks.
Strictly country-agnostic.
"""

import os
import sys
import re
import time
import random
import argparse
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_leading_digits(tokens) -> str:
    """Extracts first sequence of digits found in leading address tokens."""
    if tokens is None:
        return ""
    try:
        t_len = len(tokens)
    except TypeError:
        return ""
    if t_len == 0:
        return ""
    for t in tokens:
        m = re.search(r"(\d+)", str(t))
        if m:
            return m.group(1)
    return ""


def build_s1_country_cache(
    s1_df_country,
    s1_cand_count_map: Dict[str, int],
    common_tokens: Set[str],
):
    """Precomputes lookup structures and TF-IDF matrix for S1 records of a single country."""
    s1_ids = s1_df_country["entity_id"].tolist()
    n_s1 = len(s1_ids)
    s1_id_to_idx = {eid: i for i, eid in enumerate(s1_ids)}

    s1_nf_strs = [
        " ".join(list(toks)) if toks is not None and len(toks) > 0 else ""
        for toks in s1_df_country["name_full"]
    ]
    s1_nf_sets = [
        set(list(toks)) if toks is not None and len(toks) > 0 else set()
        for toks in s1_df_country["name_full"]
    ]
    s1_nc_sets = [
        set(list(toks)) if toks is not None and len(toks) > 0 else set()
        for toks in s1_df_country["name_core"]
    ]
    s1_ad_sets = [
        set(list(toks)) if toks is not None and len(toks) > 0 else set()
        for toks in s1_df_country["address"]
    ]
    s1_nf_lens = [len(s) for s in s1_nf_sets]
    s1_sufs = [s1_nf_sets[i] - s1_nc_sets[i] for i in range(n_s1)]
    s1_inits = [
        sorted([t[0] for t in list(toks) if t]) if toks is not None and len(toks) > 0 else []
        for toks in s1_df_country["name_full"]
    ]
    s1_house = [
        extract_leading_digits(toks)
        for toks in s1_df_country["address"]
    ]
    s1_cities = [
        " ".join(list(toks)[-2:]) if (toks is not None and len(toks) >= 2) else (" ".join(list(toks)) if (toks is not None and len(toks) > 0) else "")
        for toks in s1_df_country["address"]
    ]
    s1_num_cands = [s1_cand_count_map.get(eid, 0) for eid in s1_ids]

    # Precompute distinctive token sets (excluding generic + doc_freq > 90th percentile)
    s1_dist_sets = [s - common_tokens for s in s1_nf_sets]

    # Char 3-gram TF-IDF
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
    X_s1 = vec.fit_transform(s1_nf_strs)

    return {
        "s1_ids": s1_ids,
        "n_s1": n_s1,
        "s1_id_to_idx": s1_id_to_idx,
        "s1_nf_strs": s1_nf_strs,
        "s1_nf_sets": s1_nf_sets,
        "s1_nc_sets": s1_nc_sets,
        "s1_ad_sets": s1_ad_sets,
        "s1_nf_lens": s1_nf_lens,
        "s1_sufs": s1_sufs,
        "s1_inits": s1_inits,
        "s1_house": s1_house,
        "s1_cities": s1_cities,
        "s1_num_cands": s1_num_cands,
        "s1_dist_sets": s1_dist_sets,
        "vec": vec,
        "X_s1": X_s1,
    }


def extract_features(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    idf_path: str,
    candidate_record_path: str,
    candidate_s1_path: str,
    gt_path: str,
    out_features_path: str,
    batch_size: int = 10000,
    seed: int = 42,
):
    t_start = time.time()
    print("=== Phase 5: Pairwise Feature Extraction (v2 with Distinctive Token Features) ===")

    # 1. Load Ground Truth pairs
    print(f"Loading ground truth from: {gt_path}")
    gt_tbl = pq.read_table(gt_path).to_pandas()
    gt_pairs = set()
    for s1_id, m_str in zip(gt_tbl["source1_entity_id"], gt_tbl["matched_entity_ids"]):
        if m_str is not None and str(m_str).strip():
            for m in str(m_str).split(","):
                m_clean = m.strip()
                if m_clean:
                    gt_pairs.add((s1_id, m_clean))
    print(f"Loaded {len(gt_pairs)} true match pairs.")

    # 2. Load candidate counts per S1 from candidates_s1_grouped.parquet
    print(f"Loading S1 candidates from: {candidate_s1_path}")
    s1_grp_tbl = pq.read_table(candidate_s1_path).to_pandas()
    s1_cand_count_map = {
        s1_id: len(cands) for s1_id, cands in zip(s1_grp_tbl["source1_entity_id"], s1_grp_tbl["candidate_entity_ids"])
    }

    # 3. Load IDF weights per country and identify tokens > 90th percentile doc_freq
    print(f"Loading IDF weights from: {idf_path}")
    idf_tbl = pq.read_table(idf_path).to_pandas()
    idf_by_country = {}
    common_tokens_by_country = {}
    for country, group in idf_tbl.groupby("country"):
        idf_by_country[country] = dict(zip(group["token"], group["idf"]))
        p90 = np.percentile(group["doc_freq"], 90)
        common_tokens = set(group[(group["doc_freq"] > p90) | (group["is_generic"] == True)]["token"])
        common_tokens_by_country[country] = common_tokens
        print(f"[{country}] p90 doc_freq={p90:.1f} -> {len(common_tokens)} common/generic tokens identified")

    # 4. Load S1, S2, S3 normalized tables
    req_cols = ["entity_id", "country", "name_full", "name_core", "address"]
    print("Loading normalized source tables...")
    s1_tbl = pq.read_table(s1_path, columns=req_cols).to_pandas()
    s2_tbl = pq.read_table(s2_path, columns=req_cols).to_pandas()
    s3_tbl = pq.read_table(s3_path, columns=req_cols).to_pandas()

    query_sources_df = {
        "S2": s2_tbl,
        "S3": s3_tbl,
    }

    # 5. Deterministic 70/15/15 S1 entity split (stratified by country)
    print(f"Assigning entity splits (70% train, 15% val, 15% test, seed={seed})...")
    s1_split_map = {}
    split_rows = []
    rng = random.Random(seed)
    for country, grp in s1_tbl.groupby("country"):
        country_s1_ids = grp["entity_id"].tolist()
        rng.shuffle(country_s1_ids)
        n_c = len(country_s1_ids)
        n_train = int(0.70 * n_c)
        n_val = int(0.15 * n_c)
        for i, eid in enumerate(country_s1_ids):
            if i < n_train:
                sp = "train"
            elif i < n_train + n_val:
                sp = "val"
            else:
                sp = "test"
            s1_split_map[eid] = sp
            split_rows.append({"source1_entity_id": eid, "split": sp, "country": country})

    splits_path = os.path.join(os.path.dirname(os.path.abspath(out_features_path)), "s1_splits.parquet")
    pq.write_table(pa.Table.from_pylist(split_rows), splits_path)
    print(f"S1 splits saved to: {splits_path}")

    # 6. Load candidate_pairs_by_record.parquet
    print(f"Loading record candidates from: {candidate_record_path}")
    record_cands_tbl = pq.read_table(candidate_record_path).to_pandas()
    distinct_countries = sorted(record_cands_tbl["country"].unique())
    print(f"Countries to process: {distinct_countries}")

    # Setup PyArrow schema and ParquetWriter
    feature_schema = pa.schema([
        ("source1_entity_id", pa.string()),
        ("record_id", pa.string()),
        ("country", pa.string()),
        ("split", pa.string()),
        ("label", pa.int8()),
        ("token_jaccard_full", pa.float32()),
        ("token_jaccard_core", pa.float32()),
        ("char_3gram_cosine", pa.float32()),
        ("levenshtein_ratio", pa.float32()),
        ("token_sort_ratio", pa.float32()),
        ("token_set_ratio", pa.float32()),
        ("initials_match", pa.float32()),
        ("name_length_diff", pa.float32()),
        ("legal_suffix_agreement", pa.float32()),
        ("token_jaccard_addr", pa.float32()),
        ("is_s1_address_blank", pa.float32()),
        ("is_cand_address_blank", pa.float32()),
        ("idf_weighted_token_overlap_addr", pa.float32()),
        ("house_number_soft_match", pa.float32()),
        ("city_fuzzy_match", pa.float32()),
        ("rank_in_candidates", pa.float32()),
        ("score_gap_to_next_best", pa.float32()),
        ("num_candidates_for_this_s1", pa.float32()),
        ("name_token_rarity", pa.float32()),
        ("distinctive_token_jaccard", pa.float32()),
        ("shared_distinctive_token_count", pa.float32()),
    ])

    os.makedirs(os.path.dirname(os.path.abspath(out_features_path)), exist_ok=True)
    writer = pq.ParquetWriter(out_features_path, schema=feature_schema, compression="snappy")

    total_pairs_written = 0
    total_positives_written = 0

    for country in distinct_countries:
        c_t0 = time.time()
        print(f"\n[{country}] Caching S1 lookup structures...")
        s1_country = s1_tbl[s1_tbl["country"] == country].reset_index(drop=True)
        common_toks = common_tokens_by_country.get(country, set())
        s1_cache = build_s1_country_cache(s1_country, s1_cand_count_map, common_toks)
        idf_map = idf_by_country.get(country, {})
        vec = s1_cache["vec"]
        X_s1 = s1_cache["X_s1"]
        s1_ids = s1_cache["s1_ids"]
        s1_id_to_idx = s1_cache["s1_id_to_idx"]

        # Filter candidate records for this country
        cands_country = record_cands_tbl[record_cands_tbl["country"] == country].reset_index(drop=True)
        n_records = len(cands_country)
        print(f"[{country}] Processing {n_records} records...")

        # Build query record lookup indexed by record_id
        q_record_map = {}
        for src_label in ["S2", "S3"]:
            src_df = query_sources_df[src_label]
            src_country_df = src_df[src_df["country"] == country]
            for eid, nf, nc, ad in zip(
                src_country_df["entity_id"],
                src_country_df["name_full"],
                src_country_df["name_core"],
                src_country_df["address"],
            ):
                nf_list = list(nf) if nf is not None and len(nf) > 0 else []
                nc_list = list(nc) if nc is not None and len(nc) > 0 else []
                ad_list = list(ad) if ad is not None and len(ad) > 0 else []

                nf_set = set(nf_list)
                nc_set = set(nc_list)
                ad_set = set(ad_list)
                q_record_map[eid] = {
                    "nf_str": " ".join(nf_list),
                    "nf_set": nf_set,
                    "nc_set": nc_set,
                    "ad_set": ad_set,
                    "nf_len": len(nf_set),
                    "suf": nf_set - nc_set,
                    "dist_set": nf_set - common_toks,
                    "inits": sorted([t[0] for t in nf_list if t]),
                    "house": extract_leading_digits(ad_list),
                    "city": " ".join(ad_list[-2:]) if len(ad_list) >= 2 else (" ".join(ad_list)),
                }

        # Batch buffer lists
        batch_col_s1_id = []
        batch_col_rec_id = []
        batch_col_country = []
        batch_col_split = []
        batch_col_label = []
        batch_col_j_full = []
        batch_col_j_core = []
        batch_col_cos = []
        batch_col_lev = []
        batch_col_ts = []
        batch_col_tset = []
        batch_col_init = []
        batch_col_ldiff = []
        batch_col_suf = []
        batch_col_j_ad = []
        batch_col_b1 = []
        batch_col_b2 = []
        batch_col_w_ad = []
        batch_col_house = []
        batch_col_city = []
        batch_col_rank = []
        batch_col_gap = []
        batch_col_num_cands = []
        batch_col_rarity = []
        batch_col_dist_j = []
        batch_col_dist_cnt = []

        for b_start in range(0, n_records, batch_size):
            b_end = min(b_start + batch_size, n_records)
            chunk = cands_country.iloc[b_start:b_end]

            chunk_rec_ids = chunk["record_id"].tolist()
            chunk_cand_s1_lists = chunk["candidate_s1_ids"].tolist()

            # Pre-transform chunk query strings
            chunk_q_strs = [q_record_map[rid]["nf_str"] for rid in chunk_rec_ids]
            X_q_chunk = vec.transform(chunk_q_strs)

            for i in range(len(chunk_rec_ids)):
                r_id = chunk_rec_ids[i]
                cand_s1s = list(chunk_cand_s1_lists[i]) if chunk_cand_s1_lists[i] is not None else []
                if len(cand_s1s) == 0:
                    continue

                q_data = q_record_map[r_id]
                q_str = q_data["nf_str"]
                q_nf_set = q_data["nf_set"]
                q_nc_set = q_data["nc_set"]
                q_ad_set = q_data["ad_set"]
                q_dist_set = q_data["dist_set"]
                q_len = q_data["nf_len"]
                q_suf = q_data["suf"]
                q_inits = q_data["inits"]
                q_house = q_data["house"]
                q_city = q_data["city"]
                is_q_ad_blank = 1.0 if not q_ad_set else 0.0

                # Compute char_3gram_cosine for all candidates of this record
                cand_idx_list = [s1_id_to_idx[cid] for cid in cand_s1s]
                q_row_vec = X_q_chunk[i]
                cos_sims = X_s1[cand_idx_list].dot(q_row_vec.T).toarray().ravel()

                # Sort candidates by cos_sim descending to derive rank and score gaps
                sorted_order = np.argsort(-cos_sims)
                s_top1 = cos_sims[sorted_order[0]]
                s_top2 = cos_sims[sorted_order[1]] if len(sorted_order) > 1 else 0.0

                for rank_0, c_pos in enumerate(sorted_order):
                    s1_id = cand_s1s[c_pos]
                    s1_idx = cand_idx_list[c_pos]
                    cos_val = float(cos_sims[c_pos])
                    rank_1 = float(rank_0 + 1)

                    # Score gap: leader gets s_top1 - s_top2; runner-ups get score - s_top1
                    if rank_0 == 0:
                        score_gap = float(s_top1 - s_top2)
                    else:
                        score_gap = float(cos_val - s_top1)

                    # S1 cached attributes
                    s1_s = s1_cache["s1_nf_strs"][s1_idx]
                    s1_nf_set = s1_cache["s1_nf_sets"][s1_idx]
                    s1_nc_set = s1_cache["s1_nc_sets"][s1_idx]
                    s1_ad_set = s1_cache["s1_ad_sets"][s1_idx]
                    s1_dist_set = s1_cache["s1_dist_sets"][s1_idx]
                    s1_len = s1_cache["s1_nf_lens"][s1_idx]
                    s1_suf = s1_cache["s1_sufs"][s1_idx]
                    s1_inits = s1_cache["s1_inits"][s1_idx]
                    s1_house = s1_cache["s1_house"][s1_idx]
                    s1_city = s1_cache["s1_cities"][s1_idx]

                    # Token Jaccard Name Full
                    u_f = len(s1_nf_set | q_nf_set)
                    j_full = (len(s1_nf_set & q_nf_set) / float(u_f)) if u_f > 0 else 0.0

                    # Token Jaccard Name Core
                    u_c = len(s1_nc_set | q_nc_set)
                    j_core = (len(s1_nc_set & q_nc_set) / float(u_c)) if u_c > 0 else 0.0

                    # Distinctive Token Jaccard & Count (task 1 additions)
                    shared_dist = s1_dist_set & q_dist_set
                    u_dist = s1_dist_set | q_dist_set
                    dist_j = (len(shared_dist) / float(len(u_dist))) if len(u_dist) > 0 else 0.0
                    dist_cnt = float(len(shared_dist))

                    # RapidFuzz string metrics
                    lev = fuzz.ratio(s1_s, q_str) / 100.0
                    ts = fuzz.token_sort_ratio(s1_s, q_str) / 100.0
                    tset = fuzz.token_set_ratio(s1_s, q_str) / 100.0

                    # Initials
                    im = 1.0 if (s1_inits and s1_inits == q_inits) else 0.0

                    # Name length diff
                    ldiff = float(abs(s1_len - q_len))

                    # Legal suffix agreement
                    if s1_suf and s1_suf == q_suf:
                        suf_agree = 1.0
                    elif not s1_suf and not q_suf:
                        suf_agree = 0.5
                    else:
                        suf_agree = 0.0

                    # Address features
                    is_s1_ad_blank = 1.0 if not s1_ad_set else 0.0
                    u_a = len(s1_ad_set | q_ad_set)
                    j_ad = (len(s1_ad_set & q_ad_set) / float(u_a)) if (s1_ad_set and q_ad_set and u_a > 0) else 0.0

                    # Address IDF weighted overlap
                    overlap_ad = s1_ad_set & q_ad_set
                    w_ad = float(sum(idf_map.get(t, 1.0) for t in overlap_ad))

                    # House number soft match (Levenshtein distance <= 2)
                    if s1_house and q_house:
                        hm = 1.0 if distance.Levenshtein.distance(s1_house, q_house) <= 2 else 0.0
                    else:
                        hm = 0.0

                    # City fuzzy match
                    if s1_city and q_city:
                        cm = fuzz.token_sort_ratio(s1_city, q_city) / 100.0
                    else:
                        cm = 0.0

                    # Context features
                    num_cands = float(s1_cache["s1_num_cands"][s1_idx])
                    shared_name_tokens = s1_nf_set & q_nf_set
                    if shared_name_tokens:
                        rarity = float(sum(idf_map.get(t, 1.0) for t in shared_name_tokens) / len(shared_name_tokens))
                    else:
                        rarity = 0.0

                    # Split assignment
                    s1_split = s1_split_map[s1_id]

                    # Label
                    is_match = 1 if (s1_id, r_id) in gt_pairs else 0
                    if is_match:
                        total_positives_written += 1

                    # Append to batch lists
                    batch_col_s1_id.append(s1_id)
                    batch_col_rec_id.append(r_id)
                    batch_col_country.append(country)
                    batch_col_split.append(s1_split)
                    batch_col_label.append(is_match)
                    batch_col_j_full.append(j_full)
                    batch_col_j_core.append(j_core)
                    batch_col_cos.append(cos_val)
                    batch_col_lev.append(lev)
                    batch_col_ts.append(ts)
                    batch_col_tset.append(tset)
                    batch_col_init.append(im)
                    batch_col_ldiff.append(ldiff)
                    batch_col_suf.append(suf_agree)
                    batch_col_j_ad.append(j_ad)
                    batch_col_b1.append(is_s1_ad_blank)
                    batch_col_b2.append(is_q_ad_blank)
                    batch_col_w_ad.append(w_ad)
                    batch_col_house.append(hm)
                    batch_col_city.append(cm)
                    batch_col_rank.append(rank_1)
                    batch_col_gap.append(score_gap)
                    batch_col_num_cands.append(num_cands)
                    batch_col_rarity.append(rarity)
                    batch_col_dist_j.append(dist_j)
                    batch_col_dist_cnt.append(dist_cnt)

            # Flush to parquet if batch list grows large
            if len(batch_col_label) >= 200000:
                n_b = len(batch_col_label)
                batch_tbl = pa.Table.from_arrays(
                    [
                        pa.array(batch_col_s1_id, type=pa.string()),
                        pa.array(batch_col_rec_id, type=pa.string()),
                        pa.array(batch_col_country, type=pa.string()),
                        pa.array(batch_col_split, type=pa.string()),
                        pa.array(batch_col_label, type=pa.int8()),
                        pa.array(batch_col_j_full, type=pa.float32()),
                        pa.array(batch_col_j_core, type=pa.float32()),
                        pa.array(batch_col_cos, type=pa.float32()),
                        pa.array(batch_col_lev, type=pa.float32()),
                        pa.array(batch_col_ts, type=pa.float32()),
                        pa.array(batch_col_tset, type=pa.float32()),
                        pa.array(batch_col_init, type=pa.float32()),
                        pa.array(batch_col_ldiff, type=pa.float32()),
                        pa.array(batch_col_suf, type=pa.float32()),
                        pa.array(batch_col_j_ad, type=pa.float32()),
                        pa.array(batch_col_b1, type=pa.float32()),
                        pa.array(batch_col_b2, type=pa.float32()),
                        pa.array(batch_col_w_ad, type=pa.float32()),
                        pa.array(batch_col_house, type=pa.float32()),
                        pa.array(batch_col_city, type=pa.float32()),
                        pa.array(batch_col_rank, type=pa.float32()),
                        pa.array(batch_col_gap, type=pa.float32()),
                        pa.array(batch_col_num_cands, type=pa.float32()),
                        pa.array(batch_col_rarity, type=pa.float32()),
                        pa.array(batch_col_dist_j, type=pa.float32()),
                        pa.array(batch_col_dist_cnt, type=pa.float32()),
                    ],
                    schema=feature_schema,
                )
                writer.write_table(batch_tbl)
                total_pairs_written += n_b

                # Reset batch lists
                batch_col_s1_id.clear()
                batch_col_rec_id.clear()
                batch_col_country.clear()
                batch_col_split.clear()
                batch_col_label.clear()
                batch_col_j_full.clear()
                batch_col_j_core.clear()
                batch_col_cos.clear()
                batch_col_lev.clear()
                batch_col_ts.clear()
                batch_col_tset.clear()
                batch_col_init.clear()
                batch_col_ldiff.clear()
                batch_col_suf.clear()
                batch_col_j_ad.clear()
                batch_col_b1.clear()
                batch_col_b2.clear()
                batch_col_w_ad.clear()
                batch_col_house.clear()
                batch_col_city.clear()
                batch_col_rank.clear()
                batch_col_gap.clear()
                batch_col_num_cands.clear()
                batch_col_rarity.clear()
                batch_col_dist_j.clear()
                batch_col_dist_cnt.clear()

                print(f"  [{country}] {b_end}/{n_records} records processed ({total_pairs_written:,} pairs written)...")

        # Flush any remaining rows for this country
        if batch_col_label:
            n_b = len(batch_col_label)
            batch_tbl = pa.Table.from_arrays(
                [
                    pa.array(batch_col_s1_id, type=pa.string()),
                    pa.array(batch_col_rec_id, type=pa.string()),
                    pa.array(batch_col_country, type=pa.string()),
                    pa.array(batch_col_split, type=pa.string()),
                    pa.array(batch_col_label, type=pa.int8()),
                    pa.array(batch_col_j_full, type=pa.float32()),
                    pa.array(batch_col_j_core, type=pa.float32()),
                    pa.array(batch_col_cos, type=pa.float32()),
                    pa.array(batch_col_lev, type=pa.float32()),
                    pa.array(batch_col_ts, type=pa.float32()),
                    pa.array(batch_col_tset, type=pa.float32()),
                    pa.array(batch_col_init, type=pa.float32()),
                    pa.array(batch_col_ldiff, type=pa.float32()),
                    pa.array(batch_col_suf, type=pa.float32()),
                    pa.array(batch_col_j_ad, type=pa.float32()),
                    pa.array(batch_col_b1, type=pa.float32()),
                    pa.array(batch_col_b2, type=pa.float32()),
                    pa.array(batch_col_w_ad, type=pa.float32()),
                    pa.array(batch_col_house, type=pa.float32()),
                    pa.array(batch_col_city, type=pa.float32()),
                    pa.array(batch_col_rank, type=pa.float32()),
                    pa.array(batch_col_gap, type=pa.float32()),
                    pa.array(batch_col_num_cands, type=pa.float32()),
                    pa.array(batch_col_rarity, type=pa.float32()),
                    pa.array(batch_col_dist_j, type=pa.float32()),
                    pa.array(batch_col_dist_cnt, type=pa.float32()),
                ],
                schema=feature_schema,
            )
            writer.write_table(batch_tbl)
            total_pairs_written += n_b
            batch_col_s1_id.clear()
            batch_col_rec_id.clear()
            batch_col_country.clear()
            batch_col_split.clear()
            batch_col_label.clear()
            batch_col_j_full.clear()
            batch_col_j_core.clear()
            batch_col_cos.clear()
            batch_col_lev.clear()
            batch_col_ts.clear()
            batch_col_tset.clear()
            batch_col_init.clear()
            batch_col_ldiff.clear()
            batch_col_suf.clear()
            batch_col_j_ad.clear()
            batch_col_b1.clear()
            batch_col_b2.clear()
            batch_col_w_ad.clear()
            batch_col_house.clear()
            batch_col_city.clear()
            batch_col_rank.clear()
            batch_col_gap.clear()
            batch_col_num_cands.clear()
            batch_col_rarity.clear()
            batch_col_dist_j.clear()
            batch_col_dist_cnt.clear()

        print(f"[{country}] Completed in {time.time() - c_t0:.2f}s")

    writer.close()
    elapsed = time.time() - t_start
    print(f"\n=== Pairwise Feature Extraction Complete in {elapsed:.2f}s ===")
    print(f"Total pairs written: {total_pairs_written:,} (Positives: {total_positives_written:,}, Negatives: {total_pairs_written - total_positives_written:,})")
    print(f"Features file: {out_features_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract pairwise features for matching.")
    parser.add_argument("--s1-path", type=str, default="data_mini/source1_normalized.parquet")
    parser.add_argument("--s2-path", type=str, default="data_mini/source2_normalized.parquet")
    parser.add_argument("--s3-path", type=str, default="data_mini/source3_normalized.parquet")
    parser.add_argument("--idf-path", type=str, default="data_mini/idf_weights.parquet")
    parser.add_argument(
        "--candidate-record-path",
        type=str,
        default="data_mini/candidate_pairs_by_record.parquet",
    )
    parser.add_argument(
        "--candidate-s1-path",
        type=str,
        default="data_mini/candidates_s1_grouped.parquet",
    )
    parser.add_argument("--ground-truth-path", type=str, default="data_mini/ground_truth.parquet")
    parser.add_argument(
        "--out-features-path",
        type=str,
        default="data_mini/pairwise_features.parquet",
    )
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extract_features(
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        idf_path=args.idf_path,
        candidate_record_path=args.candidate_record_path,
        candidate_s1_path=args.candidate_s1_path,
        gt_path=args.ground_truth_path,
        out_features_path=args.out_features_path,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
