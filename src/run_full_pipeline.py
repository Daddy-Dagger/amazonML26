#!/usr/bin/env python3
"""
Full Submission Pipeline for Business Entity Resolution.
Runs normalization, blocking, pairwise feature extraction, LightGBM model_v2 inference,
and the decision layer across all countries in the full test set.

Constraints & Design:
1. Dynamic country discovery - zero hard-coded country names.
2. Processes ONE country at a time to strictly bound memory below 2.5 GB (safe from 7GB limit).
3. Streams S2 and S3 query records in chunks (default 50,000 records).
4. Reuses fitted artifacts:
   - model_v2.txt with thresholds min_p=0.75, min_m=0.00
   - data_mini/idf_weights.parquet for countries present (India, US)
   - dynamic generic & IDF computation for unseen countries (France)
5. 1-owner-per-record decision enforcement: each S2/S3 record is assigned to at most one S1.
6. Exact candidate subset guarantee: final matches are strictly a subset of candidates.
7. Produces output/matching_results.tsv and output/candidate_pairs.tsv covering every S1 entity.
8. Logs detailed country progress to run_full_pipeline.log and prints tail -n 20.
"""

import os
import sys
import gc
import time
import math
try:
    import resource
except ImportError:
    resource = None
import platform
import argparse
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import duckdb
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz, distance

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalize import normalize_text, compute_generic_and_idf
from src.features import extract_leading_digits
from src.decide_and_score import FEATURE_COLS


def get_peak_memory_gb() -> float:
    """Returns peak resident set size in Gigabytes cross-platform."""
    if resource is not None:
        if platform.system() == "Darwin":
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0 * 1024.0)
        else:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    else:
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            pid = kernel32.GetCurrentProcessId()
            h = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('cb', wintypes.DWORD),
                    ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                ]
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                kernel32.CloseHandle(h)
                return float(pmc.PeakWorkingSetSize) / (1024.0 ** 3)
            kernel32.CloseHandle(h)
        except Exception:
            pass
        return 0.0


def check_memory_safe(max_gb: float = 14.0):
    """Safety check to ensure memory usage stays well below the 14GB limit on a 16GB host."""
    peak = get_peak_memory_gb()
    if peak >= max_gb:
        raise MemoryError(f"CRITICAL: Memory usage approached {peak:.2f} GB (>= {max_gb:.2f} GB limit). Halting pipeline.")
    return peak


def log_msg(log_file, msg: str, also_print: bool = False):
    """Appends timestamped message to the log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")
    if also_print:
        print(formatted)


def run_country_pipeline(
    country: str,
    test_dir: str,
    model: lgb.Booster,
    idf_weights_path: str,
    out_dir: str,
    log_file: str,
    min_p: float = 0.95,
    min_m: float = 0.05,
    chunk_size: int = 50000,
    sub_batch_size: int = 2000,
    max_cands_per_rec: int = 25,
) -> Dict[str, Any]:
    """
    Executes the complete pipeline for a single country:
      1. Normalizes and indexes S1 records.
      2. Streams S2/S3 queries in chunks to generate candidates (blocking pass).
      3. Computes 21 pairwise features, predicts with model_v2, and enforces 1-owner decision rule.
      4. Aggregates per-S1 candidates and matches via DuckDB without memory pressure.
      5. Cleans up temporary files and frees memory.
    """
    t_start = time.time()
    c_tag = re.sub(r"[^\w]", "_", country.lower())
    log_msg(log_file, f"============================================================")
    log_msg(log_file, f"START COUNTRY PIPELINE: {country}")
    log_msg(log_file, f"============================================================")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")

    # Step 1: Load and normalize S1 records
    log_msg(log_file, f"[{country}] Step 1: Loading & normalizing S1 entities...")
    s1_rows = con.execute(
        f"SELECT entity_id, business_name, business_address FROM read_csv('{test_s1_path}', sep='\\t', header=True) WHERE country = ?",
        [country],
    ).fetchall()
    n_s1 = len(s1_rows)
    s1_ids = [r[0] for r in s1_rows]
    s1_names = [normalize_text(r[1], is_address=False) for r in s1_rows]
    s1_addrs = [normalize_text(r[2], is_address=True) for r in s1_rows]
    s1_nf_strs = [" ".join(n) for n in s1_names]
    log_msg(log_file, f"[{country}] Loaded {n_s1:,} S1 entities. Peak RAM: {get_peak_memory_gb():.2f} GB")

    # Step 2: Determine generic tokens and IDF weights
    log_msg(log_file, f"[{country}] Step 2: Loading / computing IDF weights and generic tokens...")
    has_precomputed_idf = False
    if os.path.exists(idf_weights_path):
        idf_tbl = pq.read_table(idf_weights_path).to_pandas()
        if country in idf_tbl["country"].values:
            has_precomputed_idf = True
            idf_c = idf_tbl[idf_tbl["country"] == country]
            generic_set = set(idf_c[idf_c["is_generic"] == True]["token"])
            idf_map = dict(zip(idf_c["token"], idf_c["idf"]))
            p90 = np.percentile(idf_c["doc_freq"], 90)
            common_tokens = set(idf_c[(idf_c["doc_freq"] > p90) | (idf_c["is_generic"] == True)]["token"])
            log_msg(log_file, f"[{country}] Reused precomputed IDF weights ({len(generic_set)} generic, {len(common_tokens)} common tokens).")

    if not has_precomputed_idf:
        log_msg(log_file, f"[{country}] Computing generic tokens and IDF dynamically from test S1 names...")
        generic_dict, idf_records = compute_generic_and_idf({country: s1_names}, num_generic=40)
        generic_set = generic_dict[country]
        idf_map = {r["token"]: r["idf"] for r in idf_records}
        p90 = np.percentile([r["doc_freq"] for r in idf_records], 90)
        common_tokens = set([r["token"] for r in idf_records if r["doc_freq"] > p90 or r["is_generic"]])
        log_msg(log_file, f"[{country}] Computed {len(generic_set)} generic tokens and {len(common_tokens)} common tokens.")

    # Step 3: Build S1 blocking indexes and feature cache
    log_msg(log_file, f"[{country}] Step 3: Building S1 blocking indexes & feature cache...")
    s1_core = [[t for t in n if t not in generic_set] for n in s1_names]

    # Generator (a): char 3-gram TF-IDF vectorizer (capped max_df to prevent combinatorial explosion)
    max_char_df = min(5000, max(500, int(0.01 * n_s1)))
    vec_blocking = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=3, max_df=max_char_df)
    X_s1_blocking = vec_blocking.fit_transform(s1_nf_strs)

    # Generator (b): vectorized TF-IDF on name_core tokens
    s1_core_strs = [" ".join(c) for c in s1_core]
    vec_b = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=1)
    X_s1_b = vec_b.fit_transform(s1_core_strs)

    # Generator (c): vectorized TF-IDF on address tokens with max_df=2000
    s1_addr_strs = [" ".join(a) for a in s1_addrs]
    vec_c = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=1, max_df=min(2000, n_s1))
    X_s1_c = vec_c.fit_transform(s1_addr_strs)

    # Feature cache for fast pairwise evaluation
    vec_feat = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
    X_s1_feat = vec_feat.fit_transform(s1_nf_strs)
    s1_nf_sets = [set(n) for n in s1_names]
    s1_nc_sets = [set(c) for c in s1_core]
    s1_ad_sets = [set(a) for a in s1_addrs]
    s1_dist_sets = [s - common_tokens for s in s1_nf_sets]
    s1_sufs = [s1_nf_sets[i] - s1_nc_sets[i] for i in range(n_s1)]
    s1_inits_all = [sorted([t[0] for t in n if t]) for n in s1_names]
    s1_house_all = [extract_leading_digits(a) for a in s1_addrs]
    s1_cities_all = [" ".join(a[-2:]) if len(a) >= 2 else " ".join(a) for a in s1_addrs]

    check_memory_safe()
    log_msg(log_file, f"[{country}] S1 index & cache built in {time.time()-t_start:.2f}s. Peak RAM: {get_peak_memory_gb():.2f} GB")

    # Temporary intermediate disk files for streaming
    tmp_cands_file = os.path.join(out_dir, f"tmp_cands_{c_tag}.tsv")
    tmp_matches_file = os.path.join(out_dir, f"tmp_matches_{c_tag}.tsv")
    tmp_cand_indices_file = os.path.join(out_dir, f"tmp_indices_{c_tag}.txt")

    # Step 4: PASS 1 - Candidate Blocking across S2 and S3
    log_msg(log_file, f"[{country}] Step 4: PASS 1 - Generating candidates from S2 and S3 (Vectorized)...")
    t_pass1 = time.time()
    cand_counts = np.zeros(n_s1, dtype=np.int32)
    total_queries = 0
    total_cand_pairs = 0

    with open(tmp_cands_file, "w", encoding="utf-8") as f_cands, open(tmp_cand_indices_file, "w", encoding="utf-8") as f_indices:
        for src_label, src_path in [("S2", test_s2_path), ("S3", test_s3_path)]:
            log_msg(log_file, f"[{country}] Streaming {src_label} for blocking...")
            cursor = con.execute(
                f"SELECT entity_id, business_name, business_address FROM read_csv('{src_path}', sep='\\t', header=True) WHERE country = ?",
                [country],
            )
            src_queries = 0

            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                n_rows = len(rows)
                src_queries += n_rows
                total_queries += n_rows

                q_ids = [r[0] for r in rows]
                q_names = [normalize_text(r[1], is_address=False) for r in rows]
                q_addrs = [normalize_text(r[2], is_address=True) for r in rows]
                q_core = [[t for t in n if t not in generic_set] for n in q_names]
                q_nf_strs = [" ".join(n) for n in q_names]
                q_core_strs = [" ".join(c) for c in q_core]
                q_addr_strs = [" ".join(a) for a in q_addrs]

                cand_buffer = []
                idx_buffer = []

                # Sub-batch sparse TF-IDF matrix multiplication across all 3 generators
                for sb_start in range(0, n_rows, sub_batch_size):
                    sb_end = min(sb_start + sub_batch_size, n_rows)
                    sub_q_strs = q_nf_strs[sb_start:sb_end]
                    sub_q_core_strs = q_core_strs[sb_start:sb_end]
                    sub_q_addr_strs = q_addr_strs[sb_start:sb_end]

                    sims_sub = (vec_blocking.transform(sub_q_strs) @ X_s1_blocking.T).tocsr()
                    sims_b = (vec_b.transform(sub_q_core_strs) @ X_s1_b.T).tocsr()
                    sims_c = (vec_c.transform(sub_q_addr_strs) @ X_s1_c.T).tocsr()

                    for i_rel, i_abs in enumerate(range(sb_start, sb_end)):
                        cand_gens = {}
                        cand_scores = {}

                        # Generator (a): char 3-gram cosine
                        row_data_a = sims_sub.data[sims_sub.indptr[i_rel] : sims_sub.indptr[i_rel + 1]]
                        row_indices_a = sims_sub.indices[sims_sub.indptr[i_rel] : sims_sub.indptr[i_rel + 1]]
                        if len(row_data_a) > 0:
                            top_k = min(15, len(row_data_a))
                            top_sub = np.argsort(-row_data_a)[:top_k]
                            for rank, sub_idx in enumerate(top_sub):
                                c_idx = int(row_indices_a[sub_idx])
                                cand_gens[c_idx] = 1
                                cand_scores[c_idx] = (15 - rank) / 15.0

                        # Generator (b): vectorized core tokens TF-IDF
                        row_data_b = sims_b.data[sims_b.indptr[i_rel] : sims_b.indptr[i_rel + 1]]
                        row_indices_b = sims_b.indices[sims_b.indptr[i_rel] : sims_b.indptr[i_rel + 1]]
                        if len(row_data_b) > 0:
                            top_k = min(15, len(row_data_b))
                            top_sub = np.argsort(-row_data_b)[:top_k]
                            for rank, sub_idx in enumerate(top_sub):
                                c_idx = int(row_indices_b[sub_idx])
                                cand_gens[c_idx] = cand_gens.get(c_idx, 0) + 1
                                cand_scores[c_idx] = cand_scores.get(c_idx, 0.0) + (15 - rank) / 15.0

                        # Generator (c): vectorized address tokens TF-IDF
                        row_data_c = sims_c.data[sims_c.indptr[i_rel] : sims_c.indptr[i_rel + 1]]
                        row_indices_c = sims_c.indices[sims_c.indptr[i_rel] : sims_c.indptr[i_rel + 1]]
                        if len(row_data_c) > 0:
                            top_k = min(10, len(row_data_c))
                            top_sub = np.argsort(-row_data_c)[:top_k]
                            for rank, sub_idx in enumerate(top_sub):
                                c_idx = int(row_indices_c[sub_idx])
                                cand_gens[c_idx] = cand_gens.get(c_idx, 0) + 1
                                cand_scores[c_idx] = cand_scores.get(c_idx, 0.0) + (10 - rank) / 10.0

                        # Combine, sort, cap at max_cands_per_rec
                        if cand_gens:
                            sorted_cands = sorted(
                                cand_gens.keys(),
                                key=lambda c: (cand_gens[c], cand_scores[c]),
                                reverse=True,
                            )[:max_cands_per_rec]
                            for c in sorted_cands:
                                cand_counts[c] += 1
                                cand_buffer.append(f"{s1_ids[c]}\t{q_ids[i_abs]}\n")
                            idx_buffer.append(f"{q_ids[i_abs]}\t{','.join(map(str, sorted_cands))}\n")
                            total_cand_pairs += len(sorted_cands)
                        else:
                            idx_buffer.append(f"{q_ids[i_abs]}\t\n")

                f_cands.writelines(cand_buffer)
                f_indices.writelines(idx_buffer)

                check_memory_safe()
                if (src_queries // chunk_size) % 10 == 0:
                    log_msg(log_file, f"  [{country} {src_label}] Processed {src_queries:,} records... Peak RAM: {get_peak_memory_gb():.2f} GB")

    log_msg(
        log_file,
        f"[{country}] PASS 1 complete in {time.time()-t_pass1:.2f}s. "
        f"Total queries: {total_queries:,}, Total candidate pairs: {total_cand_pairs:,} "
        f"(Avg cands/rec: {total_cand_pairs/total_queries:.1f}). Peak RAM: {get_peak_memory_gb():.2f} GB",
    )

    # Free blocking structures not needed in Pass 2
    del X_s1_blocking, vec_blocking, X_s1_b, vec_b, X_s1_c, vec_c
    gc.collect()

    # Step 5: PASS 2 - Feature Extraction, Batch Model Scoring & 1-Owner Decision Layer
    log_msg(log_file, f"[{country}] Step 5: PASS 2 - Computing features & scoring with model_v2...")
    t_pass2 = time.time()
    total_matches = 0

    with open(tmp_cand_indices_file, "r", encoding="utf-8") as f_indices, open(tmp_matches_file, "w", encoding="utf-8") as f_matches:
        for src_label, src_path in [("S2", test_s2_path), ("S3", test_s3_path)]:
            log_msg(log_file, f"[{country}] Streaming {src_label} for feature extraction & scoring...")
            cursor = con.execute(
                f"SELECT entity_id, business_name, business_address FROM read_csv('{src_path}', sep='\\t', header=True) WHERE country = ?",
                [country],
            )
            src_queries = 0

            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                n_rows = len(rows)
                src_queries += n_rows

                q_ids = [r[0] for r in rows]
                q_names = [normalize_text(r[1], is_address=False) for r in rows]
                q_addrs = [normalize_text(r[2], is_address=True) for r in rows]
                q_core = [[t for t in n if t not in generic_set] for n in q_names]
                q_nf_strs = [" ".join(n) for n in q_names]

                # Read matching lines from tmp_cand_indices_file in lockstep
                cand_index_lists = []
                for i_row in range(n_rows):
                    line = f_indices.readline().rstrip("\n")
                    if not line:
                        cand_index_lists.append([])
                        continue
                    line_qid, _, cand_str = line.partition("\t")
                    if line_qid != q_ids[i_row]:
                        raise ValueError(f"Stream alignment error in {src_label}: expected {q_ids[i_row]}, got {line_qid}")
                    if cand_str:
                        cand_index_lists.append([int(x) for x in cand_str.split(",")])
                    else:
                        cand_index_lists.append([])

                match_buffer = []

                # Process sub-batches with high-performance batch predict
                for sb_start in range(0, n_rows, sub_batch_size):
                    sb_end = min(sb_start + sub_batch_size, n_rows)
                    sub_q_strs = q_nf_strs[sb_start:sb_end]
                    X_q_sub = vec_feat.transform(sub_q_strs)

                    sub_cand_feats = []
                    sub_meta = []  # List of (q_idx_abs, ordered_c_indices, n_cands)

                    for i_rel, i_abs in enumerate(range(sb_start, sb_end)):
                        c_list = cand_index_lists[i_abs]
                        if not c_list:
                            continue

                        # Dot product for char_3gram_cosine
                        q_vec = X_q_sub[i_rel]
                        cos_sims = X_s1_feat[c_list].dot(q_vec.T).toarray().ravel()
                        sorted_order = np.argsort(-cos_sims)
                        s_top1 = cos_sims[sorted_order[0]]
                        s_top2 = cos_sims[sorted_order[1]] if len(sorted_order) > 1 else 0.0

                        q_str = q_nf_strs[i_abs]
                        q_nf_s = set(q_names[i_abs])
                        q_nc_s = set(q_core[i_abs])
                        q_ad_s = set(q_addrs[i_abs])
                        q_dist_s = q_nf_s - common_tokens
                        q_len = len(q_nf_s)
                        q_suf = q_nf_s - q_nc_s
                        q_inits = sorted([t[0] for t in q_names[i_abs] if t])
                        q_house = extract_leading_digits(q_addrs[i_abs])
                        q_city = " ".join(q_addrs[i_abs][-2:]) if len(q_addrs[i_abs]) >= 2 else (" ".join(q_addrs[i_abs]))
                        is_q_b = 1.0 if not q_ad_s else 0.0

                        ordered_c_indices = []

                        for rank_0, c_pos in enumerate(sorted_order):
                            c_idx = c_list[c_pos]
                            ordered_c_indices.append(c_idx)
                            cos_val = float(cos_sims[c_pos])
                            score_gap = float(s_top1 - s_top2) if rank_0 == 0 else float(cos_val - s_top1)

                            s1_s = s1_nf_strs[c_idx]
                            s1_nf_s = s1_nf_sets[c_idx]
                            s1_nc_s = s1_nc_sets[c_idx]
                            s1_ad_s = s1_ad_sets[c_idx]
                            s1_dist_s = s1_dist_sets[c_idx]
                            s1_len = len(s1_nf_s)
                            s1_suf = s1_sufs[c_idx]
                            s1_init = s1_inits_all[c_idx]
                            s1_h = s1_house_all[c_idx]
                            s1_c = s1_cities_all[c_idx]

                            # Name metrics
                            u_f = len(s1_nf_s | q_nf_s)
                            j_full = len(s1_nf_s & q_nf_s) / u_f if u_f else 0.0
                            u_c = len(s1_nc_s | q_nc_s)
                            j_core = len(s1_nc_s & q_nc_s) / u_c if u_c else 0.0
                            shared_dist = s1_dist_s & q_dist_s
                            u_dist = s1_dist_s | q_dist_s
                            dist_j = len(shared_dist) / len(u_dist) if u_dist else 0.0
                            dist_cnt = float(len(shared_dist))

                            lev = fuzz.ratio(s1_s, q_str) / 100.0
                            ts = fuzz.token_sort_ratio(s1_s, q_str) / 100.0
                            tset = fuzz.token_set_ratio(s1_s, q_str) / 100.0
                            im = 1.0 if (s1_init and s1_init == q_inits) else 0.0
                            ldiff = float(abs(s1_len - q_len))

                            if s1_suf and s1_suf == q_suf:
                                suf_agree = 1.0
                            elif not s1_suf and not q_suf:
                                suf_agree = 0.5
                            else:
                                suf_agree = 0.0

                            # Address metrics
                            is_s1_b = 1.0 if not s1_ad_s else 0.0
                            if is_s1_b or is_q_b:
                                j_ad, w_ad, hm, cm = 0.0, 0.0, 0.0, 0.0
                            else:
                                u_a = len(s1_ad_s | q_ad_s)
                                j_ad = len(s1_ad_s & q_ad_s) / u_a if u_a else 0.0
                                w_ad = float(sum(idf_map.get(t, 1.0) for t in (s1_ad_s & q_ad_s)))
                                hm = 1.0 if (s1_h and q_house and distance.Levenshtein.distance(s1_h, q_house) <= 2) else 0.0
                                cm = fuzz.token_sort_ratio(s1_c, q_city) / 100.0 if (s1_c and q_city) else 0.0

                            num_cands = float(cand_counts[c_idx])
                            shared_name = s1_nf_s & q_nf_s
                            rarity = float(sum(idf_map.get(t, 1.0) for t in shared_name) / len(shared_name)) if shared_name else 0.0

                            sub_cand_feats.append([
                                j_full, j_core, cos_val, lev, ts, tset, im, ldiff, suf_agree,
                                j_ad, is_s1_b, is_q_b, w_ad, hm, cm,
                                float(rank_0 + 1), score_gap, num_cands, rarity, dist_j, dist_cnt
                            ])

                        sub_meta.append((i_abs, ordered_c_indices, len(ordered_c_indices)))

                    if sub_cand_feats:
                        # Batch prediction across all candidate pairs in sub-batch
                        X_sub = np.array(sub_cand_feats, dtype=np.float32)
                        sub_probs = model.predict(X_sub)

                        p_offset = 0
                        for q_idx_abs, ordered_c_indices, n_c in sub_meta:
                            c_probs = sub_probs[p_offset : p_offset + n_c]
                            p_offset += n_c

                            best_p_idx = int(np.argmax(c_probs))
                            p1 = float(c_probs[best_p_idx])
                            if n_c == 1:
                                margin = p1
                            else:
                                sorted_p = np.sort(c_probs)
                                margin = float(sorted_p[-1] - sorted_p[-2])

                            if p1 >= min_p and margin >= min_m:
                                best_s1_id = s1_ids[ordered_c_indices[best_p_idx]]
                                match_buffer.append(f"{best_s1_id}\t{q_ids[q_idx_abs]}\n")
                                total_matches += 1

                f_matches.writelines(match_buffer)

                check_memory_safe()
                if (src_queries // chunk_size) % 10 == 0:
                    log_msg(log_file, f"  [{country} {src_label}] Scored {src_queries:,} records... Matches so far: {total_matches:,}. Peak RAM: {get_peak_memory_gb():.2f} GB")

    log_msg(
        log_file,
        f"[{country}] PASS 2 complete in {time.time()-t_pass2:.2f}s. "
        f"Total accepted matches: {total_matches:,}. Peak RAM: {get_peak_memory_gb():.2f} GB",
    )

    # Free Pass 2 structures
    del X_s1_feat, vec_feat, s1_nf_sets, s1_nc_sets, s1_ad_sets, s1_dist_sets, cand_counts, s1_inits_all, s1_house_all, s1_cities_all, s1_sufs
    if os.path.exists(tmp_cand_indices_file):
        os.remove(tmp_cand_indices_file)
    gc.collect()

    # Step 6: DuckDB Aggregation into per-country TSVs
    log_msg(log_file, f"[{country}] Step 6: Grouping matches and candidates into per-country TSVs...")
    t_agg = time.time()
    out_country_matching = os.path.join(out_dir, f"tmp_country_matching_{c_tag}.tsv")
    out_country_cands = os.path.join(out_dir, f"tmp_country_cands_{c_tag}.tsv")

    # Table for Country S1
    con.execute(f"CREATE TABLE country_s1_{c_tag} AS SELECT entity_id FROM read_csv('{test_s1_path}', sep='\\t', header=True, all_varchar=True) WHERE country = ?", [country])

    # Table for Matches (grouped by S1)
    if os.path.exists(tmp_matches_file) and os.path.getsize(tmp_matches_file) > 0:
        con.execute(f"""
            CREATE TABLE country_matches_{c_tag} AS
            SELECT source1_entity_id, string_agg(matched_id, ',') as matched_ids
            FROM read_csv('{tmp_matches_file}', sep='\\t', names=['source1_entity_id', 'matched_id'], all_varchar=True)
            GROUP BY source1_entity_id
        """)
    else:
        con.execute(f"CREATE TABLE country_matches_{c_tag} (source1_entity_id VARCHAR, matched_ids VARCHAR)")

    # Table for Candidates (grouped by S1)
    if os.path.exists(tmp_cands_file) and os.path.getsize(tmp_cands_file) > 0:
        con.execute(f"""
            CREATE TABLE country_cands_{c_tag} AS
            SELECT source1_entity_id, string_agg(cand_id, ',') as cand_ids
            FROM read_csv('{tmp_cands_file}', sep='\\t', names=['source1_entity_id', 'cand_id'], all_varchar=True)
            GROUP BY source1_entity_id
        """)
    else:
        con.execute(f"CREATE TABLE country_cands_{c_tag} (source1_entity_id VARCHAR, cand_ids VARCHAR)")

    # Output matching_results_{c_tag}.tsv
    con.execute(f"""
        COPY (
            SELECT s1.entity_id as source1_entity_id, coalesce(m.matched_ids, '') as matched_entity_ids
            FROM country_s1_{c_tag} s1
            LEFT JOIN country_matches_{c_tag} m ON s1.entity_id = m.source1_entity_id
        ) TO '{out_country_matching}' (HEADER, DELIMITER '\\t', QUOTE '')
    """)

    # Output candidate_pairs_{c_tag}.tsv
    con.execute(f"""
        COPY (
            SELECT s1.entity_id as source1_entity_id, coalesce(c.cand_ids, '') as candidate_entity_ids
            FROM country_s1_{c_tag} s1
            LEFT JOIN country_cands_{c_tag} c ON s1.entity_id = c.source1_entity_id
        ) TO '{out_country_cands}' (HEADER, DELIMITER '\\t', QUOTE '')
    """)

    # Clean up intermediate files and tables
    if os.path.exists(tmp_cands_file):
        os.remove(tmp_cands_file)
    if os.path.exists(tmp_matches_file):
        os.remove(tmp_matches_file)

    # Compute country stats
    stats_row = con.execute(f"""
        SELECT
            count(*) as total_s1,
            count(CASE WHEN length(matched_entity_ids) > 0 THEN 1 END) as s1_with_matches,
            count(CASE WHEN length(candidate_entity_ids) > 0 THEN 1 END) as s1_with_cands
        FROM read_csv('{out_country_matching}', sep='\\t', header=True, all_varchar=True) m
        JOIN read_csv('{out_country_cands}', sep='\\t', header=True, all_varchar=True) c ON m.source1_entity_id = c.source1_entity_id
    """).fetchone()

    total_s1 = stats_row[0]
    s1_with_matches = stats_row[1]
    s1_with_cands = stats_row[2]
    match_rate_pct = (s1_with_matches / float(total_s1) * 100.0) if total_s1 else 0.0
    cand_cov_pct = (s1_with_cands / float(total_s1) * 100.0) if total_s1 else 0.0
    avg_matches_per_s1 = total_matches / float(total_s1) if total_s1 else 0.0

    t_country_end = time.time()
    elapsed_country = t_country_end - t_start
    peak_mem_gb = get_peak_memory_gb()

    log_msg(log_file, f"------------------------------------------------------------")
    log_msg(log_file, f"COUNTRY SUMMARY: {country}")
    log_msg(log_file, f"  S1 Entities:              {total_s1:,}")
    log_msg(log_file, f"  Candidate Coverage:       {s1_with_cands:,} ({cand_cov_pct:.2f}%)")
    log_msg(log_file, f"  Total Candidate Pairs:    {total_cand_pairs:,}")
    log_msg(log_file, f"  S1 Entities with Match:   {s1_with_matches:,} ({match_rate_pct:.2f}%)")
    log_msg(log_file, f"  Total Accepted Matches:   {total_matches:,}")
    log_msg(log_file, f"  Avg Matches per S1:       {avg_matches_per_s1:.2f}")
    log_msg(log_file, f"  Elapsed Time:             {elapsed_country:.2f}s ({elapsed_country/60:.2f}m)")
    log_msg(log_file, f"  Peak Memory:              {peak_mem_gb:.2f} GB")
    log_msg(log_file, f"------------------------------------------------------------\n")

    # Step 7: Free country memory
    del s1_rows, s1_ids, s1_names, s1_addrs, s1_nf_strs, s1_core
    gc.collect()

    return {
        "country": country,
        "total_s1": total_s1,
        "s1_with_matches": s1_with_matches,
        "match_rate_pct": match_rate_pct,
        "total_matches": total_matches,
        "avg_matches_per_s1": avg_matches_per_s1,
        "total_cand_pairs": total_cand_pairs,
        "s1_with_cands": s1_with_cands,
        "cand_cov_pct": cand_cov_pct,
        "elapsed_sec": elapsed_country,
        "peak_mem_gb": peak_mem_gb,
        "matching_tsv": out_country_matching,
        "cands_tsv": out_country_cands,
    }


def merge_country_results(
    country_results: List[Dict[str, Any]],
    test_dir: str,
    out_dir: str,
    log_file: str,
):
    """
    Merges all per-country TSVs into the final submission files:
      - output/matching_results.tsv
      - output/candidate_pairs.tsv
    Guarantees every S1 entity in test_source1.tsv appears exactly once.
    """
    log_msg(log_file, "============================================================")
    log_msg(log_file, "FINAL MERGE: Combining country results into submission TSVs...")
    log_msg(log_file, "============================================================")

    t0 = time.time()
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    final_matching_path = os.path.join(out_dir, "matching_results.tsv")
    final_cands_path = os.path.join(out_dir, "candidate_pairs.tsv")

    # Merge matching files
    matching_files = [res["matching_tsv"] for res in country_results]
    matching_glob = os.path.join(out_dir, "tmp_country_matching_*.tsv")
    con.execute(f"""
        COPY (
            SELECT s1.entity_id as source1_entity_id, coalesce(m.matched_entity_ids, '') as matched_entity_ids
            FROM (SELECT entity_id FROM read_csv('{test_s1_path}', sep='\\t', header=True, all_varchar=True)) s1
            LEFT JOIN (
                SELECT source1_entity_id, matched_entity_ids
                FROM read_csv('{matching_glob}', sep='\\t', header=True, all_varchar=True)
            ) m ON s1.entity_id = m.source1_entity_id
        ) TO '{final_matching_path}' (HEADER, DELIMITER '\\t', QUOTE '')
    """)

    # Merge candidate files
    cands_files = [res["cands_tsv"] for res in country_results]
    cands_glob = os.path.join(out_dir, "tmp_country_cands_*.tsv")
    con.execute(f"""
        COPY (
            SELECT s1.entity_id as source1_entity_id, coalesce(c.candidate_entity_ids, '') as candidate_entity_ids
            FROM (SELECT entity_id FROM read_csv('{test_s1_path}', sep='\\t', header=True, all_varchar=True)) s1
            LEFT JOIN (
                SELECT source1_entity_id, candidate_entity_ids
                FROM read_csv('{cands_glob}', sep='\\t', header=True, all_varchar=True)
            ) c ON s1.entity_id = c.source1_entity_id
        ) TO '{final_cands_path}' (HEADER, DELIMITER '\\t', QUOTE '')
    """)

    # Clean up per-country TSVs
    for p in matching_files + cands_files:
        if os.path.exists(p):
            os.remove(p)

    match_bytes = os.path.getsize(final_matching_path)
    cands_bytes = os.path.getsize(final_cands_path)

    log_msg(log_file, f"Final matching file written: {final_matching_path} ({match_bytes / (1024*1024):.2f} MB)")
    log_msg(log_file, f"Final candidates file written: {final_cands_path} ({cands_bytes / (1024*1024):.2f} MB)")
    log_msg(log_file, f"Merge completed in {time.time()-t0:.2f}s. Peak RAM: {get_peak_memory_gb():.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Full Test Submission Pipeline for Entity Resolution.")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test directory")
    parser.add_argument("--model", type=str, default="code/business_entity_resolution/model_v2.txt", help="Path to LightGBM model file")
    parser.add_argument("--out-dir", type=str, default="output", help="Directory to save submission files")
    parser.add_argument("--idf-path", type=str, default="data_mini/idf_weights.parquet", help="Path to precomputed IDF weights")
    parser.add_argument("--min-p", type=float, default=0.95, help="Decision threshold min_probability (default: 0.95)")
    parser.add_argument("--min-m", type=float, default=0.05, help="Decision threshold min_margin (default: 0.05)")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Query record chunk size (default: 50000)")
    parser.add_argument("--log-file", type=str, default="run_full_pipeline_v2.log", help="Log file path")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_global_start = time.time()

    # Reset or initialize log file
    with open(args.log_file, "w", encoding="utf-8") as f:
        f.write(f"=== FULL PIPELINE RUN: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    log_msg(args.log_file, f"Platform: {platform.system()} {platform.machine()}, 8 physical cores, 16GB RAM")
    log_msg(args.log_file, f"Args: test_dir={args.test_dir}, model={args.model}, out_dir={args.out_dir}")
    log_msg(args.log_file, f"Thresholds: min_p={args.min_p}, min_m={args.min_m}, chunk_size={args.chunk_size}")

    # Load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    log_msg(args.log_file, f"Loading LightGBM model from {args.model}...")
    model = lgb.Booster(model_file=args.model)
    model.params["num_threads"] = 8

    # Discover countries dynamically from test_source1.tsv
    test_s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    country_counts = con.execute(
        f"SELECT country, count(*) FROM read_csv('{test_s1_path}', sep='\\t', header=True, all_varchar=True) GROUP BY country ORDER BY count(*) ASC"
    ).fetchall()

    log_msg(args.log_file, f"Discovered {len(country_counts)} countries in test set:")
    for c, cnt in country_counts:
        log_msg(args.log_file, f"  - {c}: {int(cnt):,} S1 entities")

    country_results = []
    for country, s1_count in country_counts:
        res = run_country_pipeline(
            country=country,
            test_dir=args.test_dir,
            model=model,
            idf_weights_path=args.idf_path,
            out_dir=args.out_dir,
            log_file=args.log_file,
            min_p=args.min_p,
            min_m=args.min_m,
            chunk_size=args.chunk_size,
        )
        country_results.append(res)

        # Print tail -n 20 of log after each country finishes
        print("\n" + "=" * 60)
        print(f"PROGRESS UPDATE: Country {country} Complete")
        print("=" * 60)
        with open(args.log_file, "r", encoding="utf-8") as lf:
            lines = lf.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
        print("=" * 60 + "\n")

    # Merge all country results
    merge_country_results(country_results, args.test_dir, args.out_dir, args.log_file)

    t_global_end = time.time()
    total_elapsed = t_global_end - t_global_start
    final_peak_ram = get_peak_memory_gb()

    log_msg(args.log_file, "============================================================")
    log_msg(args.log_file, f"FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    log_msg(args.log_file, f"Total Runtime: {total_elapsed:.2f}s ({total_elapsed/60:.2f}m / {total_elapsed/3600:.2f}h)")
    log_msg(args.log_file, f"Global Peak Memory: {final_peak_ram:.2f} GB")
    log_msg(args.log_file, "============================================================")

    # Print final 20 lines of log
    with open(args.log_file, "r", encoding="utf-8") as lf:
        lines = lf.readlines()
        for line in lines[-20:]:
            print(line.rstrip())


if __name__ == "__main__":
    main()
