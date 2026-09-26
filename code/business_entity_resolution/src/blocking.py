#!/usr/bin/env python3
"""
Blocking (Candidate Generation) Stage for Business Entity Resolution.
Generates candidates from the record side for each S2 and S3 record against S1 entities.
Combines 3 independent candidate generators per country:
  (a) Character 3-gram TF-IDF (top-15 cosine similarity)
  (b) Rare token inverted index on name_core weighted by IDF (top-15)
  (c) Address token inverted index weighted by address IDF (top-10)
Takes the union per record (deduped), caps at 25 candidates, and inverts to S1-grouped format.
Strictly country-agnostic.
"""

import os
import sys
import math
import time
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer


def build_country_s1_indices(
    s1_df_country,
    idf_weights_map: Dict[str, float],
    max_addr_df: int = 2000,
):
    """
    Builds the search indices on S1 records for a single country:
      (a) TF-IDF vectorizer and sparse matrix for char 3-grams on name_full
      (b) Inverted index on name_core tokens
      (c) Inverted index on address tokens with computed address IDF
    """
    s1_ids = s1_df_country["entity_id"].tolist()
    n_s1 = len(s1_ids)

    # (a) Char 3-gram TF-IDF
    s1_name_full_texts = [
        " ".join(toks) if toks is not None and len(toks) > 0 else ""
        for toks in s1_df_country["name_full"]
    ]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
    X_s1 = vec.fit_transform(s1_name_full_texts)

    # (b) Inverted index for name_core
    inv_b = defaultdict(list)
    for idx, core_toks in enumerate(s1_df_country["name_core"]):
        if core_toks is not None:
            for t in set(core_toks):
                inv_b[t].append(idx)

    # (c) Address IDF and inverted index
    df_addr = Counter()
    for addr_toks in s1_df_country["address"]:
        if addr_toks is not None and len(addr_toks) > 0:
            df_addr.update(set(addr_toks))

    idf_addr = {}
    for t, cnt in df_addr.items():
        if cnt <= max_addr_df:
            idf_addr[t] = math.log((n_s1 + 1.0) / (cnt + 1.0)) + 1.0

    inv_c = defaultdict(list)
    for idx, addr_toks in enumerate(s1_df_country["address"]):
        if addr_toks is not None and len(addr_toks) > 0:
            for t in set(addr_toks):
                if t in idf_addr:
                    inv_c[t].append(idx)

    return {
        "s1_ids": s1_ids,
        "n_s1": n_s1,
        "vec": vec,
        "X_s1": X_s1,
        "inv_b": inv_b,
        "idf_weights_map": idf_weights_map,
        "inv_c": inv_c,
        "idf_addr": idf_addr,
    }


def query_batch_candidates(
    batch_df,
    country_index: Dict[str, Any],
    top_k_a: int = 15,
    top_k_b: int = 15,
    top_k_c: int = 10,
    max_cands: int = 25,
) -> Tuple[List[List[str]], List[List[List[str]]]]:
    """
    Runs the 3 candidate generators for a batch of query records and returns
    deduped, prioritized, capped candidates and their generating sources.
    """
    s1_ids = country_index["s1_ids"]
    vec = country_index["vec"]
    X_s1 = country_index["X_s1"]
    inv_b = country_index["inv_b"]
    idf_weights_map = country_index["idf_weights_map"]
    inv_c = country_index["inv_c"]
    idf_addr = country_index["idf_addr"]

    n_queries = len(batch_df)
    batch_name_full = [
        " ".join(toks) if toks is not None and len(toks) > 0 else ""
        for toks in batch_df["name_full"]
    ]
    batch_name_core = batch_df["name_core"].tolist()
    batch_address = batch_df["address"].tolist()

    # Generator (a): TF-IDF char 3-gram sparse cosine similarity
    X_q = vec.transform(batch_name_full)
    sims = (X_q @ X_s1.T).tocsr()

    batch_candidate_ids = []
    batch_candidate_gens = []

    for i in range(n_queries):
        cand_gens = {}
        cand_scores = {}

        # --- Generator (a) ---
        row_data = sims.data[sims.indptr[i] : sims.indptr[i + 1]]
        row_indices = sims.indices[sims.indptr[i] : sims.indptr[i + 1]]
        if len(row_data) > 0:
            if len(row_data) <= top_k_a:
                top_sub = np.argsort(-row_data)
            else:
                p = np.argpartition(row_data, -top_k_a)[-top_k_a:]
                top_sub = p[np.argsort(-row_data[p])]

            for rank, sub_idx in enumerate(top_sub):
                c_idx = row_indices[sub_idx]
                cand_gens[c_idx] = ["a"]
                cand_scores[c_idx] = (top_k_a - rank) / float(top_k_a)

        # --- Generator (b) ---
        q_core = batch_name_core[i]
        if q_core is not None and len(q_core) > 0:
            scores_b = defaultdict(float)
            for t in set(q_core):
                w = idf_weights_map.get(t, 1.0)
                if w > 0.0 and t in inv_b:
                    for s1_idx in inv_b[t]:
                        scores_b[s1_idx] += w

            if scores_b:
                top_b = sorted(scores_b.items(), key=lambda x: -x[1])[:top_k_b]
                for rank, (c_idx, _) in enumerate(top_b):
                    norm_score = (top_k_b - rank) / float(top_k_b)
                    if c_idx in cand_gens:
                        cand_gens[c_idx].append("b")
                        cand_scores[c_idx] += norm_score
                    else:
                        cand_gens[c_idx] = ["b"]
                        cand_scores[c_idx] = norm_score

        # --- Generator (c) ---
        q_addr = batch_address[i]
        if q_addr is not None and len(q_addr) > 0:
            scores_c = defaultdict(float)
            for t in set(q_addr):
                w = idf_addr.get(t)
                if w is not None and t in inv_c:
                    for s1_idx in inv_c[t]:
                        scores_c[s1_idx] += w

            if scores_c:
                top_c = sorted(scores_c.items(), key=lambda x: -x[1])[:top_k_c]
                for rank, (c_idx, _) in enumerate(top_c):
                    norm_score = (top_k_c - rank) / float(top_k_c)
                    if c_idx in cand_gens:
                        cand_gens[c_idx].append("c")
                        cand_scores[c_idx] += norm_score
                    else:
                        cand_gens[c_idx] = ["c"]
                        cand_scores[c_idx] = norm_score

        # Combine, rank by (number of generators desc, combined rank score desc), cap at max_cands
        if cand_gens:
            sorted_candidates = sorted(
                cand_gens.keys(),
                key=lambda c: (len(cand_gens[c]), cand_scores[c]),
                reverse=True,
            )[:max_cands]
            batch_candidate_ids.append([s1_ids[c] for c in sorted_candidates])
            batch_candidate_gens.append([cand_gens[c] for c in sorted_candidates])
        else:
            batch_candidate_ids.append([])
            batch_candidate_gens.append([])

    return batch_candidate_ids, batch_candidate_gens


def run_blocking(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    idf_path: str,
    out_record_candidates: str,
    out_s1_candidates: str,
    top_k_a: int = 15,
    top_k_b: int = 15,
    top_k_c: int = 10,
    max_cands: int = 25,
    batch_size: int = 5000,
    max_addr_df: int = 2000,
):
    t_start = time.time()
    print("=== Phase 4: Candidate Generation (Blocking) ===")
    print(f"Loading S1 from: {s1_path}")
    req_cols = ["entity_id", "country", "name_full", "name_core", "address"]
    s1_table = pq.read_table(s1_path, columns=req_cols).to_pandas()
    all_s1_ids = s1_table["entity_id"].tolist()
    distinct_countries = sorted(s1_table["country"].unique())
    print(f"Loaded {len(s1_table)} S1 records across countries: {distinct_countries}")

    print(f"Loading IDF weights from: {idf_path}")
    idf_table = pq.read_table(idf_path).to_pandas()
    idf_by_country = {}
    for country, group in idf_table.groupby("country"):
        idf_by_country[country] = dict(zip(group["token"], group["idf"]))

    print(f"Loading S2 from: {s2_path}")
    s2_table = pq.read_table(s2_path, columns=req_cols).to_pandas()
    print(f"Loading S3 from: {s3_path}")
    s3_table = pq.read_table(s3_path, columns=req_cols).to_pandas()

    # Setup ParquetWriter for record candidates
    os.makedirs(os.path.dirname(os.path.abspath(out_record_candidates)), exist_ok=True)
    record_schema = pa.schema([
        ("record_id", pa.string()),
        ("record_source", pa.string()),
        ("country", pa.string()),
        ("candidate_s1_ids", pa.list_(pa.string())),
        ("which_generators_found_it", pa.list_(pa.list_(pa.string()))),
    ])
    record_writer = pq.ParquetWriter(out_record_candidates, schema=record_schema, compression="snappy")

    # Inverted mapping: S1 ID -> list of candidate record IDs
    s1_to_record_candidates = {s1_id: [] for s1_id in all_s1_ids}
    total_record_count = 0

    for country in distinct_countries:
        c_start = time.time()
        s1_country = s1_table[s1_table["country"] == country].reset_index(drop=True)
        print(f"\n[{country}] Building search index for {len(s1_country)} S1 records...")
        country_index = build_country_s1_indices(
            s1_country,
            idf_weights_map=idf_by_country.get(country, {}),
            max_addr_df=max_addr_df,
        )
        print(f"[{country}] Index built in {time.time() - c_start:.2f}s")

        for source_label, source_df in [("S2", s2_table), ("S3", s3_table)]:
            src_country_df = source_df[source_df["country"] == country].reset_index(drop=True)
            n_records = len(src_country_df)
            print(f"[{country}] Processing {source_label}: {n_records} records...")

            for b_start in range(0, n_records, batch_size):
                b_end = min(b_start + batch_size, n_records)
                batch_df = src_country_df.iloc[b_start:b_end]
                batch_cands, batch_gens = query_batch_candidates(
                    batch_df,
                    country_index,
                    top_k_a=top_k_a,
                    top_k_b=top_k_b,
                    top_k_c=top_k_c,
                    max_cands=max_cands,
                )

                batch_ids = batch_df["entity_id"].tolist()
                for rec_id, cands in zip(batch_ids, batch_cands):
                    for s1_cand in cands:
                        s1_to_record_candidates[s1_cand].append(rec_id)

                # Write batch table to parquet writer
                n_b = len(batch_ids)
                batch_table = pa.Table.from_arrays(
                    [
                        pa.array(batch_ids, type=pa.string()),
                        pa.array([source_label] * n_b, type=pa.string()),
                        pa.array([country] * n_b, type=pa.string()),
                        pa.array(batch_cands, type=pa.list_(pa.string())),
                        pa.array(batch_gens, type=pa.list_(pa.list_(pa.string()))),
                    ],
                    schema=record_schema,
                )
                record_writer.write_table(batch_table)
                total_record_count += n_b

                if (b_end // batch_size) % 10 == 0 or b_end == n_records:
                    print(f"  [{country} {source_label}] {b_end}/{n_records} processed...")

    record_writer.close()
    print(f"\nWrote {total_record_count} record candidate rows to: {out_record_candidates}")

    # Write candidates_s1_grouped.parquet
    print(f"Writing {len(all_s1_ids)} S1 grouped candidate rows to: {out_s1_candidates}")
    os.makedirs(os.path.dirname(os.path.abspath(out_s1_candidates)), exist_ok=True)
    grouped_record_candidates = [s1_to_record_candidates[s1_id] for s1_id in all_s1_ids]

    s1_table_out = pa.Table.from_arrays(
        [
            pa.array(all_s1_ids, type=pa.string()),
            pa.array(grouped_record_candidates, type=pa.list_(pa.string())),
        ],
        names=["source1_entity_id", "candidate_entity_ids"],
    )
    pq.write_table(s1_table_out, out_s1_candidates)

    print(f"\n=== Blocking Complete in {time.time() - t_start:.2f}s ===")


def main():
    parser = argparse.ArgumentParser(description="Multi-generator candidate blocking.")
    parser.add_argument("--s1-path", type=str, default="data_mini/source1_normalized.parquet")
    parser.add_argument("--s2-path", type=str, default="data_mini/source2_normalized.parquet")
    parser.add_argument("--s3-path", type=str, default="data_mini/source3_normalized.parquet")
    parser.add_argument("--idf-path", type=str, default="data_mini/idf_weights.parquet")
    parser.add_argument(
        "--out-record-candidates",
        type=str,
        default="data_mini/candidate_pairs_by_record.parquet",
    )
    parser.add_argument(
        "--out-s1-candidates",
        type=str,
        default="data_mini/candidates_s1_grouped.parquet",
    )
    parser.add_argument("--top-k-a", type=int, default=15)
    parser.add_argument("--top-k-b", type=int, default=15)
    parser.add_argument("--top-k-c", type=int, default=10)
    parser.add_argument("--max-cands-per-record", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-addr-df", type=int, default=2000)
    args = parser.parse_args()

    run_blocking(
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        idf_path=args.idf_path,
        out_record_candidates=args.out_record_candidates,
        out_s1_candidates=args.out_s1_candidates,
        top_k_a=args.top_k_a,
        top_k_b=args.top_k_b,
        top_k_c=args.top_k_c,
        max_cands=args.max_cands_per_record,
        batch_size=args.batch_size,
        max_addr_df=args.max_addr_df,
    )


if __name__ == "__main__":
    main()
