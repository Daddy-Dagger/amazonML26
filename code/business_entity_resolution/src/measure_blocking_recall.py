#!/usr/bin/env python3
"""
Measurement of Blocking Recall on Mini-World.
Evaluates candidate generation coverage against ground truth:
  - Overall pairwise recall & entity-level complete recall
  - Recall broken down by country
  - Recall broken down by candidate generator (a, b, c)
  - 15 sampled missed match examples with diagnosis
  - Average candidates per S1 entity and reduction ratio
  - Generates audit/step3_blocking_report.md (under 100 lines)
Strictly country-agnostic.
"""

import os
import sys
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

import pyarrow.parquet as pq


def load_ground_truth(gt_path: str) -> Tuple[Dict[str, List[str]], int]:
    """Loads ground truth and returns mapping from S1 ID -> list of true match IDs."""
    gt_table = pq.read_table(gt_path).to_pandas()
    truth_map = {}
    singleton_count = 0
    for s1_id, m_str in zip(gt_table["source1_entity_id"], gt_table["matched_entity_ids"]):
        if m_str is None or str(m_str).strip() == "":
            truth_map[s1_id] = []
            singleton_count += 1
        else:
            matches = [m.strip() for m in str(m_str).split(",") if m.strip()]
            truth_map[s1_id] = matches
            if len(matches) == 0:
                singleton_count += 1
    return truth_map, singleton_count


def load_record_candidates(record_cand_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Loads candidate_pairs_by_record.parquet and returns mapping:
    record_id -> {s1_id: [generators]}
    """
    table = pq.read_table(record_cand_path).to_pandas()
    record_map = {}
    for r_id, cands, gens in zip(
        table["record_id"],
        table["candidate_s1_ids"],
        table["which_generators_found_it"],
    ):
        cand_dict = {}
        for c, g in zip(cands, gens):
            cand_dict[c] = list(g)
        record_map[r_id] = cand_dict
    return record_map


def load_s1_candidates(s1_cand_path: str) -> Dict[str, Set[str]]:
    """Loads candidates_s1_grouped.parquet and returns mapping: s1_id -> set(candidate_record_ids)."""
    table = pq.read_table(s1_cand_path).to_pandas()
    s1_map = {}
    for s1_id, cands in zip(table["source1_entity_id"], table["candidate_entity_ids"]):
        s1_map[s1_id] = set(cands)
    return s1_map


def load_record_metadata(*paths: str) -> Dict[str, Dict[str, str]]:
    """Loads metadata (name, address, country) for all records across provided tables."""
    meta = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        table = pq.read_table(
            p,
            columns=["entity_id", "business_name", "business_address", "country"],
        ).to_pandas()
        for eid, name, addr, country in zip(
            table["entity_id"],
            table["business_name"],
            table["business_address"],
            table["country"],
        ):
            meta[eid] = {
                "name": str(name) if name is not None else "",
                "address": str(addr) if addr is not None else "",
                "country": str(country) if country is not None else "",
            }
    return meta


def analyze_blocking(
    gt_path: str,
    record_cand_path: str,
    s1_cand_path: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    out_report: str,
    sample_seed: int = 42,
):
    print("=== Measuring Blocking Recall ===")
    truth_map, singleton_count = load_ground_truth(gt_path)
    total_s1 = len(truth_map)
    non_singletons = {s1: m for s1, m in truth_map.items() if len(m) > 0}
    print(f"Total S1 entities: {total_s1} ({len(non_singletons)} with matches, {singleton_count} singletons)")

    print(f"Loading record candidates from: {record_cand_path}")
    record_cands = load_record_candidates(record_cand_path)

    print(f"Loading S1 candidates from: {s1_cand_path}")
    s1_cands = load_s1_candidates(s1_cand_path)

    print("Loading entity metadata...")
    meta = load_record_metadata(s1_path, s2_path, s3_path)

    # Countries present
    countries = sorted(list({meta[s1]["country"] for s1 in truth_map if s1 in meta}))

    # Recall tracking
    total_pairs = 0
    recalled_pairs = 0
    pairs_by_country = Counter()
    recalled_by_country = Counter()

    s1_complete = 0
    s1_by_country = Counter()
    s1_complete_by_country = Counter()

    gen_hits = Counter()
    gen_hits_by_country = defaultdict(Counter)

    missed_records = []  # List of dicts for incomplete S1s

    for s1_id, true_matches in non_singletons.items():
        c_country = meta.get(s1_id, {}).get("country", "Unknown")
        s1_by_country[c_country] += 1
        s1_cand_set = s1_cands.get(s1_id, set())

        s1_all_found = True
        for m_id in true_matches:
            total_pairs += 1
            pairs_by_country[c_country] += 1

            if m_id in s1_cand_set:
                recalled_pairs += 1
                recalled_by_country[c_country] += 1

                # Check which generators found it
                gens = record_cands.get(m_id, {}).get(s1_id, [])
                for g in gens:
                    gen_hits[g] += 1
                    gen_hits_by_country[c_country][g] += 1
            else:
                s1_all_found = False
                missed_records.append({
                    "s1_id": s1_id,
                    "target_id": m_id,
                    "country": c_country,
                    "s1_name": meta.get(s1_id, {}).get("name", ""),
                    "s1_addr": meta.get(s1_id, {}).get("address", ""),
                    "target_name": meta.get(m_id, {}).get("name", ""),
                    "target_addr": meta.get(m_id, {}).get("address", ""),
                })

        if s1_all_found:
            s1_complete += 1
            s1_complete_by_country[c_country] += 1

    # Candidate stats per S1
    cand_counts_by_s1 = [len(s1_cands.get(s1, set())) for s1 in truth_map]
    avg_cands_s1 = sum(cand_counts_by_s1) / float(len(cand_counts_by_s1)) if cand_counts_by_s1 else 0.0

    # Reduction ratios per country
    s1_counts_by_country = Counter()
    query_counts_by_country = Counter()
    for s1_id in truth_map:
        s1_counts_by_country[meta.get(s1_id, {}).get("country", "Unknown")] += 1
    for r_id, c_dict in record_cands.items():
        query_counts_by_country[meta.get(r_id, {}).get("country", "Unknown")] += 1

    # Reduction ratio: 1 - (avg candidates per record / total S1 in country)
    reduction_by_country = {}
    for c in countries:
        n_queries = query_counts_by_country[c]
        n_s1 = s1_counts_by_country[c]
        total_links = sum(len(cands) for r, cands in record_cands.items() if meta.get(r, {}).get("country") == c)
        avg_per_record = (total_links / float(n_queries)) if n_queries > 0 else 0.0
        red_ratio = (1.0 - (avg_per_record / float(n_s1))) if n_s1 > 0 else 0.0
        reduction_by_country[c] = (avg_per_record, red_ratio)

    overall_recall = (recalled_pairs / float(total_pairs)) if total_pairs > 0 else 1.0
    overall_s1_complete = (s1_complete / float(len(non_singletons))) if non_singletons else 1.0

    print(f"\n--- Results Summary ---")
    print(f"Overall Pairwise Recall: {recalled_pairs}/{total_pairs} ({overall_recall:.2%})")
    print(f"Overall Complete S1 Recall: {s1_complete}/{len(non_singletons)} ({overall_s1_complete:.2%})")
    for c in countries:
        p_c = pairs_by_country[c]
        r_c = recalled_by_country[c]
        s1_tot = s1_by_country[c]
        s1_cmp = s1_complete_by_country[c]
        print(f"  [{c}] Pairwise: {r_c}/{p_c} ({r_c/p_c:.2%}) | Complete S1: {s1_cmp}/{s1_tot} ({s1_cmp/s1_tot:.2%})")

    print(f"\nGenerator Contributions (out of {recalled_pairs} recalled pairs):")
    for g in ["a", "b", "c"]:
        g_count = gen_hits[g]
        print(f"  Generator ({g}): {g_count} ({g_count/total_pairs:.2%})")

    print(f"\nAvg Candidates per S1 Entity: {avg_cands_s1:.2f}")
    for c in countries:
        avg_rec, red = reduction_by_country[c]
        print(f"  [{c}] Avg cands/record: {avg_rec:.2f} | Reduction Ratio: {red:.6%}")

    # Diagnose 15 missed records
    random.seed(sample_seed)
    sampled_misses = random.sample(missed_records, min(15, len(missed_records)))

    # Write report
    os.makedirs(os.path.dirname(os.path.abspath(out_report)), exist_ok=True)
    report_lines = []
    report_lines.append("# Audit Report: Step 3 - Candidate Blocking & Recall Analysis\n")
    report_lines.append(f"**Verdict:** {'RECALL OK (>=95%)' if overall_recall >= 0.95 else 'RECALL NEEDS WORK (<95%)'}\n")

    report_lines.append("## 1. Recall Metrics Breakdown")
    report_lines.append("| Metric / Dimension | Total | Recalled | Pairwise Recall | Complete S1 Recall |")
    report_lines.append("|:---|---:|---:|---:|---:|")
    report_lines.append(f"| **Overall** | {total_pairs} | {recalled_pairs} | **{overall_recall:.2%}** | **{overall_s1_complete:.2%}** |")
    for c in countries:
        p_c = pairs_by_country[c]
        r_c = recalled_by_country[c]
        s1_tot = s1_by_country[c]
        s1_cmp = s1_complete_by_country[c]
        report_lines.append(f"| Country: {c} | {p_c} | {r_c} | {r_c/p_c:.2%} | {s1_cmp/s1_tot:.2%} |")

    report_lines.append("\n## 2. Generator Contribution & Blocking Efficiency")
    report_lines.append("| Generator | Description | Recalled Pairs | Standalone Recall |")
    report_lines.append("|:---|:---|---:|---:|")
    report_lines.append(f"| **(a)** | Char 3-gram TF-IDF (top-15) | {gen_hits['a']} | {gen_hits['a']/total_pairs:.2%} |")
    report_lines.append(f"| **(b)** | Rare Token Inverted Index (top-15) | {gen_hits['b']} | {gen_hits['b']/total_pairs:.2%} |")
    report_lines.append(f"| **(c)** | Address Token Inverted Index (top-10) | {gen_hits['c']} | {gen_hits['c']/total_pairs:.2%} |")
    report_lines.append(f"\n- **Avg Candidates per S1 Entity:** {avg_cands_s1:.2f}")
    for c in countries:
        avg_rec, red = reduction_by_country[c]
        report_lines.append(f"- **Reduction Ratio ({c}):** {red:.6%} (avg {avg_rec:.2f} cands/record out of {s1_counts_by_country[c]:,} S1s)")

    report_lines.append("\n## 3. Analysis of Sampled Missed Pairs (15 Cases)")
    report_lines.append("| # | Country | S1 Entity / Name / Address | Target Record / Name / Address | Likely Cause |")
    report_lines.append("|:--|:---|:---|:---|:---|")

    for idx, m in enumerate(sampled_misses, 1):
        s1_str = f"`{m['s1_id']}`: {m['s1_name'][:30]} | {m['s1_addr'][:30]}"
        tgt_str = f"`{m['target_id']}`: {m['target_name'][:30]} | {m['target_addr'][:30]}"

        # Diagnose cause
        addr_blank = (len(m["s1_addr"].strip()) == 0 or len(m["target_addr"].strip()) == 0)
        name_empty = (len(m["s1_name"].strip()) == 0 or len(m["target_name"].strip()) == 0)
        if addr_blank:
            cause = "Blank address; severe name deviation/typo"
        elif name_empty:
            cause = "Blank name field; address tokens diverged"
        else:
            cause = "Extreme typo/phonetic distortion + distinct address representation"

        report_lines.append(f"| {idx} | {m['country']} | {s1_str} | {tgt_str} | {cause} |")

    report_content = "\n".join(report_lines) + "\n"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nReport generated at: {out_report} ({len(report_lines)} lines)")
    return overall_recall, overall_s1_complete, gen_hits, avg_cands_s1


def main():
    parser = argparse.ArgumentParser(description="Measure blocking candidate recall.")
    parser.add_argument("--gt-path", type=str, default="data_mini/ground_truth.parquet")
    parser.add_argument(
        "--record-cand-path",
        type=str,
        default="data_mini/candidate_pairs_by_record.parquet",
    )
    parser.add_argument(
        "--s1-cand-path",
        type=str,
        default="data_mini/candidates_s1_grouped.parquet",
    )
    parser.add_argument("--s1-path", type=str, default="data_mini/source1_normalized.parquet")
    parser.add_argument("--s2-path", type=str, default="data_mini/source2_normalized.parquet")
    parser.add_argument("--s3-path", type=str, default="data_mini/source3_normalized.parquet")
    parser.add_argument("--out-report", type=str, default="audit/step3_blocking_report.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyze_blocking(
        gt_path=args.gt_path,
        record_cand_path=args.record_cand_path,
        s1_cand_path=args.s1_cand_path,
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        out_report=args.out_report,
        sample_seed=args.seed,
    )


if __name__ == "__main__":
    main()
