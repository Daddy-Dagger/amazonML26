#!/usr/bin/env python3
"""
Streaming fast validator to cross-check all rules locally in < 30 seconds.
"""
import sys
import time

def main():
    t0 = time.time()
    print("Starting streaming validation of submission files...")

    # 1. Load required S1 IDs
    print("Loading test_source1 S1 IDs...")
    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        required_s1 = set(line.split("\t", 1)[0].strip() for line in f)
    print(f"Required S1 entities: {len(required_s1):,}")

    # 2. Check matching_results.tsv
    print("\nValidating matching_results.tsv...")
    matching_seen = set()
    s2_s3_assigned = set()
    multi_assigned = 0
    matching_map = {}
    m_intra_dupes = 0
    m_self_matches = 0
    m_wrong_prefix = 0
    m_empty_count = 0

    with open("output/matching_results.tsv", "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        assert header == "source1_entity_id\tmatched_entity_ids", f"Bad header: {header}"
        for line in f:
            s1, tab, rest = line.partition("\t")
            matching_seen.add(s1)
            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                m_empty_count += 1
                matching_map[s1] = set()
                continue
            if len(ids) != len(set(ids)):
                m_intra_dupes += 1
            id_set = set(ids)
            matching_map[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    m_self_matches += 1
                elif not mid.startswith(("S2-", "S3-")):
                    m_wrong_prefix += 1
                if mid in s2_s3_assigned:
                    multi_assigned += 1
                s2_s3_assigned.add(mid)

    print(f"  matching rows: {len(matching_seen):,}")
    print(f"  empty rows (singletons): {m_empty_count:,}")
    print(f"  total accepted matches: {len(s2_s3_assigned):,}")
    print(f"  missing S1: {len(required_s1 - matching_seen)}")
    print(f"  extra S1: {len(matching_seen - required_s1)}")
    print(f"  multi-assigned S2/S3 queries: {multi_assigned}")
    print(f"  intra-dupes: {m_intra_dupes}")
    print(f"  self-matches: {m_self_matches}")
    print(f"  wrong-prefix: {m_wrong_prefix}")

    assert required_s1 == matching_seen, "Mismatch in S1 IDs!"
    assert multi_assigned == 0, f"Found {multi_assigned} multi-assigned queries!"
    assert m_intra_dupes == 0, "Intra-dupes found!"
    assert m_self_matches == 0, "Self matches found!"
    assert m_wrong_prefix == 0, "Wrong prefix found!"
    print(">> matching_results.tsv: 100% PERFECT!")

    # 3. Stream candidate_pairs.tsv and check subset property on the fly
    print("\nValidating candidate_pairs.tsv & subset check (streaming)...")
    cand_seen = set()
    c_intra_dupes = 0
    c_self_matches = 0
    c_wrong_prefix = 0
    c_empty_count = 0
    not_in_cands = 0
    total_cands = 0

    with open("output/candidate_pairs.tsv", "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        assert header == "source1_entity_id\tcandidate_entity_ids", f"Bad header: {header}"
        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition("\t")
            cand_seen.add(s1)
            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                c_empty_count += 1
                c_set = set()
            else:
                if len(ids) != len(set(ids)):
                    c_intra_dupes += 1
                c_set = set(ids)
                total_cands += len(ids)
                for cid in c_set:
                    if cid.startswith("S1-"):
                        c_self_matches += 1
                    elif not cid.startswith(("S2-", "S3-")):
                        c_wrong_prefix += 1

            # Check that matching is subset of candidates for this S1
            m_set = matching_map.get(s1, set())
            diff = m_set - c_set
            if diff:
                not_in_cands += len(diff)
                if not_in_cands <= 5:
                    print(f"  Mismatch at {s1}: matched {diff} not in candidates!")

            if line_num % 500000 == 0:
                print(f"  Processed {line_num:,} candidate rows...")

    print(f"  candidate rows: {len(cand_seen):,}")
    print(f"  empty candidate rows: {c_empty_count:,}")
    print(f"  total candidate pairs: {total_cands:,}")
    print(f"  missing S1: {len(required_s1 - cand_seen)}")
    print(f"  extra S1: {len(cand_seen - required_s1)}")
    print(f"  intra-dupes: {c_intra_dupes}")
    print(f"  self-matches: {c_self_matches}")
    print(f"  wrong-prefix: {c_wrong_prefix}")
    print(f"  matches NOT in candidates: {not_in_cands}")

    assert required_s1 == cand_seen, "Mismatch in S1 IDs!"
    assert c_intra_dupes == 0, "Intra-dupes found in candidates!"
    assert c_self_matches == 0, "Self matches found in candidates!"
    assert c_wrong_prefix == 0, "Wrong prefix found in candidates!"
    assert not_in_cands == 0, f"Found {not_in_cands} matches not in candidates!"
    print(">> candidate_pairs.tsv: 100% PERFECT!")

    print(f"\nALL CHECKS PASSED in {time.time()-t0:.2f}s! Safe to submit.")

if __name__ == "__main__":
    main()
