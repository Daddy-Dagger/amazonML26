#!/usr/bin/env python3
"""
Groups India matches and candidates into per-country TSVs:
  - output/tmp_country_matching_india.tsv
  - output/tmp_country_cands_india.tsv

Uses 16-bucket streaming partition to bound peak memory < 1.0 GB RAM.
"""

import os
import sys
import time
import gc
import collections
import resource

def get_peak_memory_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 ** 3)
    else:
        return usage / (1024.0 ** 2)

def log_msg(msg: str, log_file: str = "run_full_pipeline.log"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

def main():
    test_dir = "dataset/test"
    out_dir = "output"
    log_file = "run_full_pipeline.log"
    
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    tmp_matches_file = os.path.join(out_dir, "tmp_matches_india.tsv")
    tmp_cands_file = os.path.join(out_dir, "tmp_cands_india.tsv")
    
    out_country_matching = os.path.join(out_dir, "tmp_country_matching_india.tsv")
    out_country_cands = os.path.join(out_dir, "tmp_country_cands_india.tsv")
    
    if not os.path.exists(tmp_matches_file):
        raise FileNotFoundError(f"Missing {tmp_matches_file}")
    if not os.path.exists(tmp_cands_file):
        raise FileNotFoundError(f"Missing {tmp_cands_file}")
        
    log_msg("[India] Starting memory-safe grouping for India...", log_file)
    t0 = time.time()
    
    # 1. Load India S1 entity IDs from test_source1.tsv
    log_msg("[India] Loading India S1 IDs...", log_file)
    india_s1_list = []
    with open(test_s1_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == "India":
                india_s1_list.append(parts[0])
                
    total_s1 = len(india_s1_list)
    log_msg(f"[India] Loaded {total_s1:,} India S1 IDs. Peak RAM: {get_peak_memory_gb():.2f} GB", log_file)
    
    # Partition India S1 list into 16 buckets
    num_buckets = 16
    bucket_s1 = [[] for _ in range(num_buckets)]
    for s1 in india_s1_list:
        b = hash(s1) & (num_buckets - 1)
        bucket_s1[b].append(s1)
        
    # 2. Group Matches (3.25M rows)
    log_msg("[India] Grouping matches...", log_file)
    t_match = time.time()
    matches_by_s1 = collections.defaultdict(list)
    total_matches = 0
    with open(tmp_matches_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.rstrip("\n")
            if not line_str:
                continue
            s1, m = line_str.split("\t")
            matches_by_s1[s1].append(m)
            total_matches += 1
            
    s1_with_matches = len(matches_by_s1)
    match_rate_pct = (s1_with_matches / float(total_s1) * 100.0) if total_s1 else 0.0
    avg_matches_per_s1 = total_matches / float(total_s1) if total_s1 else 0.0
    
    # Write matching TSV in bucket order
    with open(out_country_matching, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for b in range(num_buckets):
            for s1 in bucket_s1[b]:
                m_list = matches_by_s1.get(s1, [])
                f_out.write(f"{s1}\t{','.join(m_list)}\n")
                
    del matches_by_s1
    gc.collect()
    log_msg(f"[India] Matches grouped in {time.time()-t_match:.2f}s. Total matches: {total_matches:,}, S1 with match: {s1_with_matches:,} ({match_rate_pct:.2f}%). Peak RAM: {get_peak_memory_gb():.2f} GB", log_file)
    
    # 3. Partition Candidates into 16 buckets
    log_msg("[India] Partitioning 116.8M candidates into 16 buckets...", log_file)
    t_part = time.time()
    bucket_files = [open(os.path.join(out_dir, f"tmp_cand_b_{b}.tsv"), "w", buffering=2*1024*1024, encoding="utf-8") for b in range(num_buckets)]
    
    cand_line_count = 0
    with open(tmp_cands_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            tab = line.find("\t")
            if tab == -1:
                continue
            s1 = line[:tab]
            b = hash(s1) & (num_buckets - 1)
            bucket_files[b].write(line)
            cand_line_count += 1
            
    for bf in bucket_files:
        bf.close()
    log_msg(f"[India] Partitioned {cand_line_count:,} candidates into 16 buckets in {time.time()-t_part:.2f}s. Peak RAM: {get_peak_memory_gb():.2f} GB", log_file)
    
    # 4. Group Each Candidate Bucket & Stream to Final TSV
    log_msg("[India] Grouping candidate buckets and writing candidate_pairs...", log_file)
    t_grp = time.time()
    s1_with_cands = 0
    with open(out_country_cands, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for b in range(num_buckets):
            b_path = os.path.join(out_dir, f"tmp_cand_b_{b}.tsv")
            cands_dict = collections.defaultdict(list)
            with open(b_path, "r", encoding="utf-8") as f_b:
                for line in f_b:
                    tab = line.find("\t")
                    if tab == -1:
                        continue
                    s1 = line[:tab]
                    cand = line[tab+1:-1]
                    cands_dict[s1].append(cand)
                    
            if os.path.exists(b_path):
                os.remove(b_path)
                
            # Write bucket rows
            for s1 in bucket_s1[b]:
                c_list = cands_dict.get(s1, [])
                if c_list:
                    s1_with_cands += 1
                    # deduplicate preserving order
                    deduped = list(dict.fromkeys(c_list))
                    f_out.write(f"{s1}\t{','.join(deduped)}\n")
                else:
                    f_out.write(f"{s1}\t\n")
                    
            del cands_dict
            gc.collect()
            
    cand_cov_pct = (s1_with_cands / float(total_s1) * 100.0) if total_s1 else 0.0
    log_msg(f"[India] Candidates grouped in {time.time()-t_grp:.2f}s. S1 with cands: {s1_with_cands:,} ({cand_cov_pct:.2f}%). Peak RAM: {get_peak_memory_gb():.2f} GB", log_file)
    
    # 5. Clean up large tmp candidate/match pair files
    log_msg("[India] Cleaning up tmp candidate and match pair files...", log_file)
    if os.path.exists(tmp_cands_file):
        os.remove(tmp_cands_file)
    if os.path.exists(tmp_matches_file):
        os.remove(tmp_matches_file)
        
    elapsed_total = time.time() - t0
    peak_mem = get_peak_memory_gb()
    
    log_msg("------------------------------------------------------------", log_file)
    log_msg("COUNTRY SUMMARY: India", log_file)
    log_msg(f"  S1 Entities:              {total_s1:,}", log_file)
    log_msg(f"  Candidate Coverage:       {s1_with_cands:,} ({cand_cov_pct:.2f}%)", log_file)
    log_msg(f"  Total Candidate Pairs:    {cand_line_count:,}", log_file)
    log_msg(f"  S1 Entities with Match:   {s1_with_matches:,} ({match_rate_pct:.2f}%)", log_file)
    log_msg(f"  Total Accepted Matches:   {total_matches:,}", log_file)
    log_msg(f"  Avg Matches per S1:       {avg_matches_per_s1:.2f}", log_file)
    log_msg(f"  Elapsed Time:             {elapsed_total:.2f}s ({elapsed_total/60:.2f}m)", log_file)
    log_msg(f"  Peak Memory:              {peak_mem:.2f} GB", log_file)
    log_msg("------------------------------------------------------------\n", log_file)

if __name__ == "__main__":
    main()
