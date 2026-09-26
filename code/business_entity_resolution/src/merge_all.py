#!/usr/bin/env python3
"""
Merges France, US, and India per-country matching and candidate files
into the final submission files:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv

Verifies row counts and formats.
"""

import os
import sys
import time

def log_msg(msg: str, log_file: str = "run_full_pipeline.log"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

def merge_files(country_files, out_file, header, log_file):
    t0 = time.time()
    total_rows = 0
    with open(out_file, "w", encoding="utf-8") as f_out:
        f_out.write(header + "\n")
        for path in country_files:
            log_msg(f"Streaming {os.path.basename(path)} into {os.path.basename(out_file)}...", log_file)
            with open(path, "r", encoding="utf-8") as f_in:
                # verify header
                in_header = f_in.readline().rstrip("\n")
                if in_header != header:
                    raise ValueError(f"Header mismatch in {path}: expected '{header}', got '{in_header}'")
                for line in f_in:
                    f_out.write(line)
                    total_rows += 1
    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_file) / (1024.0 * 1024.0)
    log_msg(f"Created {out_file} ({size_mb:.2f} MB, {total_rows:,} data rows) in {elapsed:.2f}s.", log_file)
    return total_rows

def main():
    out_dir = "output"
    log_file = "run_full_pipeline.log"
    
    matching_countries = [
        os.path.join(out_dir, "tmp_country_matching_france.tsv"),
        os.path.join(out_dir, "tmp_country_matching_us.tsv"),
        os.path.join(out_dir, "tmp_country_matching_india.tsv"),
    ]
    
    cands_countries = [
        os.path.join(out_dir, "tmp_country_cands_france.tsv"),
        os.path.join(out_dir, "tmp_country_cands_us.tsv"),
        os.path.join(out_dir, "tmp_country_cands_india.tsv"),
    ]
    
    for p in matching_countries + cands_countries:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required country file: {p}")
            
    final_matching = os.path.join(out_dir, "matching_results.tsv")
    final_cands = os.path.join(out_dir, "candidate_pairs.tsv")
    
    log_msg("============================================================", log_file)
    log_msg("MERGING ALL COUNTRIES INTO FINAL SUBMISSION FILES", log_file)
    log_msg("============================================================", log_file)
    
    m_rows = merge_files(
        matching_countries,
        final_matching,
        "source1_entity_id\tmatched_entity_ids",
        log_file
    )
    
    c_rows = merge_files(
        cands_countries,
        final_cands,
        "source1_entity_id\tcandidate_entity_ids",
        log_file
    )
    
    expected_s1 = 1732544
    if m_rows != expected_s1:
        raise ValueError(f"matching_results.tsv has {m_rows} rows, expected {expected_s1}")
    if c_rows != expected_s1:
        raise ValueError(f"candidate_pairs.tsv has {c_rows} rows, expected {expected_s1}")
        
    log_msg("All files merged and row count verified (1,732,544 rows each).", log_file)

if __name__ == "__main__":
    main()
