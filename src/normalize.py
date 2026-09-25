#!/usr/bin/env python3
"""
Country-agnostic Text Normalization for Business Entity Resolution.
Applies NFKD, unidecode, lowercase, & -> and, punctuation stripping, space collapsing,
leading 'the' stripping, abbreviation expansion, consecutive duplicate removal,
and data-driven top-40 generic token identification per country.
"""

import os
import re
import math
import argparse
import unicodedata
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Any

import unidecode
import pyarrow as pa
import pyarrow.parquet as pq

# Pre-compiled regexes
AMP_RE = re.compile(r"&")
PUNCT_RE = re.compile(r"[^\w\s]")
SPACE_RE = re.compile(r"\s+")

# Common abbreviation dictionary (applied to all fields)
COMMON_ABBR = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "dr": "drive",
    "ln": "lane",
    "blvd": "boulevard",
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "co": "company",
}

# Address-specific abbreviations (applied in address fields only)
ADDR_EXTRA_ABBR = {
    "r": "rue",
    "av": "avenue",
    "bd": "boulevard",
}

ADDR_ABBR = {**COMMON_ABBR, **ADDR_EXTRA_ABBR}


def normalize_text(text: Any, is_address: bool = False) -> List[str]:
    """
    Normalizes a text string into a list of cleaned tokens.
    Country-agnostic: never branches on country.
    """
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []

    # 1. NFKD normalization + unidecode transliteration
    s = unicodedata.normalize("NFKD", s)
    s = unidecode.unidecode(s)

    # 2. Lowercase
    s = s.lower()

    # 3. & -> and
    s = AMP_RE.sub(" and ", s)

    # 4. Strip punctuation (replace with space to preserve word boundaries)
    s = PUNCT_RE.sub(" ", s)

    # 5. Collapse spaces
    s = SPACE_RE.sub(" ", s).strip()
    if not s:
        return []

    # 6. Tokenize & drop leading 'the'
    toks = s.split()
    if toks and toks[0] == "the":
        toks = toks[1:]
    if not toks:
        return []

    # 7. Abbreviation lookup
    abbr_map = ADDR_ABBR if is_address else COMMON_ABBR
    toks = [abbr_map.get(t, t) for t in toks]

    # 8. Remove repeated identical tokens (consecutive)
    dedup_toks = []
    for t in toks:
        if not dedup_toks or t != dedup_toks[-1]:
            dedup_toks.append(t)

    return dedup_toks


def compute_generic_and_idf(
    records_by_country: Dict[str, List[List[str]]],
    num_generic: int = 40,
) -> Tuple[Dict[str, Set[str]], List[Dict[str, Any]]]:
    """
    Computes document frequency and IDF weights per country over all provided name token lists.
    Identifies the top num_generic tokens per country as generic.
    Country-agnostic: operates purely on the distinct country keys present in input.
    """
    generic_by_country = {}
    idf_rows = []

    for country, doc_tokens_list in records_by_country.items():
        doc_count = len(doc_tokens_list)
        df_counter = Counter()

        for toks in doc_tokens_list:
            unique_toks = set(toks)
            for t in unique_toks:
                df_counter[t] += 1

        # Sort tokens by doc_freq descending, then token ascending for deterministic ordering
        sorted_tokens = sorted(df_counter.items(), key=lambda x: (-x[1], x[0]))

        top_generic = {t for t, _ in sorted_tokens[:num_generic]}
        generic_by_country[country] = top_generic

        for t, df in sorted_tokens:
            # Standard smoothed IDF: ln((N + 1) / (df + 1)) + 1.0
            idf_val = math.log((doc_count + 1.0) / (df + 1.0)) + 1.0
            idf_rows.append({
                "country": country,
                "token": t,
                "doc_freq": df,
                "idf": idf_val,
                "is_generic": (t in top_generic),
            })

    return generic_by_country, idf_rows


def process_table_normalization(
    table: pa.Table,
    generic_by_country: Dict[str, Set[str]],
) -> pa.Table:
    """
    Given a PyArrow table with entity_id, business_name, business_address, country,
    adds name_full, name_core, and address token columns.
    """
    names = table["business_name"].to_pylist()
    addrs = table["business_address"].to_pylist()
    countries = table["country"].to_pylist()

    name_full_list = []
    name_core_list = []
    addr_list = []

    for name, addr, country in zip(names, addrs, countries):
        norm_name = normalize_text(name, is_address=False)
        norm_addr = normalize_text(addr, is_address=True)

        generic_set = generic_by_country.get(country, set())
        core_name = [t for t in norm_name if t not in generic_set]

        name_full_list.append(norm_name)
        name_core_list.append(core_name)
        addr_list.append(norm_addr)

    # Append new columns to original table
    res_table = table.append_column("name_full", pa.array(name_full_list, type=pa.list_(pa.string())))
    res_table = res_table.append_column("name_core", pa.array(name_core_list, type=pa.list_(pa.string())))
    res_table = res_table.append_column("address", pa.array(addr_list, type=pa.list_(pa.string())))

    return res_table


def normalize_dataset(in_dir: str, out_dir: str, num_generic: int = 40):
    os.makedirs(out_dir, exist_ok=True)

    source_names = ["source1.parquet", "source2.parquet", "source3.parquet"]
    source_tables = {}

    for s_name in source_names:
        p = os.path.join(in_dir, s_name)
        if not os.path.exists(p):
            alt_p = os.path.join(in_dir, f"train_{s_name}")
            if os.path.exists(alt_p):
                p = alt_p
            else:
                raise FileNotFoundError(f"Could not locate {s_name} in {in_dir}")
        source_tables[s_name] = pq.read_table(p)

    # Pass 1: Normalize all business names across all sources to collect document frequencies per country
    names_by_country = defaultdict(list)
    for s_name, table in source_tables.items():
        names = table["business_name"].to_pylist()
        countries = table["country"].to_pylist()
        for name, country in zip(names, countries):
            norm_tokens = normalize_text(name, is_address=False)
            names_by_country[country].append(norm_tokens)

    # Pass 2: Compute per-country generic tokens and IDF weights
    generic_by_country, idf_records = compute_generic_and_idf(names_by_country, num_generic=num_generic)

    # Save IDF weights table to parquet
    idf_table = pa.Table.from_pylist(idf_records)
    idf_path = os.path.join(out_dir, "idf_weights.parquet")
    pq.write_table(idf_table, idf_path)

    # Pass 3: Produce normalized tables and write to parquet
    for s_name, table in source_tables.items():
        norm_table = process_table_normalization(table, generic_by_country)
        base_root = s_name.replace(".parquet", "")
        out_path = os.path.join(out_dir, f"{base_root}_normalized.parquet")
        pq.write_table(norm_table, out_path)

        # Also create symlink for normalized_*.parquet
        alt_out = os.path.join(out_dir, f"normalized_{base_root}.parquet")
        if os.path.exists(alt_out):
            os.remove(alt_out)
        os.link(out_path, alt_out)

    return generic_by_country, idf_path


def main():
    parser = argparse.ArgumentParser(description="Normalize dataset tables and extract generic/IDF features.")
    parser.add_argument("--in-dir", type=str, default="data_mini", help="Input directory containing parquet files")
    parser.add_argument("--out-dir", type=str, default="data_mini", help="Output directory for normalized parquet")
    parser.add_argument("--num-generic", type=int, default=40, help="Number of generic tokens to mark per country")
    args = parser.parse_args()

    generic_by_country, idf_path = normalize_dataset(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        num_generic=args.num_generic,
    )

    print("=== Normalization Completed ===")
    print(f"IDF weights saved to: {idf_path}")
    for country, generics in sorted(generic_by_country.items()):
        print(f"Country: {country} — Identified {len(generics)} generic tokens")


if __name__ == "__main__":
    main()
