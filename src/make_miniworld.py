#!/usr/bin/env python3
"""
Make Mini-World Dataset for Business Entity Resolution.
Reads TSVs via DuckDB streaming without full pandas loads.
Samples stratified fraction of S1, retains ground truth, matched S2/S3,
and proportional unmatched S2/S3 distractors.
"""

import os
import sys
import argparse
import duckdb


def resolve_input_paths(data_dir):
    """Find source TSVs in data_dir, supporting train/ subdirectory or root."""
    paths = {}
    candidates = {
        "s1": ["train/train_source1.tsv", "train_source1.tsv"],
        "s2": ["train/train_source2.tsv", "train_source2.tsv"],
        "s3": ["train/train_source3.tsv", "train_source3.tsv"],
        "gt": ["train/train_ground_truth.tsv", "train_ground_truth.tsv"],
    }
    for key, options in candidates.items():
        found = None
        for opt in options:
            p = os.path.join(data_dir, opt)
            if os.path.exists(p):
                found = p
                break
        if not found:
            raise FileNotFoundError(f"Could not find TSV for {key} in {data_dir} (checked: {options})")
        paths[key] = found
    return paths


def make_miniworld(data_dir: str, out_dir: str, frac: float = 0.05, seed: int = 42):
    paths = resolve_input_paths(data_dir)
    os.makedirs(out_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")

    # 1. Read S1 and stratify sample by country
    con.execute(f"""
        CREATE TEMP TABLE s1_all AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{paths["s1"]}', delim='\\t', header=true, all_varchar=true, quote='', escape='');
    """)

    con.execute(f"""
        CREATE TEMP TABLE s1_sampled AS
        SELECT entity_id, business_name, business_address, country
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY country
                    ORDER BY hash(entity_id || '_s1_{seed}')
                ) AS rn,
                COUNT(*) OVER (PARTITION BY country) AS total_count
            FROM s1_all
        )
        WHERE rn <= ROUND(total_count * {frac});
    """)

    # 2. Read full Ground Truth and get sampled GT
    con.execute(f"""
        CREATE TEMP TABLE gt_all AS
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('{paths["gt"]}', delim='\\t', header=true, all_varchar=true, quote='', escape='');
    """)

    con.execute("""
        CREATE TEMP TABLE gt_sampled AS
        SELECT g.source1_entity_id, g.matched_entity_ids
        FROM gt_all g
        INNER JOIN s1_sampled s ON g.source1_entity_id = s.entity_id;
    """)

    # 3. All matched IDs across the full dataset (to identify true distractors)
    con.execute("""
        CREATE TEMP TABLE all_matched_ids AS
        SELECT DISTINCT match_id
        FROM (
            SELECT unnest(string_split(matched_entity_ids, ',')) AS match_id
            FROM gt_all
        )
        WHERE match_id != '' AND match_id IS NOT NULL;
    """)

    # 4. Matched IDs for the sampled S1 entities
    con.execute("""
        CREATE TEMP TABLE sampled_matched_ids AS
        SELECT DISTINCT match_id
        FROM (
            SELECT unnest(string_split(matched_entity_ids, ',')) AS match_id
            FROM gt_sampled
        )
        WHERE match_id != '' AND match_id IS NOT NULL;
    """)

    # 5. Process S2: matched + stratified sample of unmatched
    con.execute(f"""
        CREATE TEMP TABLE s2_all AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{paths["s2"]}', delim='\\t', header=true, all_varchar=true, quote='', escape='');
    """)

    con.execute("""
        CREATE TEMP TABLE s2_matched_sampled AS
        SELECT s.entity_id, s.business_name, s.business_address, s.country, 0 AS is_distractor
        FROM s2_all s
        INNER JOIN sampled_matched_ids m ON s.entity_id = m.match_id;
    """)

    con.execute(f"""
        CREATE TEMP TABLE s2_unmatched_sampled AS
        SELECT entity_id, business_name, business_address, country, 1 AS is_distractor
        FROM (
            SELECT s.*,
                ROW_NUMBER() OVER (
                    PARTITION BY country
                    ORDER BY hash(entity_id || '_s2_dist_{seed}')
                ) AS rn,
                COUNT(*) OVER (PARTITION BY country) AS total_count
            FROM s2_all s
            ANTI JOIN all_matched_ids m ON s.entity_id = m.match_id
        )
        WHERE rn <= ROUND(total_count * {frac});
    """)

    con.execute("""
        CREATE TEMP TABLE mini_s2 AS
        SELECT * FROM s2_matched_sampled
        UNION ALL
        SELECT * FROM s2_unmatched_sampled;
    """)

    # 6. Process S3: matched + stratified sample of unmatched
    con.execute(f"""
        CREATE TEMP TABLE s3_all AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{paths["s3"]}', delim='\\t', header=true, all_varchar=true, quote='', escape='');
    """)

    con.execute("""
        CREATE TEMP TABLE s3_matched_sampled AS
        SELECT s.entity_id, s.business_name, s.business_address, s.country, 0 AS is_distractor
        FROM s3_all s
        INNER JOIN sampled_matched_ids m ON s.entity_id = m.match_id;
    """)

    con.execute(f"""
        CREATE TEMP TABLE s3_unmatched_sampled AS
        SELECT entity_id, business_name, business_address, country, 1 AS is_distractor
        FROM (
            SELECT s.*,
                ROW_NUMBER() OVER (
                    PARTITION BY country
                    ORDER BY hash(entity_id || '_s3_dist_{seed}')
                ) AS rn,
                COUNT(*) OVER (PARTITION BY country) AS total_count
            FROM s3_all s
            ANTI JOIN all_matched_ids m ON s.entity_id = m.match_id
        )
        WHERE rn <= ROUND(total_count * {frac});
    """)

    con.execute("""
        CREATE TEMP TABLE mini_s3 AS
        SELECT * FROM s3_matched_sampled
        UNION ALL
        SELECT * FROM s3_unmatched_sampled;
    """)

    # 7. Export Parquet files
    s1_out = os.path.join(out_dir, "source1.parquet")
    s2_out = os.path.join(out_dir, "source2.parquet")
    s3_out = os.path.join(out_dir, "source3.parquet")
    gt_out = os.path.join(out_dir, "ground_truth.parquet")

    con.execute(f"COPY s1_sampled TO '{s1_out}' (FORMAT PARQUET);")
    con.execute(f"COPY (SELECT entity_id, business_name, business_address, country FROM mini_s2) TO '{s2_out}' (FORMAT PARQUET);")
    con.execute(f"COPY (SELECT entity_id, business_name, business_address, country FROM mini_s3) TO '{s3_out}' (FORMAT PARQUET);")
    con.execute(f"COPY gt_sampled TO '{gt_out}' (FORMAT PARQUET);")

    # Also link train_source*.parquet for compatibility
    for src, alias in [
        (s1_out, "train_source1.parquet"),
        (s2_out, "train_source2.parquet"),
        (s3_out, "train_source3.parquet"),
        (gt_out, "train_ground_truth.parquet"),
    ]:
        alias_path = os.path.join(out_dir, alias)
        if os.path.exists(alias_path):
            os.remove(alias_path)
        os.link(src, alias_path)

    # 8. Compute and print statistics: row counts, singleton share, distractor share per country
    stats_query = """
    WITH
    c_list AS (
        SELECT DISTINCT country FROM s1_sampled
    ),
    s1_stats AS (
        SELECT
            s.country,
            COUNT(*) AS s1_cnt,
            COUNT(CASE WHEN g.matched_entity_ids = '' OR g.matched_entity_ids IS NULL THEN 1 END) AS singleton_cnt
        FROM s1_sampled s
        JOIN gt_sampled g ON s.entity_id = g.source1_entity_id
        GROUP BY s.country
    ),
    s2_stats AS (
        SELECT
            country,
            COUNT(*) AS s2_cnt,
            SUM(is_distractor) AS s2_dist_cnt
        FROM mini_s2
        GROUP BY country
    ),
    s3_stats AS (
        SELECT
            country,
            COUNT(*) AS s3_cnt,
            SUM(is_distractor) AS s3_dist_cnt
        FROM mini_s3
        GROUP BY country
    )
    SELECT
        c.country,
        s1.s1_cnt,
        s2.s2_cnt,
        s3.s3_cnt,
        (s1.singleton_cnt * 100.0 / s1.s1_cnt) AS singleton_share,
        (s2.s2_dist_cnt * 100.0 / s2.s2_cnt) AS s2_dist_share,
        (s3.s3_dist_cnt * 100.0 / s3.s3_cnt) AS s3_dist_share,
        ((s2.s2_dist_cnt + s3.s3_dist_cnt) * 100.0 / (s2.s2_cnt + s3.s3_cnt)) AS total_dist_share
    FROM c_list c
    JOIN s1_stats s1 ON c.country = s1.country
    JOIN s2_stats s2 ON c.country = s2.country
    JOIN s3_stats s3 ON c.country = s3.country
    ORDER BY c.country;
    """

    rows = con.execute(stats_query).fetchall()

    tot_s1 = sum(r[1] for r in rows)
    tot_s2 = sum(r[2] for r in rows)
    tot_s3 = sum(r[3] for r in rows)
    tot_s1_single = con.execute("SELECT COUNT(*) FROM gt_sampled WHERE matched_entity_ids = '' OR matched_entity_ids IS NULL").fetchone()[0]
    tot_s2_dist = con.execute("SELECT SUM(is_distractor) FROM mini_s2").fetchone()[0]
    tot_s3_dist = con.execute("SELECT SUM(is_distractor) FROM mini_s3").fetchone()[0]

    tot_singleton_share = tot_s1_single * 100.0 / tot_s1 if tot_s1 else 0.0
    tot_dist_share = (tot_s2_dist + tot_s3_dist) * 100.0 / (tot_s2 + tot_s3) if (tot_s2 + tot_s3) else 0.0

    print("=== Mini-World Sampling Summary ===")
    for r in rows:
        country, s1_c, s2_c, s3_c, s_share, s2_d, s3_d, t_d = r
        print(f"Country: {country}")
        print(f"  Row counts: S1={s1_c:,}, S2={s2_c:,}, S3={s3_c:,}, GT={s1_c:,}")
        print(f"  Singleton share: {s_share:.2f}%")
        print(f"  Distractor share: S2={s2_d:.2f}%, S3={s3_d:.2f}%, Combined={t_d:.2f}%")

    print("Overall:")
    print(f"  Row counts: S1={tot_s1:,}, S2={tot_s2:,}, S3={tot_s3:,}, GT={tot_s1:,}")
    print(f"  Singleton share: {tot_singleton_share:.2f}%")
    print(f"  Distractor share: Combined={tot_dist_share:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Create mini-world sample of training dataset.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directory containing dataset files")
    parser.add_argument("--out-dir", type=str, default="data_mini", help="Output directory for mini-world parquet")
    parser.add_argument("--frac", type=float, default=0.05, help="Sampling fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    make_miniworld(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        frac=args.frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
