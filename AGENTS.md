# Agent Guidelines

- Goal: Link Source 2/3 records to each Source 1 entity, scored by per-entity macro F0.5 (singletons count; empty prediction = 1.0 if right).
- Files are tab-separated: `pd.read_csv(sep="\t", dtype=str, keep_default_na=False)`.
- Never hard-code country names; France appears only in test.
- No external APIs or data lookups; no LLM calls inside the pipeline; only MIT/Apache-2.0 models up to 8B params.
- Paths come from CLI arguments, never hard-coded.
- Every S1 entity needs one output row; matches must be a subset of candidates.
- Never print whole data files; keep tool output short.
- Append decisions and results to NOTES.md after every major step.
