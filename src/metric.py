#!/usr/bin/env python3
"""
Evaluation Metric for Business Entity Resolution.
Computes per-entity macro F0.5 over all S1 entities.
"""

from typing import Any, Dict, Set, Union


def _parse_ids(val: Any) -> Set[str]:
    """Parse comma-separated string or iterable of IDs into a set of clean strings."""
    if val is None:
        return set()
    if isinstance(val, (set, frozenset)):
        return {str(x).strip() for x in val if str(x).strip()}
    if isinstance(val, (list, tuple)):
        return {str(x).strip() for x in val if str(x).strip()}
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return set()
        return {x.strip() for x in val.split(",") if x.strip()}
    s = str(val).strip()
    return {s} if s else set()


def single_entity_f05(pred_ids: Union[Set[str], Any], truth_ids: Union[Set[str], Any]) -> float:
    """
    Computes F0.5 score for a single S1 entity.
    Rules:
      - empty truth and empty pred = 1.0
      - empty truth with any pred = 0.0
      - empty pred with non-empty truth = 0.0
      - non-empty truth and non-empty pred: F0.5 with beta=0.5
    """
    p = _parse_ids(pred_ids)
    t = _parse_ids(truth_ids)

    len_p = len(p)
    len_t = len(t)

    if len_t == 0 and len_p == 0:
        return 1.0
    if len_t == 0 and len_p > 0:
        return 0.0
    if len_p == 0 and len_t > 0:
        return 0.0

    tp = len(p & t)
    if tp == 0:
        return 0.0

    # F_beta with beta=0.5:
    # F0.5 = (1 + 0.25) * tp / (len_p + 0.25 * len_t)
    return 1.25 * tp / (len_p + 0.25 * len_t)


def per_entity_f05(pred: Dict[str, Any], truth: Dict[str, Any]) -> float:
    """
    Computes macro F0.5 over all S1 entities defined in truth.
    Any entity in truth missing from pred is treated as empty prediction.
    """
    if not truth:
        return 1.0 if not pred else 0.0

    total_score = 0.0
    for s1_id, t_val in truth.items():
        p_val = pred.get(s1_id, set())
        total_score += single_entity_f05(p_val, t_val)

    return total_score / len(truth)
