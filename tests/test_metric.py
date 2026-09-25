#!/usr/bin/env python3
"""
Unit tests for macro per-entity F0.5 metric.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metric import per_entity_f05, single_entity_f05


class TestMetric(unittest.TestCase):
    def test_statement_example(self):
        """
        pred S2-00047,S2-00193,S3-00812 vs truth S2-00047,S3-00812 must give 0.714
        """
        pred = {"S1-1": ["S2-00047", "S2-00193", "S3-00812"]}
        truth = {"S1-1": ["S2-00047", "S3-00812"]}
        score = per_entity_f05(pred, truth)
        self.assertAlmostEqual(score, 5.0 / 7.0, places=6)
        self.assertEqual(round(score, 3), 0.714)

    def test_statement_example_comma_string(self):
        """Test with comma-separated string inputs."""
        pred = {"S1-1": "S2-00047,S2-00193,S3-00812"}
        truth = {"S1-1": "S2-00047,S3-00812"}
        score = per_entity_f05(pred, truth)
        self.assertEqual(round(score, 3), 0.714)

    def test_singleton_empty_truth_empty_pred(self):
        """Empty truth and empty pred must return 1.0."""
        pred = {"S1-1": []}
        truth = {"S1-1": []}
        self.assertEqual(per_entity_f05(pred, truth), 1.0)
        self.assertEqual(single_entity_f05([], []), 1.0)
        self.assertEqual(single_entity_f05("", ""), 1.0)

    def test_singleton_empty_truth_nonempty_pred(self):
        """Empty truth with any prediction must return 0.0."""
        pred = {"S1-1": ["S2-00047"]}
        truth = {"S1-1": []}
        self.assertEqual(per_entity_f05(pred, truth), 0.0)
        self.assertEqual(single_entity_f05(["S2-00047"], []), 0.0)
        self.assertEqual(single_entity_f05("S2-00047", ""), 0.0)

    def test_singleton_empty_pred_nonempty_truth(self):
        """Empty pred with non-empty truth must return 0.0."""
        pred = {"S1-1": []}
        truth = {"S1-1": ["S2-00047"]}
        self.assertEqual(per_entity_f05(pred, truth), 0.0)
        self.assertEqual(single_entity_f05([], ["S2-00047"]), 0.0)
        self.assertEqual(single_entity_f05("", "S2-00047"), 0.0)

    def test_perfect_match(self):
        """Exact prediction match must return 1.0."""
        pred = {"S1-1": ["S2-00047", "S3-00812"]}
        truth = {"S1-1": ["S2-00047", "S3-00812"]}
        self.assertEqual(per_entity_f05(pred, truth), 1.0)

    def test_macro_average_multientity(self):
        """Test macro averaging across 5 diverse entities."""
        truth = {
            "S1-1": ["S2-1", "S3-1"],  # Score: 5/7 ~ 0.7142857
            "S1-2": [],                # Singleton correct: 1.0
            "S1-3": [],                # Singleton false positive: 0.0
            "S1-4": ["S2-4"],          # False negative (missed): 0.0
            "S1-5": ["S2-5"],          # Perfect match: 1.0
        }
        pred = {
            "S1-1": ["S2-1", "S3-1", "S2-extra"],
            "S1-2": [],
            "S1-3": ["S2-noise"],
            "S1-4": [],
            "S1-5": ["S2-5"],
        }
        expected = (5.0 / 7.0 + 1.0 + 0.0 + 0.0 + 1.0) / 5.0
        score = per_entity_f05(pred, truth)
        self.assertAlmostEqual(score, expected, places=6)

    def test_missing_entity_in_pred(self):
        """Omitted entity from pred dictionary should count as empty prediction."""
        truth = {
            "S1-1": [],         # Empty truth, missing from pred -> empty pred -> 1.0
            "S1-2": ["S2-1"],   # Non-empty truth, missing from pred -> empty pred -> 0.0
        }
        pred = {}
        self.assertEqual(per_entity_f05(pred, truth), 0.5)


if __name__ == "__main__":
    unittest.main()
