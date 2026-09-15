"""Check reported diversity statistics against hand-computed examples."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.diversity import compute_diversity, length_stats, normalize_words


class DiversityTest(unittest.TestCase):
    def test_known_repeated_text(self):
        result = compute_diversity(["A b a b"])
        self.assertEqual(result["num_tokens"], 4)
        self.assertEqual(result["vocab_size"], 2)
        self.assertEqual(result["distinct_1"], 0.5)
        self.assertAlmostEqual(result["distinct_2"], 2 / 3)
        self.assertEqual(result["distinct_3"], 1)
        self.assertEqual(result["yule_k"], 2500)
        self.assertEqual(result["mtld"], 4)

    def test_median_for_even_and_odd_sample_counts(self):
        self.assertEqual(length_stats([["a"], ["b", "c", "d"]])["median_length_words"], 2)
        self.assertEqual(length_stats([["a"], ["b", "c"], ["d"]])["median_length_words"], 1)

    def test_normalization_and_empty_input(self):
        self.assertEqual(normalize_words(" HELLO, __World! ... "), ["hello", "world"])
        result = compute_diversity(["", "..."])
        for key in ("num_tokens", "num_examples_after_strip", "distinct_1", "mtld", "median_length_words"):
            self.assertEqual(result[key], 0)

    def test_ngrams_do_not_cross_examples(self):
        self.assertEqual(compute_diversity(["a", "b"])["total_2grams"], 0)


if __name__ == "__main__":
    unittest.main()
