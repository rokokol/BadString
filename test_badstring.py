import unittest

import numpy as np

from badstring import SPEEDS, contours, evolve


class StringTests(unittest.TestCase):
    def test_circle_contours_and_empty_columns(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[2:7, 3:6] = True
        top, bottom = contours(mask)
        self.assertEqual(top[4], -2)
        self.assertEqual(bottom[4], 2)
        self.assertEqual(top[0], 0)
        self.assertEqual(bottom[-1], 0)

    def test_comparison_has_distinct_speeds(self):
        self.assertGreaterEqual(len(set(SPEEDS)), 3)
        x = np.linspace(0, 1, 101)
        initial = np.sin(np.pi * x)
        for speed in SPEEDS:
            result = evolve(initial, 0.25, speed)
            np.testing.assert_allclose(result, initial * np.cos(np.pi * speed * 0.25), atol=1e-12)

    def test_single_mode_and_fixed_ends(self):
        x = np.linspace(0, 1, 101)
        initial = np.sin(np.pi * x)
        result = evolve(initial, 0.5, 1.0)
        np.testing.assert_allclose(result, 0, atol=1e-12)
        np.testing.assert_allclose(evolve(initial, 1.0, 1.0), -initial, atol=1e-12)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 0)


if __name__ == "__main__":
    unittest.main()
