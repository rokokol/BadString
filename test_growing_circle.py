import unittest

import numpy as np

from growing_circle import advance, circle_profile, simulate


class GrowingCircleTests(unittest.TestCase):
    def test_circle_grows_and_has_fixed_ends(self):
        early = circle_profile(0.3)
        late = circle_profile(3.0)
        self.assertGreater(np.max(late), np.max(early))
        self.assertEqual(late[0], 0)
        self.assertEqual(late[-1], 0)

    def test_free_single_mode(self):
        position, velocity = advance(np.array([1.0]), np.array([0.0]), np.array([0.0]), 0.5, 0.0)
        np.testing.assert_allclose(position, [0], atol=1e-12)
        np.testing.assert_allclose(velocity, [-np.pi], atol=1e-12)

    def test_spring_moves_towards_target(self):
        position, velocity = advance(np.array([0.0]), np.array([0.0]), np.array([1.0]), 0.01, 900.0)
        self.assertGreater(position[0], 0)
        self.assertGreater(velocity[0], 0)

    def test_three_scenarios_differ_and_keep_ends(self):
        results = [simulate(mode) for mode in ("reset", "spring", "impulse")]
        for result in results:
            self.assertEqual(result.shape[0], 181)
            np.testing.assert_array_equal(result[:, [0, -1]], 0)
        self.assertFalse(np.allclose(results[0], results[1]))
        self.assertFalse(np.allclose(results[1], results[2]))


if __name__ == "__main__":
    unittest.main()
