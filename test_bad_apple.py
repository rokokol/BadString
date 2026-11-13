import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bad_apple import HEIGHT, WIDTH, boundaries, damped_advance, read_video, render_frame, simulate


class BadAppleTests(unittest.TestCase):
    def test_render_frame_draws_black_strings_on_white(self):
        strings = np.zeros((2, WIDTH))
        strings[0] = -20
        strings[1] = 20
        image = np.asarray(render_frame(strings))
        self.assertEqual(image.shape[2], 3)
        self.assertTrue(np.all(image[0, 0] == 255))
        self.assertTrue(np.any(np.all(image == 0, axis=2)))
        black_rows = np.flatnonzero(np.any(np.all(image == 0, axis=2), axis=1))
        self.assertGreater(black_rows[-1] - black_rows[0], 20)

    def test_read_video_does_not_limit_duration(self):
        data = bytes(WIDTH * HEIGHT * 105)
        with patch("bad_apple.subprocess.run", return_value=SimpleNamespace(stdout=data)) as run:
            frames = read_video("video.mp4")
        self.assertEqual(len(frames), 105)
        self.assertNotIn("-t", run.call_args.args[0])

    def test_uniform_columns_follow_background_or_fill_screen(self):
        mask = np.array([[0, 1, 0, 1, 0],
                         [0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0]], dtype=bool)
        upper, lower = boundaries(mask)
        np.testing.assert_array_equal(upper[[0, 4]], 0)
        np.testing.assert_array_equal(lower[[0, 4]], 0)
        self.assertEqual(upper[1], -1.5)
        self.assertEqual(lower[1], 1.5)
        self.assertLess(upper[2], lower[2])
        self.assertLess(upper[3], lower[3])

    def test_clipped_silhouette_keeps_boundary_on_screen_edge_for_both_colors(self):
        mask = np.zeros((5, 9), dtype=bool)
        mask[:3, 2] = True
        mask[2:, 4] = True
        mask[1:4, 6] = True
        mask[:, 7] = True
        for image in (mask, ~mask):
            with self.subTest(inverted=bool(image[0, 0])):
                upper, lower = boundaries(image)
                self.assertEqual(upper[2], -2)
                self.assertEqual(lower[2], 0.5)
                self.assertEqual(upper[4], -0.5)
                self.assertEqual(lower[4], 2)
                self.assertEqual(upper[6], -1.5)
                self.assertEqual(lower[6], 1.5)
                self.assertEqual(upper[7], -2)
                self.assertEqual(lower[7], 2)
                self.assertEqual(upper[3], 0)
                self.assertEqual(lower[3], 0)

    def test_damping_reduces_free_mode_energy(self):
        position = np.array([1.0])
        velocity = np.array([0.0])
        target = np.array([0.0])
        position, velocity = damped_advance(position, velocity, target, 1.0, 0.0, 2.0)
        energy = velocity[0] ** 2 + (np.pi * position[0]) ** 2
        self.assertLess(energy, np.pi ** 2)
        self.assertGreater(energy, 0)

    def test_overdamped_mode_stays_finite(self):
        position, velocity = damped_advance(np.array([0.0]), np.array([0.0]), np.array([1.0]), 1.0, 1.0, 100.0)
        self.assertTrue(np.isfinite(position).all() and np.isfinite(velocity).all())
        self.assertGreater(position[0], 0)

    def test_spring_pulls_between_target_updates(self):
        frames = np.zeros((3, 5, 7), dtype=bool)
        frames[1:, 1:4, 2:5] = True
        spring = simulate(frames, fps=20, update_rate=10, mode="spring")
        impulse = simulate(frames, fps=20, update_rate=10, mode="impulse")
        np.testing.assert_allclose(spring[2], 0, atol=1e-12)
        self.assertGreater(np.max(np.abs(spring[3])), 0)
        self.assertFalse(np.allclose(spring, impulse))
        np.testing.assert_array_equal(spring[:, :, [0, -1]], 0)

    def test_impulses_change_velocity_and_keep_fixed_ends(self):
        frames = np.zeros((3, 5, 7), dtype=bool)
        frames[1:, 1:4, 2:5] = True
        strings = simulate(frames, fps=20, update_rate=10)
        self.assertEqual(strings.shape, (7, 2, 7))
        np.testing.assert_array_equal(strings[:, :, [0, -1]], 0)
        np.testing.assert_allclose(strings[2], 0, atol=1e-12)
        self.assertGreater(np.max(np.abs(strings[3])), 0)


if __name__ == "__main__":
    unittest.main()
