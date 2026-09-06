"""Dimension and compatibility checks for shared waypoint utilities."""

import unittest

import numpy as np

from modules.waypoints import (
    OnlineWaypointManager,
    goal_conditioned_state,
    sample_separated_waypoint,
)


class WaypointUtilitiesTest(unittest.TestCase):
    def test_two_dimensional_sampling_remains_deterministic(self):
        waypoint = sample_separated_waypoint(
            np.random.default_rng(123),
            np.array([0.25, -0.75], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            waypoint,
            np.array([1.6411668, -4.0156107], dtype=np.float32),
        )

    def test_three_dimensional_sampling_respects_bounds_and_distance(self):
        reference = np.zeros(3, dtype=np.float32)
        waypoint = sample_separated_waypoint(
            np.random.default_rng(7),
            reference,
            bounds=(-4.0, 4.0, -3.0, 3.0, -2.0, 2.0),
            min_distance=1.0,
        )
        self.assertEqual(waypoint.shape, (3,))
        self.assertGreaterEqual(np.linalg.norm(waypoint - reference), 1.0)
        self.assertTrue(np.all(waypoint >= [-4.0, -3.0, -2.0]))
        self.assertTrue(np.all(waypoint <= [4.0, 3.0, 2.0]))

    def test_goal_conditioning_supports_two_and_three_dimensions(self):
        state_2d = np.array([[1.0, 2.0, 0.1, 0.2]], dtype=np.float32)
        conditioned_2d = goal_conditioned_state(state_2d, [3.0, 4.0])
        np.testing.assert_array_equal(
            conditioned_2d,
            np.array([[1.0, 2.0, 0.1, 0.2, 2.0, 2.0]], dtype=np.float32),
        )

        state_3d = np.array(
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32
        )
        conditioned_3d = goal_conditioned_state(state_3d, [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(
            conditioned_3d,
            np.array(
                [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 3.0, 3.0, 3.0]],
                dtype=np.float32,
            ),
        )

    def test_three_dimensional_manager_switches_waypoints(self):
        manager = OnlineWaypointManager(
            np.random.default_rng(8),
            n_waypoints=2,
            bounds=(-4.0, 4.0, -4.0, 4.0, -4.0, 4.0),
            min_distance=1.0,
            arrival_radius=0.5,
        )
        first = manager.start(np.zeros(3, dtype=np.float32))
        positions = np.repeat(first[None, :], 5, axis=0)
        second, switched, finished = manager.update(positions)

        self.assertTrue(switched)
        self.assertFalse(finished)
        self.assertEqual(second.shape, (3,))
        self.assertGreaterEqual(np.linalg.norm(second - first), 1.0)


if __name__ == "__main__":
    unittest.main()
