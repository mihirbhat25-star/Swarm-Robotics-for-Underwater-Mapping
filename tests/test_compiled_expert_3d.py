"""Parity and metadata tests for the cloud-only compiled 3D expert."""

import unittest

import numpy as np

try:
    import numba  # noqa: F401
except ImportError:  # pragma: no cover - local legacy environments may omit it
    numba = None


@unittest.skipIf(numba is None, "Numba is required by the compiled cloud expert")
class CompiledExpert3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from runtime.compiled_expert_3d import (
            FIXED_WAYPOINT_ORDER,
            rollout_waypoints,
            sample_fixed_initial_state,
        )

        cls.fixed_policy = FIXED_WAYPOINT_ORDER
        cls.rollout = staticmethod(rollout_waypoints)
        cls.sample_fixed = staticmethod(sample_fixed_initial_state)

    def test_fixed_rollout_matches_legacy_boids(self):
        """Seeded fixed-mode states, goal labels, and graphs retain parity."""
        from modules.boids_3d import Boids3D
        from runtime.compiled_expert_3d import DEFAULT_FIXED_GOALS, unpack_adjacency

        center, positions, velocities = self.sample_fixed(
            17,
            octant=7,
            n_boids=100,
            goals=DEFAULT_FIXED_GOALS,
            legacy_discarded_initialization=True,
        )
        legacy = Boids3D(
            n_boids=100,
            perception=0.1,
            crowding=0.02,
            pos_noise=0.0,
            vel_noise=0.0,
            waypoint_order_policy=self.fixed_policy,
        ).generate_trajectory(
            save_config=False,
            random_init=(positions.copy(), velocities.copy()),
        )
        compiled = self.rollout(
            positions,
            velocities,
            DEFAULT_FIXED_GOALS,
            sampled_center=center,
            arrival_radius=0.25,
            max_steps=10_000,
        )

        legacy_states = np.concatenate(
            (legacy["positions"], legacy["velocities"]), axis=-1
        ).astype(np.float32)
        np.testing.assert_allclose(compiled.states, legacy_states, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(
            compiled.active_goals,
            np.asarray(legacy["goal_positions"][:-1], dtype=np.float32),
        )

        decoded = unpack_adjacency(compiled.adjacency_bits, compiled.n_boids)
        self.assertEqual(decoded.shape[0], len(legacy["neighbors"]) - 1)
        # Check all rows while keeping the assertion failure reasonably small.
        for step, adjacency in enumerate(legacy["neighbors"][:-1]):
            np.testing.assert_array_equal(decoded[step], adjacency.toarray().astype(bool))

    def test_online_goal_metadata_is_transition_aligned(self):
        from runtime.compiled_expert_3d import unpack_adjacency

        rng = np.random.RandomState(3)
        positions = np.asarray([-1.2, -1.2, -1.2])[None, :] + 0.1 * rng.rand(24, 3)
        velocities = np.zeros((24, 3), dtype=np.float64)
        velocities[:, 0] = 0.01
        waypoints = np.asarray(
            [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32
        )
        trajectory = self.rollout(
            positions,
            velocities,
            waypoints,
            arrival_radius=0.25,
            max_steps=5_000,
            perception=0.15,
        )

        segment_ids = trajectory.goal_segment_ids
        self.assertTrue(np.any(segment_ids == 0))
        self.assertTrue(np.any(segment_ids == 1))
        switch = int(np.flatnonzero(segment_ids == 1)[0])
        self.assertTrue(np.all(segment_ids[:switch] == 0))
        self.assertTrue(np.all(segment_ids[switch:] == 1))
        np.testing.assert_array_equal(
            trajectory.active_goals[:switch],
            np.broadcast_to(waypoints[0], (switch, 3)),
        )
        np.testing.assert_array_equal(
            trajectory.active_goals[switch:],
            np.broadcast_to(waypoints[1], (trajectory.steps - switch, 3)),
        )
        self.assertTrue(
            np.all(trajectory.max_previous_goal_distances[:switch] == -1.0)
        )
        previous_distances = trajectory.max_previous_goal_distances[switch:]
        self.assertTrue(np.all(previous_distances >= 0.0))
        self.assertTrue(np.all(np.diff(previous_distances) >= -1e-7))

        # The packed graph must remain the graph of x_t, not x_(t-1).
        decoded = unpack_adjacency(trajectory.adjacency_bits, trajectory.n_boids)
        for step in (0, switch, trajectory.steps - 1):
            current_positions = trajectory.states[step, :, :3]
            squared = np.sum(
                (
                    current_positions[:, None, :]
                    - current_positions[None, :, :]
                )
                ** 2,
                axis=-1,
            )
            expected = squared < 0.15**2
            np.fill_diagonal(expected, False)
            np.testing.assert_array_equal(decoded[step], expected)

    def test_random_online_waypoints_obey_bounds_and_separation(self):
        from runtime.compiled_expert_3d import sample_online_initial_state_and_goals

        _, positions, _, goals = sample_online_initial_state_and_goals(
            29,
            n_boids=20,
            n_waypoints=2,
            start_bounds=(-5, 5, -5, 5, -5, 5),
            goal_bounds=(-4.5, 4.5, -4.5, 4.5, -4.5, 4.5),
            goal_min_distance=2.5,
        )
        self.assertTrue(np.all(goals >= -4.5))
        self.assertTrue(np.all(goals <= 4.5))
        self.assertGreaterEqual(np.linalg.norm(goals[0] - positions.mean(axis=0)), 2.5)
        self.assertGreaterEqual(np.linalg.norm(goals[1] - goals[0]), 2.5)

    def test_public_warmup_compiles_production_signature(self):
        from runtime.compiled_expert_3d import warm_up_compiled_expert_3d

        elapsed = warm_up_compiled_expert_3d()
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
