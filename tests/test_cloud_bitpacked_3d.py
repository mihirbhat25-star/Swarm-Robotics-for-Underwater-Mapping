"""Contract tests for the lazy bit-packed 3D cloud adapter."""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import tensorflow as tf

    from runtime.cloud_bitpacked_3d import (
        _flatten_sources,
        _make_model_batch,
        _selected_steps,
    )
except ImportError:  # pragma: no cover - local lightweight environments
    tf = None


def _fixture():
    n_boids = 2
    steps = 3
    states = np.arange(
        (steps + 1) * n_boids * 6, dtype=np.float32
    ).reshape(steps + 1, n_boids, 6)
    states[..., :3] *= 0.01
    active_goals = np.asarray(
        [[4.0, 4.0, 4.0], [0.1, 0.1, 0.1], [-4.0, -4.0, -4.0]],
        dtype=np.float32,
    )
    previous_goals = np.asarray(
        [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
        dtype=np.float32,
    )
    max_previous_distances = np.asarray([-1.0, 0.2, 0.5], dtype=np.float32)
    masks = np.asarray(
        [
            [[False, True], [True, False]],
            [[False, False], [True, False]],
            [[False, True], [False, False]],
        ]
    )
    bits = np.packbits(
        masks.reshape(steps, -1), axis=-1, bitorder="little"
    )
    values = {
        "states": states,
        "active_goals": active_goals,
        "previous_goals": previous_goals,
        "max_previous_goal_distances": max_previous_distances,
        "adjacency_bits": bits,
    }
    return n_boids, masks, values


@unittest.skipIf(tf is None, "TensorFlow is required by the cloud adapter")
class CloudBitpacked3DTests(unittest.TestCase):
    def test_near_goal_selection_uses_transition_active_goal(self):
        _, _, values = _fixture()
        states = values["states"].copy()
        states[:, :, :3] = 0.0
        selected = _selected_steps(
            states,
            values["active_goals"],
            timestep_stride=3,
            near_goal_radius=0.2,
        )
        np.testing.assert_array_equal(selected, np.asarray([0, 1]))

    def test_accepts_legacy_mapping_in_fixed_mode(self):
        n_boids, masks, values = _fixture()
        legacy = {
            "states": values["states"],
            "goals": values["active_goals"],
            "adjacency_bits": values["adjacency_bits"],
        }
        flattened = _flatten_sources(
            [legacy], n_boids, 1, 0.0, goal_conditioned=False
        )
        self.assertEqual(flattened[0].shape, (4, n_boids, 6))
        self.assertEqual(flattened[2].shape, (3, 3))
        self.assertIsNone(flattened[3])
        self.assertIsNone(flattened[4])

        inputs, targets = _make_model_batch(
            tf.constant([0, 1], dtype=tf.int32),
            tf.constant(flattened[0]),
            tf.constant(flattened[1]),
            tf.constant(flattened[2]),
            None,
            None,
            tf.constant(flattened[5]),
            batch_size=2,
            n_boids=n_boids,
            goal_conditioned=False,
        )
        x, adjacency, graph_ids = inputs
        self.assertEqual(x.shape, (4, 6))
        self.assertEqual(targets.shape, (4, 16))
        np.testing.assert_array_equal(graph_ids.numpy(), [0, 0, 1, 1])
        expected_adjacency = np.zeros((4, 4), dtype=np.float32)
        expected_adjacency[:2, :2] = masks[0]
        expected_adjacency[2:, 2:] = masks[1]
        np.testing.assert_array_equal(
            tf.sparse.to_dense(adjacency).numpy(), expected_adjacency
        )

    def test_goal_conditioned_object_has_exact_model_contract(self):
        n_boids, masks, values = _fixture()
        trajectory = SimpleNamespace(**values)
        (
            states,
            state_ids,
            active_goals,
            previous_goals,
            max_previous_distances,
            adjacency_bits,
        ) = _flatten_sources(
            [trajectory], n_boids, 1, 0.0, goal_conditioned=True
        )
        inputs, targets = _make_model_batch(
            tf.constant([0, 1], dtype=tf.int32),
            tf.constant(states),
            tf.constant(state_ids),
            tf.constant(active_goals),
            tf.constant(previous_goals),
            tf.constant(max_previous_distances),
            tf.constant(adjacency_bits),
            batch_size=2,
            n_boids=n_boids,
            goal_conditioned=True,
        )
        x, adjacency, graph_ids = inputs
        self.assertEqual(x.shape, (4, 9))
        self.assertEqual(targets.shape, (4, 20))
        np.testing.assert_array_equal(graph_ids.numpy(), [0, 0, 1, 1])

        expected_x = np.concatenate(
            (values["states"][0, 0], values["active_goals"][0]
             - values["states"][0, 0, :3])
        )
        np.testing.assert_allclose(x.numpy()[0], expected_x)
        expected_y = np.concatenate((
            values["states"][0, 0],
            values["states"][1, 0],
            values["active_goals"][0],
            values["previous_goals"][0],
            values["max_previous_goal_distances"][:1],
            np.asarray([0.0], dtype=np.float32),
        ))
        np.testing.assert_allclose(targets.numpy()[0], expected_y)

        expected_adjacency = np.zeros((4, 4), dtype=np.float32)
        expected_adjacency[:2, :2] = masks[0]
        expected_adjacency[2:, 2:] = masks[1]
        np.testing.assert_array_equal(
            tf.sparse.to_dense(adjacency).numpy(), expected_adjacency
        )


if __name__ == "__main__":
    unittest.main()
