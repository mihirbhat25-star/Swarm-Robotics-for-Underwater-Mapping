"""Tensor-contract tests for online-goal 3D GNCA training."""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import tensorflow as tf

    from runtime.online_goal_3d import (
        build_online_goal_model_3d,
        build_online_goal_variables_3d,
        make_online_goal_loss_3d,
    )
except ImportError:  # pragma: no cover - lightweight environments omit TF
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is required by the online 3D trainer")
class OnlineGoal3DTests(unittest.TestCase):
    def test_model_maps_nine_conditioned_features_to_six_physical_features(self):
        args = SimpleNamespace()
        model = build_online_goal_model_3d(args, 1e-3)
        build_online_goal_variables_3d(model, 5)
        adjacency = tf.SparseTensor(
            indices=tf.zeros((0, 2), dtype=tf.int64),
            values=tf.zeros((0,), dtype=tf.float32),
            dense_shape=(5, 5),
        )
        output = model(
            [
                tf.zeros((5, 9), dtype=tf.float32),
                adjacency,
                tf.zeros((5,), dtype=tf.int64),
            ],
            training=False,
        )
        self.assertEqual(output.shape, (5, 6))

    def test_transition_aware_loss_is_finite_and_per_node(self):
        args = SimpleNamespace(
            loss_type="newl",
            critical_distance=2.5,
            distance_weight=2.5,
        )
        # Two graphs with two nodes each. Target layout is current state (6),
        # next state (6), active goal (3), previous goal (3), previous-goal
        # maximum distance (1), and graph id (1).
        targets = np.zeros((4, 20), dtype=np.float32)
        targets[:, 12:15] = np.asarray([1.0, 1.0, 1.0])
        targets[:, 18] = -1.0
        targets[:, 19] = np.asarray([0.0, 0.0, 1.0, 1.0])
        predictions = tf.zeros((4, 6), dtype=tf.float32)
        losses = make_online_goal_loss_3d(args)(
            tf.constant(targets), predictions
        )
        self.assertEqual(losses.shape, (4,))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(losses))))


if __name__ == "__main__":
    unittest.main()
