"""CPU-only helpers for cloud in-memory 3D expert generation.

The public training entry point remains ``boids.run_boids_3d``.  Functions in
this module are kept importable so Joblib's process workers can generate
compact trajectories without executing the training CLI.
"""

import time

import numpy as np

from modules.boids_3d import Boids3D


OCTANT_BOUNDS_3D = (
    (0.0, 5.0, 0.0, 5.0, 0.0, 5.0),
    (-5.0, 0.0, 0.0, 5.0, 0.0, 5.0),
    (-5.0, 0.0, -5.0, 0.0, 0.0, 5.0),
    (0.0, 5.0, -5.0, 0.0, 0.0, 5.0),
    (0.0, 5.0, 0.0, 5.0, -5.0, 0.0),
    (-5.0, 0.0, 0.0, 5.0, -5.0, 0.0),
    (-5.0, 0.0, -5.0, 0.0, -5.0, 0.0),
    (0.0, 5.0, -5.0, 0.0, -5.0, 0.0),
)


def _capsule_exclusion(goal_positions, radius):
    return None if radius <= 0 else (np.asarray(goal_positions), float(radius))


def canonical_ragged_adjacency_3d(states, perception, block_steps=64):
    """Build row-major ragged COO edges from the exact float32 model states."""
    positions = np.asarray(states[..., :3], dtype=np.float32)
    n_steps, n_boids = positions.shape[:2]
    edge_parts = []
    edge_lengths = np.zeros(n_steps, dtype=np.int64)
    diagonal = np.eye(n_boids, dtype=bool)[None, :, :]
    threshold = np.float32(perception)

    for start in range(0, n_steps, block_steps):
        stop = min(start + block_steps, n_steps)
        block = positions[start:stop]
        distances = np.linalg.norm(
            block[:, :, None, :] - block[:, None, :, :], axis=-1
        )
        neighbor_mask = (distances < threshold) & (~diagonal)
        for local_idx in range(stop - start):
            row, col = np.nonzero(neighbor_mask[local_idx])
            edges = np.column_stack((row, col)).astype(np.int32, copy=False)
            edge_parts.append(edges)
            edge_lengths[start + local_idx] = len(edges)

    edge_offsets = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(edge_lengths, dtype=np.int64),
    ))
    if edge_parts and edge_offsets[-1] > 0:
        edge_values = np.concatenate(edge_parts, axis=0)
    else:
        edge_values = np.zeros((0, 2), dtype=np.int32)
    return edge_values, edge_offsets


def history_ragged_adjacency_3d(neighbors):
    """Pack the exact adjacency matrices already produced by the expert."""
    edge_parts = []
    edge_lengths = np.zeros(len(neighbors), dtype=np.int64)
    for step, adjacency in enumerate(neighbors):
        adjacency = adjacency.tocoo(copy=False)
        edges = np.column_stack((adjacency.row, adjacency.col)).astype(
            np.int32, copy=False
        )
        if len(edges) > 1:
            order = np.lexsort((edges[:, 1], edges[:, 0]))
            edges = edges[order]
        edge_parts.append(edges)
        edge_lengths[step] = len(edges)

    edge_offsets = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(edge_lengths, dtype=np.int64),
    ))
    if edge_parts and edge_offsets[-1] > 0:
        edge_values = np.concatenate(edge_parts, axis=0)
    else:
        edge_values = np.zeros((0, 2), dtype=np.int32)
    return edge_values, edge_offsets


def _compact_history_3d(history, _perception):
    states = np.concatenate(
        (history["positions"], history["velocities"]), axis=-1
    ).astype(np.float32, copy=False)
    x = np.ascontiguousarray(states[:-1])
    next_state = states[1:]
    goals = np.asarray(history["goal_positions"][:-1], dtype=np.float32)
    goal_broadcast = np.broadcast_to(
        goals[:, None, :], (len(goals), x.shape[1], 3)
    )
    y = np.ascontiguousarray(
        np.concatenate((x, next_state, goal_broadcast), axis=-1),
        dtype=np.float32,
    )
    # Boids3D computes neighbors from the current positions before every
    # update. Reusing those exact matrices avoids repeating an O(T*N^2)
    # distance calculation while preserving the training graph byte-for-byte.
    edge_values, edge_offsets = history_ragged_adjacency_3d(
        history["neighbors"][:-1]
    )
    return {
        "x": x,
        "y": y,
        "edge_values": edge_values,
        "edge_offsets": edge_offsets,
    }


def generate_compact_task_3d(config):
    """Generate a deterministic group of trajectories for one CPU process."""
    started = time.time()
    seed = int(config["seed"]) % (2**32 - 1)
    np.random.seed(seed)

    boids = Boids3D(
        n_boids=int(config["n_boids"]),
        perception=float(config["perception"]),
        pos_noise=float(config["pos_noise"]),
        vel_noise=float(config["vel_noise"]),
        waypoint_order_policy=config["waypoint_order_policy"],
    )
    octant = int(config["octant"])
    bounds = OCTANT_BOUNDS_3D[octant]
    exclusion_zone = _capsule_exclusion(
        boids.goal_positions, config["goal_exclusion_size"]
    )

    trajectories = []
    centers = []
    for _ in range(int(config["count"])):
        _, _, _, center = boids.get_random_init(
            boids.n_boids,
            save_config=False,
            bounds=bounds,
            exclusion_zone=exclusion_zone,
        )
        history = boids.generate_trajectory(
            random_init=np.asarray(center), save_config=False
        )
        trajectories.append(_compact_history_3d(history, boids.perception))
        centers.append(np.asarray(center, dtype=np.float32))

    return {
        "octant": octant,
        "count": len(trajectories),
        "seed": seed,
        "seconds": time.time() - started,
        "centers": np.stack(centers, axis=0),
        "trajectories": trajectories,
    }
