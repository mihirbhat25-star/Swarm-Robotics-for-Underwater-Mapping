"""CPU-only data adapter used by the 3D cloud runtime.

``Boids3D`` remains the owner of simulation and trajectory generation. This
module only invokes it in CPU workers and compacts its output for cloud RAM.
It does not import the GPU training runtime.
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


def _history_ragged_adjacency(neighbors):
    """Pack expert adjacency matrices as row-major ragged COO edges."""
    edge_parts = []
    edge_lengths = np.zeros(len(neighbors), dtype=np.int64)
    for step, adjacency in enumerate(neighbors):
        adjacency = adjacency.tocoo(copy=False)
        edges = np.column_stack((adjacency.row, adjacency.col)).astype(
            np.int32, copy=False
        )
        if len(edges) > 1:
            edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
        edge_parts.append(edges)
        edge_lengths[step] = len(edges)

    edge_offsets = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(edge_lengths, dtype=np.int64),
    ))
    edge_values = (
        np.concatenate(edge_parts, axis=0)
        if edge_parts and edge_offsets[-1] > 0
        else np.zeros((0, 2), dtype=np.int32)
    )
    return edge_values, edge_offsets


def _compact_history(history):
    states = np.concatenate(
        (history["positions"], history["velocities"]), axis=-1
    ).astype(np.float32, copy=False)
    x = np.ascontiguousarray(states[:-1])
    goals = np.asarray(history["goal_positions"][:-1], dtype=np.float32)
    goal_broadcast = np.broadcast_to(
        goals[:, None, :], (len(goals), x.shape[1], 3)
    )
    y = np.ascontiguousarray(
        np.concatenate((x, states[1:], goal_broadcast), axis=-1),
        dtype=np.float32,
    )
    edge_values, edge_offsets = _history_ragged_adjacency(
        history["neighbors"][:-1]
    )
    return {
        "x": x,
        "y": y,
        "edge_values": edge_values,
        "edge_offsets": edge_offsets,
    }


def simulate_compact_task(config):
    """Invoke ``Boids3D`` and compact the result in one deterministic task."""
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
    exclusion_zone = _capsule_exclusion(
        boids.goal_positions, config["goal_exclusion_size"]
    )

    trajectories = []
    centers = []
    for _ in range(int(config["count"])):
        _, _, _, center = boids.get_random_init(
            boids.n_boids,
            save_config=False,
            bounds=OCTANT_BOUNDS_3D[octant],
            exclusion_zone=exclusion_zone,
        )
        history = boids.generate_trajectory(
            random_init=np.asarray(center), save_config=False
        )
        trajectories.append(_compact_history(history))
        centers.append(np.asarray(center, dtype=np.float32))

    return {
        "octant": octant,
        "count": len(trajectories),
        "seed": seed,
        "seconds": time.time() - started,
        "centers": np.stack(centers, axis=0),
        "trajectories": trajectories,
    }
