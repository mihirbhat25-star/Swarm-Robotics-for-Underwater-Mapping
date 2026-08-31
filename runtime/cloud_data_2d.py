"""CPU-only expert generation for cacheless 2D online-goal training."""

import time

import numpy as np

from modules.boids import Boids, ensure_goal_transition_metadata
from modules.waypoints import goal_conditioned_state


def _ragged_adjacency(neighbors):
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
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(edge_lengths, dtype=np.int64))
    )
    values = (
        np.concatenate(edge_parts, axis=0)
        if edge_parts and offsets[-1] > 0
        else np.zeros((0, 2), dtype=np.int32)
    )
    return values, offsets


def compact_online_history(history):
    """Compact one two-waypoint episode without constructing Graph objects."""
    physical = np.concatenate(
        (history["positions"], history["velocities"]), axis=-1
    ).astype(np.float32, copy=False)
    goals = np.asarray(history["goal_positions"], dtype=np.float32)
    x = np.stack(
        [goal_conditioned_state(state, goal) for state, goal in zip(physical[:-1], goals[:-1])]
    )
    goal_features = np.broadcast_to(
        goals[:-1, None, :], (len(goals) - 1, physical.shape[1], 2)
    )
    y = np.concatenate(
        (physical[:-1], physical[1:], goal_features), axis=-1
    ).astype(np.float32, copy=False)
    y = ensure_goal_transition_metadata(x, y)
    edge_values, edge_offsets = _ragged_adjacency(history["neighbors"][:-1])
    return {
        "x": np.ascontiguousarray(x, dtype=np.float32),
        "y": np.ascontiguousarray(y, dtype=np.float32),
        "edge_values": edge_values,
        "edge_offsets": edge_offsets,
        "waypoints": np.asarray(history["waypoints"], dtype=np.float32),
    }


def simulate_online_goal_task(config):
    """Generate one deterministic two-waypoint expert trajectory."""
    started = time.time()
    seed = int(config["seed"]) % (2**32 - 1)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    boids = Boids(
        n_boids=int(config["n_boids"]),
        perception=float(config["perception"]),
        pos_noise=float(config["pos_noise"]),
        vel_noise=float(config["vel_noise"]),
        waypoint_order_policy="fixed_waypoint_order",
    )
    history = boids.generate_online_goal_trajectory(
        save_config=False,
        random_init=True,
        rng=rng,
        n_waypoints=2,
        goal_bounds=tuple(config["goal_bounds"]),
        start_bounds=tuple(config["start_bounds"]),
        goal_min_distance=float(config["goal_min_distance"]),
        goal_arrival_radius=float(config["goal_arrival_radius"]),
        max_steps=int(config["expert_max_steps"]),
    )
    center = history["positions"][0].mean(axis=0).astype(np.float32)
    return {
        "seed": seed,
        "start_quadrant": int(config.get("start_quadrant", -1)),
        "seconds": time.time() - started,
        "center": center,
        "trajectory": compact_online_history(history),
    }
