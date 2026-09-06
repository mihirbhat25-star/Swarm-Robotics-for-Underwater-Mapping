"""Import-light, compiled expert rollouts for cloud 3D Boids training.

This module deliberately depends only on NumPy and Numba.  In particular, it
does not import TensorFlow, Spektral, SciPy, h5py, or Matplotlib, so many CPU
workers can use it without each worker initializing the ML stack.

The rollout kernel preserves the equations implemented by
``modules.boids_3d.Boids3D`` while avoiding a SciPy sparse matrix per timestep:

* the radius graph is computed once for each state and carried to the next
  update;
* the same graph and degree counts are reused by every force component; and
* row-major adjacency is retained as an eightfold bit-packed mask.

The returned goal and transition metadata are aligned with state ``x[t]`` and
the transition ``x[t] -> x[t + 1]``.  This is suitable for both the existing
fixed-waypoint task and a future two-online-waypoint, goal-conditioned task.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Optional, Sequence

import numpy as np
from numba import njit


FIXED_WAYPOINT_ORDER = "fixed_waypoint_order"
NEAREST_CCW_ORDER = "nearest_then_counterclockwise_xy"

DEFAULT_FIXED_GOALS = np.asarray(
    [[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]], dtype=np.float32
)

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


@dataclass(frozen=True)
class CompiledTrajectory3D:
    """Compact transition-aligned output of one compiled rollout.

    ``states`` has one more row than the other temporal arrays.  For transition
    ``t``, use ``states[t]``, ``states[t + 1]``, ``active_goals[t]``, and
    ``adjacency_bits[t]``.
    """

    states: np.ndarray
    active_goals: np.ndarray
    previous_goals: np.ndarray
    max_previous_goal_distances: np.ndarray
    goal_segment_ids: np.ndarray
    adjacency_bits: np.ndarray
    waypoints: np.ndarray
    sampled_center: np.ndarray

    @property
    def steps(self) -> int:
        return int(self.active_goals.shape[0])

    @property
    def n_boids(self) -> int:
        return int(self.states.shape[1])


def _validate_bounds(bounds: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"{name} must have six values, got shape {values.shape}.")
    if not (
        values[0] < values[1]
        and values[2] < values[3]
        and values[4] < values[5]
    ):
        raise ValueError(f"{name} must contain three increasing min/max pairs.")
    return values


def _validate_goals(goals: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(goals, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 1:
        raise ValueError(f"goals must have shape (G, 3), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("goals must contain only finite values.")
    return values


def order_fixed_waypoints(
    goals: Sequence[Sequence[float]],
    initial_positions: np.ndarray,
    policy: str,
) -> np.ndarray:
    """Match ``Boids3D`` fixed or nearest-then-counterclockwise ordering."""
    goals = _validate_goals(goals)
    initial_positions = np.asarray(initial_positions, dtype=np.float64)
    if initial_positions.ndim != 2 or initial_positions.shape[1] != 3:
        raise ValueError("initial_positions must have shape (N, 3).")
    if policy == FIXED_WAYPOINT_ORDER:
        return goals.copy()
    if policy != NEAREST_CCW_ORDER:
        raise ValueError(
            f"policy must be '{FIXED_WAYPOINT_ORDER}' or '{NEAREST_CCW_ORDER}'."
        )

    start = initial_positions.mean(axis=0)
    nearest = int(np.argmin(np.linalg.norm(goals - start[None, :], axis=1)))
    center_xy = goals[:, :2].mean(axis=0)
    relative_xy = goals[:, :2] - center_xy[None, :]
    angles = np.mod(np.arctan2(relative_xy[:, 1], relative_xy[:, 0]), 2 * np.pi)
    ccw_order = np.argsort(angles, kind="stable")
    nearest_position = int(np.flatnonzero(ccw_order == nearest)[0])
    return goals[np.roll(ccw_order, -nearest_position)].copy()


def _inside_cyclic_capsule(center: np.ndarray, goals: np.ndarray, radius: float) -> bool:
    if radius <= 0:
        return False
    for idx in range(len(goals)):
        start = goals[idx].astype(np.float64, copy=False)
        end = goals[(idx + 1) % len(goals)].astype(np.float64, copy=False)
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared < 1e-10:
            if np.linalg.norm(center - start) < radius:
                return True
            continue
        fraction = np.clip(
            np.dot(center - start, segment) / length_squared, 0.0, 1.0
        )
        if np.linalg.norm(center - (start + fraction * segment)) < radius:
            return True
    return False


def sample_fixed_initial_state(
    seed: int,
    *,
    octant: int,
    n_boids: int = 100,
    goals: Sequence[Sequence[float]] = DEFAULT_FIXED_GOALS,
    exclusion_radius: float = 0.5,
    init_scatter: float = 0.325,
    max_speed: float = 0.01,
    legacy_discarded_initialization: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the legacy cloud fixed-task center and initial flock.

    The old cloud adapter calls ``get_random_init`` only to obtain its center,
    discards the flock sampled by that call, and samples another flock at the
    same center in ``generate_trajectory``.  The compatibility flag consumes
    that discarded draw so seeded parity tests can reproduce the old path.
    """
    if octant < 0 or octant >= len(OCTANT_BOUNDS_3D):
        raise ValueError(f"octant must be in [0, 7], got {octant}.")
    if n_boids < 1:
        raise ValueError("n_boids must be positive.")
    goals_array = _validate_goals(goals)
    bounds = np.asarray(OCTANT_BOUNDS_3D[octant], dtype=np.float64)
    rng = np.random.RandomState(int(seed) % (2**32 - 1))

    for _ in range(10_000):
        center = np.asarray(
            [
                rng.uniform(bounds[0], bounds[1]),
                rng.uniform(bounds[2], bounds[3]),
                rng.uniform(bounds[4], bounds[5]),
            ],
            dtype=np.float64,
        )
        if not _inside_cyclic_capsule(center, goals_array, exclusion_radius):
            break
    else:
        raise RuntimeError("Could not sample a center outside the exclusion capsule.")

    if legacy_discarded_initialization:
        rng.rand(n_boids, 3)
    positions = center + float(init_scatter) * rng.rand(n_boids, 3)
    velocities = np.zeros((n_boids, 3), dtype=np.float64)
    velocities[:, 0] = float(max_speed)
    return center, positions, velocities


def sample_online_initial_state_and_goals(
    seed: int,
    *,
    n_boids: int = 100,
    n_waypoints: int = 2,
    start_bounds: Sequence[float] = (-5.0, 5.0, -5.0, 5.0, -5.0, 5.0),
    goal_bounds: Sequence[float] = (-4.5, 4.5, -4.5, 4.5, -4.5, 4.5),
    goal_min_distance: float = 2.5,
    init_scatter: float = 0.325,
    max_speed: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a 3D clump and sequential externally supplied random waypoints.

    This mirrors the 2D online experiment: the initial goal is separated from
    the initial flock centroid; each later goal is separated from the goal it
    replaces.  The initial-state RNG and waypoint RNG are independent streams
    initialized with the same seed, matching ``runtime.cloud_data_2d``.
    """
    if n_boids < 1 or n_waypoints < 1:
        raise ValueError("n_boids and n_waypoints must be positive.")
    if goal_min_distance < 0:
        raise ValueError("goal_min_distance cannot be negative.")
    start = _validate_bounds(start_bounds, "start_bounds")
    goal_box = _validate_bounds(goal_bounds, "goal_bounds")
    seed = int(seed) % (2**32 - 1)

    state_rng = np.random.RandomState(seed)
    center = np.asarray(
        [
            state_rng.uniform(start[0], start[1]),
            state_rng.uniform(start[2], start[3]),
            state_rng.uniform(start[4], start[5]),
        ],
        dtype=np.float64,
    )
    positions = center + float(init_scatter) * state_rng.rand(n_boids, 3)
    velocities = np.zeros((n_boids, 3), dtype=np.float64)
    velocities[:, 0] = float(max_speed)

    waypoint_rng = np.random.default_rng(seed)
    waypoints = np.empty((n_waypoints, 3), dtype=np.float32)
    reference = positions.mean(axis=0)
    for waypoint_idx in range(n_waypoints):
        for _ in range(10_000):
            candidate = np.asarray(
                [
                    waypoint_rng.uniform(goal_box[0], goal_box[1]),
                    waypoint_rng.uniform(goal_box[2], goal_box[3]),
                    waypoint_rng.uniform(goal_box[4], goal_box[5]),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(candidate - reference) >= goal_min_distance:
                waypoints[waypoint_idx] = candidate
                reference = candidate
                break
        else:
            raise RuntimeError(
                "Could not sample separated online waypoint; enlarge goal bounds "
                "or reduce goal_min_distance."
            )
    return center, positions, velocities, waypoints


@njit(cache=True, nogil=True)
def _fill_graph_and_sums(
    positions,
    velocities,
    perception_squared,
    crowding_squared,
    neighbors,
    degrees,
    position_sums,
    velocity_sums,
    separation,
):
    n_boids = positions.shape[0]
    for idx in range(n_boids):
        degrees[idx] = 0
        for dim in range(3):
            position_sums[idx, dim] = 0.0
            velocity_sums[idx, dim] = 0.0
            separation[idx, dim] = 0.0
        for other in range(n_boids):
            neighbors[idx, other] = 0

    # The graph is undirected, so evaluate each pair once and update both rows.
    for left in range(n_boids):
        for right in range(left + 1, n_boids):
            dx = positions[left, 0] - positions[right, 0]
            dy = positions[left, 1] - positions[right, 1]
            dz = positions[left, 2] - positions[right, 2]
            distance_squared = dx * dx + dy * dy + dz * dz
            if distance_squared < perception_squared:
                neighbors[left, right] = 1
                neighbors[right, left] = 1
                degrees[left] += 1
                degrees[right] += 1
                for dim in range(3):
                    position_sums[left, dim] += positions[right, dim]
                    position_sums[right, dim] += positions[left, dim]
                    velocity_sums[left, dim] += velocities[right, dim]
                    velocity_sums[right, dim] += velocities[left, dim]
                if distance_squared < crowding_squared:
                    separation[left, 0] += dx
                    separation[left, 1] += dy
                    separation[left, 2] += dz
                    separation[right, 0] -= dx
                    separation[right, 1] -= dy
                    separation[right, 2] -= dz


@njit(cache=True, nogil=True)
def _clamp_vector_in_place(vector, maximum):
    norm = np.sqrt(
        vector[0] * vector[0]
        + vector[1] * vector[1]
        + vector[2] * vector[2]
    )
    if norm > maximum:
        factor = maximum / norm
        vector[0] *= factor
        vector[1] *= factor
        vector[2] *= factor


@njit(cache=True, nogil=True)
def _store_packed_adjacency(neighbors, destination):
    for byte_idx in range(destination.shape[0]):
        destination[byte_idx] = 0
    n_boids = neighbors.shape[0]
    for row in range(n_boids):
        base = row * n_boids
        for col in range(n_boids):
            if neighbors[row, col] != 0:
                flat_idx = base + col
                destination[flat_idx >> 3] |= np.uint8(1 << (flat_idx & 7))


@njit(cache=True, nogil=True)
def _rollout_kernel_3d(
    initial_positions,
    initial_velocities,
    waypoints,
    arrival_radius,
    max_steps,
    min_speed,
    max_speed,
    max_force,
    max_turn_degrees,
    perception,
    crowding,
    dt,
    borders,
    pos_noise,
    vel_noise,
    noise_seed,
):
    n_boids = initial_positions.shape[0]
    n_goals = waypoints.shape[0]
    packed_width = (n_boids * n_boids + 7) // 8

    states = np.empty((max_steps + 1, n_boids, 6), dtype=np.float32)
    active_goals = np.empty((max_steps, 3), dtype=np.float32)
    previous_goals = np.zeros((max_steps, 3), dtype=np.float32)
    previous_max_distances = np.full(max_steps, -1.0, dtype=np.float32)
    goal_segments = np.empty(max_steps, dtype=np.int16)
    adjacency_bits = np.empty((max_steps, packed_width), dtype=np.uint8)

    positions = initial_positions.copy()
    velocities = initial_velocities.copy()
    neighbors = np.zeros((n_boids, n_boids), dtype=np.uint8)
    degrees = np.zeros(n_boids, dtype=np.int32)
    position_sums = np.zeros((n_boids, 3), dtype=np.float64)
    velocity_sums = np.zeros((n_boids, 3), dtype=np.float64)
    separation = np.zeros((n_boids, 3), dtype=np.float64)

    for boid in range(n_boids):
        for dim in range(3):
            states[0, boid, dim] = positions[boid, dim]
            states[0, boid, dim + 3] = velocities[boid, dim]

    current_goal = 0
    running_previous_distance = -1.0
    perception_squared = perception * perception
    crowding_squared = crowding * crowding
    max_turn_radians = max_turn_degrees * np.pi / 180.0
    np.random.seed(noise_seed)

    for step in range(max_steps):
        _fill_graph_and_sums(
            positions,
            velocities,
            perception_squared,
            crowding_squared,
            neighbors,
            degrees,
            position_sums,
            velocity_sums,
            separation,
        )
        _store_packed_adjacency(neighbors, adjacency_bits[step])

        for dim in range(3):
            active_goals[step, dim] = waypoints[current_goal, dim]
        goal_segments[step] = current_goal

        if current_goal > 0:
            centroid0 = 0.0
            centroid1 = 0.0
            centroid2 = 0.0
            for boid in range(n_boids):
                centroid0 += positions[boid, 0]
                centroid1 += positions[boid, 1]
                centroid2 += positions[boid, 2]
            centroid0 /= n_boids
            centroid1 /= n_boids
            centroid2 /= n_boids
            pdx = centroid0 - waypoints[current_goal - 1, 0]
            pdy = centroid1 - waypoints[current_goal - 1, 1]
            pdz = centroid2 - waypoints[current_goal - 1, 2]
            previous_distance = np.sqrt(pdx * pdx + pdy * pdy + pdz * pdz)
            if previous_distance > running_previous_distance:
                running_previous_distance = previous_distance
            for dim in range(3):
                previous_goals[step, dim] = waypoints[current_goal - 1, dim]
            previous_max_distances[step] = running_previous_distance

        for boid in range(n_boids):
            degree = degrees[boid]

            # Separation, alignment, and cohesion are clamped independently,
            # exactly as in Boids3D before their scalar weights are applied.
            sep = separation[boid].copy()
            _clamp_vector_in_place(sep, max_force)

            alignment = np.empty(3, dtype=np.float64)
            cohesion = np.empty(3, dtype=np.float64)
            if degree > 0:
                inverse_degree = 1.0 / degree
                for dim in range(3):
                    alignment[dim] = (
                        velocity_sums[boid, dim] * inverse_degree
                        - velocities[boid, dim]
                    )
                    cohesion[dim] = (
                        position_sums[boid, dim] * inverse_degree
                        - positions[boid, dim]
                    )
            else:
                for dim in range(3):
                    alignment[dim] = -velocities[boid, dim]
                    cohesion[dim] = -positions[boid, dim]
            _clamp_vector_in_place(alignment, max_force)
            _clamp_vector_in_place(cohesion, max_force)

            goal_force = np.zeros(3, dtype=np.float64)
            if degree > 0:
                denominator = degree + 1.0
                for dim in range(3):
                    local_centroid = (
                        position_sums[boid, dim] + positions[boid, dim]
                    ) / denominator
                    goal_force[dim] = waypoints[current_goal, dim] - local_centroid
                _clamp_vector_in_place(goal_force, max_force)

            proposed0 = velocities[boid, 0] + dt * (
                0.35 * sep[0]
                + 0.35 * alignment[0]
                + 0.001 * cohesion[0]
                + 0.01 * goal_force[0]
            )
            proposed1 = velocities[boid, 1] + dt * (
                0.35 * sep[1]
                + 0.35 * alignment[1]
                + 0.001 * cohesion[1]
                + 0.01 * goal_force[1]
            )
            proposed2 = velocities[boid, 2] + dt * (
                0.35 * sep[2]
                + 0.35 * alignment[2]
                + 0.001 * cohesion[2]
                + 0.01 * goal_force[2]
            )

            old0 = velocities[boid, 0]
            old1 = velocities[boid, 1]
            old2 = velocities[boid, 2]
            old_speed = np.sqrt(old0 * old0 + old1 * old1 + old2 * old2)
            new_speed = np.sqrt(
                proposed0 * proposed0
                + proposed1 * proposed1
                + proposed2 * proposed2
            )

            if old_speed > 1e-8:
                old_dir0 = old0 / old_speed
                old_dir1 = old1 / old_speed
                old_dir2 = old2 / old_speed
            else:
                old_dir0, old_dir1, old_dir2 = 1.0, 0.0, 0.0
            if new_speed > 1e-8:
                new_dir0 = proposed0 / new_speed
                new_dir1 = proposed1 / new_speed
                new_dir2 = proposed2 / new_speed
            else:
                new_dir0, new_dir1, new_dir2 = old_dir0, old_dir1, old_dir2

            cosine = (
                old_dir0 * new_dir0
                + old_dir1 * new_dir1
                + old_dir2 * new_dir2
            )
            cosine = min(1.0, max(-1.0, cosine))
            angle = np.arccos(cosine)
            if angle > max_turn_radians:
                fraction = max_turn_radians / max(angle, 1e-8)
                blend0 = (1.0 - fraction) * old_dir0 + fraction * new_dir0
                blend1 = (1.0 - fraction) * old_dir1 + fraction * new_dir1
                blend2 = (1.0 - fraction) * old_dir2 + fraction * new_dir2
                blend_norm = np.sqrt(
                    blend0 * blend0 + blend1 * blend1 + blend2 * blend2
                )
                if blend_norm > 1e-8:
                    proposed0 = blend0 / blend_norm * new_speed
                    proposed1 = blend1 / blend_norm * new_speed
                    proposed2 = blend2 / blend_norm * new_speed
                else:
                    proposed0 = old_dir0 * new_speed
                    proposed1 = old_dir1 * new_speed
                    proposed2 = old_dir2 * new_speed

            limited_speed = np.sqrt(
                proposed0 * proposed0
                + proposed1 * proposed1
                + proposed2 * proposed2
            )
            if limited_speed < min_speed and limited_speed > 0.0:
                factor = min_speed / limited_speed
                proposed0 *= factor
                proposed1 *= factor
                proposed2 *= factor
            elif limited_speed > max_speed:
                factor = max_speed / limited_speed
                proposed0 *= factor
                proposed1 *= factor
                proposed2 *= factor

            velocities[boid, 0] = proposed0
            velocities[boid, 1] = proposed1
            velocities[boid, 2] = proposed2

        for boid in range(n_boids):
            for dim in range(3):
                positions[boid, dim] += velocities[boid, dim] * dt
                positions[boid, dim] = min(
                    borders[dim + 3], max(borders[dim], positions[boid, dim])
                )
                if pos_noise > 0.0:
                    positions[boid, dim] += np.random.uniform(-pos_noise, pos_noise)
                if vel_noise > 0.0:
                    velocities[boid, dim] += np.random.uniform(-vel_noise, vel_noise)
                states[step + 1, boid, dim] = positions[boid, dim]
                states[step + 1, boid, dim + 3] = velocities[boid, dim]

        mean_agent_distance = 0.0
        for boid in range(n_boids):
            dx = positions[boid, 0] - waypoints[current_goal, 0]
            dy = positions[boid, 1] - waypoints[current_goal, 1]
            dz = positions[boid, 2] - waypoints[current_goal, 2]
            mean_agent_distance += np.sqrt(dx * dx + dy * dy + dz * dz)
        mean_agent_distance /= n_boids
        if mean_agent_distance < arrival_radius:
            current_goal += 1
            running_previous_distance = 0.0
            if current_goal == n_goals:
                completed_steps = step + 1
                return (
                    states[: completed_steps + 1].copy(),
                    active_goals[:completed_steps].copy(),
                    previous_goals[:completed_steps].copy(),
                    previous_max_distances[:completed_steps].copy(),
                    goal_segments[:completed_steps].copy(),
                    adjacency_bits[:completed_steps].copy(),
                    current_goal,
                )

    return (
        states.copy(),
        active_goals.copy(),
        previous_goals.copy(),
        previous_max_distances.copy(),
        goal_segments.copy(),
        adjacency_bits.copy(),
        current_goal,
    )


def rollout_waypoints(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    waypoints: Sequence[Sequence[float]],
    *,
    sampled_center: Optional[Sequence[float]] = None,
    arrival_radius: float = 0.25,
    max_steps: int = 50_000,
    min_speed: float = 0.0001,
    max_speed: float = 0.01,
    max_force: float = 0.1,
    max_turn_degrees: float = 15.0,
    perception: float = 0.1,
    crowding: float = 0.02,
    dt: float = 1.0,
    borders: Sequence[float] = (-5.0, -5.0, -5.0, 5.0, 5.0, 5.0),
    pos_noise: float = 0.0,
    vel_noise: float = 0.0,
    noise_seed: int = 0,
) -> CompiledTrajectory3D:
    """Roll out arbitrary sequential 3D waypoints with unchanged Boids rules."""
    positions = np.ascontiguousarray(initial_positions, dtype=np.float64)
    velocities = np.ascontiguousarray(initial_velocities, dtype=np.float64)
    goals = np.ascontiguousarray(_validate_goals(waypoints), dtype=np.float64)
    border_array = np.ascontiguousarray(borders, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("initial_positions must have shape (N, 3).")
    if velocities.shape != positions.shape:
        raise ValueError("initial_velocities must match initial_positions.")
    if border_array.shape != (6,):
        raise ValueError("borders must have shape (6,).")
    if arrival_radius <= 0 or max_steps < 1:
        raise ValueError("arrival_radius and max_steps must be positive.")

    result = _rollout_kernel_3d(
        positions,
        velocities,
        goals,
        float(arrival_radius),
        int(max_steps),
        float(min_speed),
        float(max_speed),
        float(max_force),
        float(max_turn_degrees),
        float(perception),
        float(crowding),
        float(dt),
        border_array,
        float(pos_noise),
        float(vel_noise),
        int(noise_seed) % (2**32 - 1),
    )
    (
        states,
        active_goals,
        previous_goals,
        previous_max_distances,
        goal_segments,
        adjacency_bits,
        completed_goals,
    ) = result
    if completed_goals != len(goals):
        raise RuntimeError(
            f"Compiled expert reached {completed_goals}/{len(goals)} waypoints "
            f"within max_steps={max_steps}."
        )
    center = (
        positions.mean(axis=0)
        if sampled_center is None
        else np.asarray(sampled_center, dtype=np.float32)
    )
    return CompiledTrajectory3D(
        states=states,
        active_goals=active_goals,
        previous_goals=previous_goals,
        max_previous_goal_distances=previous_max_distances,
        goal_segment_ids=goal_segments,
        adjacency_bits=adjacency_bits,
        waypoints=np.asarray(goals, dtype=np.float32),
        sampled_center=np.asarray(center, dtype=np.float32),
    )


def generate_fixed_trajectory(
    config: Mapping[str, object],
) -> CompiledTrajectory3D:
    """Generate one fixed-task trajectory from a cloud-worker-style config."""
    seed = int(config["seed"])
    n_boids = int(config.get("n_boids", 100))
    goals = _validate_goals(config.get("goals", DEFAULT_FIXED_GOALS))
    center, positions, velocities = sample_fixed_initial_state(
        seed,
        octant=int(config["octant"]),
        n_boids=n_boids,
        goals=goals,
        exclusion_radius=float(config.get("goal_exclusion_size", 0.5)),
        init_scatter=float(config.get("init_scatter", 0.325)),
        max_speed=float(config.get("max_speed", 0.01)),
        legacy_discarded_initialization=bool(
            config.get("legacy_discarded_initialization", True)
        ),
    )
    ordered_goals = order_fixed_waypoints(
        goals,
        positions,
        str(config.get("waypoint_order_policy", FIXED_WAYPOINT_ORDER)),
    )
    return rollout_waypoints(
        positions,
        velocities,
        ordered_goals,
        sampled_center=center,
        arrival_radius=float(config.get("goal_arrival_radius", 0.25)),
        max_steps=int(config.get("expert_max_steps", 50_000)),
        min_speed=float(config.get("min_speed", 0.0001)),
        max_speed=float(config.get("max_speed", 0.01)),
        max_force=float(config.get("max_force", 0.1)),
        max_turn_degrees=float(config.get("max_turn", 15.0)),
        perception=float(config.get("perception", 0.1)),
        crowding=float(config.get("crowding", 0.02)),
        dt=float(config.get("dt", 1.0)),
        pos_noise=float(config.get("pos_noise", 0.0)),
        vel_noise=float(config.get("vel_noise", 0.0)),
        noise_seed=seed,
    )


def generate_online_trajectory(
    config: Mapping[str, object],
) -> CompiledTrajectory3D:
    """Generate one arbitrary two-online-waypoint 3D expert episode."""
    seed = int(config["seed"])
    center, positions, velocities, waypoints = sample_online_initial_state_and_goals(
        seed,
        n_boids=int(config.get("n_boids", 100)),
        n_waypoints=int(config.get("n_waypoints", 2)),
        start_bounds=config.get(
            "start_bounds", (-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)
        ),
        goal_bounds=config.get(
            "goal_bounds", (-4.5, 4.5, -4.5, 4.5, -4.5, 4.5)
        ),
        goal_min_distance=float(config.get("goal_min_distance", 2.5)),
        init_scatter=float(config.get("init_scatter", 0.325)),
        max_speed=float(config.get("max_speed", 0.01)),
    )
    return rollout_waypoints(
        positions,
        velocities,
        waypoints,
        sampled_center=center,
        arrival_radius=float(config.get("goal_arrival_radius", 0.5)),
        max_steps=int(config.get("expert_max_steps", 10_000)),
        min_speed=float(config.get("min_speed", 0.0001)),
        max_speed=float(config.get("max_speed", 0.01)),
        max_force=float(config.get("max_force", 0.1)),
        max_turn_degrees=float(config.get("max_turn", 15.0)),
        perception=float(config.get("perception", 0.1)),
        crowding=float(config.get("crowding", 0.02)),
        dt=float(config.get("dt", 1.0)),
        pos_noise=float(config.get("pos_noise", 0.0)),
        vel_noise=float(config.get("vel_noise", 0.0)),
        noise_seed=seed,
    )


def unpack_adjacency(adjacency_bits: np.ndarray, n_boids: int) -> np.ndarray:
    """Decode little-endian packed rows to ``(..., N, N)`` Boolean masks."""
    packed = np.asarray(adjacency_bits, dtype=np.uint8)
    flat = np.unpackbits(packed, axis=-1, bitorder="little")
    flat = flat[..., : n_boids * n_boids]
    return flat.reshape(packed.shape[:-1] + (n_boids, n_boids)).astype(bool)


def materialize_goal_conditioned_arrays(
    trajectory: CompiledTrajectory3D,
) -> tuple[np.ndarray, np.ndarray]:
    """Build online-GNCA inputs and transition-aware targets on demand.

    Inputs are ``[position, velocity, active_goal - position]`` (9 features).
    Targets are ``[current physical state, next physical state, active goal,
    previous goal, max distance from previous goal]`` (19 features).
    """
    current = trajectory.states[:-1]
    following = trajectory.states[1:]
    goals = trajectory.active_goals[:, None, :]
    goal_vectors = goals - current[..., :3]
    inputs = np.concatenate((current, goal_vectors), axis=-1).astype(
        np.float32, copy=False
    )
    n_boids = trajectory.n_boids
    active = np.broadcast_to(goals, (trajectory.steps, n_boids, 3))
    previous = np.broadcast_to(
        trajectory.previous_goals[:, None, :], (trajectory.steps, n_boids, 3)
    )
    previous_max = np.broadcast_to(
        trajectory.max_previous_goal_distances[:, None, None],
        (trajectory.steps, n_boids, 1),
    )
    targets = np.concatenate(
        (current, following, active, previous, previous_max), axis=-1
    ).astype(np.float32, copy=False)
    return np.ascontiguousarray(inputs), np.ascontiguousarray(targets)


def warm_up_compiled_expert_3d() -> float:
    """Compile the zero-noise rollout before starting the measured pipeline.

    Numba specializes on dtype and rank rather than the concrete number of
    boids, so this tiny one-step rollout compiles the same kernel signature used
    by a production 100-boid trajectory without generating a training sample.
    The returned value is compilation wall time in seconds.
    """
    positions = np.asarray(
        [
            [-0.015, 0.000, 0.000],
            [0.015, 0.000, 0.000],
            [0.000, -0.015, 0.000],
            [0.000, 0.015, 0.000],
        ],
        dtype=np.float64,
    )
    velocities = np.zeros((4, 3), dtype=np.float64)
    velocities[:, 0] = 0.01
    started = time.perf_counter()
    rollout_waypoints(
        positions,
        velocities,
        np.zeros((1, 3), dtype=np.float32),
        arrival_radius=100.0,
        max_steps=1,
        pos_noise=0.0,
        vel_noise=0.0,
    )
    return time.perf_counter() - started
