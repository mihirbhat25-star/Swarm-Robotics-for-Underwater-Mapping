import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spektral import utils
from spektral.data import Dataset, Graph
from tqdm import tqdm
import tensorflow as tf
import h5py


BOIDS_GOAL_POSITIONS_3D = np.array(
    [[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]],
    dtype=np.float32,
)
NEAREST_CCW_POLICY = "nearest_then_counterclockwise_xy"
FIXED_ORDER_POLICY = "fixed_waypoint_order"


def nearest_then_counterclockwise_order(goal_positions, start_position):
    """Order waypoints nearest-first, then counterclockwise in the XY plane."""
    goals = np.asarray(goal_positions)
    start = np.asarray(start_position)
    if goals.ndim != 2 or len(goals) == 0 or goals.shape[1] < 2:
        raise ValueError("goal_positions must have shape (N, D) with D >= 2.")
    if start.shape != (goals.shape[1],):
        raise ValueError(
            f"start_position must have shape ({goals.shape[1]},), got {start.shape}."
        )

    nearest_idx = int(np.argmin(np.linalg.norm(goals - start[None, :], axis=1)))
    center_xy = np.mean(goals[:, :2], axis=0)
    relative_xy = goals[:, :2] - center_xy[None, :]
    angles = np.mod(np.arctan2(relative_xy[:, 1], relative_xy[:, 0]), 2 * np.pi)
    ccw_order = np.argsort(angles, kind="stable")
    nearest_position = int(np.flatnonzero(ccw_order == nearest_idx)[0])
    return np.roll(ccw_order, -nearest_position).astype(np.int64)


def _in_exclusion_zone_3d(center, exclusion_zone):
    """Return True if center is within capsule exclusion zone."""
    if exclusion_zone is None:
        return False
    goals, radius = exclusion_zone
    for i in range(len(goals)):
        a = goals[i]
        b = goals[(i + 1) % len(goals)]
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-10:
            if np.linalg.norm(center - a) < radius:
                return True
            continue
        t = np.clip(np.dot(center - a, ab) / ab_len_sq, 0.0, 1.0)
        closest = a + t * ab
        if np.linalg.norm(center - closest) < radius:
            return True
    return False

class Boids3D:
    def __init__(
        self,
        min_speed=0.0001,
        max_speed=0.01,
        max_force=0.1,
        max_turn=15,
        perception=0.1,
        crowding=0.02,
        n_boids=100,
        dt=1,
        canvas_scale=1,
        boundary_size_pctg=0.2,
        wrap=False,
        limits=True,
        show=False,
        pos_noise=0.000,  # Max magnitude of bounded uniform noise added to positions at each step
        vel_noise=0.0000,  # Max magnitude of bounded uniform noise added to velocities at each step
        waypoint_order_policy=NEAREST_CCW_POLICY,
    ):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_force = max_force
        self.max_turn = max_turn
        self.perception = perception
        self.crowding = crowding
        self.n_boids = n_boids
        self.dt = dt
        self.canvas_scale = canvas_scale
        self.boundary_size_pctg = boundary_size_pctg
        self.wrap = wrap
        self.limits = limits
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.init_scatter = 0.325  # Spread of boids around the initial center

        # 3D canvas: [xmin, ymin, zmin, xmax, ymax, zmax]
        self.borders = canvas_scale * np.array([-5, -5, -5, 5, 5, 5])
        self.center = (self.borders[3:] + self.borders[:3]) / 2
        self.canonical_goal_positions = BOIDS_GOAL_POSITIONS_3D.copy()
        self.goal_positions = self.canonical_goal_positions.copy()
        self.goal_order = np.arange(len(self.goal_positions), dtype=np.int64)
        if waypoint_order_policy not in (NEAREST_CCW_POLICY, FIXED_ORDER_POLICY):
            raise ValueError(
                "waypoint_order_policy must be either "
                f"'{NEAREST_CCW_POLICY}' or '{FIXED_ORDER_POLICY}'."
            )
        self.waypoint_order_policy = waypoint_order_policy
        self.current_goal = 0
        self.reached_goal = False
        self.loiter_timer = 0
        self.goals_completed = 0
        self.rand_configs = []
        self.unseen_configs = []  # Centers from cache NOT used in training

        self.boundary_margins = self.borders * boundary_size_pctg
        self.boundaries = self.borders - self.boundary_margins

        self.show = show
        self.figure = None
        self.print_neighbors_warning = True

    def update_boids(self, positions, velocities, return_accel=False):
        accelerations = np.zeros_like(velocities)

        if self.wrap:
            positions = ((positions + 1) % 2) - 1
        # else:
        #     accelerations += self.avoid_borders(positions)

        neighbors = self.get_neighbors(positions)

        SEPARATION_WEIGHT = 0.35
        ALIGNMENT_WEIGHT = 0.35
        COHESION_WEIGHT = 0.001
        GOAL_WEIGHT = 0.01

        accelerations += SEPARATION_WEIGHT * self.get_separation(neighbors, positions)
        accelerations += ALIGNMENT_WEIGHT * self.get_alignment(neighbors, velocities)
        accelerations += COHESION_WEIGHT * self.get_cohesion(neighbors, positions)
        accelerations += GOAL_WEIGHT * self.get_goal(neighbors, positions)

        velocities_new = velocities + accelerations * self.dt
        if self.limits:
            velocities = self.enforce_limits(velocities, velocities_new)
        else:
            velocities = velocities_new

        positions = positions + velocities * self.dt

        # Clip to 3D borders
        for dim in range(3):
            positions[:, dim] = np.clip(positions[:, dim], self.borders[dim], self.borders[dim + 3])

        # Add a bounded random noise to positions and velocities
        positions += np.random.uniform(-self.pos_noise, self.pos_noise, positions.shape)
        velocities += np.random.uniform(-self.vel_noise, self.vel_noise, velocities.shape)

        if self.show:
            self.plot(positions)

        return (positions, velocities, neighbors) + (
            (accelerations,) if return_accel else ()
        )

    def generate_trajectory(self, save_config, random_init, return_accel=False):
        # Reset goal state for each new trajectory
        self.current_goal = 0
        self.goals_completed = 0
        self.reached_goal = False

        if random_init is False:
            positions, velocities, neighbors = self.get_fixed_init(self.n_boids)
        elif random_init is True:
            positions, velocities, neighbors, _ = self.get_random_init(self.n_boids, save_config)
        elif isinstance(random_init, np.ndarray) and random_init.shape == (3,):
            # Treat as a center for the flock (3D)
            center = random_init
            positions = center + 0.325 * np.random.rand(self.n_boids, 3)
            direction = np.array([1.0, 0.0, 0.0])
            velocity = direction * self.max_speed
            velocities = np.tile(velocity, (self.n_boids, 1))
            neighbors = self.get_neighbors(positions)
        else:
            assert (
                len(random_init) == 2
            ), "Expected random_init to have length 2 (positions, velocities) or a center (3,)"
            positions, velocities = random_init
            neighbors = self.get_neighbors(positions)

        self._set_goal_order_from_positions(positions)

        history = {
            "positions": [positions],
            "velocities": [velocities],
            "neighbors": [neighbors],
            "goal_positions": [self.goal_positions[self.current_goal].copy()],
        }
        if return_accel:
            history["accelerations"] = []

        while True:
            output = self.update_boids(positions, velocities, return_accel=return_accel)
            positions, velocities, _ = output[:3]
            # update_boids() uses the pre-update graph to compute the expert
            # dynamics.  A training sample at the new state, however, must be
            # paired with the graph of that new state.  Storing the graph
            # returned by update_boids() made every sample after t=0 use the
            # previous timestep's adjacency matrix.
            neighbors = self.get_neighbors(positions)
            history["positions"].append(positions)
            history["velocities"].append(velocities)
            history["neighbors"].append(neighbors)
            if return_accel:
                history["accelerations"].append(output[3])
            self.update_goal(positions)
            # The goal stored with state x_t must be the goal used to produce
            # x_{t+1}. Update the goal before labeling the newly appended state.
            history["goal_positions"].append(self.goal_positions[self.current_goal].copy())
            if self.check_termination():
                break

        history["positions"] = np.array(history["positions"])
        history["velocities"] = np.array(history["velocities"])
        history["goal_positions"] = np.array(history["goal_positions"])
        if return_accel:
            history["accelerations"] = np.array(history["accelerations"])

        return history

    def _set_goal_order_from_positions(self, positions):
        """Apply the configured fixed or nearest-then-CCW waypoint policy."""
        if self.waypoint_order_policy == FIXED_ORDER_POLICY:
            self.goal_order = np.arange(
                len(self.canonical_goal_positions), dtype=np.int64
            )
            self.goal_positions = self.canonical_goal_positions.copy()
            return

        initial_flock_centroid = np.mean(positions, axis=0)
        self.goal_order = nearest_then_counterclockwise_order(
            self.canonical_goal_positions,
            initial_flock_centroid,
        )
        self.goal_positions = self.canonical_goal_positions[self.goal_order].copy()

    def avoid_borders(self, positions):
        """Steer boids away from 3D boundaries."""
        in_margin = (
            np.any(positions < self.boundaries[:3], -1) |
            np.any(positions > self.boundaries[3:], -1)
        )
        steering = np.zeros_like(positions)
        steering[in_margin] += self.center - positions[in_margin]
        return steering

    def get_neighbors(self, positions):
        neighbors_matrix = (
            np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
            < self.perception
        )
        np.fill_diagonal(neighbors_matrix, 0)
        neighbors = sp.coo_matrix(neighbors_matrix, dtype=int)
        return neighbors

    def get_separation(self, neighbors, positions):
        self_idx, neig_idx = neighbors.row, neighbors.col
        distances = np.linalg.norm(positions[self_idx] - positions[neig_idx], axis=-1)
        mask = distances < self.crowding
        steering = np.zeros_like(positions)
        changes = -(positions[neig_idx[mask]] - positions[self_idx[mask]])
        np.add.at(steering, self_idx[mask], changes)
        steering = self.clamp(steering)
        return steering

    def get_alignment(self, neighbors, velocities):
        degree = utils.degree_power(neighbors, -1.0)
        steering = degree @ neighbors @ velocities
        steering -= velocities
        steering = self.clamp(steering)
        return steering

    def get_cohesion(self, neighbors, positions):
        return self.get_alignment(neighbors, positions)

    def get_goal(self, neighbors, positions):
        neighbor_mask = neighbors.toarray().astype(float)  # (n, n)
        n_neighbors = neighbor_mask.sum(axis=1)  # (n,)
        has_neighbors = n_neighbors > 0
        neighbor_pos_sum = neighbor_mask @ positions  # (n, 3)
        centroid = (neighbor_pos_sum + positions) / (n_neighbors[:, None] + 1)
        goal_vec = self.goal_positions[self.current_goal] - centroid  # (n, 3)
        steering = np.where(has_neighbors[:, None], goal_vec, 0.0)
        steering = self.clamp(steering)
        return steering

    def update_goal(self, positions):
        goal_vec = self.goal_positions[self.current_goal] - positions
        goal_dist = np.linalg.norm(goal_vec, axis=-1)
        if np.mean(goal_dist) < 0.25 and not self.reached_goal:
            self.reached_goal = True
        if self.reached_goal:
            self.current_goal = (self.current_goal + 1) % len(self.goal_positions)
            self.goals_completed += 1
            self.reached_goal = False

    def check_termination(self):
        return self.goals_completed == len(self.goal_positions)

    def enforce_limits(self, velocities_old, velocities_new):
        """Enforce speed AND turn limits in 3D."""
        velocities = velocities_new.copy()

        # ── Turn limit ───────────────────────────────────────────────────────
        speed_old = np.linalg.norm(velocities_old, axis=-1, keepdims=True)
        speed_new = np.linalg.norm(velocities, axis=-1, keepdims=True)
        # Avoid division by zero
        safe_old = np.where(speed_old > 1e-8, velocities_old / speed_old, np.array([1., 0., 0.]))
        safe_new = np.where(speed_new > 1e-8, velocities / speed_new, safe_old)
        cos_angle = np.clip((safe_old * safe_new).sum(axis=-1), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)  # (n,)
        max_turn_rad = np.deg2rad(self.max_turn)
        mask = angle_rad > max_turn_rad
        if mask.any():
            t = np.where(mask, max_turn_rad / np.maximum(angle_rad, 1e-8), 1.0)[:, None]
            # Slerp between old and new direction, then renormalize
            blended = (1.0 - t) * safe_old + t * safe_new
            norm = np.linalg.norm(blended, axis=-1, keepdims=True)
            blended = np.where(norm > 1e-8, blended / norm, safe_old)
            velocities = np.where(mask[:, None], blended * speed_new, velocities)

        # ── Speed limit ──────────────────────────────────────────────────────
        speed = np.linalg.norm(velocities, axis=-1)
        velocities[speed < self.min_speed] = scale(velocities[speed < self.min_speed], self.min_speed)
        velocities[speed > self.max_speed] = scale(velocities[speed > self.max_speed], self.max_speed)
        return velocities

    def clamp(self, force):
        to_clamp = np.linalg.norm(force, axis=-1) > self.max_force
        force[to_clamp] = scale(force[to_clamp], length=self.max_force)
        return force

    def get_random_init(self, n_boids, save_config, bounds=None, exclusion_zone=None, center=None, max_attempts=10000):
        """
        Spawns boids in a tight clump at a random location.
        bounds:         optional (x_min, x_max, y_min, y_max, z_min, z_max)
        exclusion_zone: optional list of (center_3d, radius) spheres to reject
        center:         if provided, skip sampling and use directly
        """
        if center is None:
            if bounds is not None:
                x_min, x_max, y_min, y_max, z_min, z_max = bounds
            else:
                x_min, x_max, y_min, y_max, z_min, z_max = -5, 0, -5, 0, -5, 0
            for _ in range(max_attempts):
                c = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max),
                    np.random.uniform(z_min, z_max),
                ])
                if exclusion_zone is None or not _in_exclusion_zone_3d(c, exclusion_zone):
                    center = c
                    break
            else:
                raise RuntimeError(f"Could not sample valid center after {max_attempts} attempts.")
        if save_config:
            self.rand_configs.append(center)
        positions = center + self.init_scatter * np.random.rand(n_boids, 3)
        direction = np.array([1.0, 0.0, 0.0])
        velocities = np.tile(direction * self.max_speed, (n_boids, 1))
        neighbors = self.get_neighbors(positions)
        return positions, velocities, neighbors, center

    def get_fixed_init(self, n_boids):
        center = np.array([-3.0, -3.0, -3.0])
        # print("Using fixed initial configuration for reproducibility. Center of flock: ", center)
        self.rand_configs.append(center)
        positions = center + 0.325 * np.random.rand(n_boids, 3)
        direction = np.array([1.0, 0.0, 0.0])
        velocity = direction * self.max_speed
        velocities = np.tile(velocity, (n_boids, 1))
        neighbors = self.get_neighbors(positions)
        return positions, velocities, neighbors

    def plot(self, positions):
        if self.figure is None:
            plt.ion()
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.scatter = self.ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                marker=".", edgecolor="k", lw=0.5
            )
            goal = self.goal_positions[self.current_goal]
            self.goal_marker = self.ax.scatter([goal[0]], [goal[1]], [goal[2]],
                                               c='red', marker='*', s=200, label='Goal')
            self.ax.set_xlim(self.borders[0], self.borders[3])
            self.ax.set_ylim(self.borders[1], self.borders[4])
            self.ax.set_zlim(self.borders[2], self.borders[5])
            self.ax.legend()
            plt.show()
        self.scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        goal = self.goal_positions[self.current_goal]
        self.goal_marker._offsets3d = ([goal[0]], [goal[1]], [goal[2]])
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

class BoidsDataset3D(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.graphs = dataset

    def read(self):
        return []


def scale(x, length=1.0):
    return length * x / np.linalg.norm(x, axis=-1, keepdims=True)

def history_to_samples_3d(history, accel=False):
    """Convert 3D trajectory history to graph samples.
    y is [current_state (6D), next_state (6D)] if goal present, else just next_state (6D).
    """
    inputs = np.concatenate((history["positions"], history["velocities"]), axis=-1)
    neighbors = history["neighbors"]
    n_boids = inputs.shape[1]
    if accel and "accelerations" in history:
        targets = history["accelerations"]
        return [(x, a, y) for x, a, y in zip(inputs[:-1], neighbors[:-1], targets)]
    else:
        base_targets = np.concatenate([inputs[:-1], inputs[1:]], axis=-1)  # [T-1, n_boids, 12]
        if "goal_positions" in history:
            # goal_positions: [T, 3] -> broadcast to [T-1, n_boids, 3]
            goal_pos = history["goal_positions"][:-1]  # [T-1, 3]
            goal_broadcast = np.broadcast_to(goal_pos[:, None, :], (len(goal_pos), n_boids, 3)).copy()
            targets = np.concatenate([base_targets, goal_broadcast], axis=-1)  # [T-1, n_boids, 15]
        else:
            targets = base_targets  # [T-1, n_boids, 12]
        return [(x, a, y_) for x, a, y_ in zip(inputs[:-1], neighbors[:-1], targets)]


def _load_adaptive_stride_3d(f, start, end, timestep_stride, near_goal_radius):
    """Load timesteps with adaptive stride: dense (stride=1) near any goal, sparse elsewhere.
    y[:, 0, 12:15] = current goal position (same for all boids at each timestep).
    """
    if timestep_stride <= 1 or near_goal_radius <= 0:
        return (
            f['x'][start:end:timestep_stride],
            f['y'][start:end:timestep_stride],
            f['a_row'][start:end:timestep_stride],
            f['a_col'][start:end:timestep_stride],
            f['a_len'][start:end:timestep_stride],
        )

    x_full = f['x'][start:end]   # (T, n_boids, 6)
    y_full = f['y'][start:end]   # (T, n_boids, 15)

    mean_pos = x_full[:, :, :3].mean(axis=1)   # (T, 3)
    dist = np.min(
        np.linalg.norm(
            mean_pos[:, None, :] - BOIDS_GOAL_POSITIONS_3D[None, :, :],
            axis=-1,
        ),
        axis=1,
    )

    T = end - start
    strided = np.zeros(T, dtype=bool)
    strided[::timestep_stride] = True
    keep = np.where(strided | (dist < near_goal_radius))[0]
    abs_keep = keep + start

    return (
        x_full[keep],
        y_full[keep],
        f['a_row'][abs_keep],
        f['a_col'][abs_keep],
        f['a_len'][abs_keep],
    )


def load_chunk_from_cache_3d(cache_path, flat_indices, boundaries, n_boids_cache,
                              timestep_stride=1, near_goal_radius=1.0):
    """Load a subset of 3D trajectories from HDF5 into a BoidsDataset3D."""
    graphs = []
    with h5py.File(cache_path, 'r') as f:
        for flat_idx in flat_indices:
            start = int(boundaries[flat_idx])
            end   = int(boundaries[flat_idx + 1])
            x_c, y_c, arow_c, acol_c, alen_c = _load_adaptive_stride_3d(
                f, start, end, timestep_stride, near_goal_radius
            )
            for k in range(len(x_c)):
                length = int(alen_c[k])
                row = arow_c[k, :length]
                col = acol_c[k, :length]
                a = sp.coo_matrix(
                    (np.ones(length, dtype=np.float32), (row, col)),
                    shape=(n_boids_cache, n_boids_cache),
                )
                graphs.append(Graph(x=x_c[k], a=a, y=y_c[k]))
    return BoidsDataset3D(graphs)


def make_dataset_3d(unique_reps, repeat_reps, save_config, trajectory_len=None, random_init=True, return_boids=False, accel=False, **kwargs):
    kwargs.pop("n_jobs", 1)
    kwargs.pop("init", None)
    boids_cache_npz = kwargs.pop("boids_cache_npz", None)
    timestep_stride = kwargs.pop("timestep_stride", 1)

    boids = Boids3D(**kwargs)
    all_graphs = []

    # Fast path: load precomputed graph tuples from HDF5 cache
    if boids_cache_npz is not None:
        
        print(f">>> Loading precomputed 3D training data from '{boids_cache_npz}'...")
        with h5py.File(boids_cache_npz, 'r') as f:
            centers       = f['centers'][:]
            cache_unique  = int(f.attrs['unique_reps'])
            cache_repeats = int(f.attrs['repeats'])
            n_boids_cache = int(f.attrs['n_boids'])

            n_train = min(unique_reps, cache_unique)
            train_indices  = np.random.choice(cache_unique, size=n_train, replace=False)
            unseen_indices = np.array([i for i in range(cache_unique) if i not in set(train_indices)])

            boids.rand_configs   = [np.array(centers[i], dtype=np.float32) for i in train_indices]
            boids.unseen_configs = [np.array(centers[i], dtype=np.float32) for i in unseen_indices]
            print(f"✅ Loaded cache: {n_train} training centers, {len(unseen_indices)} unseen centers, timestep_stride={timestep_stride}")

            if cache_repeats < 1:
                raise ValueError("Cache must contain at least one trajectory per center.")
            n_reps = repeat_reps

            if 'traj_lengths' in f:
                traj_lengths = f['traj_lengths'][:]
                boundaries = np.concatenate([[0], np.cumsum(traj_lengths)])
            else:
                total_samples    = f['x'].shape[0]
                samples_per_traj = total_samples // (cache_unique * cache_repeats)
                boundaries = np.arange(cache_unique * cache_repeats + 1) * samples_per_traj

            for traj_i in train_indices:
                for rep_j in range(n_reps):
                    cached_repeat_idx = rep_j % cache_repeats
                    flat_idx = traj_i * cache_repeats + cached_repeat_idx
                    start = int(boundaries[flat_idx])
                    end   = int(boundaries[flat_idx + 1])
                    x_chunk    = f['x'][start:end:timestep_stride]
                    y_chunk    = f['y'][start:end:timestep_stride]
                    arow_chunk = f['a_row'][start:end:timestep_stride]
                    acol_chunk = f['a_col'][start:end:timestep_stride]
                    alen_chunk = f['a_len'][start:end:timestep_stride]
                    for k in range(len(x_chunk)):
                        length = int(alen_chunk[k])
                        row  = arow_chunk[k, :length]
                        col  = acol_chunk[k, :length]
                        data = np.ones(length, dtype=np.float32)
                        a = sp.coo_matrix((data, (row, col)), shape=(n_boids_cache, n_boids_cache))
                        all_graphs.append(Graph(x=x_chunk[k], a=a, y=y_chunk[k]))

        dataset = BoidsDataset3D(all_graphs)
        return (dataset, boids) if return_boids else dataset

    # Normal path: simulate trajectories
    print(f">>> Generating {unique_reps} unique 3D trajectories with {repeat_reps} repeats each...")

    use_init_list = isinstance(random_init, list) and len(random_init) == unique_reps

    for i in tqdm(range(unique_reps)):
        # Sample one center per unique rep, reuse for all repeats
        if use_init_list:
            center = random_init[i]
        elif random_init is True:
            _, _, _, center = boids.get_random_init(boids.n_boids, save_config=save_config)
        else:
            center = None  # fixed init — handled inside generate_trajectory

        for j in range(repeat_reps):
            if center is not None:
                history = boids.generate_trajectory(
                    random_init=center,
                    return_accel=accel,
                    save_config=False,
                )
            else:
                history = boids.generate_trajectory(
                    random_init=random_init,
                    return_accel=accel,
                    save_config=(save_config and j == 0),
                )
            samples = history_to_samples_3d(history, accel=accel)
            for x, a, y in samples:
                all_graphs.append(Graph(x=x, a=a, y=y))
            del history, samples

    dataset = BoidsDataset3D(all_graphs)
    return (dataset, boids) if return_boids else dataset
