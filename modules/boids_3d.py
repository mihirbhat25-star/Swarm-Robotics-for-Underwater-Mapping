import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spektral import utils
from spektral.data import Dataset, Graph
from tqdm import tqdm
import tensorflow as tf


class Boids3D:
    def __init__(
        self,
        min_speed=0.0001,
        max_speed=0.01,
        max_force=0.1,
        max_turn=5,
        perception=0.1,
        crowding=0.02,
        n_boids=100,
        dt=1,
        canvas_scale=1,
        boundary_size_pctg=0.2,
        wrap=False,
        limits=True,
        show=False,
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

        # 3D canvas: [xmin, ymin, zmin, xmax, ymax, zmax]
        self.borders = canvas_scale * np.array([-5, -5, -5, 5, 5, 5])
        self.center = (self.borders[3:] + self.borders[:3]) / 2
        self.goal_positions = np.array([[3.0, 3.0, 3.0]])  # One goal at far corner for full 3D diagonal trajectory
        self.current_goal = 0
        self.reached_goal = False
        self.loiter_timer = 0
        self.goals_completed = 0
        self.rand_configs = []

        self.boundary_margins = self.borders * boundary_size_pctg
        self.boundaries = self.borders - self.boundary_margins

        self.show = show
        self.figure = None

    def update_boids(self, positions, velocities, return_accel=False):
        accelerations = np.zeros_like(velocities)

        if self.wrap:
            positions = ((positions + 1) % 2) - 1
        else:
            accelerations += self.avoid_borders(positions)

        neighbors = self.get_neighbors(positions)

        SEPARATION_WEIGHT = 0.35
        ALIGNMENT_WEIGHT = 0.70
        COHESION_WEIGHT = 0.002
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

        if self.show:
            self.plot(positions)

        return (positions, velocities, neighbors) + (
            (accelerations,) if return_accel else ()
        )

    def generate_trajectory(self, save_config, random_init, return_accel=False):
        if random_init is False:
            positions, velocities, neighbors = self.get_fixed_init(self.n_boids)
        elif random_init is True:
            positions, velocities, neighbors = self.get_random_init(self.n_boids, save_config)
        else:
            assert len(random_init) == 2
            positions, velocities = random_init
            neighbors = self.get_neighbors(positions)

        history = {
            "positions": [positions],
            "velocities": [velocities],
            "neighbors": [neighbors],
        }
        if return_accel:
            history["accelerations"] = []

        while True:
            output = self.update_boids(positions, velocities, return_accel=return_accel)
            positions, velocities, neighbors = output[:3]
            history["positions"].append(positions)
            history["velocities"].append(velocities)
            history["neighbors"].append(neighbors)
            if return_accel:
                history["accelerations"].append(output[3])
            self.update_goal(positions)
            if self.check_termination():
                break

        history["positions"] = np.array(history["positions"])
        history["velocities"] = np.array(history["velocities"])
        if return_accel:
            history["accelerations"] = np.array(history["accelerations"])

        return history

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
        neighbor_mask = neighbors.toarray()
        steering = np.zeros_like(positions)
        for i in range(self.n_boids):
            neighbor_indices = np.where(neighbor_mask[i] == 1)[0]
            if len(neighbor_indices) == 0:
                continue
            neighbor_positions = positions[neighbor_indices]
            group_positions = np.vstack([neighbor_positions, positions[i]])
            centroid = np.mean(group_positions, axis=0)
            goal_vec = self.goal_positions[self.current_goal] - centroid
            steering[i] = goal_vec
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
        return self.goals_completed == 1

    def enforce_limits(self, velocities_old, velocities_new):
        """Enforce speed limits in 3D (no turn limit in 3D polar form)."""
        velocities = velocities_new.copy()
        speed = np.linalg.norm(velocities, axis=-1)
        velocities[speed < self.min_speed] = scale(
            velocities[speed < self.min_speed], self.min_speed
        )
        velocities[speed > self.max_speed] = scale(
            velocities[speed > self.max_speed], self.max_speed
        )
        return velocities

    def clamp(self, force):
        to_clamp = np.linalg.norm(force, axis=-1) > self.max_force
        force[to_clamp] = scale(force[to_clamp], length=self.max_force)
        return force

    def get_random_init(self, n_boids, save_config):
        center = np.array([
            np.random.uniform(-5, -2.5),
            np.random.uniform(-5, -2.5),
            np.random.uniform(-2.5, 0.0),
        ])
        if save_config:
            self.rand_configs.append(center)
        positions = center + 0.325 * np.random.rand(n_boids, 3)
        direction = np.array([1.0, 0.0, 0.0])
        velocity = direction * self.max_speed
        velocities = np.tile(velocity, (n_boids, 1))
        neighbors = self.get_neighbors(positions)
        return positions, velocities, neighbors

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
            self.ax.set_xlim(self.borders[0], self.borders[3])
            self.ax.set_ylim(self.borders[1], self.borders[4])
            self.ax.set_zlim(self.borders[2], self.borders[5])
            plt.show()
        self.scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
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
    """Standard MSE version: y is just the next state (6 features)."""
    inputs = np.concatenate((history["positions"], history["velocities"]), axis=-1)
    neighbors = history["neighbors"]
    if accel and "accelerations" in history:
        targets = history["accelerations"]
        return [(x, a, y) for x, a, y in zip(inputs[:-1], neighbors[:-1], targets)]
    else:
        targets = inputs[1:]
        return [(x, a, y_) for x, a, y_ in zip(inputs[:-1], neighbors[:-1], targets)]


def history_to_samples_3d_weighted(history, accel=False):
    """Custom weighted loss version: y is [current_state, next_state] (12 features)."""
    inputs = np.concatenate((history["positions"], history["velocities"]), axis=-1)
    neighbors = history["neighbors"]
    if accel and "accelerations" in history:
        targets = history["accelerations"]
        return [(x, a, y) for x, a, y in zip(inputs[:-1], neighbors[:-1], targets)]
    else:
        targets = np.concatenate([
            inputs[:-1],  # current state (t)
            inputs[1:]    # next state (t+1)
        ], axis=-1)
        return [(x, a, y_) for x, a, y_ in zip(inputs[:-1], neighbors[:-1], targets)]


def make_dataset_3d(unique_reps, repeat_reps, save_config, trajectory_len=None, random_init=False, return_boids=False, accel=False, custom_loss=False, **kwargs):
    kwargs.pop("n_jobs", 1)
    kwargs.pop("init", None)

    boids = Boids3D(**kwargs)
    all_graphs = []
    sample_fn = history_to_samples_3d_weighted if custom_loss else history_to_samples_3d

    print(f">>> Generating {unique_reps} unique 3D trajectories with {repeat_reps} repeats each, random_init={random_init}, custom_loss={custom_loss}...")

    for i in tqdm(range(unique_reps)):
        print(f"\nGenerating trajectory {i+1}/{unique_reps}...")
        for j in range(repeat_reps):
            history = boids.generate_trajectory(
                random_init=random_init,
                return_accel=accel,
                save_config=(save_config and j == 0)
            )
            samples = sample_fn(history, accel=accel)
            for x, a, y in samples:
                all_graphs.append(Graph(x=x, a=a, y=y))
            del history
            del samples

    dataset = BoidsDataset3D(all_graphs)
    return (dataset, boids) if return_boids else dataset
