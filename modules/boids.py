import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from spektral import utils
from spektral.data import Dataset, Graph
from tqdm import tqdm
import tensorflow as tf

class Boids:
    def __init__(
        self,
        min_speed=0.0001,  # Min speed of the boids
        max_speed=0.01,  # Max speed of the boids
        max_force=0.1,  # Max amount of steering that any single update is allowed to add
        max_turn=5,  # How many degrees is a boid allowed to turn
        perception=0.1,  # How distant must two boids be in order to be neighbors
        crowding=0.02,  # How much groups are pushed apart (lower = tighter groups)
        n_boids=100,  # How many boids in the environment
        dt=1,  # Size of a time step (lower = more precise simulation)
        canvas_scale=1,  # Canvas is rescaled by this amount (used to control size)
        boundary_size_pctg=0.2,  # Relative size of the soft boundary
        wrap=False,  # If True, wrap around instead of avoiding boundary
        limits=True,  # If True, enforce speed and turn limits
        show=False,  # Show an animated plot of the boid everytime update_boids is called
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

        self.borders = canvas_scale * np.array([-5, -5, 5, 5])  # Hard borders of canvas
        self.center = (self.borders[2:] + self.borders[:2]) / 2  # Center of the canvas
        self.goal_positions = np.array([[3.0, -3.0], [3.0, 3.0], [0.0, 0.0]])  # Set multiple goals in a triangular fashion at (3, -3), (3, 3) and (0, 0). Fix this if the canvas is [-5,5]^2.
        self.current_goal = 0
        self.reached_goal = False  # Flag to track if boids have reached the current goal
        self.loiter_timer = 0  # Timer to track how long boids have been loitering at the current goal
        self.goals_completed = 0  # Counter to track how many goals have been completed

        # Soft boundary inside which boids are pushed towards the center to avoid leaving the canvas
        self.boundary_margins = self.borders * boundary_size_pctg
        self.boundaries = self.borders - self.boundary_margins

        self.show = show
        self.figure = None
        self.print_neighbors_warning = True  # Flag to control printing neighbors warning

    def update_boids(self, positions, velocities, return_accel=False):
        accelerations = np.zeros_like(velocities)

        if self.wrap:
            positions = ((positions + 1) % 2) - 1  # Wrap around
        else:
            accelerations += self.avoid_borders(positions)  # Avoid edge collisions

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

        # Update positions
        positions = positions + velocities * self.dt

        positions[:, 0] = np.clip(positions[:, 0], self.borders[0], self.borders[2])
        positions[:, 1] = np.clip(positions[:, 1], self.borders[1], self.borders[3])

        # Plot if needed
        if self.show:
            self.plot(positions)

        return (positions, velocities, neighbors) + (
            (accelerations,) if return_accel else ()
        )

    def generate_trajectory(self, random_init, return_accel=False):
        if random_init is False:
            positions, velocities, neighbors = self.get_fixed_init(self.n_boids)
        elif random_init is True:
            positions, velocities, neighbors = self.get_random_init(self.n_boids)
        else:
            assert (
                len(random_init) == 2
            ), "Expected random_init to have length 2 (positions, velocities)"
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
                # print("Trajectory generation terminating...")
                break

        history["positions"] = np.array(history["positions"])
        history["velocities"] = np.array(history["velocities"])
        if return_accel:
            history["accelerations"] = np.array(history["accelerations"])

        return history

    def avoid_borders(self, positions):
        """If a boid is within the external margins, steer it towards the centre"""
        in_margin = np.any(positions < self.boundaries[:2], -1) | np.any(
            positions > self.boundaries[2:], -1
        )
        steering = np.zeros_like(positions)

        steering[in_margin] += self.center - positions[in_margin]

        return steering

    def get_neighbors(self, positions):
        neighbors_matrix = (
            np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
            < self.perception
        )
        # Exclude self-neighbors before converting to sparse
        np.fill_diagonal(neighbors_matrix, 0)

        # Debug: print average number of neighbors per boid
        avg_neighbors = np.sum(neighbors_matrix, axis=1).mean()
        if avg_neighbors < 10 and self.print_neighbors_warning:
            print(f"[DEBUG] Problematic! Too low of neighbors: {avg_neighbors:.2f}.")
            # self.print_neighbors_warning = False  # Only print once

        # Also print debug to see if any boids are overlapping each other (which is bad)
        # num_overlaps = np.sum(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) < 0.01) - self.n_boids  # Subtract self-count
        # if num_overlaps > 0:
        #     avg_overlap = np.mean(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)[np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) < 0.01])
        #     print(f"[DEBUG] Warning! {num_overlaps} pairs of boids are overlapping (distance < 0.01). The average distance of overlap is {avg_overlap:.4f}.")
        
        # Convert to sparse matrix for return
        neighbors = sp.coo_matrix(neighbors_matrix, dtype=int)
        return neighbors

    def get_separation(self, neighbors, positions):
        """
        Get the steering component to separate boids that are too close
        """
        self_idx, neig_idx = neighbors.row, neighbors.col
        distances = np.linalg.norm(positions[self_idx] - positions[neig_idx], axis=-1)
        mask = distances < self.crowding
        steering = np.zeros_like(positions)
        changes = -(positions[neig_idx[mask]] - positions[self_idx[mask]])
        np.add.at(steering, self_idx[mask], changes)
        steering = self.clamp(steering)

        return steering

    def get_alignment(self, neighbors, velocities):
        """
        Get the steering component to align the velocities of neighbors
        """
        degree = utils.degree_power(neighbors, -1.0)
        steering = degree @ neighbors @ velocities
        steering -= velocities
        steering = self.clamp(steering)

        return steering

    def get_cohesion(self, neighbors, positions):
        """
        Get the steering component to align the positions of neighbors
        """
        return self.get_alignment(neighbors, positions)
    
    def get_goal(self, neighbors, positions):
        
        """
        Get the steering component to move towards a goal position.
        """
        neighbor_mask = neighbors.toarray()
        steering = np.zeros_like(positions)
        for i in range(self.n_boids):
            neighbor_indices = np.where(neighbor_mask[i] == 1)[0]
            if len(neighbor_indices) == 0:
                continue
            # For each neighbor, compute distance to goal and direction vector
            neighbor_positions = positions[neighbor_indices]
            group_positions = np.vstack([neighbor_positions, positions[i]])
            centroid = np.mean(group_positions, axis=0)
            goal_vec = self.goal_positions[self.current_goal] - centroid  # (num_neighbors, 2)
            steering[i] = goal_vec
        steering = self.clamp(steering)
        return steering
    
    def update_goal(self, positions):
        """
        Update the goal position after boids reach it and loiter for 1 step.
        """
        # Fetch current positions of boids and compute distance to goal
        goal_vec = self.goal_positions[self.current_goal] - positions  # (num_boids, 2)
        goal_dist = np.linalg.norm(goal_vec, axis=-1)  # (num_boids,)
        # If minimum distance to goal is less than 0.25, start loitering timer
        if np.mean(goal_dist) < 0.25 and not self.reached_goal:
            self.reached_goal = True

        if self.reached_goal:
            self.current_goal = (self.current_goal + 1) % len(self.goal_positions)
            self.goals_completed += 1
            self.reached_goal = False
            
    def check_termination(self):
        """
        Terminate the trajectory if boids finish loitering at all goals for 1 step.
        """
        
        if self.goals_completed == 3:  # Assuming 3 goals
            # print("Termination condition met. All goals completed.")
            return True
        else:
            return False

    def enforce_limits(self, velocities_old, velocities_new):
        # Update velocities
        velocities_old_polar = to_polar(velocities_old)
        velocities_new_polar = to_polar(velocities_new)

        # Enforce turn limit
        phi_diff = (
            180 - (180 - velocities_new_polar[:, 1] + velocities_old_polar[:, 1]) % 360
        )
        mask = np.abs(phi_diff) > self.max_turn
        velocities_new_polar[mask, 1] = (
            velocities_old_polar[mask, 1] + np.sign(phi_diff[mask]) * self.max_turn
        )
        velocities = to_cartesian(velocities_new_polar)

        # Enforce speed limit
        speed = np.linalg.norm(velocities, axis=-1)
        velocities[speed < self.min_speed] = scale(
            velocities[speed < self.min_speed], self.min_speed
        )
        velocities[speed > self.max_speed] = scale(
            velocities[speed > self.max_speed], self.max_speed
        )

        return velocities

    def clamp(self, force):
        """
        Clamp a given force (steering) to the maximum value allowed (to make things
        more stable)
        """
        to_clamp = np.linalg.norm(force, axis=-1) > self.max_force
        force[to_clamp] = scale(force[to_clamp], length=self.max_force)

        return force

    def get_random_init(self, n_boids):
        """
        Spawns boids in a tight clump at a random location on the canvas.
        """
        # 1. Pick a random 'center' for the flock (between [-5 and 0])
        center = np.random.uniform(-5, 0, size=2)
        # print(f"Center of flock: {center}")
        positions = center + 0.325 * np.random.rand(n_boids, 2)
        # Set all boids to have the same initial velocity
        # Example: all boids move right at max_speed
        direction = np.array([1.0, 0.0])  # unit vector to the right
        velocity = direction * self.max_speed
        velocities = np.tile(velocity, (n_boids, 1))
        neighbors = self.get_neighbors(positions)

        return positions, velocities, neighbors
    
    def get_fixed_init(self, n_boids):
        """
        Set a fixed initial position of (-2.5 -2.5).
        :param n_boids: int, number of boids
        """
        positions = np.full((n_boids, 2), -2.5) + 0.325 * np.random.rand(n_boids, 2)
        print(f"Using fixed initial configuration for trajectory generation.")
        # Set all boids to have the same initial velocity
        # Example: all boids move right at max_speed
        direction = np.array([1.0, 0.0])  # unit vector to the right
        velocity = direction * self.max_speed
        velocities = np.tile(velocity, (n_boids, 1))
        neighbors = self.get_neighbors(positions)

        return positions, velocities, neighbors

    def plot(self, positions, **kwargs):
        if self.figure is None:
            plt.ion()
            self.figure = plt.figure()
            axes = plt.axes(xlim=self.borders[::2], ylim=self.borders[1::2])
            self.scatter = axes.scatter(
                positions[:, 0],
                positions[:, 1],
                marker=".",
                edgecolor="k",
                lw=0.5,
                **kwargs
            )
            self.goal_marker, = axes.plot([], [], 'r*', markersize=15, label='Goal')
            axes.legend()
            axes.set_title(
                f"Boids Evolution with Goal"
            )
            plt.show()
        # Always update the goal marker position
        self.goal_marker.set_data(
            [self.goal_positions[self.current_goal][0]],
            [self.goal_positions[self.current_goal][1]]
        )
        self.scatter.set_offsets(positions)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
class BoidsDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.graphs = dataset

    def read(self):
        return []

def to_polar(cartesian_coords):
    x, y = cartesian_coords.T
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x) * 180 / np.pi
    return np.stack((rho, phi), -1)

def to_cartesian(polar_coords):
    rho, phi = polar_coords.T
    phi *= np.pi / 180
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.stack((x, y), -1)

def scale(x, length=1.0):
    return length * x / np.linalg.norm(x, axis=-1, keepdims=True)

def history_to_samples(history, accel=False):
    inputs = np.concatenate((history["positions"], history["velocities"]), axis=-1)
    neighbors = history["neighbors"]
    if accel and "accelerations" in history:
        targets = history["accelerations"]
    else:
        targets = inputs[1:]

    return [(x, a, y) for x, a, y in zip(inputs[:-1], neighbors[:-1], targets)]

def make_dataset(reps, trajectory_len=None, random_init=True, return_boids=False, accel=False, **kwargs):
    # 1. Clean up kwargs so Boids() doesn't crash
    n_jobs = kwargs.pop("n_jobs", 1)
    # If evaluate() passes 'init', we'll use it as our random_init data
    init_data = kwargs.pop("init", None)

    boids = Boids(**kwargs)
    all_graphs = []

    print(f">>> Generating {reps} trajectories with random_init={random_init}...")
    
    for i in tqdm(range(reps)):
        # 2. Generate trajectory using the 'init' data from evaluate
        history = boids.generate_trajectory(
            random_init=random_init,
            return_accel=accel
        )
        
        samples = history_to_samples(history, accel=accel)
        
        for x, a, y in samples:
            all_graphs.append(Graph(x=x, a=a, y=y))
            
        del history
        del samples

    dataset = BoidsDataset(all_graphs)
    return (dataset, boids) if return_boids else dataset