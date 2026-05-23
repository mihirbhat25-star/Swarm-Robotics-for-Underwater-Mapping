"""
Evaluates the trained 3D GNCA.
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
from modules.boids_3d import Boids3D
import random


@tf.function(experimental_relax_shapes=True)
def forward(model, x, a, i, training=None):
    """Computes one forward pass of the 3D GNCA"""
    x_pred = model((x, a, i[:, None]), training=training)
    return x_pred


def to_tf_sparse(a):
    indices = np.stack([a.row, a.col], axis=1)
    a_tf = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape
    )
    return tf.sparse.reorder(a_tf)


def run_autoregressive(model, forward, boids, positions, velocities, n_boids, max_trajectory_len):
    """Runs the 3D GNCA autoregressively from a given start position."""
    x = np.concatenate([positions, velocities], axis=-1)
    a_scipy = boids.get_neighbors(positions)
    a = to_tf_sparse(a_scipy)
    trajectory = [x.astype(np.float32)]
    for t in range(max_trajectory_len - 1):
        x_last = trajectory[-1]
        a_scipy = boids.get_neighbors(x_last[:, :3])
        a = to_tf_sparse(a_scipy)
        x_next = forward(model, x_last, a, np.zeros((n_boids, 1)), training=False)
        trajectory.append(x_next.numpy())
    return np.array(trajectory)


def evaluate_3d(model, forward, max_trajectory_len, n_boids, use_saved_config, saved_boids):
    """
    Evaluation for 3D GNCA: generates autoregressive trajectories and saves PDF + MP4.
    """
    np.random.seed(0)

    boids = saved_boids if saved_boids is not None else Boids3D(n_boids=n_boids)
    goals = boids.goal_positions  # shape (N, 3)

    # --- 5-trajectory 3D PDF ---
    colors = ["g", "b", "m", "c", "y"]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    num_trajectories = min(5, len(boids.rand_configs))
    for i in range(num_trajectories):
        center = boids.rand_configs[i]
        print("Using start config centered at:", center)
        positions = center + 0.325 * np.random.rand(n_boids, 3)
        direction = np.array([1.0, 0.0, 0.0])
        velocity = direction * boids.max_speed
        velocities = np.tile(velocity, (n_boids, 1))

        traj = run_autoregressive(model, forward, boids, positions, velocities, n_boids, max_trajectory_len)
        # Plot one boid's path per trajectory
        ax.plot(traj[:, 0, 0], traj[:, 0, 1], traj[:, 0, 2], c=colors[i % len(colors)], lw=2, label=f"GNCA {i+1}")

    # Plot goals
    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], c='red', marker='*', s=150, label='Goals')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title("3D Autoregressive (Full Flight) Paths (5 starts)")
    ax.legend()
    plt.savefig("boids_auto_rand_3d.pdf")
    plt.close()
    print("✅ Saved boids_auto_rand_3d.pdf")

    # --- MP4 animation using the first rand_config ---
    if len(boids.rand_configs) > 0:
        center = boids.rand_configs[0]
    else:
        center = np.array([-2.5, -2.5, 0.0])

    positions = center + 0.325 * np.random.rand(n_boids, 3)
    direction = np.array([1.0, 0.0, 0.0])
    velocity = direction * boids.max_speed
    velocities = np.tile(velocity, (n_boids, 1))
    traj = run_autoregressive(model, forward, boids, positions, velocities, n_boids, max_trajectory_len)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    writer = FFMpegWriter(fps=20)
    print("🎬 Saving 3D GNCA flight to gnca_boids_rand_3d.mp4...")
    with writer.saving(fig, "gnca_boids_rand_3d.mp4", dpi=100):
        for i in range(len(traj)):
            ax.cla()
            pos = traj[i][:, :3]
            ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], c='red', marker='*', s=150)
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='lime', s=20, edgecolors='k')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-5, 5)
            ax.set_title(f"Step {i}")
            writer.grab_frame()
    print("✅ Done! Check your workspace folder for gnca_boids_rand_3d.mp4")
