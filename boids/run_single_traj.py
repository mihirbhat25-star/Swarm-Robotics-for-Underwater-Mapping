"""
Load saved model weights and run a single GNCA trajectory for a fixed number of steps
from a specified starting center.

Usage:
    python -m boids.run_single_traj \
        --weights saved_models/best_weights_200x5_newl_cd_2.5_dw_2.5_nw_2 \
        --center -2.5 -2.5 \
        --steps 5000 \
        --output traj_run.npz
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.gnn_ca_simple_boids import GNNCASimpleBoids
from modules.boids import Boids

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True,
                    help="Path to saved weight checkpoint (no extension needed).")
parser.add_argument("--center", type=float, nargs=2, default=[-2.5, -2.5],
                    metavar=("X", "Y"),
                    help="Starting center of the flock. Default: -2.5 -2.5")
parser.add_argument("--steps", type=int, default=5000,
                    help="Number of autoregressive steps. Default: 5000")
parser.add_argument("--n_boids", type=int, default=100)
parser.add_argument("--output", type=str, default="traj_single_run.npz",
                    help="Path to save the trajectory (.npz).")
parser.add_argument("--no_plot", action="store_true",
                    help="Skip saving a PDF plot.")
args = parser.parse_args()

# ── Build Boids env ────────────────────────────────────────────────────────────
boids = Boids(n_boids=args.n_boids)
center = np.array(args.center, dtype=np.float32)
print(f">>> Starting center: {center}")
print(f">>> Steps: {args.steps}  |  n_boids: {args.n_boids}")

# Initialise flock at the requested center
positions  = center + boids.init_scatter * np.random.rand(args.n_boids, 2).astype(np.float32)
velocities = np.tile(np.array([1.0, 0.0], dtype=np.float32) * boids.max_speed, (args.n_boids, 1))

# ── Build & load model ─────────────────────────────────────────────────────────
model = GNNCASimpleBoids(
    activation="linear",
    batch_norm=False,
    hidden=256,
    hidden_activation="relu",
    connectivity="cat",
    aggregate="mean",
)

# Build model weights via one dummy forward pass before loading checkpoint
def _to_tf_sparse(a):
    indices = np.stack([a.row, a.col], axis=1)
    a_tf = tf.SparseTensor(indices=indices, values=a.data.astype(np.float32), dense_shape=a.shape)
    return tf.sparse.reorder(a_tf)

dummy_x = tf.constant(np.concatenate([positions, velocities], axis=-1), dtype=tf.float32)
dummy_a = _to_tf_sparse(boids.get_neighbors(positions))
dummy_i = tf.zeros(args.n_boids, dtype=tf.int64)
model([dummy_x, dummy_a, dummy_i[:, None]], training=False)

model.load_weights(args.weights)
print(f">>> Loaded weights from '{args.weights}'")

# ── Autoregressive rollout ─────────────────────────────────────────────────────
print(f">>> Running {args.steps}-step GNCA trajectory...")
i_tf = tf.zeros(args.n_boids, dtype=tf.int64)

traj = [np.concatenate([positions, velocities], axis=-1).astype(np.float32)]
for step in range(args.steps - 1):
    x_last = traj[-1]
    a_tf   = _to_tf_sparse(boids.get_neighbors(x_last[:, :2]))
    x_next = model([tf.constant(x_last, dtype=tf.float32), a_tf, i_tf[:, None]], training=False)
    x_next_np = x_next.numpy()

    # Clip positions to canvas
    x_next_np[:, :2] = np.clip(x_next_np[:, :2], -5.0, 5.0)
    # Clamp velocity magnitude
    speeds = np.linalg.norm(x_next_np[:, 2:], axis=-1, keepdims=True)
    too_fast = speeds > boids.max_speed
    x_next_np[:, 2:] = np.where(
        too_fast,
        x_next_np[:, 2:] * (boids.max_speed / (speeds + 1e-8)),
        x_next_np[:, 2:]
    )
    traj.append(x_next_np)

    if (step + 1) % 500 == 0:
        print(f"  step {step+1}/{args.steps - 1}")

traj = np.array(traj)  # (steps, n_boids, 4)
print(f">>> Trajectory shape: {traj.shape}")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
np.savez(args.output, traj=traj, center=center, goals=boids.goal_positions)
print(f">>> Saved trajectory to '{args.output}'")

# ── Plot ───────────────────────────────────────────────────────────────────────
if not args.no_plot:
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString, MultiPolygon

    centroid = traj[:, :, :2].mean(axis=1)  # (steps, 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(centroid[:, 0], centroid[:, 1], color='#8B0000', lw=1.2, label='Flock centroid')
    ax.scatter([center[0]], [center[1]], c='blue', marker='*', s=200, zorder=5, label='Start')
    ax.scatter(boids.goal_positions[:, 0], boids.goal_positions[:, 1],
               c='red', marker='*', s=200, zorder=5, label='Goals')
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
    ax.axhline(0, color='gray', lw=0.4, ls='--'); ax.axvline(0, color='gray', lw=0.4, ls='--')
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"GNCA — {args.steps} steps from {tuple(center)}")
    ax.set_aspect('equal'); ax.legend()
    plt.tight_layout()

    plot_path = args.output.replace('.npz', '.pdf')
    plt.savefig(plot_path)
    plt.close()
    print(f">>> Saved plot to '{plot_path}'")
