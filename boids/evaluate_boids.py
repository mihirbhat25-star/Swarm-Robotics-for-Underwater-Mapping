"""
Evaluates the trained GNCA by comparing it to the true Boids GCA.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import Animation, FFMpegWriter
import nolds
import numpy as np
import os
import tensorflow as tf
from spektral.data import DisjointLoader
from spektral.layers import ops
from tensorflow.keras.models import load_model
from modules.boids import make_dataset
from modules.boids import Boids
from shapely.geometry import LineString
import random

@tf.function(experimental_relax_shapes=True)
def forward(model, x, a, i, training=None):
    """Computes one forward pass of the GNCA"""
    x_pred = model((x, a, i[:, None]), training=training)
    return x_pred

def avg_measure(trajectory, measure_fn, n_boids=None, coord=0, **kwargs):
    n_boids_total = trajectory.shape[-2]
    measures = []
    # Ensure we don't try to sample more boids than exist
    sample_size = min(n_boids if n_boids else 5, n_boids_total)
    for i in np.random.permutation(n_boids_total)[:sample_size]:
        measures.append(measure_fn(trajectory[:, i, coord], **kwargs))

    mn, std = np.mean(measures), np.std(measures)
    print(f"{measure_fn.__name__} {mn} +- {std}")
    return np.array(measures)

def convert_to_tf_sparse(a):
    """Safe conversion from scipy sparse to TF SparseTensor using standard API"""
    # 1. Get the indices where the connections are (row, col)
    # a.row and a.col come from the Scipy COO matrix
    indices = np.stack([a.row, a.col], axis=1)
    
    # 2. Create the SparseTensor
    a_tf = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape
    )
    
    # 3. Always reorder to ensure the sparse indices are in canonical order
    return tf.sparse.reorder(a_tf)


def _get_tube_exterior(tube):
    """Return (x, y) arrays for the exterior of a shapely Polygon or the largest polygon in a MultiPolygon."""
    from shapely.geometry import MultiPolygon
    if isinstance(tube, MultiPolygon):
        tube = max(tube.geoms, key=lambda p: p.area)
    return tube.exterior.xy


def _plot_tubular(trajs, goals, n_boids):
    """
    Tubular band visualization.
    trajs: list of (T, n_boids, 4) arrays, one per run.
    Plots the mean swarm centroid curve with a 99% tubular band.
    Uses shapely.buffer() for a correct Euclidean tube (no self-intersection artifacts).
    """

    centroids = np.array([t[:, :, :2].mean(axis=1) for t in trajs])  # (N, T, 2)
    mean_curve = centroids.mean(axis=0)  # (T, 2)

    # Per-run trimmed-sup distance: Q0.95 of pointwise Euclidean distances over time
    dists = np.linalg.norm(centroids - mean_curve[None], axis=-1)  # (N, T)
    d_i = np.percentile(dists, 95, axis=1)  # (N,)
    r_99 = np.percentile(d_i, 99)  # 99% tube radius

    # Build true Euclidean tube using shapely: union of circles of radius r_99 along mean_curve
    line = LineString(mean_curve)
    tube = line.buffer(r_99)
    tube_x, tube_y = _get_tube_exterior(tube)

    return mean_curve, np.array(tube_x), np.array(tube_y), r_99


def _plot_per_boid(trajs, goals, n_boids):
    """
    Per-boid individual path visualization.
    trajs: list of (T, n_boids, 4) arrays, one per run.
    Picks one random run and plots all individual boid paths from that rollout.
    """
    run_idx = np.random.randint(len(trajs))
    traj = trajs[run_idx]  # (T, n_boids, 4)
    print(f"Per-boid viz using run {run_idx} of {len(trajs)}")

    fig, ax = plt.subplots(figsize=(7, 7))
    for b in range(n_boids):
        ax.plot(
            traj[:, b, 0], traj[:, b, 1],
            color='#8B0000', lw=0.6, alpha=0.4,
            label='Boid path' if b == 0 else None
        )
    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"GNCA Swarm — Individual Boid Paths (run {run_idx})")
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig("boids_auto_rand.pdf")
    plt.close()


def _plot_multi_tubular(trajs, goals, n_boids, n_show=10, filename="boids_multi_tube.pdf"):
    """
    Plots n_show randomly selected runs as separate tubular bands on the same canvas.
    Each run gets its own per-boid centroid curve + individual tube.
    trajs: list of (T, n_boids, 4) arrays, one per run.
    """
    from shapely.geometry import LineString
    import matplotlib.cm as cm

    indices = np.random.choice(len(trajs), min(n_show, len(trajs)), replace=False)
    colors = cm.Reds(np.linspace(0.4, 0.9, len(indices)))

    fig, ax = plt.subplots(figsize=(7, 7))
    for color, run_idx in zip(colors, indices):
        traj = trajs[run_idx]  # (T, n_boids, 4)
        centroid = traj[:, :, :2].mean(axis=1)  # (T, 2)

        boid_dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)  # (T, n_boids)
        r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)

        line = LineString(centroid)
        tube = line.buffer(r)
        tx, ty = _get_tube_exterior(tube)
        ax.fill(tx, ty, color=color, alpha=0.25)
        ax.plot(centroid[:, 0], centroid[:, 1], color=color, lw=1.2,
                label=f'Run {run_idx}')

    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"GNCA Swarm — {len(indices)} Separate Tubular Bands")
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def _plot_single_tubular(trajs, goals, n_boids, filename="boids_single_tube.pdf"):
    """
    Picks ONE random run and plots its per-boid centroid as a single tubular band.
    """
    from shapely.geometry import LineString

    run_idx = np.random.randint(len(trajs))
    traj = trajs[run_idx]  # (T, n_boids, 4)
    centroid = traj[:, :, :2].mean(axis=1)  # (T, 2)
    print(f"Single-tube viz using run {run_idx} of {len(trajs)}")

    boid_dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)  # (T, n_boids)
    r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)

    line = LineString(centroid)
    tube = line.buffer(r)
    tx, ty = _get_tube_exterior(tube)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.fill(tx, ty, color='#8B0000', alpha=0.3, label=f'99% tube (r={r:.2f})')
    ax.plot(centroid[:, 0], centroid[:, 1], color='#8B0000', lw=1.5, label=f'Run {run_idx}')
    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"GNCA Swarm — Single Run Tube (run {run_idx})")
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def _model_hash(model):
    """Compute a short hash of the model's trainable weights for cache keying."""
    import hashlib
    h = hashlib.md5()
    for w in model.trainable_variables:
        h.update(w.numpy().tobytes())
    return h.hexdigest()[:12]


def evaluate(model, forward, max_trajectory_len, n_boids, use_saved_config, saved_boids,
             init_blob=False, viz_mode='tubular', max_viz_runs=50,
             traj_cache_path="viz_trajectories.npz"):
    """
    Evaluate GNCA trajectories and produce a visualization PDF.

    viz_mode:        'tubular'       — mean over all runs with 99% tubular band (default)
                     'per_boid'      — all 100 boid paths from one random run
                     'multi_tubular' — 10 random runs each as a separate tube
    max_viz_runs:    cap on number of runs used for visualization (default 20)
    traj_cache_path: path to cache computed trajectories; reused if model unchanged
    """
    np.random.seed(0)  # seed for reproducible trajectory generation only

    if saved_boids is not None:
        boids = saved_boids
    else:
        boids = Boids(n_boids=n_boids)

    goals = boids.goal_positions
    borders = boids.borders

    def to_tf_sparse(a):
        indices = np.stack([a.row, a.col], axis=1)
        a_tf = tf.SparseTensor(indices=indices, values=a.data.astype(np.float32), dense_shape=a.shape)
        return tf.sparse.reorder(a_tf)

    def run_gnca_traj(positions, velocities):
        x = np.concatenate([positions, velocities], axis=-1)
        a = to_tf_sparse(boids.get_neighbors(positions))
        traj = [x.astype(np.float32)]
        for t in range(max_trajectory_len - 1):
            x_last = traj[-1]
            a = to_tf_sparse(boids.get_neighbors(x_last[:, :2]))
            x_next = forward(model, x_last, a, np.zeros((n_boids, 1)), training=False)
            traj.append(x_next.numpy())
        return np.array(traj)

    # Cap runs at max_viz_runs
    if use_saved_config and len(boids.rand_configs) > 0:
        all_centers = np.array(boids.rand_configs)
        n_runs = min(max_viz_runs, len(all_centers))
        selected_centers = all_centers[:n_runs]
    else:
        selected_centers = None
        n_runs = max_viz_runs

    # Check trajectory cache
    model_tag = _model_hash(model)
    trajs = None
    if os.path.exists(traj_cache_path):
        cache = np.load(traj_cache_path, allow_pickle=True)
        if cache.get("model_tag", "") == model_tag and cache.get("n_runs", 0) == n_runs:
            print(f"✅ Loaded cached trajectories ({n_runs} runs) from '{traj_cache_path}'")
            trajs = list(cache["trajs"])

    if trajs is None:
        print(f"🔄 Computing {n_runs} trajectories (model_tag={model_tag})...")
        trajs = []
        for i in range(n_runs):
            if selected_centers is not None:
                center = selected_centers[i]
                pos = center + 0.325 * np.random.rand(n_boids, 2)
                vel = np.tile(np.array([1.0, 0.0]) * boids.max_speed, (n_boids, 1))
            else:
                pos, vel, _ = boids.get_random_init(n_boids, save_config=False)
            trajs.append(run_gnca_traj(pos, vel))
        np.savez(traj_cache_path, trajs=np.array(trajs), model_tag=model_tag, n_runs=n_runs)
        print(f"💾 Cached trajectories to '{traj_cache_path}'")

    # Reset seed so visualization run/index selection is truly random each call
    np.random.seed(None)

    if viz_mode == 'tubular':
        # 1. Mean over all runs
        mean_curve, poly_x, poly_y, r_99 = _plot_tubular(trajs, goals, n_boids)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.fill(poly_x, poly_y, color='#8B0000', alpha=0.3, label=f'99% tube (r={r_99:.2f})')
        ax.plot(mean_curve[:, 0], mean_curve[:, 1], color='#8B0000', lw=1.5, label='Mean path')
        ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"GNCA Swarm — Mean over {len(trajs)} Runs (Tubular Band)")
        ax.set_aspect('equal')
        ax.legend()
        plt.tight_layout()
        plt.savefig("boids_tube_mean.pdf")
        plt.close()
        print("Saved boids_tube_mean.pdf")
        # 2. 10 random runs, each as a separate tube
        _plot_multi_tubular(trajs, goals, n_boids, n_show=10, filename="boids_tube_multi.pdf")
        # 3. Single random run as a tube
        _plot_single_tubular(trajs, goals, n_boids, filename="boids_tube_single.pdf")
    elif viz_mode == 'per_boid':
        _plot_per_boid(trajs, goals, n_boids)
    else:
        raise ValueError(f"Unknown viz_mode '{viz_mode}'. Choose 'tubular' or 'per_boid'.")

    # Animation: use the median-y trajectory
    # mid_positions = configs[selected_indices[1]] + 0.325 * np.random.rand(n_boids, 2) if use_saved_config and len(boids.rand_configs) > 0 else configs[1][0]
    # mid_velocities = np.tile(np.array([1.0, 0.0]) * boids.max_speed, (n_boids, 1)) if use_saved_config and len(boids.rand_configs) > 0 else configs[1][1]
    # boid_trajectory_auto = run_gnca_traj(mid_positions, mid_velocities)

    # fig, ax = plt.subplots(figsize=(7, 7))
    # writer = FFMpegWriter(fps=20)
    # print("🎬 Saving GNCA flight to gnca_boids_rand.mp4...")
    # with writer.saving(fig, "gnca_boids_rand.mp4", dpi=100):
    #     for i in range(len(boid_trajectory_auto)):
    #         ax.clear()
    #         pos = boid_trajectory_auto[i][:, :2]
    #         ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150)
    #         ax.scatter(pos[:, 0], pos[:, 1], c='lime', s=20, edgecolors='k')
    #         ax.set_xlim(borders[0], borders[2])
    #         ax.set_ylim(borders[1], borders[3])
    #         ax.set_title(f"Step {i}")
    #         writer.grab_frame()
    # print("✅ Done! Check your workspace folder for gnca_boids_rand.mp4")

def evaluate_complexity(model, forward, te_set_size, trajectory_len, n_boids, init_blob=False):
    """
    Runs multiple randomized test trajectories to calculate average SampEn and CorrDim.
    """
    np.random.seed(0)
    all_measures = []
    
    for i in range(te_set_size):
        # We rely on get_random_init() for the 'clump', so we keep init=None
        data_te, boids_te = make_dataset(
            1,
            trajectory_len,
            random_init=True,
            return_boids=True,
            n_boids=n_boids,
            n_jobs=1,
            init=None, 
        )
        loader_te = DisjointLoader(data_te, node_level=True, epochs=1, shuffle=False)

        boid_trajectory_true = []
        boid_trajectory_auto = []
        
        for sample in loader_te:
            inputs, x_next = sample
            
            if len(boid_trajectory_auto) == 0:
                # Synchronize start: use the first prediction as the seed
                x_start_pred = forward(model, *inputs, training=False)
                boid_trajectory_auto.append(x_start_pred)
            else:
                x_last = boid_trajectory_auto[-1]
                # Re-calculate neighbors for the GNN's current position
                a_scipy = boids_te.get_neighbors(x_last[:, :2])
                a = convert_to_tf_sparse(a_scipy)
                
                # Forward pass using GNN's own previous output
                inputs_auto = [x_last, a, inputs[-1]]
                x_next_auto = forward(model, *inputs_auto, training=False)
                boid_trajectory_auto.append(x_next_auto)

            boid_trajectory_true.append(x_next)

        # Convert to numpy for complexity analysis
        traj_true = np.array(boid_trajectory_true)
        traj_auto = np.array(boid_trajectory_auto)

        # Calculate metrics for this specific trajectory
        m_true_samp = avg_measure(traj_true, nolds.sampen)
        m_auto_samp = avg_measure(traj_auto, nolds.sampen)
        m_true_corr = avg_measure(traj_true, nolds.corr_dim, emb_dim=10)
        m_auto_corr = avg_measure(traj_auto, nolds.corr_dim, emb_dim=10)
        
        all_measures.append((m_true_samp, m_auto_samp, m_true_corr, m_auto_corr))

    # Convert to array and calculate final averages/stds
    measures = np.array(all_measures)
    measures_mean = np.mean(measures, (0, -1))
    measures_std = np.std(measures, (0, -1))
    
    print(f"\nFINAL COMPLEXITY STATS OVER {te_set_size} RUNS:")
    print(f"SampEn True: {measures_mean[0]:.6f} +- {measures_std[0]:.6f}")
    print(f"SampEn GNCA: {measures_mean[1]:.6f} +- {measures_std[1]:.6f}")
    print(f"CorrDim True: {measures_mean[2]:.6f} +- {measures_std[2]:.6f}")
    print(f"CorrDim GNCA: {measures_mean[3]:.6f} +- {measures_std[3]:.6f}")
    return measures_mean, measures_std