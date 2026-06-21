"""
Visualization functions and standalone script for 2D GNCA evaluation.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from shapely.geometry import LineString, MultiPolygon
import matplotlib.cm as cm
import joblib
import tensorflow as tf

# Make sure the workspace root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from boids.forward import forward


# ============================================================================
# Visualization / Plotting Functions
# ============================================================================

def _get_tube_exterior(tube):
    """Return (x, y) arrays for the exterior of a shapely Polygon or largest MultiPolygon."""
    if isinstance(tube, MultiPolygon):
        tube = max(tube.geoms, key=lambda p: p.area)
    return tube.exterior.xy


def _plot_tubular(trajs, goals, n_boids):
    """
    Tubular band visualization.
    trajs: list of (T, n_boids, 4) arrays, one per run.
    Returns: (mean_curve, poly_x, poly_y, r_99)
    """
    centroids = np.array([t[:, :, :2].mean(axis=1) for t in trajs])  # (N, T, 2)
    mean_curve = centroids.mean(axis=0)  # (T, 2)

    dists = np.linalg.norm(centroids - mean_curve[None], axis=-1)  # (N, T)
    d_i = np.percentile(dists, 95, axis=1)  # (N,)
    r_99 = np.percentile(d_i, 99)

    line = LineString(mean_curve)
    tube = line.buffer(r_99)
    tube_x, tube_y = _get_tube_exterior(tube)

    return mean_curve, np.array(tube_x), np.array(tube_y), r_99


def _plot_per_boid(trajs, goals, n_boids, filename="boids_auto_rand.pdf"):
    """
    Per-boid individual path visualization: one random run, all boid paths.
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
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def _plot_single_tubular(trajs, goals, n_boids, filename="boids_single_tube.pdf"):
    """Pick one random run and plot its centroid as a single tubular band."""
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


def _plot_multi_tubular(trajs, goals, n_boids, n_show=10, filename="boids_multi_tube.pdf", specific_runs=None, hide_legend=False):
    """Plot n_show runs as separate tubular bands on same canvas."""

    if specific_runs is not None:
        indices = [i for i in specific_runs if i < len(trajs)]
    else:
        indices = np.random.choice(len(trajs), min(n_show, len(trajs)), replace=False)
    colors = cm.hsv(np.linspace(0, 0.9, len(indices)))

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
        ax.plot(centroid[:, 0], centroid[:, 1], color=color, lw=1.2, label=f'Run {run_idx}')

    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"GNCA Swarm — {len(indices)} Separate Tubular Bands")
    ax.set_aspect('equal')
    if not hide_legend:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def _plot_individual_ranked(trajs, goals, n_boids, selected_centers=None, run_tag="", output_dir="."):
    """
    Generate individual PDFs for each run, ranked by tube radius.
    Each PDF shows a single 2D visualization with tube.
    """
    run_stats = []
    for idx, traj in enumerate(trajs):
        centroid = traj[:, :, :2].mean(axis=1)
        boid_dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)
        r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)
        center = selected_centers[idx] if selected_centers is not None and len(selected_centers) > idx else None
        run_stats.append((idx, r, centroid, boid_dists, center))
    
    run_stats.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- Runs sorted by tube radius r (greatest → least) ---")
    for rank, (idx, r, centroid, boid_dists, center) in enumerate(run_stats):
        print(f"Rank {rank+1:02d} | Run {idx:02d} | r={r:.4f} | center: {np.round(center, 4) if center is not None else 'unknown'}")
        line = LineString(centroid)
        tube = line.buffer(r)
        tx, ty = _get_tube_exterior(tube)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.fill(tx, ty, color='#8B0000', alpha=0.3, label=f'99% tube (r={r:.2f})')
        ax.plot(centroid[:, 0], centroid[:, 1], color='#8B0000', lw=1.5, label=f'Run {idx}')
        ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title(f"Rank {rank+1:02d} | Run {idx:02d} | r={r:.3f} | center: {np.round(center, 2) if center is not None else ''}")
        ax.set_aspect('equal'); ax.legend()
        plt.tight_layout()
        tag = f"_{run_tag}" if run_tag else ""
        fpath = os.path.join(output_dir, f"boids_tube_run{tag}_rank{rank+1:02d}_run{idx:02d}.pdf")
        plt.savefig(fpath); plt.close()
        print(f"  → Saved {fpath}")


def save_tubular_triplet(trajs, goals, n_boids, tag, label, output_dir):
    """Save mean, single, and all-runs tubular PDFs for a trajectory set."""
    mean_curve, poly_x, poly_y, r_99 = _plot_tubular(trajs, goals, n_boids)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.fill(poly_x, poly_y, color='#8B0000', alpha=0.3, label=f'99% tube (r={r_99:.2f})')
    ax.plot(mean_curve[:, 0], mean_curve[:, 1], color='#8B0000', lw=1.5, label='Mean path')
    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"GNCA Swarm — Mean over {len(trajs)} Runs [{label.upper()}]")
    ax.set_aspect('equal'); ax.legend()
    plt.tight_layout()
    mean_fpath = os.path.join(output_dir, f"boids_tube_mean_{label}{tag}.pdf")
    plt.savefig(mean_fpath); plt.close()
    print(f"Saved {mean_fpath}")
    _plot_single_tubular(trajs, goals, n_boids,
                         filename=os.path.join(output_dir, f"boids_tube_single_{label}{tag}.pdf"))
    # Cap all-runs to 50 max, randomly selected
    n_runs_to_show = min(50, len(trajs))
    selected_runs = list(np.random.choice(len(trajs), n_runs_to_show, replace=False))
    _plot_multi_tubular(trajs, goals, n_boids, n_show=n_runs_to_show,
                        filename=os.path.join(output_dir, f"boids_tube_50_runs_{label}{tag}.pdf"),
                        specific_runs=selected_runs, hide_legend=True)

# ============================================================================
# Standalone Script
# ============================================================================

def visualize_2d(model, max_trajectory_len, n_boids, boids_obj,
                viz_mode='tubular', test_centers=None, run_tag="",
                traj_cache_path="viz_trajectories.npz", output_dir=".",
                n_show=50, specific_runs=None):
    """
    Main 2D visualization orchestrator (mirrors 3D's visualize_3d).
    Runs evaluation from cached or computed trajectories.
    Imports orchestration functions from evaluate_boids.
    """
    from boids.evaluate_boids import (
        _model_hash, _make_gnca_runner, _load_or_compute_trained_trajs,
        _compute_test_trajs
    )
    
    np.random.seed(0)
    goals = boids_obj.goal_positions

    run_gnca_traj = _make_gnca_runner(model, boids_obj, n_boids, max_trajectory_len)
    model_tag = _model_hash(model)
    trajs, selected_centers = _load_or_compute_trained_trajs(
        boids_obj, n_boids, run_gnca_traj, traj_cache_path, model_tag)
    np.random.seed(None)

    tag = f"_{run_tag}" if run_tag else ""

    if viz_mode == 'tubular':
        save_tubular_triplet(trajs, goals, n_boids, tag, 'trained', output_dir)
        trajs_test = _compute_test_trajs(boids_obj, n_boids, test_centers, run_gnca_traj, selected_centers)
        save_tubular_triplet(trajs_test, goals, n_boids, tag, 'random', output_dir)

    elif viz_mode == 'per_boid':
        _plot_per_boid(trajs, goals, n_boids, filename=os.path.join(output_dir, f"boids_auto_rand{tag}.pdf"))

    elif viz_mode == 'individual':
        _plot_individual_ranked(trajs, goals, n_boids, selected_centers, run_tag, output_dir)

    elif viz_mode == 'multi_tubular':
        _plot_multi_tubular(trajs, goals, n_boids, n_show=n_show,
                           filename=os.path.join(output_dir, f"boids_multi_tubular{tag}.pdf"),
                           specific_runs=specific_runs if specific_runs else list(range(min(n_show, len(trajs)))), hide_legend=False)

    else:
        raise ValueError(f"Unknown viz_mode '{viz_mode}'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_mode", default="tubular", choices=["tubular", "per_boid", "multi_tubular", "individual"],
                        help="Visualization style: 'tubular' (mean over all 50), 'per_boid' (one random run, all boids), 'multi_tubular' (multiple separate tubes)")
    parser.add_argument("--run_tag", default=None,
                        help="Training run tag, e.g. '50x20'. If set, infers model_path and boids_path automatically.")
    parser.add_argument("--model_path", default="gnca_model",
                        help="Path to the saved TF model directory")
    parser.add_argument("--boids_path", default="boids_tr.pkl",
                        help="Path to the saved boids_tr pickle")
    parser.add_argument("--traj_cache_path", default="viz_trajectories.npz",
                        help="Path to the trajectory cache .npz file")
    parser.add_argument("--max_trajectory_len", type=int, default=1850)
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument("--n_show", type=int, default=50,
                        help="Number of tubes to show in multi_tubular mode")
    parser.add_argument("--runs", type=int, nargs="+", default=None,
                        help="Specific run indices to plot in multi_tubular mode")
    args = parser.parse_args()

    if args.run_tag:
        model_path = f"saved_models/gnca_model_{args.run_tag}"
        boids_path = f"saved_boids_tr/boids_tr_{args.run_tag}.pkl"
        traj_cache_path = f"saved_trajectory/viz_trajectories_{args.run_tag}.npz"
        output_dir = "."
    else:
        model_path = args.model_path
        boids_path = args.boids_path
        traj_cache_path = args.traj_cache_path
        output_dir = "."

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"Loading model from '{model_path}'...")
    model = tf.saved_model.load(model_path)

    print(f"Loading boids from '{boids_path}'...")
    boids_tr = joblib.load(boids_path)

    print(f"Running 2D visualization with viz_mode='{args.viz_mode}'...")
    visualize_2d(
        model=model,
        max_trajectory_len=args.max_trajectory_len,
        n_boids=args.n_boids,
        boids_obj=boids_tr,
        viz_mode=args.viz_mode,
        run_tag=args.run_tag or "",
        traj_cache_path=traj_cache_path,
        output_dir=output_dir,
        n_show=args.n_show,
        specific_runs=args.runs,
    )


if __name__ == "__main__":
    main()
