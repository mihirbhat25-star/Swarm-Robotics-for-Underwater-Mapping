"""
Visualization functions and standalone script for 2D GNCA evaluation.
"""
import argparse
import h5py
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
    start = traj[0, :, :2].mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 7))
    for b in range(n_boids):
        ax.plot(
            traj[:, b, 0], traj[:, b, 1],
            color='#8B0000', lw=0.6, alpha=0.4,
            label='Boid path' if b == 0 else None
        )
    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.scatter([start[0]], [start[1]], c='blue', marker='*', s=150, zorder=5, label='Start')
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
    ax.scatter([centroid[0, 0]], [centroid[0, 1]], c='blue', marker='*', s=150, zorder=5, label='Start')
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
        ax.scatter([centroid[0, 0]], [centroid[0, 1]], c='blue', marker='*', s=100, zorder=6)

    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.scatter([], [], c='blue', marker='*', s=100, label='Start')
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
        # Ensure traj is a properly formatted float array
        traj = np.asarray(traj, dtype=np.float32)
        if len(traj.shape) != 3 or traj.shape[1] != n_boids:
            print(f"Warning: Trajectory {idx} has unexpected shape {traj.shape}, skipping")
            continue
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
        start = np.array(center) if center is not None else centroid[0]
        ax.scatter([start[0]], [start[1]], c='blue', marker='*', s=150, zorder=5, label='Start')
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

def _load_gt_trajs_from_h5(h5_path, n_per_quadrant=10, rep_idx=0):
    """Load ground-truth boids trajectories from an H5 cache,
    sampling n_per_quadrant centers from each of the 4 quadrants.

    Returns:
        trajs:            list of (T, n_boids, 4) numpy arrays  [pos_x, pos_y, vel_x, vel_y]
        selected_centers: (N, 2) numpy array of flock start centers
    """
    from boids.generate_boids_cache import QUADRANTS

    with h5py.File(h5_path, 'r') as f:
        centers       = f['centers'][:]
        cache_unique  = int(f.attrs['unique_reps'])
        cache_repeats = int(f.attrs['repeats'])

        if 'traj_lengths' in f:
            boundaries = np.concatenate([[0], np.cumsum(f['traj_lengths'][:])])
        else:
            total_samples    = f['x'].shape[0]
            samples_per_traj = total_samples // (cache_unique * cache_repeats)
            boundaries = np.arange(cache_unique * cache_repeats + 1) * samples_per_traj

        # Assign each unique center to its quadrant
        quadrant_buckets = {q: [] for q in range(len(QUADRANTS))}
        for i, c in enumerate(centers):
            for q, (xmn, xmx, ymn, ymx, _) in enumerate(QUADRANTS):
                if xmn <= c[0] <= xmx and ymn <= c[1] <= ymx:
                    quadrant_buckets[q].append(i)
                    break

        trajs, selected_centers = [], []
        for q in range(len(QUADRANTS)):
            bucket = quadrant_buckets[q]
            picked = np.random.choice(bucket, size=min(n_per_quadrant, len(bucket)), replace=False)
            for center_idx in picked:
                flat_idx = int(center_idx) * cache_repeats + rep_idx
                start = int(boundaries[flat_idx])
                end   = int(boundaries[flat_idx + 1])
                trajs.append(f['x'][start:end])   # (T, n_boids, 4)
                selected_centers.append(centers[center_idx])

    return trajs, np.array(selected_centers)


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

    # Only compute trained-center trajectories when no test_centers override is present.
    need_trained = (test_centers is None and viz_mode in ('tubular', 'per_boid', 'individual')) or \
                   (viz_mode == 'multi_tubular' and test_centers is None)
    if need_trained:
        trajs, selected_centers = _load_or_compute_trained_trajs(
            boids_obj, n_boids, run_gnca_traj, traj_cache_path, model_tag)
    else:
        trajs = []
        selected_centers = np.array(boids_obj.rand_configs) if boids_obj.rand_configs else None
    np.random.seed(None)

    tag = f"_{run_tag}" if run_tag else ""

    if viz_mode == 'tubular':
        save_tubular_triplet(trajs, goals, n_boids, tag, 'trained', output_dir)
        trajs_test = _compute_test_trajs(boids_obj, n_boids, test_centers, run_gnca_traj, selected_centers)
        save_tubular_triplet(trajs_test, goals, n_boids, tag, 'random', output_dir)

    elif viz_mode == 'per_boid':
        _plot_per_boid(trajs, goals, n_boids, filename=os.path.join(output_dir, f"boids_auto_rand{tag}.pdf"))

    elif viz_mode == 'individual':
        if test_centers is not None:
            # Use provided test centers — compute only those trajectories
            viz_trajs    = _compute_test_trajs(boids_obj, n_boids, test_centers, run_gnca_traj, selected_centers)
            viz_centers  = list(test_centers)
        else:
            viz_trajs    = trajs
            viz_centers  = list(selected_centers) if selected_centers is not None else []
        _plot_individual_ranked(viz_trajs, goals, n_boids, viz_centers, run_tag, output_dir)

    elif viz_mode == 'multi_tubular':
        # If fresh test_centers are provided, visualize those instead of trained centers.
        if test_centers is not None:
            trajs_plot = _compute_test_trajs(boids_obj, n_boids, test_centers, run_gnca_traj, selected_centers)
            centers_plot = list(test_centers)
        else:
            trajs_plot  = trajs
            centers_plot = list(selected_centers) if selected_centers is not None else []
        _plot_multi_tubular(trajs_plot, goals, n_boids, n_show=n_show,
                           filename=os.path.join(output_dir, f"boids_multi_tubular{tag}.pdf"),
                           specific_runs=specific_runs if specific_runs else list(range(min(n_show, len(trajs_plot)))), hide_legend=False)

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
    parser.add_argument("--n_per_quadrant", type=int, default=10,
                        help="Centers per quadrant: for --gt_h5 selects from H5; otherwise samples fresh unseen test centers. Set 0 to skip.")
    parser.add_argument("--viz_n_centers", type=int, default=None,
                        help="Total number of test centers to generate (used with --inference_only). "
                             "Distributed equally across --selected_quadrants.")
    parser.add_argument("--test_npz", default=None,
                        help="Path to a saved test-trajectory NPZ (e.g. saved_trajectory/viz_trajectories_<tag>_test.npz). "
                             "When set, loads those exact trajectories instead of recomputing.")
    # ── Ground-truth H5 visualization ──────────────────────────────────────
    parser.add_argument("--gt_h5", default=None,
                        help="Path to an H5 cache file. When set, visualizes ground-truth "
                             "boids trajectories instead of running GNCA inference.")
    parser.add_argument("--inference_only", action="store_true", default=False,
                        help="Load model from run_tag, generate fresh test centers, run inference, and visualize. "
                             "Requires --run_tag, --selected_quadrants, and --viz_n_centers.")
    parser.add_argument("--exclusion_size", type=float, default=0.2,
                        help="Buffer radius around goal triangle for excluding test centers (default: 0.2). "
                             "Should match the cache exclusion size.")
    parser.add_argument("--selected_quadrants", type=int, nargs="+", default=None,
                        help="Which quadrants to sample test centers from (0=Q1, 1=Q2, 2=Q3, 3=Q4). "
                             "E.g. '--selected_quadrants 1 2 3' for Q2, Q3, Q4. Used with --inference_only.")
    args = parser.parse_args()

    # ── Test-NPZ branch (no model needed — load exact saved test trajs) ────
    if args.test_npz:
        from modules.boids import Boids
        print(f"Loading test trajectories from '{args.test_npz}'...")
        cache = np.load(args.test_npz, allow_pickle=True)
        trajs   = list(cache['trajs'])
        centers = cache['centers'] if 'centers' in cache else None
        goals   = Boids().goal_positions
        tag     = args.run_tag or ""
        print(f"Loaded {len(trajs)} test trajectories.")
        if args.viz_mode == 'individual':
            _plot_individual_ranked(trajs, goals, args.n_boids, centers, tag, ".")
        elif args.viz_mode == 'multi_tubular':
            _plot_multi_tubular(trajs, goals, args.n_boids, n_show=args.n_show,
                               filename=f"boids_multi_tubular_test_{tag}.pdf" if tag else "boids_multi_tubular_test.pdf")
        else:
            print(f"viz_mode '{args.viz_mode}' not supported in test_npz mode. Use 'individual' or 'multi_tubular'.")
        return

    # ── Inference-only branch (load model, generate test centers, run inference) ──
    if args.inference_only:
        import pickle
        import glob
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
        from modules.boids import Boids
        from boids.generate_boids_cache import QUADRANTS, build_exclusion_zone
        from models.gnn_ca_simple_boids import GNNCASimpleBoids
        from boids.run_boids import custom_weighted_mse
        
        if not args.run_tag:
            raise ValueError("--inference_only requires --run_tag to load model")
        if not args.selected_quadrants:
            raise ValueError("--inference_only requires --selected_quadrants (e.g. 1 2 3)")
        if not args.viz_n_centers:
            raise ValueError("--inference_only requires --viz_n_centers")
        
        print(f">>> Inference-only mode: searching for weights with run_tag '{args.run_tag}'")
        
        # Try to find weights
        weights_candidates = glob.glob(f"best_weights_{args.run_tag}*")
        if not weights_candidates:
            print(f">>> Available weight files:")
            for f in glob.glob("best_weights_*"):
                print(f"    {f}")
            raise FileNotFoundError(f"Weights not found matching run_tag '{args.run_tag}'")
        
        weights_path = weights_candidates[0].replace('.index', '')  # remove .index suffix if present
        print(f"  Loading weights from: {weights_path}")
        
        # Build fresh model and load weights
        model = GNNCASimpleBoids(
            activation="linear",
            batch_norm=False,
            hidden=256,
            hidden_activation="relu",
            connectivity="cat",
            aggregate="mean",
        )
        model.compile(optimizer=Adam(learning_rate=1e-3), loss=custom_weighted_mse, run_eagerly=True)
        model.load_weights(weights_path)
        
        boids_tr = Boids(n_boids=args.n_boids)
        
        goals = boids_tr.goal_positions
        n_boids = args.n_boids
        
        # Generate fresh test centers from selected quadrants with exclusion
        excl = build_exclusion_zone(goals, args.exclusion_size)
        n_per_q = max(1, args.viz_n_centers // len(args.selected_quadrants))
        test_centers = []
        sampler = Boids(n_boids=n_boids)
        
        for q_idx in args.selected_quadrants:
            xmn, xmx, ymn, ymx, qlabel = QUADRANTS[q_idx]
            q_centers = []
            while len(q_centers) < n_per_q:
                _, _, _, c = sampler.get_random_init(
                    n_boids, save_config=False,
                    bounds=(xmn, xmx, ymn, ymx),
                    exclusion_zone=excl,
                )
                q_centers.append(c)
            test_centers.extend(q_centers)
        
        print(f">>> Generated {len(test_centers)} test centers: {n_per_q} per quadrant × {len(args.selected_quadrants)} quadrants")
        print(f">>> Exclusion zone: {args.exclusion_size}")
        
        # Run inference
        def _make_gnca_runner(model, boids_obj, n_boids, max_traj_len):
            def run_gnca_traj(positions, velocities):
                def to_tf_sparse(a):
                    indices = np.stack([a.row, a.col], axis=1)
                    a_tf = tf.SparseTensor(indices=indices, values=a.data.astype(np.float32), dense_shape=a.shape)
                    return tf.sparse.reorder(a_tf)
                traj = [np.concatenate([positions, velocities], axis=-1).astype(np.float32)]
                for _ in range(max_traj_len - 1):
                    x_last = traj[-1]
                    a = to_tf_sparse(boids_obj.get_neighbors(x_last[:, :2]))
                    x_next = model([tf.constant(x_last, dtype=tf.float32), a, i], training=False)
                    traj.append(x_next.numpy())
                return np.array(traj)
            return run_gnca_traj
        
        i = tf.constant(0)
        run_gnca_traj = _make_gnca_runner(model, boids_tr, n_boids, args.max_trajectory_len)
        
        print(f"\n🔄 Running inference on {len(test_centers)} test centers...")
        trajs = []
        for idx, center in enumerate(test_centers):
            pos = center + boids_tr.init_scatter * np.random.rand(n_boids, 2)
            vel = np.tile(np.array([1.0, 0.0]) * boids_tr.max_speed, (n_boids, 1))
            trajs.append(run_gnca_traj(pos, vel))
            print(f"  [{idx+1}/{len(test_centers)}] center=({center[0]:.2f}, {center[1]:.2f})")
        
        print(f"✅ Generated {len(trajs)} trajectories")
        
        # Visualize
        tag = f"_{args.run_tag}" if args.run_tag else ""
        if args.viz_mode == 'individual':
            _plot_individual_ranked(trajs, goals, n_boids, test_centers, args.run_tag, ".")
        elif args.viz_mode == 'multi_tubular':
            _plot_multi_tubular(trajs, goals, n_boids, n_show=args.viz_n_centers,
                               filename=f"boids_multi_tubular_inference{tag}.pdf")
        else:
            print(f"viz_mode '{args.viz_mode}' not supported in inference_only mode. Use 'individual' or 'multi_tubular'.")
        return

    # ── Ground-truth H5 branch (no model needed) ───────────────────────────
    if args.gt_h5:
        from modules.boids import Boids
        output_dir = "boids_gt_ind"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Loading ground-truth trajectories from '{args.gt_h5}'...")
        trajs, selected_centers = _load_gt_trajs_from_h5(
            args.gt_h5, n_per_quadrant=args.n_per_quadrant
        )
        goals = Boids().goal_positions
        print(f"Loaded {len(trajs)} trajectories. Plotting individual ranked PDFs "
              f"to ./{output_dir}/ ...")
        _plot_individual_ranked(
            trajs, goals, args.n_boids, selected_centers,
            run_tag="gt", output_dir=output_dir
        )
        return

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
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.gnn_ca_simple_boids import GNNCASimpleBoids
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'GNNCASimpleBoids': GNNCASimpleBoids},
        compile=False,
    )

    print(f"Loading boids from '{boids_path}'...")
    boids_tr = joblib.load(boids_path)

    # Build fresh per-quadrant test centers if requested
    test_centers = None
    if args.n_per_quadrant > 0:
        from modules.boids import Boids as _Boids
        from boids.generate_boids_cache import QUADRANTS, build_exclusion_zone
        _excl      = build_exclusion_zone(boids_tr.goal_positions, 0.2)
        _train_set = {tuple(np.round(c, 6)) for c in boids_tr.rand_configs}
        _sampler   = _Boids(n_boids=args.n_boids)
        test_centers = []
        for _xmn, _xmx, _ymn, _ymx, _qlabel in QUADRANTS:
            _q = []
            while len(_q) < args.n_per_quadrant:
                _, _, _, _c = _sampler.get_random_init(
                    args.n_boids, save_config=False,
                    bounds=(_xmn, _xmx, _ymn, _ymx),
                    exclusion_zone=_excl,
                )
                if tuple(np.round(_c, 6)) not in _train_set:
                    _q.append(_c)
            test_centers.extend(_q)
        print(f"Sampled {len(test_centers)} fresh test centers "
              f"({args.n_per_quadrant}/quadrant), none in training set.")

    print(f"Running 2D visualization with viz_mode='{args.viz_mode}'...")
    visualize_2d(
        model=model,
        max_trajectory_len=args.max_trajectory_len,
        n_boids=args.n_boids,
        boids_obj=boids_tr,
        viz_mode=args.viz_mode,
        test_centers=test_centers,
        run_tag=args.run_tag or "",
        traj_cache_path=traj_cache_path,
        output_dir=output_dir,
        n_show=args.n_show,
        specific_runs=args.runs,
    )


if __name__ == "__main__":
    main()
