"""
Standalone inference script.
Loads saved GNCA weights, generates fresh test centers, runs inference, and plots.

Usage: python -m boids.run_inference
Edit the CONFIG block at the top to change settings.
"""
import glob
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from modules.boids import Boids
from models.gnn_ca_simple_boids import GNNCASimpleBoids
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── CONFIG (edit these) ──────────────────────────────────────────────────────
RUN_TAG         = "200x5_newl_cd_2.5_dw_2.5"
INTERACTIVE     = True        # prompt user to type starting (x, y) coordinates
QUADRANTS       = [1, 2, 3]
N_CENTERS       = 15
EXCLUSION       = 0.5
VIZ_MODE        = "individual"
N_BOIDS         = 100
MAX_STEPS       = 2000
NEAR_GOAL_VERBOSE = False
NEAR_GOAL_RADIUS  = 1.0
SKIP_QUADRANT_INFERENCE = True
OUTPUT_DIR      = "."
# ─────────────────────────────────────────────────────────────────────────────

QUADRANT_BOUNDS = {
    0: ( 0,  5,  0,  5),
    1: (-5,  0,  0,  5),
    2: (-5,  0, -5,  0),
    3: ( 0,  5, -5,  0),
}


def _get_tube_exterior(tube):
    if isinstance(tube, MultiPolygon):
        tube = max(tube.geoms, key=lambda p: p.area)
    return tube.exterior.xy


def _plot_individual_ranked(trajs, goals, n_boids, centers, run_tag, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tube_radii = []
    for traj in trajs:
        centroid = traj[:, :, :2].mean(axis=1)
        dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)
        r = np.percentile(np.percentile(dists, 95, axis=1), 99)
        tube_radii.append(r)

    ranked = sorted(range(len(trajs)), key=lambda i: tube_radii[i])
    for rank, idx in enumerate(ranked):
        traj = trajs[idx]
        centroid = traj[:, :, :2].mean(axis=1)
        r = tube_radii[idx]
        line = LineString(centroid)
        tube = line.buffer(r)
        tx, ty = _get_tube_exterior(tube)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.fill(tx, ty, color='#8B0000', alpha=0.3, label=f'Tube r={r:.3f}')
        ax.plot(centroid[:, 0], centroid[:, 1], color='#8B0000', lw=1.5)
        ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5, label='Goals')
        if centers is not None:
            c = centers[idx]
            ax.scatter([c[0]], [c[1]], c='blue', marker='o', s=80, zorder=6, label='Start center')
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_title(f"Run {idx} | rank {rank+1}/{len(trajs)} | r={r:.3f}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"inference_{run_tag}_rank{rank+1:03d}_run{idx}.pdf")
        plt.savefig(fname)
        plt.close()
        print(f"  Saved {fname}")


def _plot_multi_tubular(trajs, goals, run_tag, output_dir):
    colors = cm.hsv(np.linspace(0, 0.9, len(trajs)))
    fig, ax = plt.subplots(figsize=(7, 7))
    for color, traj in zip(colors, trajs):
        centroid = traj[:, :, :2].mean(axis=1)
        dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)
        r = np.percentile(np.percentile(dists, 95, axis=1), 99)
        line = LineString(centroid)
        tube = line.buffer(r)
        tx, ty = _get_tube_exterior(tube)
        ax.fill(tx, ty, color=color, alpha=0.25)
        ax.plot(centroid[:, 0], centroid[:, 1], color=color, lw=1.2)
        ax.scatter([centroid[0, 0]], [centroid[0, 1]], c='blue', marker='*', s=80, zorder=6)
    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5, label='Goals')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(f"GNCA Inference — {len(trajs)} runs")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(output_dir, f"inference_{run_tag}_multi.pdf")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


def main():

    # ── Load weights ────────────────────────────────────────────────────────
    candidates = glob.glob(f"best_weights_{RUN_TAG}*") or glob.glob(f"saved_models/best_weights_{RUN_TAG}*")
    if not candidates:
        raise FileNotFoundError(f"No files matching 'best_weights_{RUN_TAG}*' in cwd: {os.getcwd()}")
    weights_path = candidates[0]
    for ext in ('.index', '.data-00000-of-00001'):
        weights_path = weights_path.replace(ext, '')
    print(f"Loading weights: {weights_path}")

    model = GNNCASimpleBoids(
        activation="linear", batch_norm=False, hidden=256,
        hidden_activation="relu", connectivity="cat", aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", run_eagerly=True)
    model.load_weights(weights_path).expect_partial()
    print("Weights loaded.")

    # ── Build boids + exclusion zone ─────────────────────────────────────────
    boids = Boids(n_boids=N_BOIDS)
    goals = boids.goal_positions
    excl  = Polygon(goals.tolist()).buffer(EXCLUSION)

    # ── Sample test centers ──────────────────────────────────────────────────
    n_per_q = max(1, N_CENTERS // len(QUADRANTS))
    test_centers = []
    for q_idx in QUADRANTS:
        xmn, xmx, ymn, ymx = QUADRANT_BOUNDS[q_idx]
        q_centers = []
        while len(q_centers) < n_per_q:
            c = np.array([np.random.uniform(xmn, xmx), np.random.uniform(ymn, ymx)])
            if not excl.contains(Point(c)):
                q_centers.append(c)
        test_centers.extend(q_centers)
    print(f"Generated {len(test_centers)} test centers ({n_per_q} per quadrant, quadrants={QUADRANTS})")

    # ── Run inference ────────────────────────────────────────────────────────
    def to_tf_sparse(a):
        indices = np.stack([a.row, a.col], axis=1)
        sp = tf.SparseTensor(indices=indices, values=a.data.astype(np.float32), dense_shape=a.shape)
        return tf.sparse.reorder(sp)

    step_i = tf.constant(0)
    trajs = []
    q_labels = {0: "Q1(top-right)", 1: "Q2(top-left)", 2: "Q3(bot-left)", 3: "Q4(bot-right)"}
    center_q_map = [QUADRANTS[k // n_per_q] for k in range(len(test_centers))]
    trajs = []

    if not SKIP_QUADRANT_INFERENCE:
        for k, center in enumerate(test_centers):
            q_idx = center_q_map[k]
            print(f"  Inference {k+1}/{len(test_centers)} | {q_labels[q_idx]} | center=({center[0]:.2f}, {center[1]:.2f})")
            pos, vel, _, _ = boids.get_random_init(N_BOIDS, save_config=False, center=center)
            frames = [np.concatenate([pos, vel], axis=-1).astype(np.float32)]
            goal0 = goals[0]
            for step in range(MAX_STEPS - 1):
                x = frames[-1]
                a = to_tf_sparse(boids.get_neighbors(x[:, :2]))
                x_next = model([tf.constant(x, dtype=tf.float32), a, step_i], training=False)
                frames.append(x_next.numpy())
                if (step + 1) % 500 == 0:
                    print(f"    step {step+1}/{MAX_STEPS}")
                if NEAR_GOAL_VERBOSE:
                    centroid_now = x_next.numpy()[:, :2].mean(axis=0)
                    mean_vel     = x_next.numpy()[:, 2:].mean(axis=0)
                    dist_g0 = np.linalg.norm(centroid_now - goal0)
                    if dist_g0 < NEAR_GOAL_RADIUS:
                        angle = np.degrees(np.arctan2(mean_vel[1], mean_vel[0]))
                        print(f"    [step {step+1}] near goal0 | centroid=({centroid_now[0]:.3f},{centroid_now[1]:.3f}) dist={dist_g0:.3f} | mean_vel=({mean_vel[0]:.4f},{mean_vel[1]:.4f}) angle={angle:.1f}°")
            traj = np.array(frames)
            trajs.append(traj)
            positions = traj[:, :, :2]
            centroid  = positions.mean(axis=1)
            for g_idx, g in enumerate(goals):
                dists = np.linalg.norm(centroid - g[None, :], axis=-1)
                print(f"    goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest mean dist = {dists.min():.4f}")
        print(f"Done. {len(trajs)} trajectories.")

    # ── Interactive mode ──────────────────────────────────────────────────────
    if INTERACTIVE:
        print(f"\n>>> Interactive mode — canvas is [-5,5]×[-5,5], goals at {goals.tolist()}")
        print("    Type 'q' to quit.\n")
        while True:
            try:
                raw = input("  Enter start center (x y): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if raw.lower() in ('q', 'quit', 'exit'):
                break
            parts = raw.split()
            if len(parts) != 2:
                print("  Please enter exactly two numbers, e.g.: -3 2")
                continue
            try:
                cx, cy = float(parts[0]), float(parts[1])
            except ValueError:
                print("  Invalid numbers, try again.")
                continue
            center = np.array([cx, cy])
            print(f"  Running GNCA from ({cx:.2f}, {cy:.2f}) for {MAX_STEPS} steps...")
            pos, vel, _, _ = boids.get_random_init(N_BOIDS, save_config=False, center=center)
            frames = [np.concatenate([pos, vel], axis=-1).astype(np.float32)]
            for step in range(MAX_STEPS - 1):
                x = frames[-1]
                a = to_tf_sparse(boids.get_neighbors(x[:, :2]))
                x_next = model([tf.constant(x, dtype=tf.float32), a, step_i], training=False)
                frames.append(x_next.numpy())
            traj = np.array(frames)
            centroid = traj[:, :, :2].mean(axis=1)
            for g_idx, g in enumerate(goals):
                dists = np.linalg.norm(centroid - g[None, :], axis=-1)
                print(f"    goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest mean dist = {dists.min():.4f}")
            # Plot
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(centroid[:, 0], centroid[:, 1], lw=1.5, color="#8B0000")
            ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5)
            for g_idx, g in enumerate(goals):
                ax.annotate(f"G{g_idx}", g, textcoords="offset points", xytext=(6, 6), fontsize=10)
            ax.scatter([cx], [cy], c='blue', marker='o', s=100, zorder=6, label=f'Start ({cx:.1f},{cy:.1f})')
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.set_title(f"GNCA from ({cx:.2f},{cy:.2f})")
            ax.legend()
            plt.tight_layout()
            fname = os.path.join(OUTPUT_DIR, f"interactive_{cx:.1f}_{cy:.1f}.pdf")
            plt.savefig(fname); plt.close()
            print(f"  Saved {fname}\n")

    # ── DEBUG: start flock AT goal 0 and see if GNCA navigates onward ────────
    # print("\n>>> DEBUG: initializing flock at goal 0 (3,-3) ...")
    # goal0_center = np.array([3.0, -3.0])
    # pos_g0, vel_g0, _, _ = boids.get_random_init(N_BOIDS, save_config=False, center=goal0_center)
    # frames_g0 = [np.concatenate([pos_g0, vel_g0], axis=-1).astype(np.float32)]
    # for step in range(MAX_STEPS - 1):
    #     x = frames_g0[-1]
    #     a = to_tf_sparse(boids.get_neighbors(x[:, :2]))
    #     x_next = model([tf.constant(x, dtype=tf.float32), a, step], training=False)
    #     frames_g0.append(x_next.numpy())
    #     if (step + 1) % 500 == 0:
    #         centroid_now = x_next.numpy()[:, :2].mean(axis=0)
    #         print(f"    step {step+1} | centroid=({centroid_now[0]:.3f},{centroid_now[1]:.3f})")
    # traj_g0 = np.array(frames_g0)
    # centroid_g0 = traj_g0[:, :, :2].mean(axis=1)
    # for g_idx, g in enumerate(goals):
    #     dists = np.linalg.norm(centroid_g0 - g[None, :], axis=-1)
    #     print(f"  goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest mean dist = {dists.min():.4f}")
    # # plot it
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.plot(centroid_g0[:, 0], centroid_g0[:, 1], lw=1.5, color="#8B0000")
    # ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5)
    # for g_idx, g in enumerate(goals):
    #     ax.annotate(f"G{g_idx}", g, textcoords="offset points", xytext=(6, 6), fontsize=10)
    # ax.scatter([goal0_center[0]], [goal0_center[1]], c='blue', marker='o', s=100, zorder=6, label='Start (goal 0)')
    # ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    # ax.set_aspect('equal')
    # ax.set_title("GNCA starting AT goal 0 — does it reach goal 1?")
    # ax.legend()
    # plt.tight_layout()
    # fname = os.path.join(OUTPUT_DIR, "debug_start_at_goal0.pdf")
    # plt.savefig(fname); plt.close()
    # print(f"  Saved {fname}")

    # ── Ground-truth boids on same centers ───────────────────────────────────
    if not SKIP_QUADRANT_INFERENCE:
        print("\n>>> Running ground-truth boids on same centers for comparison...")
        gt_trajs = []
        for k, center in enumerate(test_centers):
            q_idx = center_q_map[k]
            print(f"  GT {k+1}/{len(test_centers)} | {q_labels[q_idx]} | center=({center[0]:.2f}, {center[1]:.2f})")
            gt_boids = Boids(n_boids=N_BOIDS)
            history = gt_boids.generate_trajectory(save_config=False, random_init=center)
            pos_arr = history["positions"]
            vel_arr = history["velocities"]
            traj_gt = np.concatenate([pos_arr, vel_arr], axis=-1)
            gt_trajs.append(traj_gt)
            if NEAR_GOAL_VERBOSE:
                goal0 = goals[0]
                for step in range(len(traj_gt)):
                    centroid_now = traj_gt[step, :, :2].mean(axis=0)
                    mean_vel     = traj_gt[step, :, 2:].mean(axis=0)
                    dist_g0 = np.linalg.norm(centroid_now - goal0)
                    if dist_g0 < NEAR_GOAL_RADIUS:
                        angle = np.degrees(np.arctan2(mean_vel[1], mean_vel[0]))
                        print(f"    [GT step {step}] near goal0 | centroid=({centroid_now[0]:.3f},{centroid_now[1]:.3f}) dist={dist_g0:.3f} | mean_vel=({mean_vel[0]:.4f},{mean_vel[1]:.4f}) angle={angle:.1f}°")
            centroid_gt = traj_gt[:, :, :2].mean(axis=1)
            for g_idx, g in enumerate(goals):
                dists = np.linalg.norm(centroid_gt - g[None, :], axis=-1)
                print(f"    goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest mean dist = {dists.min():.4f}")

        for k, (center, traj_gnca, traj_gt) in enumerate(zip(test_centers, trajs, gt_trajs)):
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            for ax, traj, title in [(axes[0], traj_gnca, "GNCA"), (axes[1], traj_gt, "Ground Truth")]:
                centroid = traj[:, :, :2].mean(axis=1)
                ax.plot(centroid[:, 0], centroid[:, 1], lw=1.5, color="#8B0000", label="centroid path")
                ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5, label='Goals')
                for g_idx, g in enumerate(goals):
                    ax.annotate(f"G{g_idx}", g, textcoords="offset points", xytext=(6, 6), fontsize=9)
                ax.scatter([centroid[0, 0]], [centroid[0, 1]], c='blue', marker='o', s=80, zorder=6, label='Start')
                ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
                ax.set_aspect('equal')
                ax.set_title(title)
                ax.legend(fontsize=8)
            fig.suptitle(f"Center ({center[0]:.2f},{center[1]:.2f}) | {q_labels[center_q_map[k]]}")
            plt.tight_layout()
            fname = os.path.join(OUTPUT_DIR, f"compare_gnca_vs_gt_center{k}.pdf")
            plt.savefig(fname); plt.close()
            print(f"Saved {fname}")

        if VIZ_MODE == "individual":
            _plot_individual_ranked(trajs, goals, N_BOIDS, test_centers, RUN_TAG, OUTPUT_DIR)
        else:
            _plot_multi_tubular(trajs, goals, RUN_TAG, OUTPUT_DIR)


if __name__ == "__main__":
    main()
