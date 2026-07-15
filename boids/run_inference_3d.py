"""
Standalone 3D inference script.
Loads saved GNCA 3D weights, generates fresh test centers, runs inference, and plots.

Usage: python -m boids.run_inference_3d
Edit the CONFIG block at the top to change settings.
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from modules.boids_3d import Boids3D
from models.gnn_ca_simple_boids_3d import GNNCASimpleBoids3D
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── CONFIG (edit these) ──────────────────────────────────────────────────────
RUN_TAG         = "50x5_3d_newl_cd_2.5_dw_2.5"
OCTANTS         = [6]          # 6 = bottom-left-back [-5,0]³
N_CENTERS       = 40           # total test centers (divided across OCTANTS)
EXCLUSION       = 0.5          # sphere radius around each goal to exclude
VIZ_MODE        = "individual" # "individual" or "multi_tubular"
N_BOIDS         = 100
MAX_STEPS       = 3000
OUTPUT_DIR      = "."
# ─────────────────────────────────────────────────────────────────────────────

OCTANT_BOUNDS = {
    0: ( 0,  5,  0,  5,  0,  5),
    1: (-5,  0,  0,  5,  0,  5),
    2: (-5,  0, -5,  0,  0,  5),
    3: ( 0,  5, -5,  0,  0,  5),
    4: ( 0,  5,  0,  5, -5,  0),
    5: (-5,  0,  0,  5, -5,  0),
    6: (-5,  0, -5,  0, -5,  0),
    7: ( 0,  5, -5,  0, -5,  0),
}


def _in_exclusion_zone_3d(center, goals, radius):
    for g in goals:
        if np.linalg.norm(center - g) < radius:
            return True
    return False


def _plot_individual_ranked(trajs, goals, centers, run_tag, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tube_radii = []
    for traj in trajs:
        centroid = traj[:, :, :3].mean(axis=1)
        dists = np.linalg.norm(traj[:, :, :3] - centroid[:, None, :], axis=-1)
        r = np.percentile(np.percentile(dists, 95, axis=1), 99)
        tube_radii.append(r)

    ranked = sorted(range(len(trajs)), key=lambda i: tube_radii[i])
    for rank, idx in enumerate(ranked):
        traj = trajs[idx]
        centroid = traj[:, :, :3].mean(axis=1)  # (T, 3)
        r = tube_radii[idx]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(centroid[:, 0], centroid[:, 1], centroid[:, 2],
                color='#8B0000', lw=1.5, label='centroid path')
        ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2],
                   c='red', marker='*', s=200, zorder=5, label='Goals')
        for g_idx, g in enumerate(goals):
            ax.text(g[0], g[1], g[2], f"G{g_idx}", fontsize=9)
        if centers is not None:
            c = centers[idx]
            ax.scatter([c[0]], [c[1]], [c[2]], c='blue', marker='o', s=80, zorder=6, label='Start')
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-5, 5)
        ax.set_title(f"Run {idx} | rank {rank+1}/{len(trajs)} | r={r:.3f}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"inference3d_{run_tag}_rank{rank+1:03d}_run{idx}.pdf")
        plt.savefig(fname); plt.close()
        print(f"  Saved {fname}")


def _plot_multi_tubular(trajs, goals, run_tag, output_dir):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.hsv(np.linspace(0, 0.9, len(trajs)))
    for color, traj in zip(colors, trajs):
        centroid = traj[:, :, :3].mean(axis=1)
        ax.plot(centroid[:, 0], centroid[:, 1], centroid[:, 2], color=color, lw=1.0, alpha=0.7)
        ax.scatter([centroid[0, 0]], [centroid[0, 1]], [centroid[0, 2]],
                   c='blue', marker='*', s=60, zorder=6)
    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2],
               c='red', marker='*', s=200, zorder=5, label='Goals')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-5, 5)
    ax.set_title(f"3D GNCA Inference — {len(trajs)} runs")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(output_dir, f"inference3d_{run_tag}_multi.pdf")
    plt.savefig(fname); plt.close()
    print(f"Saved {fname}")


def main():

    def custom_loss(y_true, y_pred):
        n = tf.shape(y_pred)[-1]
        next_state = y_true[..., n:2*n]
        return tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)

    # ── Load weights ─────────────────────────────────────────────────────────
    candidates = (glob.glob(f"saved_models/gnca_model_3d_{RUN_TAG}*") +
                  glob.glob(f"saved_models/best_weights_3d_{RUN_TAG}*"))
    if not candidates:
        raise FileNotFoundError(
            f"No weight files matching '*{RUN_TAG}*' in saved_models/\n"
            f"Available: {glob.glob('saved_models/*')}"
        )
    weights_path = candidates[0]
    for ext in ('.index', '.data-00000-of-00001'):
        weights_path = weights_path.replace(ext, '')
    print(f"Loading weights: {weights_path}")

    model = GNNCASimpleBoids3D(
        activation="linear", batch_norm=False, hidden=256,
        hidden_activation="relu", connectivity="cat", aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=custom_loss, run_eagerly=True)
    model.load_weights(weights_path).expect_partial()
    print("Weights loaded.")

    # ── Build boids + sample test centers ────────────────────────────────────
    boids = Boids3D(n_boids=N_BOIDS)
    goals = boids.goal_positions

    n_per_oct = max(1, N_CENTERS // len(OCTANTS))
    test_centers = []
    for o_idx in OCTANTS:
        xmn, xmx, ymn, ymx, zmn, zmx = OCTANT_BOUNDS[o_idx]
        q_centers = []
        while len(q_centers) < n_per_oct:
            c = np.array([np.random.uniform(xmn, xmx),
                          np.random.uniform(ymn, ymx),
                          np.random.uniform(zmn, zmx)])
            if not _in_exclusion_zone_3d(c, goals, EXCLUSION):
                q_centers.append(c)
        test_centers.extend(q_centers)
    print(f"Generated {len(test_centers)} test centers ({n_per_oct} per octant, octants={OCTANTS})")

    # ── Run inference ─────────────────────────────────────────────────────────

    def to_tf_sparse(a):
        indices = np.stack([a.row, a.col], axis=1)
        s = tf.SparseTensor(indices=indices, values=a.data.astype(np.float32), dense_shape=a.shape)
        return tf.sparse.reorder(s)

    step_i = tf.constant(0)
    trajs = []
    for k, center in enumerate(test_centers):
        print(f"  Inference {k+1}/{len(test_centers)} | center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})")
        pos, vel, _, _ = boids.get_random_init(N_BOIDS, save_config=False, center=center)
        frames = [np.concatenate([pos, vel], axis=-1).astype(np.float32)]
        for step in range(MAX_STEPS - 1):
            x = frames[-1]
            a = to_tf_sparse(boids.get_neighbors(x[:, :3]))
            x_next = model([tf.constant(x, dtype=tf.float32), a, step_i], training=False)
            frames.append(x_next.numpy())
            if (step + 1) % 500 == 0:
                centroid_now = x_next.numpy()[:, :3].mean(axis=0)
                print(f"    step {step+1} | centroid=({centroid_now[0]:.2f},{centroid_now[1]:.2f},{centroid_now[2]:.2f})")
        traj = np.array(frames)
        trajs.append(traj)
        centroid = traj[:, :, :3].mean(axis=1)
        for g_idx, g in enumerate(goals):
            dists = np.linalg.norm(centroid - g[None, :], axis=-1)
            print(f"    goal {g_idx} {g.tolist()}: closest mean dist = {dists.min():.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if VIZ_MODE == "individual":
        _plot_individual_ranked(trajs, goals, test_centers, RUN_TAG, OUTPUT_DIR)
    else:
        _plot_multi_tubular(trajs, goals, RUN_TAG, OUTPUT_DIR)


if __name__ == "__main__":
    main()
