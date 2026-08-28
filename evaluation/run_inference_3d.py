"""Standalone 3D GNCA inference and reporting.

Examples:
  python -m evaluation.run_inference_3d \
    --run_tag 500x2_3d_newl_cd_2.5_dw_2.5_500total_o01234567_bal_b256_p10_s5_fixedstepi_freshval7 \
    --octants 0 1 2 3 4 5 6 7 \
    --centers_per_octant 5 \
    --output_dir inference_3d_500x2_all_octants_fixedstepi \
    --save_multi \
    --save_individual
"""

import argparse
import glob
import os
import re
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models.gnn_ca_simple_boids_3d import GNNCASimpleBoids3D
from modules.boids_3d import Boids3D
from boids.generate_boids_cache_3d import (
    build_exclusion_zone_3d,
    _in_exclusion_zone_3d,
)


OCTANT_BOUNDS = {
    0: (0, 5, 0, 5, 0, 5),
    1: (-5, 0, 0, 5, 0, 5),
    2: (-5, 0, -5, 0, 0, 5),
    3: (0, 5, -5, 0, 0, 5),
    4: (0, 5, 0, 5, -5, 0),
    5: (-5, 0, 0, 5, -5, 0),
    6: (-5, 0, -5, 0, -5, 0),
    7: (0, 5, -5, 0, -5, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D GNCA inference on unseen centers.")
    parser.add_argument("--run_tag", required=True, help="Run tag suffix after gnca_model_3d_ / best_weights_3d_.")
    parser.add_argument("--weights_path", default="", help="Optional explicit checkpoint prefix.")
    parser.add_argument("--octants", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--centers_per_octant", type=int, default=5)
    parser.add_argument(
        "--centers_file",
        default="",
        help=(
            "Optional text/CSV/NPY file of explicit test centers. Text files must "
            "contain x y z or x y z octant per row. When provided, random center "
            "sampling arguments are ignored."
        ),
    )
    parser.add_argument("--exclusion", type=float, default=0.5)
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_success_r",
        type=float,
        default=2.0,
        help="Maximum tube radius r for a trajectory to count as successful.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default="inference_3d_outputs")
    parser.add_argument("--individual_dir", default="", help="Defaults to <output_dir>/individual.")
    parser.add_argument("--save_multi", action="store_true", default=False)
    parser.add_argument("--save_individual", action="store_true", default=False)
    parser.add_argument("--view_elev", type=float, default=24.0)
    parser.add_argument("--view_azim", type=float, default=-58.0)
    parser.add_argument("--debug_weight_paths", action="store_true", default=False)
    return parser.parse_args()


def checkpoint_prefix(path):
    for ext in (".index", ".data-00000-of-00001"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def resolve_weights_path(run_tag, explicit_path="", debug=False):
    if explicit_path:
        prefix = checkpoint_prefix(explicit_path)
        print(f"Loading explicit weights prefix: {prefix}")
        return prefix

    patterns = [
        f"saved_models/gnca_model_3d_{run_tag}",
        f"saved_models/best_weights_3d_{run_tag}",
        f"saved_models/gnca_model_3d_{run_tag}*",
        f"saved_models/best_weights_3d_{run_tag}*",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    prefixes = []
    for candidate in candidates:
        prefix = checkpoint_prefix(candidate)
        if prefix not in prefixes:
            prefixes.append(prefix)

    if debug:
        print("Weight search patterns:")
        for pattern in patterns:
            print(f"  {pattern}")
        print("Weight candidates:")
        for prefix in prefixes:
            print(f"  {prefix}")

    if not prefixes:
        raise FileNotFoundError(
            f"No weights found for run_tag '{run_tag}' in saved_models/."
        )

    prefixes.sort(key=lambda p: (0 if os.path.basename(p).startswith("gnca_model_3d_") else 1, p))
    print(f"Loading weights: {prefixes[0]}")
    return prefixes[0]


def build_model(n_boids, weights_path):
    def custom_loss(y_true, y_pred):
        n = tf.shape(y_pred)[-1]
        next_state = y_true[..., n:2 * n]
        return tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)

    model = GNNCASimpleBoids3D(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=custom_loss, run_eagerly=True)

    x_dummy = tf.zeros((n_boids, 6), dtype=tf.float32)
    a_dummy = tf.SparseTensor(
        indices=tf.zeros((0, 2), dtype=tf.int64),
        values=tf.zeros((0,), dtype=tf.float32),
        dense_shape=(n_boids, n_boids),
    )
    model([x_dummy, tf.sparse.reorder(a_dummy), tf.constant(0)], training=False)
    model.load_weights(weights_path).expect_partial()
    print("Weights loaded.")
    return model


def center_octant(center):
    for octant, (xmn, xmx, ymn, ymx, zmn, zmx) in OCTANT_BOUNDS.items():
        if xmn <= center[0] < xmx and ymn <= center[1] < ymx and zmn <= center[2] < zmx:
            return octant
    return -1


def sample_centers(octants, centers_per_octant, goals, exclusion):
    centers, labels = [], []
    exclusion_zone = build_exclusion_zone_3d(goals, exclusion)
    for octant in octants:
        xmn, xmx, ymn, ymx, zmn, zmx = OCTANT_BOUNDS[octant]
        made = 0
        while made < centers_per_octant:
            center = np.array([
                np.random.uniform(xmn, xmx),
                np.random.uniform(ymn, ymx),
                np.random.uniform(zmn, zmx),
            ], dtype=np.float32)
            if not _in_exclusion_zone_3d(center, exclusion_zone):
                centers.append(center)
                labels.append(octant)
                made += 1
    print(f"Generated {len(centers)} test centers with counts {dict((o, labels.count(o)) for o in octants)}")
    return centers, labels


def load_centers_file(path):
    """Load centers from numeric data or a run_inference_3d success report."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Centers file not found: {path}")

    if path.lower().endswith(".npy"):
        values = np.load(path)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            report_text = handle.read()
        report_matches = re.findall(
            r"run\s+\d+\s+\|\s+octant\s+(-?\d+)\s+\|\s+"
            r"center=\(([^)]+)\)",
            report_text,
        )
        if report_matches:
            centers = []
            labels = []
            for octant_text, center_text in report_matches:
                center = np.asarray(
                    [float(value.strip()) for value in center_text.split(",")],
                    dtype=np.float32,
                )
                if center.shape != (3,):
                    raise ValueError(
                        f"Invalid center in success report '{path}': {center_text}"
                    )
                centers.append(center)
                labels.append(int(octant_text))
            print(
                f"Loaded {len(centers)} explicit test centers from success report "
                f"{path} with counts "
                f"{dict((o, labels.count(o)) for o in sorted(set(labels)))}"
            )
            return centers, labels

        try:
            values = np.loadtxt(path, comments="#", delimiter=",")
        except ValueError:
            values = np.loadtxt(path, comments="#")

    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] not in (3, 4) or len(values) == 0:
        raise ValueError(
            f"Centers file must have shape (N, 3) or (N, 4); got {values.shape}."
        )

    centers = [row[:3].copy() for row in values]
    if values.shape[1] == 4:
        labels = [int(row[3]) for row in values]
        if any(label not in OCTANT_BOUNDS for label in labels):
            raise ValueError("Every explicit octant label must be between 0 and 7.")
    else:
        labels = [center_octant(center) for center in centers]

    print(
        f"Loaded {len(centers)} explicit test centers from {path} with counts "
        f"{dict((o, labels.count(o)) for o in sorted(set(labels)))}"
    )
    return centers, labels


def to_tf_sparse(a):
    indices = np.stack([a.row, a.col], axis=1)
    sparse = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape,
    )
    return tf.sparse.reorder(sparse)


def _update_online_success_3d(state, goals, closest_goal_distances, spread_radii,
                              success_threshold, max_success_r):
    """Update cumulative success statistics for one rollout state."""
    positions = state[:, :3]
    centroid = positions.mean(axis=0)
    closest_goal_distances[:] = np.minimum(
        closest_goal_distances,
        np.linalg.norm(goals - centroid[None, :], axis=1),
    )
    agent_distances = np.linalg.norm(positions - centroid[None, :], axis=1)
    spread_radii.append(float(np.percentile(agent_distances, 95)))

    if not np.all(closest_goal_distances <= success_threshold):
        return False
    current_r = float(np.percentile(spread_radii, 99))
    return bool(np.isfinite(current_r) and current_r <= max_success_r)


def run_one_trajectory(model, boids, center, n_boids, max_steps, goals,
                       success_threshold, max_success_r):
    pos, vel, _, _ = boids.get_random_init(n_boids, save_config=False, center=center)
    frames = [np.concatenate([pos, vel], axis=-1).astype(np.float32)]
    step_i = tf.constant(0)
    closest_goal_distances = np.full(len(goals), np.inf, dtype=np.float64)
    spread_radii = []

    success = _update_online_success_3d(
        frames[0], goals, closest_goal_distances, spread_radii,
        success_threshold, max_success_r,
    )

    for step in range(max_steps - 1):
        if success:
            break
        x = frames[-1]
        a = to_tf_sparse(boids.get_neighbors(x[:, :3]))
        x_next = model([tf.constant(x, dtype=tf.float32), a, step_i], training=False).numpy()
        frames.append(x_next)
        success = _update_online_success_3d(
            x_next, goals, closest_goal_distances, spread_radii,
            success_threshold, max_success_r,
        )
        if (step + 1) % 500 == 0:
            centroid = x_next[:, :3].mean(axis=0)
            print(f"    step {step+1} | centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f})")
        if success:
            current_r = float(np.percentile(spread_radii, 99))
            print(
                f"    success reached at step {step + 1}; "
                f"stopping rollout early (r={current_r:.4f})"
            )
    return np.array(frames)


def goal_distances(traj, goals):
    centroid = traj[:, :, :3].mean(axis=1)
    return np.array([np.min(np.linalg.norm(centroid - goal[None, :], axis=-1)) for goal in goals])


def tube_radius(traj):
    centroid = traj[:, :, :3].mean(axis=1)
    dists = np.linalg.norm(traj[:, :, :3] - centroid[:, None, :], axis=-1)
    return float(np.percentile(np.percentile(dists, 95, axis=1), 99))


def draw_tube_circles(ax, centroid, radius, color, n_sections=30, n_points=16,
                      lw=0.7, alpha=0.35):
    """Draw circular tube cross-sections around a centroid trajectory."""
    if not np.isfinite(radius) or radius <= 0:
        return
    step = max(1, len(centroid) // n_sections)
    theta = np.linspace(0, 2 * np.pi, n_points)
    for i in range(0, len(centroid), step):
        x_circ = centroid[i, 0] + radius * np.cos(theta)
        y_circ = centroid[i, 1] + radius * np.sin(theta)
        z_circ = np.ones_like(theta) * centroid[i, 2]
        ax.plot(x_circ, y_circ, z_circ, color=color, lw=lw, alpha=alpha)


def setup_3d_axes(ax, goals, elev, azim):
    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], c="red", marker="*", s=200, zorder=5, label="Goals")
    for g_idx, goal in enumerate(goals):
        ax.text(goal[0], goal[1], goal[2], f"G{g_idx}", fontsize=9)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)


def plot_multi(trajs, goals, centers, octants, run_tag, output_dir, elev, azim):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.hsv(np.linspace(0, 0.9, len(trajs)))
    for color, traj, center, octant in zip(colors, trajs, centers, octants):
        centroid = traj[:, :, :3].mean(axis=1)
        r = tube_radius(traj)
        ax.plot(centroid[:, 0], centroid[:, 1], centroid[:, 2], color=color, lw=1.4, alpha=0.85)
        draw_tube_circles(
            ax, centroid, r, color,
            n_sections=14,
            n_points=12,
            lw=0.45,
            alpha=0.22,
        )
        ax.scatter([center[0]], [center[1]], [center[2]], c=[color], marker="o", s=35, zorder=6)
        ax.text(center[0], center[1], center[2], f"O{octant}", fontsize=6)
    setup_3d_axes(ax, goals, elev, azim)
    ax.set_title(f"3D GNCA Inference | {len(trajs)} runs | {run_tag}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"inference3d_{run_tag}_multi.pdf")
    plt.savefig(path)
    plt.close()
    print(f"Saved combined PDF: {path}")
    return path


def plot_individual_trajectory(
    traj, goals, center, octant, run_idx, run_tag, output_dir, elev, azim
):
    """Save one tubular trajectory PDF immediately after its rollout finishes."""
    os.makedirs(output_dir, exist_ok=True)
    radius = tube_radius(traj)
    centroid = traj[:, :, :3].mean(axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        centroid[:, 0], centroid[:, 1], centroid[:, 2],
        color="#8B0000", lw=1.5, label="Centroid path",
    )
    draw_tube_circles(
        ax, centroid, radius, "#8B0000",
        n_sections=30,
        n_points=16,
        lw=0.8,
        alpha=0.4,
    )
    ax.plot([], [], [], color="#8B0000", lw=0.8, alpha=0.4, label="Flock tube")
    ax.scatter(
        [center[0]], [center[1]], [center[2]],
        c="blue", marker="*", s=100, zorder=6, label="Start",
    )
    setup_3d_axes(ax, goals, elev, azim)
    ax.set_title(
        f"Run {run_idx} | Octant {octant} | "
        f"start=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) | "
        f"r={radius:.3f}"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f"inference3d_{run_tag}_run{run_idx:03d}_oct{octant}.pdf",
    )
    plt.savefig(path)
    plt.close()
    print(f"  Saved individual PDF: {path}", flush=True)
    return path


def write_success_report(path, run_tag, centers, octants, dists, radii, threshold, max_success_r):
    reaches_all_goals = np.all(dists <= threshold, axis=1)
    stable = np.isfinite(radii) & (radii <= max_success_r)
    successes = reaches_all_goals & stable
    lines = [
        f"3D GNCA success report",
        f"run_tag: {run_tag}",
        f"success definition: closest centroid distance <= {threshold:g} to every goal and r <= {max_success_r:g}",
        f"overall: {int(successes.sum())}/{len(successes)} = {successes.mean() * 100:.2f}%",
        "",
        "per octant:",
    ]
    for octant in sorted(set(octants)):
        mask = np.array(octants) == octant
        n = int(mask.sum())
        s = int(successes[mask].sum())
        lines.append(f"  octant {octant}: {s}/{n} = {(s / n * 100 if n else 0):.2f}%")
    lines.extend(["", "per run:"])
    for i, (center, octant, run_dists, r, hit_goal, is_stable, success) in enumerate(
        zip(centers, octants, dists, radii, reaches_all_goals, stable, successes)
    ):
        dist_str = ", ".join(f"g{j}={d:.4f}" for j, d in enumerate(run_dists))
        lines.append(
            f"  run {i:03d} | octant {octant} | center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}) "
            f"| success={bool(success)} | reaches_all_goals={bool(hit_goal)} | stable_r={bool(is_stable)} | r={r:.4f} | {dist_str}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved success report: {path}")


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    individual_dir = args.individual_dir or os.path.join(args.output_dir, "individual")

    weights_path = resolve_weights_path(args.run_tag, args.weights_path, args.debug_weight_paths)
    model = build_model(args.n_boids, weights_path)

    boids = Boids3D(n_boids=args.n_boids)
    goals = boids.goal_positions
    if args.centers_file:
        centers, octant_labels = load_centers_file(args.centers_file)
    else:
        centers, octant_labels = sample_centers(
            args.octants, args.centers_per_octant, goals, args.exclusion
        )

    trajs, dist_rows, radii = [], [], []
    for idx, (center, octant) in enumerate(zip(centers, octant_labels), start=1):
        print(
            f"  Inference {idx}/{len(centers)} | octant={octant} | "
            f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})"
        )
        traj = run_one_trajectory(
            model,
            boids,
            center,
            args.n_boids,
            args.max_steps,
            goals,
            args.success_threshold,
            args.max_success_r,
        )
        trajs.append(traj)
        dists = goal_distances(traj, goals)
        r = tube_radius(traj)
        dist_rows.append(dists)
        radii.append(r)
        for goal_idx, dist in enumerate(dists):
            print(f"    goal {goal_idx} {goals[goal_idx].tolist()}: closest mean dist = {dist:.4f}")
        success = bool(
            np.all(dists <= args.success_threshold)
            and np.isfinite(r)
            and r <= args.max_success_r
        )
        print(f"    r = {r:.4f}")
        print(
            f"    success: {success} "
            f"(all goals <= {args.success_threshold:g} and r <= {args.max_success_r:g})"
        )
        if args.save_individual:
            plot_individual_trajectory(
                traj,
                goals,
                center,
                octant,
                idx - 1,
                args.run_tag,
                individual_dir,
                args.view_elev,
                args.view_azim,
            )

    dists = np.stack(dist_rows, axis=0)
    radii = np.array(radii, dtype=np.float32)

    if args.save_multi or not args.save_individual:
        plot_multi(trajs, goals, centers, octant_labels, args.run_tag, args.output_dir, args.view_elev, args.view_azim)

    report_path = os.path.join(args.output_dir, f"success_rate_{args.run_tag}.txt")
    write_success_report(
        report_path,
        args.run_tag,
        centers,
        octant_labels,
        dists,
        radii,
        args.success_threshold,
        args.max_success_r,
    )


if __name__ == "__main__":
    main()
