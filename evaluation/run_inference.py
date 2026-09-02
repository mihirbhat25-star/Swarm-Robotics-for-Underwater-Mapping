"""Standalone 2D GNCA inference and reporting.

Loads saved GNCA weights, generates fresh test centers, runs inference, and plots.

Canonical usage: ``python -m evaluation.run_inference``.
The original ``python -m boids.run_inference`` command remains compatible.
"""
import glob
import os
import argparse
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from modules.boids import Boids
from models.gnn_ca_simple_boids import GNNCASimpleBoids
from models.gnn_ca_goal_conditioned_boids import GoalConditionedGNNCABoids
from modules.waypoints import OnlineWaypointManager, goal_conditioned_state
from evaluation.visualize_boids import (
    _get_tube_exterior,
    _plot_individual_ranked as _plot_individual_ranked_shared,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.geometry import LineString, Point, Polygon

QUADRANT_BOUNDS = {
    0: ( 0,  5,  0,  5),
    1: (-5,  0,  0,  5),
    2: (-5,  0, -5,  0),
    3: ( 0,  5, -5,  0),
}


def _count_collision_events(traj, threshold, warmup_steps=0):
    """Count post-warmup collision onsets and unique pairs involved."""
    n_agents = traj.shape[1]
    upper = np.triu(np.ones((n_agents, n_agents), dtype=bool), k=1)
    start = min(max(0, warmup_steps), len(traj))

    def collision_mask(positions):
        delta = positions[:, None, :] - positions[None, :, :]
        return (np.linalg.norm(delta, axis=-1) < threshold) & upper

    previous = (
        collision_mask(traj[start - 1, :, :2])
        if start > 0 else np.zeros((n_agents, n_agents), dtype=bool)
    )
    total = 0
    event_pairs = np.zeros_like(previous)
    for positions in traj[start:, :, :2]:
        colliding = collision_mask(positions)
        onsets = colliding & ~previous
        total += int(np.count_nonzero(onsets))
        event_pairs |= onsets
        previous = colliding
    return total, int(np.count_nonzero(event_pairs))


def _plot_individual_ranked(trajs, goals, n_boids, centers, run_tag, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _plot_individual_ranked_shared(
        trajs,
        goals,
        n_boids,
        selected_centers=centers,
        run_tag=run_tag,
        output_dir=output_dir,
    )


def _plot_multi_tubular(trajs, goals, run_tag, output_dir):
    colors = cm.hsv(np.linspace(0, 0.9, len(trajs)))
    radii = []
    fig, ax = plt.subplots(figsize=(7, 7))
    for color, traj in zip(colors, trajs):
        centroid = traj[:, :, :2].mean(axis=1)
        dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)
        r = np.percentile(np.percentile(dists, 95, axis=1), 99)
        radii.append(r)
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
    if radii:
        ax.set_title(f"GNCA Inference - {len(trajs)} tubes | mean r={np.mean(radii):.3f}, max r={np.max(radii):.3f}")
    else:
        ax.set_title(f"GNCA Inference - {len(trajs)} tubes")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(output_dir, f"inference_{run_tag}_multi.pdf")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


def _trajectory_tube_radius(traj):
    """Return the same robust flock radius used by the tubular plots."""
    centroid = traj[:, :, :2].mean(axis=1)
    boid_dists = np.linalg.norm(
        traj[:, :, :2] - centroid[:, None, :], axis=-1
    )
    return float(np.percentile(np.percentile(boid_dists, 95, axis=1), 99))


def _goal_segment_tube_radii(traj, switch_steps):
    """Compute one robust tube radius for each active-waypoint segment."""
    final_frame = len(traj) - 1
    segment_start = 0
    radii = []
    for switch_step in switch_steps:
        segment_stop = min(max(int(switch_step), segment_start), final_frame)
        radii.append(
            _trajectory_tube_radius(traj[segment_start:segment_stop + 1])
        )
        segment_start = segment_stop
    if segment_start < final_frame or not radii:
        radii.append(_trajectory_tube_radius(traj[segment_start:]))
    return radii


def _plot_tube_on_axis(ax, traj, color="#8B0000", label="centroid path"):
    centroid = traj[:, :, :2].mean(axis=1)
    r = _trajectory_tube_radius(traj)
    line = LineString(centroid)
    tube = line.buffer(r)
    tx, ty = _get_tube_exterior(tube)
    ax.fill(tx, ty, color=color, alpha=0.3, label=f"99% tube (r={r:.3f})")
    ax.plot(centroid[:, 0], centroid[:, 1], lw=1.5, color=color, label=label)
    ax.scatter([centroid[0, 0]], [centroid[0, 1]], c='blue', marker='*', s=120, zorder=6, label='Start')
    return centroid, r


def _clean_checkpoint_prefix(path):
    path = str(path)
    for ext in ('.index', '.data-00000-of-00001'):
        path = path.replace(ext, '')
    return path


def _safe_model_name(path):
    base = os.path.basename(_clean_checkpoint_prefix(path))
    base = re.sub(r"^(best_weights_|gnca_model_)", "", base)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


def _model_size_suffix(model_name):
    match = re.search(r"\d+x\d+", model_name)
    return match.group(0) if match else "model"


def _find_weight_prefixes(weights_dir):
    prefixes = []
    for pattern in (os.path.join(weights_dir, "best_weights_*"),
                    os.path.join(weights_dir, "gnca_model_*")):
        for match in sorted(glob.glob(pattern)):
            prefix = _clean_checkpoint_prefix(match)
            if prefix not in prefixes:
                prefixes.append(prefix)
    by_tag = {}
    for prefix in prefixes:
        name = os.path.basename(prefix)
        tag = re.sub(r"^(best_weights_|gnca_model_)", "", name)
        current = by_tag.get(tag)
        # Prefer best_weights checkpoint pairs; gnca_model entries may be SavedModel dirs.
        if current is None or name.startswith("best_weights_"):
            by_tag[tag] = prefix
    return [by_tag[tag] for tag in sorted(by_tag)]


def _find_single_weights(run_tag):
    search_patterns = [
        f"best_weights_{run_tag}*",
        f"saved_models/best_weights_{run_tag}*",
    ]
    print("2D inference weight search:")
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        print(f"  pattern: {pattern}")
        print(f"  matches: {matches if matches else 'NONE'}")
    candidates = glob.glob(search_patterns[0]) or glob.glob(search_patterns[1])
    if not candidates:
        raise FileNotFoundError(f"No files matching 'best_weights_{run_tag}*' in cwd: {os.getcwd()}")
    return _clean_checkpoint_prefix(candidates[0])


def _load_model(weights_path, task="fixed_waypoints"):
    weights_path = _clean_checkpoint_prefix(weights_path)
    print(f"Loading weights: {weights_path}")
    model_class = (
        GoalConditionedGNNCABoids
        if task == "online_goals"
        else GNNCASimpleBoids
    )
    model = model_class(
        activation="linear", batch_norm=False, hidden=256,
        hidden_activation="relu", connectivity="cat", aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", run_eagerly=True)
    model.load_weights(weights_path).expect_partial()
    print("Weights loaded.")
    return model


def _manual_goal_array(values):
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 4 or len(values) % 2:
        raise ValueError("--manual_goals needs at least two x y pairs.")
    return values.reshape(-1, 2)


def _run_online_goal_trajectory(model, boids, center, args, rng, manual_goals=None):
    """Roll out a goal-conditioned GNCA under an external waypoint manager."""
    position, velocity, _, _ = boids.get_random_init(
        args.n_boids, save_config=False, center=np.asarray(center)
    )
    physical = np.concatenate((position, velocity), axis=-1).astype(np.float32)
    frames = [physical.copy()]
    switch_steps = []
    closest_distances = []
    reached = 0

    if manual_goals is None:
        manager = OnlineWaypointManager(
            rng=rng,
            n_waypoints=args.online_goal_count,
            bounds=tuple(args.goal_bounds),
            min_distance=args.goal_min_distance,
            arrival_radius=args.success_threshold,
        )
        active_goal = manager.start(position.mean(axis=0))
        sampled_goals = [active_goal.copy()]
    else:
        manager = None
        sampled_goals = [goal.copy() for goal in manual_goals]
        active_goal = sampled_goals[0]

    closest_current = float(
        np.linalg.norm(position - active_goal[None, :], axis=-1).mean()
    )
    for step in range(1, args.max_steps + 1):
        conditioned = goal_conditioned_state(physical, active_goal)
        adjacency = _to_tf_sparse(boids.get_neighbors(physical[:, :2]))
        physical = model(
            [tf.constant(conditioned), adjacency, tf.constant(0)],
            training=False,
        ).numpy()
        frames.append(physical.copy())
        mean_distance = float(
            np.linalg.norm(
                physical[:, :2] - active_goal[None, :], axis=-1
            ).mean()
        )
        closest_current = min(closest_current, mean_distance)
        if step % 500 == 0:
            centroid = physical[:, :2].mean(axis=0)
            print(
                f"    step {step}/{args.max_steps} | "
                f"centroid=({centroid[0]:.2f},{centroid[1]:.2f}) | "
                f"active_goal=({active_goal[0]:.2f},{active_goal[1]:.2f})"
            )

        if mean_distance > args.success_threshold:
            continue

        closest_distances.append(closest_current)
        reached += 1
        switch_steps.append(step)
        if manual_goals is not None:
            if reached >= len(sampled_goals):
                break
            active_goal = sampled_goals[reached]
        else:
            next_goal, switched, finished = manager.update(physical[:, :2])
            if finished:
                break
            if not switched:
                raise RuntimeError("Waypoint manager did not switch after arrival.")
            active_goal = next_goal
            sampled_goals.append(active_goal.copy())
        closest_current = float(
            np.linalg.norm(
                physical[:, :2] - active_goal[None, :], axis=-1
            ).mean()
        )

    requested = len(sampled_goals) if manual_goals is not None else args.online_goal_count
    trajectory = np.asarray(frames)
    goal_tube_radii = _goal_segment_tube_radii(trajectory, switch_steps)
    max_goal_tube_radius = max(goal_tube_radii, default=float("inf"))
    reached_all_goals = reached == requested
    cohesive = (
        np.all(np.isfinite(goal_tube_radii))
        and max_goal_tube_radius <= args.max_tube_radius
    )
    return {
        "trajectory": trajectory,
        "goals": np.asarray(sampled_goals),
        "switch_steps": switch_steps,
        "closest_distances": closest_distances,
        "reached": reached,
        "requested": requested,
        "goal_tube_radii": goal_tube_radii,
        "max_goal_tube_radius": max_goal_tube_radius,
        "reached_all_goals": reached_all_goals,
        "success": reached_all_goals and cohesive,
    }


def _save_online_goal_pdf(result, center, run_index, run_tag, output_dir):
    trajectory = result["trajectory"]
    goals = result["goals"]
    fig, axis = plt.subplots(figsize=(7, 7))
    _plot_tube_on_axis(axis, trajectory, label="GNCA centroid")
    radius = result["max_goal_tube_radius"]
    colors = cm.viridis(np.linspace(0.1, 0.9, len(goals)))
    for goal_index, (goal, color) in enumerate(zip(goals, colors)):
        axis.scatter(
            goal[0], goal[1], color=color, marker="*", s=180, zorder=7
        )
        axis.annotate(
            f"G{goal_index + 1}", goal, xytext=(6, 6),
            textcoords="offset points",
        )
    axis.set_xlim(-5, 5)
    axis.set_ylim(-5, 5)
    axis.set_aspect("equal")
    axis.set_title(
        f"Online GNCA | start=({center[0]:.2f},{center[1]:.2f}) | "
        f"goals={result['reached']}/{result['requested']} | "
        f"max goal r={radius:.3f}"
    )
    axis.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"online_{run_tag}_run{run_index:03d}.pdf"
    )
    plt.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def _save_online_goal_summary(results, run_tag, output_dir):
    fig, axis = plt.subplots(figsize=(8, 8))
    colors = cm.hsv(np.linspace(0, 0.9, len(results)))
    for result, color in zip(results, colors):
        _plot_tube_on_axis(
            axis,
            result["trajectory"],
            color=color,
            label=None,
        )
        axis.scatter(
            result["goals"][:, 0], result["goals"][:, 1],
            color=color, marker="x", s=30,
        )
    axis.set_xlim(-5, 5)
    axis.set_ylim(-5, 5)
    axis.set_aspect("equal")
    axis.set_title(f"Online GNCA - {len(results)} unseen runs")
    plt.tight_layout()
    path = os.path.join(output_dir, f"online_{run_tag}_all.pdf")
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _run_online_goal_inference(args, model):
    if args.mode != "single":
        raise ValueError("online_goals currently uses --mode single.")
    manual_goals = _manual_goal_array(args.manual_goals)
    if manual_goals is not None and args.n_centers != 1:
        print(
            ">>> Manual goal sequence will be evaluated from each sampled start center."
        )
    rng = np.random.default_rng(args.seed)
    centers = []
    center_quadrants = []
    base, remainder = divmod(args.n_centers, len(args.quadrants))
    for quadrant_index, quadrant in enumerate(args.quadrants):
        x_min, x_max, y_min, y_max = QUADRANT_BOUNDS[quadrant]
        count = base + (1 if quadrant_index < remainder else 0)
        for _ in range(count):
            centers.append(np.array([
                rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)
            ], dtype=np.float32))
            center_quadrants.append(quadrant)
    np.save(os.path.join(args.output_dir, "online_test_centers.npy"), centers)

    boids = Boids(n_boids=args.n_boids, perception=args.perception)
    individual_dir = os.path.join(args.output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)
    results = []
    for run_index, (center, quadrant) in enumerate(
        zip(centers, center_quadrants)
    ):
        print(
            f"  Online inference {run_index + 1}/{len(centers)} | "
            f"quadrant={quadrant} | center=({center[0]:.2f},{center[1]:.2f})"
        )
        result = _run_online_goal_trajectory(
            model,
            boids,
            center,
            args,
            np.random.default_rng(args.seed + 10_000 + run_index),
            manual_goals=manual_goals,
        )
        results.append(result)
        print(
            f"    reached {result['reached']}/{result['requested']} goals | "
            f"goal_r={[round(r, 4) for r in result['goal_tube_radii']]} | "
            f"max_r={result['max_goal_tube_radius']:.4f} | "
            f"success={result['success']}"
        )
        _save_online_goal_pdf(
            result, center, run_index, args.run_tag, individual_dir
        )

    _save_online_goal_summary(results, args.run_tag, args.output_dir)

    successes = sum(result["success"] for result in results)
    report = [
        "2D online-goal GNCA success report",
        f"run_tag: {args.run_tag}",
        (
            "success definition: mean per-agent distance <= "
            f"{args.success_threshold:g} for every requested waypoint and "
            f"max per-goal tube radius max(r_k) <= {args.max_tube_radius:g}"
        ),
        f"overall: {successes}/{len(results)} = "
        f"{100 * successes / max(len(results), 1):.2f}%",
        "",
    ]
    report.append("per quadrant:")
    for quadrant in args.quadrants:
        quadrant_results = [
            result
            for result, result_quadrant in zip(results, center_quadrants)
            if result_quadrant == quadrant
        ]
        quadrant_successes = sum(
            result["success"] for result in quadrant_results
        )
        quadrant_total = len(quadrant_results)
        report.append(
            f"  quadrant {quadrant}: {quadrant_successes}/{quadrant_total} = "
            f"{100 * quadrant_successes / max(quadrant_total, 1):.2f}%"
        )
    report.append("")
    for run_index, (center, quadrant, result) in enumerate(
        zip(centers, center_quadrants, results)
    ):
        report.append(
            f"run {run_index:03d} | quadrant={quadrant} | "
            f"center=({center[0]:.4f},{center[1]:.4f}) | "
            f"reached={result['reached']}/{result['requested']} | "
            f"goal_r={[round(r, 4) for r in result['goal_tube_radii']]} | "
            f"max_r={result['max_goal_tube_radius']:.4f} | "
            f"success={result['success']} | "
            f"goals={result['goals'].tolist()}"
        )
    report_path = os.path.join(args.output_dir, "online_success_rate.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    print(f"Saved {report_path}")


def _sample_centers_per_quadrant(goals, n_per_quadrant=10, quadrants=(0, 1, 2, 3), exclusion=0.5, seed=123):
    rng = np.random.default_rng(seed)
    excl = Polygon(goals.tolist()).buffer(exclusion)
    centers, center_q_map = [], []
    for q_idx in quadrants:
        xmn, xmx, ymn, ymx = QUADRANT_BOUNDS[q_idx]
        picked = 0
        while picked < n_per_quadrant:
            c = np.array([rng.uniform(xmn, xmx), rng.uniform(ymn, ymx)])
            if not excl.contains(Point(c)):
                centers.append(c)
                center_q_map.append(q_idx)
                picked += 1
    return centers, center_q_map


def _to_tf_sparse(a):
    indices = np.stack([a.row, a.col], axis=1)
    sp = tf.SparseTensor(indices=indices, values=a.data.astype(np.float32), dense_shape=a.shape)
    return tf.sparse.reorder(sp)


def _run_one_trajectory_2d(model, boids, center, max_steps, n_boids, goals,
                           success_threshold, verbose_prefix="",
                           near_goal_verbose=False, near_goal_radius=1.0):
    """Roll out until every goal is reached or the step budget is exhausted."""
    pos, vel, _, _ = boids.get_random_init(n_boids, save_config=False, center=center)
    frames = [np.concatenate([pos, vel], axis=-1).astype(np.float32)]
    closest_goal_distances = np.linalg.norm(
        goals - frames[0][:, :2].mean(axis=0)[None, :], axis=1
    )
    step_i = tf.constant(0)

    for step in range(max_steps - 1):
        if np.all(closest_goal_distances <= success_threshold):
            break
        x = frames[-1]
        a = _to_tf_sparse(boids.get_neighbors(x[:, :2]))
        x_next = model([tf.constant(x, dtype=tf.float32), a, step_i], training=False).numpy()
        frames.append(x_next)
        centroid = x_next[:, :2].mean(axis=0)
        closest_goal_distances = np.minimum(
            closest_goal_distances,
            np.linalg.norm(goals - centroid[None, :], axis=1),
        )
        if (step + 1) % 500 == 0:
            print(f"{verbose_prefix}  step {step+1}/{max_steps}")
        if near_goal_verbose:
            mean_vel = x_next[:, 2:].mean(axis=0)
            dist_g0 = np.linalg.norm(centroid - goals[0])
            if dist_g0 < near_goal_radius:
                angle = np.degrees(np.arctan2(mean_vel[1], mean_vel[0]))
                print(
                    f"{verbose_prefix}  [step {step+1}] near goal0 | "
                    f"centroid=({centroid[0]:.3f},{centroid[1]:.3f}) "
                    f"dist={dist_g0:.3f} | mean_vel=({mean_vel[0]:.4f},"
                    f"{mean_vel[1]:.4f}) angle={angle:.1f}°"
                )
        if np.all(closest_goal_distances <= success_threshold):
            print(f"{verbose_prefix}  success reached at step {step + 1}; stopping rollout early")

    return np.array(frames)


def _run_gnca_trajs(model, boids, centers, max_steps, n_boids,
                    success_threshold=0.5, verbose_prefix=""):
    trajs = []
    goals = boids.goal_positions
    for k, center in enumerate(centers):
        print(f"{verbose_prefix}Inference {k+1}/{len(centers)} | center=({center[0]:.2f}, {center[1]:.2f})")
        traj = _run_one_trajectory_2d(
            model, boids, center, max_steps, n_boids, goals,
            success_threshold, verbose_prefix=verbose_prefix,
        )
        trajs.append(traj)
        centroid = traj[:, :, :2].mean(axis=1)
        for g_idx, g in enumerate(goals):
            dists = np.linalg.norm(centroid - g[None, :], axis=-1)
            print(f"{verbose_prefix}  goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest mean dist = {dists.min():.4f}")
    return trajs


def _closest_goal_distances_2d(traj, goals):
    centroid = traj[:, :, :2].mean(axis=1)
    return [
        float(np.linalg.norm(centroid - goal[None, :], axis=-1).min())
        for goal in goals
    ]


def _write_success_rate_2d(trajs, goals, centers, center_q_map, output_path, radius=0.5):
    successes = []
    lines = [
        f"2D success criterion: closest centroid distance <= {radius:g} for all three goals",
        f"Total trajectories: {len(trajs)}",
        "",
    ]
    for idx, traj in enumerate(trajs):
        dists = _closest_goal_distances_2d(traj, goals)
        success = all(d <= radius for d in dists)
        successes.append(success)
        center = centers[idx]
        quadrant = center_q_map[idx] if center_q_map is not None else "NA"
        dist_str = ", ".join(f"goal{g_idx}={dist:.4f}" for g_idx, dist in enumerate(dists))
        lines.append(
            f"run {idx:02d} | quadrant={quadrant} | center=({center[0]:.4f},{center[1]:.4f}) | "
            f"success={success} | {dist_str}"
        )

    n_success = int(sum(successes))
    rate = 100.0 * n_success / len(successes) if successes else 0.0
    lines.insert(2, f"Successes: {n_success}/{len(successes)} ({rate:.2f}%)")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return n_success, len(successes), rate


def _run_comparison(args):
    boids = Boids(n_boids=args.n_boids, perception=args.perception)
    goals = boids.goal_positions
    weights = args.models if args.models else _find_weight_prefixes(args.weights_dir)
    if not weights:
        raise FileNotFoundError(f"No checkpoints found in {args.weights_dir}")
    print("Comparison mode weights:")
    for w in weights:
        prefix = _clean_checkpoint_prefix(w)
        print(f"  {prefix} | index={os.path.exists(prefix + '.index')} data={os.path.exists(prefix + '.data-00000-of-00001')} bare={os.path.exists(prefix)}")

    centers, center_q_map = _sample_centers_per_quadrant(
        goals,
        n_per_quadrant=args.centers_per_quadrant,
        quadrants=args.quadrants,
        exclusion=args.exclusion,
        seed=args.seed,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "comparison_centers.npy"), np.array(centers))
    np.save(os.path.join(args.output_dir, "comparison_center_quadrants.npy"), np.array(center_q_map))
    print(f"Generated {len(centers)} shared unseen centers ({args.centers_per_quadrant} per quadrant).")

    summary_lines = [
        f"2D comparison success criterion: closest centroid distance <= {args.success_threshold:g} for all three goals",
        f"Shared centers: {len(centers)}",
        "",
    ]
    for weights_path in weights:
        model_name = _safe_model_name(weights_path)
        model_dir = os.path.join(args.output_dir, model_name)
        individual_dir = os.path.join(model_dir, "individual")
        os.makedirs(model_dir, exist_ok=True)
        print(f"\n=== Running comparison model: {model_name} ===")
        model = _load_model(weights_path)
        trajs = _run_gnca_trajs(
            model,
            boids,
            centers,
            args.max_steps,
            args.n_boids,
            success_threshold=args.success_threshold,
            verbose_prefix="  ",
        )
        _plot_individual_ranked(trajs, goals, args.n_boids, centers, model_name, individual_dir)
        _plot_multi_tubular(trajs, goals, model_name, model_dir)
        n_success, total, rate = _write_success_rate_2d(
            trajs,
            goals,
            centers,
            center_q_map,
            os.path.join(model_dir, f"success_rate_{_model_size_suffix(model_name)}.txt"),
            radius=args.success_threshold,
        )
        summary_lines.append(f"{model_name}: {n_success}/{total} successful ({rate:.2f}%)")
        print(f"Saved comparison outputs for {model_name} under {model_dir}")

    with open(os.path.join(args.output_dir, "success_rate_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="2D GNCA inference")
    parser.add_argument(
        "--task",
        choices=["fixed_waypoints", "online_goals"],
        default="fixed_waypoints",
    )
    parser.add_argument("--mode", choices=["single", "comparison"], default="single")
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--weights_dir", default="important_weights_2d")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Checkpoint prefixes/files to compare. Defaults to all models in --weights_dir.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--centers_per_quadrant", type=int, default=10)
    parser.add_argument("--quadrants", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--n_centers", type=int, default=15,
                        help="Single-mode total random centers. Ignored by comparison mode.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_tube_radius",
        type=float,
        default=1.0,
        help=(
            "Maximum robust flock-tube radius allowed in every active-goal "
            "segment for online-goal success (default: 1.0)."
        ),
    )
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument(
        "--perception",
        type=float,
        default=0.1,
        help="Neighbor radius used to construct the inference graph.",
    )
    parser.add_argument("--exclusion", type=float, default=0.5)
    parser.add_argument(
        "--viz_mode",
        choices=["individual", "multi_tubular"],
        default="individual",
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--skip_quadrant_inference", action="store_true")
    parser.add_argument("--near_goal_verbose", action="store_true")
    parser.add_argument("--near_goal_radius", type=float, default=1.0)
    parser.add_argument("--collision_warmup_steps", type=int, default=100)
    parser.add_argument("--print_collisions", action="store_true", default=False)
    parser.add_argument("--compare_ground_truth", action="store_true", default=False)
    parser.add_argument("--online_goal_count", type=int, default=5)
    parser.add_argument(
        "--goal_bounds", nargs=4, type=float,
        default=[-4.5, 4.5, -4.5, 4.5],
    )
    parser.add_argument("--goal_min_distance", type=float, default=2.5)
    parser.add_argument(
        "--manual_goals",
        nargs="*",
        type=float,
        default=None,
        help="Optional online goal sequence: x1 y1 x2 y2 ...",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            "comparison_inference_2d" if args.mode == "comparison" else "."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "comparison":
        _run_comparison(args)
        return

    # ── Load weights ────────────────────────────────────────────────────────
    if not args.run_tag:
        raise ValueError("Single mode requires --run_tag.")
    np.random.seed(args.seed)
    model = _load_model(
        _find_single_weights(args.run_tag), task=args.task
    )
    if args.task == "online_goals":
        _run_online_goal_inference(args, model)
        return

    # ── Build boids + exclusion zone ─────────────────────────────────────────
    boids = Boids(n_boids=args.n_boids, perception=args.perception)
    goals = boids.goal_positions
    excl  = Polygon(goals.tolist()).buffer(args.exclusion)

    # ── Sample test centers ──────────────────────────────────────────────────
    n_per_q = max(1, args.n_centers // len(args.quadrants))
    test_centers = []
    for q_idx in args.quadrants:
        xmn, xmx, ymn, ymx = QUADRANT_BOUNDS[q_idx]
        q_centers = []
        while len(q_centers) < n_per_q:
            c = np.array([np.random.uniform(xmn, xmx), np.random.uniform(ymn, ymx)])
            if not excl.contains(Point(c)):
                q_centers.append(c)
        test_centers.extend(q_centers)
    print(f"Generated {len(test_centers)} test centers ({n_per_q} per quadrant, quadrants={args.quadrants})")

    # ── Run inference ────────────────────────────────────────────────────────
    collision_counts = []
    q_labels = {0: "Q1(top-right)", 1: "Q2(top-left)", 2: "Q3(bot-left)", 3: "Q4(bot-right)"}
    center_q_map = [args.quadrants[k // n_per_q] for k in range(len(test_centers))]
    trajs = []

    if not args.skip_quadrant_inference:
        for k, center in enumerate(test_centers):
            q_idx = center_q_map[k]
            print(f"  Inference {k+1}/{len(test_centers)} | {q_labels[q_idx]} | center=({center[0]:.2f}, {center[1]:.2f})")
            traj = _run_one_trajectory_2d(
                model,
                boids,
                center,
                args.max_steps,
                args.n_boids,
                goals,
                args.success_threshold,
                verbose_prefix="  ",
                near_goal_verbose=args.near_goal_verbose,
                near_goal_radius=args.near_goal_radius,
            )
            trajs.append(traj)
            collisions, unique_pairs = _count_collision_events(
                traj, boids.crowding, args.collision_warmup_steps
            )
            collision_counts.append((collisions, unique_pairs))
            if args.print_collisions:
                print(f"    collision events after step {args.collision_warmup_steps} (< {boids.crowding:g}): {collisions}")
                print(f"    unique agent pairs involved: {unique_pairs}")
            positions = traj[:, :, :2]
            centroid  = positions.mean(axis=1)
            for g_idx, g in enumerate(goals):
                dists = np.linalg.norm(centroid - g[None, :], axis=-1)
                print(f"    goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest mean dist = {dists.min():.4f}")
        print(f"Done. {len(trajs)} trajectories.")
        if args.viz_mode == "individual":
            _plot_individual_ranked(trajs, goals, args.n_boids, test_centers, args.run_tag, args.output_dir)
        else:
            _plot_multi_tubular(trajs, goals, args.run_tag, args.output_dir)

    # ── Interactive mode ──────────────────────────────────────────────────────
    if args.interactive:
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
            print(f"  Running GNCA from ({cx:.2f}, {cy:.2f}) for {args.max_steps} steps...")
            traj = _run_one_trajectory_2d(
                model,
                boids,
                center,
                args.max_steps,
                args.n_boids,
                goals,
                args.success_threshold,
            )
            collisions, unique_pairs = _count_collision_events(
                traj, boids.crowding, args.collision_warmup_steps
            )
            collision_counts.append((collisions, unique_pairs))
            centroid = traj[:, :, :2].mean(axis=1)
            for g_idx, g in enumerate(goals):
                dists = np.linalg.norm(centroid - g[None, :], axis=-1)
                print(f"    goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest mean dist = {dists.min():.4f}")
            if args.print_collisions:
                print(f"    collision events after step {args.collision_warmup_steps} (< {boids.crowding:g}): {collisions}")
                print(f"    unique agent pairs involved: {unique_pairs}")
            # Plot
            fig, ax = plt.subplots(figsize=(7, 7))
            centroid, r = _plot_tube_on_axis(ax, traj, label="GNCA centroid")
            ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5)
            for g_idx, g in enumerate(goals):
                ax.annotate(f"G{g_idx}", g, textcoords="offset points", xytext=(6, 6), fontsize=10)
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.set_title(f"GNCA from ({cx:.2f},{cy:.2f}) | r={r:.3f}")
            ax.legend()
            plt.tight_layout()
            fname = os.path.join(args.output_dir, f"interactive_{cx:.1f}_{cy:.1f}.pdf")
            plt.savefig(fname); plt.close()
            print(f"  Saved {fname}\n")

    # ── Ground-truth boids on same centers ───────────────────────────────────
    if args.compare_ground_truth and not args.skip_quadrant_inference:
        print("\n>>> Running ground-truth boids on same centers for comparison...")
        gt_trajs = []
        for k, center in enumerate(test_centers):
            q_idx = center_q_map[k]
            print(f"  GT {k+1}/{len(test_centers)} | {q_labels[q_idx]} | center=({center[0]:.2f}, {center[1]:.2f})")
            gt_boids = Boids(
                n_boids=args.n_boids,
                perception=args.perception,
            )
            history = gt_boids.generate_trajectory(save_config=False, random_init=center)
            pos_arr = history["positions"]
            vel_arr = history["velocities"]
            traj_gt = np.concatenate([pos_arr, vel_arr], axis=-1)
            gt_trajs.append(traj_gt)
            if args.near_goal_verbose:
                goal0 = goals[0]
                for step in range(len(traj_gt)):
                    centroid_now = traj_gt[step, :, :2].mean(axis=0)
                    mean_vel     = traj_gt[step, :, 2:].mean(axis=0)
                    dist_g0 = np.linalg.norm(centroid_now - goal0)
                    if dist_g0 < args.near_goal_radius:
                        angle = np.degrees(np.arctan2(mean_vel[1], mean_vel[0]))
                        print(f"    [GT step {step}] near goal0 | centroid=({centroid_now[0]:.3f},{centroid_now[1]:.3f}) dist={dist_g0:.3f} | mean_vel=({mean_vel[0]:.4f},{mean_vel[1]:.4f}) angle={angle:.1f}°")
            centroid_gt = traj_gt[:, :, :2].mean(axis=1)
            for g_idx, g in enumerate(goals):
                dists = np.linalg.norm(centroid_gt - g[None, :], axis=-1)
                print(f"    goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest mean dist = {dists.min():.4f}")

        for k, (center, traj_gnca, traj_gt) in enumerate(zip(test_centers, trajs, gt_trajs)):
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            for ax, traj, title in [(axes[0], traj_gnca, "GNCA"), (axes[1], traj_gt, "Ground Truth")]:
                centroid, r = _plot_tube_on_axis(ax, traj, label=f"{title} centroid")
                ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5, label='Goals')
                for g_idx, g in enumerate(goals):
                    ax.annotate(f"G{g_idx}", g, textcoords="offset points", xytext=(6, 6), fontsize=9)
                ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
                ax.set_aspect('equal')
                ax.set_title(f"{title} | r={r:.3f}")
                ax.legend(fontsize=8)
            fig.suptitle(f"Center ({center[0]:.2f},{center[1]:.2f}) | {q_labels[center_q_map[k]]}")
            plt.tight_layout()
            fname = os.path.join(args.output_dir, f"compare_gnca_vs_gt_center{k}.pdf")
            plt.savefig(fname); plt.close()
            print(f"Saved {fname}")

    if args.print_collisions:
        print(f"\nTotal post-warmup collision events across all GNCA runs: {sum(c[0] for c in collision_counts)}")
        print(f"Total unique-pair involvements across runs: {sum(c[1] for c in collision_counts)}")


if __name__ == "__main__":
    main()
