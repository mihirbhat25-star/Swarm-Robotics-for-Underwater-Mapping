"""Dimension-independent statistics for cached swarm trajectories."""

import csv
import os

import numpy as np


def load_statistics(cache_path, run_tag, dimensions, goals):
    """Load cached trajectories and compute per-run tube-radius statistics."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Trajectory cache not found: {cache_path}")
    with np.load(cache_path, allow_pickle=True) as cache:
        trajectories = cache["trajs"]
        centers = cache["centers"] if "centers" in cache.files else None

    if trajectories.ndim != 4 or trajectories.shape[-1] < dimensions:
        raise ValueError(
            f"Expected trajectories shaped (runs, steps, boids, features), "
            f"got {trajectories.shape}"
        )
    if len(trajectories) == 0:
        raise ValueError(f"Trajectory cache is empty: {cache_path}")
    if centers is not None and len(centers) != len(trajectories):
        raise ValueError(
            "Trajectory cache has different numbers of centers and trajectories"
        )

    run_stats = []
    for run_index, trajectory in enumerate(trajectories):
        centroid = trajectory[:, :, :dimensions].mean(axis=1)
        distances = np.linalg.norm(
            trajectory[:, :, :dimensions] - centroid[:, None, :], axis=-1
        )
        radius = float(np.percentile(np.percentile(distances, 95, axis=1), 99))
        center = centers[run_index] if centers is not None else centroid[0]
        run_stats.append((run_index, radius, centroid, distances, center))
    run_stats.sort(key=lambda item: item[1], reverse=True)
    radii = np.asarray([item[1] for item in run_stats], dtype=np.float64)
    return {
        "tag": run_tag,
        "dimensions": dimensions,
        "goals": np.asarray(goals, dtype=np.float32),
        "trajs": trajectories,
        "run_stats": run_stats,
        "min": float(radii.min()),
        "max": float(radii.max()),
        "mean": float(radii.mean()),
        "median": float(np.median(radii)),
        "std": float(radii.std()),
    }


def nearest_runs(target_run, all_runs, count=5):
    """Return the nearest starts to a target run, excluding the target itself."""
    target_index, _, target_centroid, _, _ = target_run
    target_start = target_centroid[0]
    candidates = []
    for run_index, radius, centroid, distances, center in all_runs:
        if run_index == target_index:
            continue
        separation = float(np.linalg.norm(target_start - centroid[0]))
        candidates.append(
            (separation, run_index, radius, centroid, distances, center)
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[:count]


def print_summary(stats, threshold):
    """Print a concise tube-radius report."""
    radii = np.asarray([run[1] for run in stats["run_stats"]])
    blowups = int(np.count_nonzero(radii >= threshold))
    print(f"Run: {stats['tag']}")
    print(
        f"r min/mean/median/max/std: {stats['min']:.6f} / "
        f"{stats['mean']:.6f} / {stats['median']:.6f} / "
        f"{stats['max']:.6f} / {stats['std']:.6f}"
    )
    print(f"Blowups (r >= {threshold:g}): {blowups}/{len(radii)}")


def write_blowup_csv(stats, output_path, threshold=2.0, neighbors=5):
    """Write blowups and nearest-start radii to one dimension-agnostic CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    neighbor_columns = [f"neighbor_{index + 1}_r" for index in range(neighbors)]
    fieldnames = [
        "run",
        "start_center",
        "r_score",
        "nearest_start_center",
        "nearest_distance",
        *neighbor_columns,
    ]
    rows = []
    for run in stats["run_stats"]:
        run_index, radius, centroid, _, _ = run
        if radius < threshold:
            continue
        nearest = nearest_runs(run, stats["run_stats"], count=neighbors)
        row = {
            "run": run_index,
            "start_center": np.array2string(centroid[0], precision=6),
            "r_score": f"{radius:.6f}",
            "nearest_start_center": "",
            "nearest_distance": "",
        }
        if nearest:
            row["nearest_start_center"] = np.array2string(
                nearest[0][3][0], precision=6
            )
            row["nearest_distance"] = f"{nearest[0][0]:.6f}"
        for column, neighbor in zip(neighbor_columns, nearest):
            row[column] = f"{neighbor[2]:.6f}"
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} blowup rows to {output_path}")
    return output_path
