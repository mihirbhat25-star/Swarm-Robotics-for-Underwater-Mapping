"""
Visualization experiments script.
Runs the individual mode visualization (separate PDFs for all 50 trajectories) 
for specific trained models using cached trajectories.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiPolygon
import argparse

def run_individual_viz(run_tag):
    """
    Load cached trajectories and run individual visualization.
    Generates separate PDFs for all 50 trajectories ranked by tube radius r.
    Returns dict with statistics (min, max, mean, median, std of r values).
    """
    print(f"\n{'='*70}")
    print(f"Running individual visualization for: {run_tag}")
    print(f"{'='*70}\n")
    
    # Load cached trajectories
    traj_cache_path = f"saved_trajectory/viz_trajectories_{run_tag}.npz"
    if not os.path.exists(traj_cache_path):
        print(f"❌ Trajectory cache not found at {traj_cache_path}")
        return
    
    print(f"Loading cached trajectories from {traj_cache_path}...")
    cache = np.load(traj_cache_path)
    trajs = cache["trajs"]
    selected_centers = cache["centers"] if "centers" in cache.files else None
    
    # print(f"Loaded {len(trajs)} trajectories")
    
    # Define goals (same as in boids.py Boids.__init__)
    goals = np.array([
        [3.0, -3.0],
        [3.0, 3.0],
        [0.0, 0.0]
    ])
    
    # Generate individual PDFs ranked by tube radius r
    run_stats = []
    r_values = []
    for idx, traj in enumerate(trajs):
        centroid = traj[:, :, :2].mean(axis=1)
        boid_dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)
        r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)
        r_values.append(r)
        center = selected_centers[idx] if selected_centers is not None and len(selected_centers) > idx else None
        run_stats.append((idx, r, centroid, boid_dists, center))
    
    run_stats.sort(key=lambda x: x[1], reverse=True)

    # print("\n--- Runs sorted by tube radius r (greatest → least) ---")
    # for rank, (idx, r, centroid, boid_dists, center) in enumerate(run_stats):
    #     print(
    #         f"Rank {rank+1:02d} | Run {idx:02d} | r={r:.4f} | start={np.round(center, 4) if center is not None else 'unknown'}"
    #     )
    
    # Compute and return statistics
    r_values = np.array(r_values)
    stats = {
        'tag': run_tag,
        'min': np.min(r_values),
        'max': np.max(r_values),
        'mean': np.mean(r_values),
        'median': np.median(r_values),
        'std': np.std(r_values),
        'run_stats': run_stats,  # Include full run_stats for analysis
        'trajs': trajs  # Include trajectories for video generation
    }
    
    print(f"\n✅ Individual visualization complete for {run_tag}")
    return stats


def print_stats_table(stats_oldl, stats_newl, old_label="50x20_oldl", new_label="50x20_newl"):
    print("\n\n============================================")
    print(f"{'Metric':<16} {old_label:>14} {new_label:>14} {'Ratio (newl/oldl)':>20}")
    print("---------------------------------------------------------------------------")
    print(f"{'Min':<16} {stats_oldl['min']:>14.4f} {stats_newl['min']:>14.4f} {stats_newl['min']/stats_oldl['min']:>19.2f}x")
    print(f"{'Max':<16} {stats_oldl['max']:>14.4f} {stats_newl['max']:>14.4f} {stats_newl['max']/stats_oldl['max']:>19.2f}x")
    print(f"{'Mean':<16} {stats_oldl['mean']:>14.4f} {stats_newl['mean']:>14.4f} {stats_newl['mean']/stats_oldl['mean']:>19.2f}x")
    print(f"{'Median':<16} {stats_oldl['median']:>14.4f} {stats_newl['median']:>14.4f} {stats_newl['median']/stats_oldl['median']:>19.2f}x")
    print(f"{'Std Dev':<16} {stats_oldl['std']:>14.4f} {stats_newl['std']:>14.4f} {stats_newl['std']/stats_oldl['std']:>19.2f}x")
    print("============================================")


def _get_tube_exterior(tube):
    """Return exterior coordinates for Polygon or largest polygon in MultiPolygon."""
    if isinstance(tube, MultiPolygon):
        tube = max(tube.geoms, key=lambda p: p.area)
    return tube.exterior.xy


def generate_individual_pdfs(stats, run_tag):
    """Save one tubular PDF per trajectory run for a given loss instance."""
    output_dir = f"individual_pdfs_{run_tag}"
    os.makedirs(output_dir, exist_ok=True)

    goals = np.array([
        [3.0, -3.0],
        [3.0, 3.0],
        [0.0, 0.0]
    ])

    run_stats_by_idx = {entry[0]: entry for entry in stats['run_stats']}
    trajs = stats['trajs']

    for run_idx, traj in enumerate(trajs):
        _, r, centroid, boid_dists, _ = run_stats_by_idx[run_idx]

        # Match evaluate_boids tubular logic: centroid curve + Shapely Euclidean buffer tube.
        tube_radius = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)
        line = LineString(centroid)
        tube = line.buffer(tube_radius)
        tx, ty = _get_tube_exterior(tube)

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.fill(tx, ty, color='#8B0000', alpha=0.3, label=f'99% tube (r={tube_radius:.2f})')
        ax.plot(centroid[:, 0], centroid[:, 1], color='#8B0000', linewidth=1.5, label=f'Run {run_idx}')
        ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f"{run_tag} | Run {run_idx:02d} | r={r:.4f}")

        filename = os.path.join(output_dir, f"trajectory_{run_tag}_run{run_idx:02d}.pdf")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    print(f"Saved {len(trajs)} trajectory PDFs to {output_dir}")


def generate_tubular_triplet(stats, run_tag):
    """Save the mean, single-run, and all-runs tubular PDFs for a given run tag."""
    goals = np.array([
        [3.0, -3.0],
        [3.0, 3.0],
        [0.0, 0.0]
    ])

    trajs = stats['trajs']
    centroids = np.array([traj[:, :, :2].mean(axis=1) for traj in trajs])
    mean_curve = centroids.mean(axis=0)

    dists = np.linalg.norm(centroids - mean_curve[None], axis=-1)
    d_i = np.percentile(dists, 95, axis=1)
    r_99 = np.percentile(d_i, 99)

    line = LineString(mean_curve)
    tube = line.buffer(r_99)
    tx, ty = _get_tube_exterior(tube)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.fill(tx, ty, color='#8B0000', alpha=0.3, label=f'99% tube (r={r_99:.2f})')
    ax.plot(mean_curve[:, 0], mean_curve[:, 1], color='#8B0000', lw=1.5, label='Mean path')
    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"GNCA Swarm — Mean over {len(trajs)} Runs (Tubular Band)")
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    mean_fname = f"boids_tube_mean_{run_tag}.pdf"
    plt.savefig(mean_fname)
    plt.close()
    print(f"Saved {mean_fname}")

    run_idx = np.random.randint(len(trajs))
    traj = trajs[run_idx]
    centroid = traj[:, :, :2].mean(axis=1)
    boid_dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)
    r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)
    line = LineString(centroid)
    tube = line.buffer(r)
    tx, ty = _get_tube_exterior(tube)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.fill(tx, ty, color='#8B0000', alpha=0.3, label=f'99% tube (r={r:.2f})')
    ax.plot(centroid[:, 0], centroid[:, 1], color='#8B0000', lw=1.5, label=f'Run {run_idx}')
    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"GNCA Swarm — Single Run Tube (run {run_idx})")
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    single_fname = f"boids_tube_single_{run_tag}.pdf"
    plt.savefig(single_fname)
    plt.close()
    print(f"Saved {single_fname}")

    colors = plt.cm.hsv(np.linspace(0, 0.9, len(trajs)))
    fig, ax = plt.subplots(figsize=(7, 7))
    for color, traj in zip(colors, trajs):
        centroid = traj[:, :, :2].mean(axis=1)
        boid_dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)
        r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)
        line = LineString(centroid)
        tube = line.buffer(r)
        tx, ty = _get_tube_exterior(tube)
        ax.fill(tx, ty, color=color, alpha=0.25)
        run_idx = len(ax.lines)
        ax.plot(centroid[:, 0], centroid[:, 1], color=color, lw=1.2, label=f'Run {run_idx:02d}')

    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"GNCA Swarm — All {len(trajs)} Runs (Tubular Bands)")
    ax.set_aspect('equal')
    ax.legend(fontsize=5, ncol=2, loc='best')
    plt.tight_layout()
    all_runs_fname = f"boids_tube_all_runs_{run_tag}.pdf"
    plt.savefig(all_runs_fname)
    plt.close()
    print(f"Saved {all_runs_fname}")


def _center_to_filename_token(center_xy):
    x, y = float(center_xy[0]), float(center_xy[1])
    return f"x{x:.4f}_y{y:.4f}"


def _k_nearest_points(blowup_run, all_run_stats, k=5):
    """Return k nearest runs to a blowup by start centroid distance (regardless of r value)."""
    blowup_idx, _, blowup_centroid, _, _ = blowup_run
    blowup_start = blowup_centroid[0]
    
    candidates = []
    for idx, r, centroid, boid_dists, center in all_run_stats:
        if idx == blowup_idx:  # Skip the blowup itself
            continue
        start = centroid[0]
        dist = np.linalg.norm(blowup_start - start)
        candidates.append((dist, idx, r, centroid, boid_dists, center))
    
    candidates.sort(key=lambda t: t[0])
    return candidates[:k]


def generate_blowup_neighbor_pdfs(stats, loss_type, base_tag="50x20_no_noise", threshold=2.0, k=5):
    """
    For each blowup run (r >= threshold), render one PDF containing:
    - the blowup trajectory
    - the k nearest trajectories by start centroid distance (regardless of r value)
    Each trajectory is plotted as mean centroid line + its tube in matching color.
    """
    # Create output folder
    output_folder = f"6_10_results/{base_tag}/blowup_analysis"
    os.makedirs(output_folder, exist_ok=True)

    goals = np.array([
        [3.0, -3.0],
        [3.0, 3.0],
        [0.0, 0.0]
    ])

    blowups = [run for run in stats['run_stats'] if run[1] >= threshold]
    print(f"Generating blowup-neighbor PDFs for {loss_type}: {len(blowups)} blowups")

    for blowup_run in blowups:
        blowup_idx, blowup_r, blowup_centroid, blowup_boid_dists, blowup_center = blowup_run
        blowup_start = blowup_centroid[0]
        neighbors = _k_nearest_points(blowup_run, stats['run_stats'], k=k)

        fig, ax = plt.subplots(figsize=(8, 8))

        # First entry is the blowup itself, followed by nearest runs.
        plot_items = [
            ("blowup", blowup_idx, blowup_r, blowup_centroid, blowup_boid_dists)
        ]
        for dist, idx, r, centroid, boid_dists, center in neighbors:
            plot_items.append((f"nn d={dist:.3f}", idx, r, centroid, boid_dists))

        colors = plt.cm.tab10(np.linspace(0, 1, len(plot_items)))

        for color, (label_prefix, idx, r, centroid, boid_dists) in zip(colors, plot_items):
            tube_r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)
            tube = LineString(centroid).buffer(tube_r)
            tx, ty = _get_tube_exterior(tube)
            ax.fill(tx, ty, color=color, alpha=0.22)
            ax.plot(centroid[:, 0], centroid[:, 1], color=color, linewidth=1.5,
                    label=f"{label_prefix} run {idx:02d} (r={r:.3f})")

        ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f"{base_tag}_{loss_type} | blowup run {blowup_idx:02d} | start={np.round(blowup_start, 4)}")

        center_token = _center_to_filename_token(blowup_start)
        filename = f"{base_tag}_{loss_type}_{center_token}.pdf"
        fpath = os.path.join(output_folder, filename)
        plt.tight_layout()
        plt.savefig(fpath)
        plt.close(fig)

    print(f"Saved {len(blowups)} blowup-neighbor PDFs to {output_folder}")


def generate_single_blowup_neighbor_pdf(stats, run_tag, threshold=2.0, k=5):
    """Save one PDF for the highest-r blowup and its k nearest non-blowup starts."""
    goals = np.array([
        [3.0, -3.0],
        [3.0, 3.0],
        [0.0, 0.0]
    ])

    blowups = [run for run in stats['run_stats'] if run[1] >= threshold]
    if not blowups:
        print(f"No blowups found for {run_tag} at threshold {threshold}")
        return None

    blowup_run = max(blowups, key=lambda run: run[1])
    blowup_idx, blowup_r, blowup_centroid, blowup_boid_dists, _ = blowup_run
    neighbors = _k_nearest_points(blowup_run, stats['run_stats'], k=k, threshold=threshold)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_items = [("blowup", blowup_idx, blowup_r, blowup_centroid, blowup_boid_dists)]
    for dist, idx, r, centroid, boid_dists, _ in neighbors:
        plot_items.append((f"nn d={dist:.3f}", idx, r, centroid, boid_dists))

    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_items)))
    for color, (label_prefix, idx, r, centroid, boid_dists) in zip(colors, plot_items):
        tube_r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)
        tube = LineString(centroid).buffer(tube_r)
        tx, ty = _get_tube_exterior(tube)
        ax.fill(tx, ty, color=color, alpha=0.22)
        ax.plot(centroid[:, 0], centroid[:, 1], color=color, linewidth=1.5,
                label=f"{label_prefix} run {idx:02d} (r={r:.3f})")

    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, zorder=5, label='Goals')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_title(f"{run_tag} | blowup run {blowup_idx:02d} + {k} nearest non-blowup starts")

    output_path = f"blowup_neighbors_{run_tag}_run{blowup_idx:02d}.pdf"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def resolve_existing_run_tag(candidates):
    """Return first run tag that has a matching cached trajectory file."""
    for tag in candidates:
        path = f"saved_trajectory/viz_trajectories_{tag}.npz"
        if os.path.exists(path):
            return tag
    return None


def debug_compare_centers_across_runs():
    """Debug-print whether cached centers are identical across selected runs."""
    run_tag_candidates = {
        "no_noise_oldl": ["50x20_oldl"],
        "no_noise_newl": ["50x20_newl"],
        "nw2_oldl": ["50x20_oldl_nw_2", "50x20_oldl_nw2"],
        "nw2_newl": ["50x20_newl_nw_2", "50x20_newl_nw2"],
    }

    resolved = {}
    centers_by_label = {}

    for label, candidates in run_tag_candidates.items():
        tag = resolve_existing_run_tag(candidates)
        resolved[label] = tag
        if tag is None:
            print(f"[missing] {label}: no cache found for candidates={candidates}")
            continue

        path = f"saved_trajectory/viz_trajectories_{tag}.npz"
        cache = np.load(path, allow_pickle=True)
        if "centers" not in cache.files:
            print(f"[missing centers] {label}: {path} has keys={list(cache.files)}")
            continue

        centers = np.array(cache["centers"], dtype=np.float32)
        centers_by_label[label] = centers
        print(f"[loaded] {label}: tag={tag}, shape={centers.shape}")

    print("\nCenter consistency checks:")
    labels = list(centers_by_label.keys())
    if len(labels) < 2:
        print("Not enough center arrays loaded to compare.")
        return

    # Report all pairwise comparisons so we can see exactly which runs match each other.
    same_pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = labels[i]
            b = labels[j]
            arr_a = centers_by_label[a]
            arr_b = centers_by_label[b]
            same_shape = arr_a.shape == arr_b.shape
            same_values = same_shape and np.allclose(arr_a, arr_b)
            max_abs_diff = float(np.max(np.abs(arr_a - arr_b))) if same_shape else float("nan")
            print(
                f"{a} vs {b}: "
                f"same_shape={same_shape}, same_values={same_values}, max_abs_diff={max_abs_diff:.8f}"
            )
            if same_values:
                same_pairs.append((a, b))

    # Build connected groups from equal pairs (if A=B and B=C, group A/B/C together).
    groups = []
    seen = set()
    adjacency = {label: set() for label in labels}
    for a, b in same_pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)

    for label in labels:
        if label in seen:
            continue
        stack = [label]
        component = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency[node] - seen)
        groups.append(sorted(component))

    print("\nRuns with identical centers:")
    any_multi = False
    for group in groups:
        if len(group) > 1:
            any_multi = True
            print(f"  SAME: {', '.join(group)}")

    if not any_multi:
        print("  None. No two loaded runs share identical centers.")

def debug_confirm_centers_match(target_run_tag, reference_run_tag):
    """Print whether centers in target run match centers in reference run."""
    target_path = f"saved_trajectory/viz_trajectories_{target_run_tag}.npz"
    reference_path = f"saved_trajectory/viz_trajectories_{reference_run_tag}.npz"

    if not os.path.exists(target_path):
        print(f"[missing] target cache not found: {target_path}")
        return False
    if not os.path.exists(reference_path):
        print(f"[missing] reference cache not found: {reference_path}")
        return False

    target_cache = np.load(target_path, allow_pickle=True)
    reference_cache = np.load(reference_path, allow_pickle=True)

    if "centers" not in target_cache.files:
        print(f"[missing centers] target {target_path} has keys={list(target_cache.files)}")
        return False
    if "centers" not in reference_cache.files:
        print(f"[missing centers] reference {reference_path} has keys={list(reference_cache.files)}")
        return False

    target_centers = np.array(target_cache["centers"], dtype=np.float32)
    reference_centers = np.array(reference_cache["centers"], dtype=np.float32)
    same_shape = target_centers.shape == reference_centers.shape
    same_values = same_shape and np.allclose(target_centers, reference_centers)
    max_abs_diff = float(np.max(np.abs(target_centers - reference_centers))) if same_shape else float("nan")

    print(
        f"Center debug: {target_run_tag} vs {reference_run_tag} -> "
        f"same_shape={same_shape}, same_values={same_values}, max_abs_diff={max_abs_diff:.8f}"
    )
    return same_values

def find_nearest_neighbor(blowup_start, all_run_stats, threshold=2.0):
    """
    Find the nearest neighbor by start position distance among non-blowup runs.
    Uses the first centroid position of each trajectory as the start state.

    blowup_start: (idx, r, centroid, boid_dists, center) tuple
    Returns: (neighbor_idx, distance, neighbor_r, neighbor_center, neighbor_start_pos)
    """
    blowup_idx, blowup_r, blowup_centroid, _, blowup_center = blowup_start
    blowup_start_pos = blowup_centroid[0]  # first timestep centroid position
    
    min_distance = float('inf')
    nearest_neighbor = None
    
    for idx, r, centroid, boid_dists, center in all_run_stats:
        if r >= threshold:  # Skip other blowups
            continue
        
        neighbor_start_pos = centroid[0]  # first timestep centroid position
        distance = np.linalg.norm(blowup_start_pos - neighbor_start_pos)
        
        if distance < min_distance:
            min_distance = distance
            nearest_neighbor = (idx, distance, r, center, neighbor_start_pos)
    
    return nearest_neighbor if nearest_neighbor else None


def format_center(center):
    if center is None:
        return ""
    return f"[{center[0]:.6f}, {center[1]:.6f}]"


def write_blowup_table(stats, run_tag, threshold=2.0):
    blowups = [run for run in stats['run_stats'] if run[1] >= threshold]
    rows = []
    distances = []

    # Create output folder
    output_folder = f"6_10_results/{run_tag}/blowup_analysis"
    os.makedirs(output_folder, exist_ok=True)

    for blowup_run in blowups:
        blowup_idx, blowup_r, blowup_centroid, _, blowup_center = blowup_run
        blowup_start_pos = blowup_centroid[0]
        neighbors = _k_nearest_points(blowup_run, stats['run_stats'], k=5)

        row = {
            'blowup_start_center': format_center(blowup_start_pos),
            'blowup_r_score': f"{blowup_r:.6f}",
            'nearest_start_center': "",
            'distance_to_nearest_start': "",
            'neighbor_1': "",
            'neighbor_2': "",
            'neighbor_3': "",
            'neighbor_4': "",
            'neighbor_5': "",
        }

        if neighbors:
            first_dist, _, _, first_centroid, _, _ = neighbors[0]
            distances.append(first_dist)
            row['nearest_start_center'] = format_center(first_centroid[0])
            row['distance_to_nearest_start'] = f"{first_dist:.6f}"

        for n, neighbor in enumerate(neighbors, start=1):
            _, _, neighbor_r, _, _, _ = neighbor
            row[f'neighbor_{n}'] = f"{neighbor_r:.6f}"

        rows.append(row)

    if distances:
        distances_arr = np.array(distances)
        print(
            f"{run_tag} nearest neighbor distance summary: "
            f"min={distances_arr.min():.6f}, "
            f"mean={distances_arr.mean():.6f}, "
            f"median={np.median(distances_arr):.6f}, "
            f"max={distances_arr.max():.6f}"
        )
    else:
        print(f"{run_tag} nearest neighbor distance summary: no blowups found")

    output_path = os.path.join(output_folder, f"blow_up_table_{run_tag}.csv")
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'blowup_start_center',
                'blowup_r_score',
                'nearest_start_center',
                'distance_to_nearest_start',
                'neighbor_1',
                'neighbor_2',
                'neighbor_3',
                'neighbor_4',
                'neighbor_5'
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {output_path} with {len(rows)} blowup rows")

    return output_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Analysis and visualization of 2D boids trajectories with tubular stats and blowup analysis"
    )
    parser.add_argument("--run_tag", type=str, required=True,
                        help="Run tag to analyze (e.g., '50x20_3d_oldl_nw_2')")
    parser.add_argument("--gen_csv", action="store_true",
                        help="Generate blowup CSV with nearest neighbor analysis")
    parser.add_argument("--gen_blowup_pdfs", action="store_true",
                        help="Generate blowup PDFs with 5 nearest non-blowup trajectories")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="R-score threshold for blowup classification (default 2.0)")
    args = parser.parse_args()
    
    run_tag = args.run_tag
    
    # Load and analyze trajectories
    print(f"\n{'='*70}")
    print(f"ANALYSIS FOR RUN: {run_tag}")
    print(f"{'='*70}\n")
    
    stats = run_individual_viz(run_tag)
    
    if stats is None:
        print(f"❌ Failed to load trajectories for {run_tag}")
        exit(1)
    
    # ALWAYS print tubular R-score stats by default
    print(f"\n{'='*70}")
    print(f"TUBULAR R-SCORE STATISTICS")
    print(f"{'='*70}")
    r_values = np.array([run[1] for run in stats['run_stats']])
    print(f"Min:    {stats['min']:.6f}")
    print(f"Max:    {stats['max']:.6f}")
    print(f"Mean:   {stats['mean']:.6f}")
    print(f"Median: {stats['median']:.6f}")
    print(f"Std:    {stats['std']:.6f}")
    print(f"Runs:   {len(r_values)}")
    blowups = np.sum(r_values >= args.threshold)
    print(f"Blowups (r >= {args.threshold}): {blowups} / {len(r_values)}")
    print(f"{'='*70}\n")
    
    # Optional: generate blowup CSV
    if args.gen_csv:
        print(f"Generating blowup CSV with threshold={args.threshold}...")
        write_blowup_table(stats, run_tag, threshold=args.threshold)
    
    # Optional: generate blowup PDFs
    if args.gen_blowup_pdfs:
        print(f"Generating blowup PDFs with threshold={args.threshold}...")
        generate_blowup_neighbor_pdfs(stats, loss_type="nn5", base_tag=run_tag, 
                                     threshold=args.threshold, k=5)