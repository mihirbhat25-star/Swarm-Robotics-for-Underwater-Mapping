"""
Visualization experiments script for 3D boids.
Runs the individual mode visualization (separate PDFs for all 50 trajectories) 
for specific trained 3D models using cached trajectories.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_individual_viz_3d(run_tag):
    """
    Load cached 3D trajectories and run individual visualization.
    Generates statistics (min, max, mean, median, std of r values).
    
    Returns dict with statistics and trajectory data.
    """
    print(f"\n{'='*70}")
    print(f"Running individual visualization for 3D: {run_tag}")
    print(f"{'='*70}\n")
    
    # Load cached trajectories
    traj_cache_path = f"saved_trajectory/viz_trajectories_3d_{run_tag}.npz"
    if not os.path.exists(traj_cache_path):
        print(f"❌ Trajectory cache not found at {traj_cache_path}")
        return None
    
    print(f"Loading cached 3D trajectories from {traj_cache_path}...")
    cache = np.load(traj_cache_path, allow_pickle=True)
    trajs = cache["trajs"]
    selected_centers = cache["centers"] if "centers" in cache.files else None
    
    print(f"Loaded {len(trajs)} 3D trajectories")
    
    # Define goals (same as in boids_3d.py Boids3D.__init__)
    goals = np.array([
        [3.0, -3.0, 0.0],
        [3.0, 3.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    # Generate statistics ranked by tube radius r
    run_stats = []
    r_values = []
    for idx, traj in enumerate(trajs):
        # Compute 3D centroid (mean position of all boids at each timestep)
        centroid = traj[:, :, :3].mean(axis=1)
        # Compute distances from centroid to all boid positions in 3D
        boid_dists = np.linalg.norm(traj[:, :, :3] - centroid[:, None, :], axis=-1)
        # Compute tube radius: 99th percentile of 95th percentile distances
        r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)
        r_values.append(r)
        center = selected_centers[idx] if selected_centers is not None and len(selected_centers) > idx else None
        run_stats.append((idx, r, centroid, boid_dists, center))
    
    run_stats.sort(key=lambda x: x[1], reverse=True)
    
    # Compute and return statistics
    r_values = np.array(r_values)
    stats = {
        'tag': run_tag,
        'min': np.min(r_values),
        'max': np.max(r_values),
        'mean': np.mean(r_values),
        'median': np.median(r_values),
        'std': np.std(r_values),
        'run_stats': run_stats,
        'trajs': trajs,
        'goals': goals
    }
    
    print(f"✅ Individual visualization complete for 3D {run_tag}")
    return stats


def _k_nearest_points_3d(blowup_run, all_run_stats, k=5):
    """Return k nearest runs to a blowup by start centroid distance in 3D (regardless of r value)."""
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


def format_center_3d(center):
    """Format 3D center for display."""
    if center is None:
        return ""
    return f"[{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]"


def write_blowup_table_3d(stats, run_tag, threshold=2.0):
    """Generate blowup CSV with nearest neighbor analysis for 3D trajectories."""
    blowups = [run for run in stats['run_stats'] if run[1] >= threshold]
    rows = []
    distances = []
    
    for blowup_run in blowups:
        blowup_idx, blowup_r, blowup_centroid, _, blowup_center = blowup_run
        blowup_start_pos = blowup_centroid[0]
        neighbors = _k_nearest_points_3d(blowup_run, stats['run_stats'], k=5)
        
        row = {
            'blowup_start_center': format_center_3d(blowup_start_pos),
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
            row['nearest_start_center'] = format_center_3d(first_centroid[0])
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
    
    output_path = f"blow_up_table_3d_{run_tag}.csv"
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
    
    print(f"✅ Saved {output_path} with {len(rows)} blowup rows")
    return output_path


def generate_blowup_neighbor_pdfs_3d(stats, run_tag, threshold=2.0, k=5):
    """
    For each blowup run (r >= threshold), render one 3D PDF containing:
    - the blowup trajectory
    - the k nearest trajectories by start centroid distance (regardless of r value)
    """
    goals = stats['goals']
    blowups = [run for run in stats['run_stats'] if run[1] >= threshold]
    print(f"Generating blowup-neighbor 3D PDFs for {run_tag}: {len(blowups)} blowups")
    
    output_dir = f"blowup_pdfs_3d_{run_tag}"
    os.makedirs(output_dir, exist_ok=True)
    
    for blowup_run in blowups:
        blowup_idx, blowup_r, blowup_centroid, blowup_boid_dists, blowup_center = blowup_run
        blowup_start = blowup_centroid[0]
        neighbors = _k_nearest_points_3d(blowup_run, stats['run_stats'], k=k)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot blowup
        ax.plot(blowup_centroid[:, 0], blowup_centroid[:, 1], blowup_centroid[:, 2],
                color='#8B0000', linewidth=2, label=f'blowup run {blowup_idx:02d} (r={blowup_r:.3f})')
        
        # Plot neighbors
        colors = plt.cm.tab10(np.linspace(0, 1, len(neighbors)))
        for color, neighbor in zip(colors, neighbors):
            dist, idx, r, centroid, boid_dists, center = neighbor
            ax.plot(centroid[:, 0], centroid[:, 1], centroid[:, 2],
                    color=color, linewidth=1.5, label=f'nn d={dist:.3f} run {idx:02d} (r={r:.3f})')
        
        # Plot goals
        ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], 
                   c='red', marker='*', s=300, zorder=5, label='Goals')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{run_tag} | blowup run {blowup_idx:02d} + {k} nearest")
        ax.legend(loc='best', fontsize=8)
        
        center_token = f"x{blowup_start[0]:.4f}_y{blowup_start[1]:.4f}_z{blowup_start[2]:.4f}"
        filename = os.path.join(output_dir, f"{run_tag}_{center_token}.pdf")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
    
    print(f"✅ Saved {len(blowups)} blowup-neighbor 3D PDFs to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analysis and visualization of 3D boids trajectories with tubular stats and blowup analysis"
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
    
    # Load and analyze 3D trajectories
    print(f"\n{'='*70}")
    print(f"3D ANALYSIS FOR RUN: {run_tag}")
    print(f"{'='*70}\n")
    
    stats = run_individual_viz_3d(run_tag)
    
    if stats is None:
        print(f"❌ Failed to load 3D trajectories for {run_tag}")
        exit(1)
    
    # ALWAYS print tubular R-score stats by default
    print(f"\n{'='*70}")
    print(f"3D TUBULAR R-SCORE STATISTICS")
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
        print(f"Generating 3D blowup CSV with threshold={args.threshold}...")
        write_blowup_table_3d(stats, run_tag, threshold=args.threshold)
    
    # Optional: generate blowup PDFs
    if args.gen_blowup_pdfs:
        print(f"Generating 3D blowup PDFs with threshold={args.threshold}...")
        generate_blowup_neighbor_pdfs_3d(stats, run_tag, threshold=args.threshold, k=5)
