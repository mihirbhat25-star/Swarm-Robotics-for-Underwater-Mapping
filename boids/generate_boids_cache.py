"""
Generates boids trajectories, converts them to training graph tuples,
and saves to an HDF5 file for memory-efficient loading during training.
Writes to disk incrementally per trajectory — no large RAM accumulation.

The HDF5 contains:
  - 'x':       (N, n_boids, 4)  — node features [pos_x, pos_y, vel_x, vel_y]
  - 'a_row':   (N, max_edges)   — COO row indices (padded with -1)
  - 'a_col':   (N, max_edges)   — COO col indices (padded with -1)
  - 'a_len':   (N,)             — actual number of edges per timestep
  - 'y':       (N, n_boids, 10) — targets [cur_state(4), next_state(4), goal(2)]
  - 'centers': (unique_reps, 2) — flock start centers for viz/testing

Usage:
    python -m boids.generate_boids_cache --unique 50 --repeats 20
    python -m boids.generate_boids_cache --unique 100 --repeats 20 --output my_cache.h5

Load in training (via --boids_cache flag in run_boids.py):
    python -m boids.run_boids --boids_cache boids_cache_50x20.h5 ...

Load centers only (for viz/testing):
    import h5py
    with h5py.File("boids_cache_50x20.h5", "r") as f:
        centers = f["centers"][:]  # (50, 2)
"""
import argparse
import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.boids import Boids, history_to_samples


def main():
    parser = argparse.ArgumentParser(description="Generate and cache boids graph training data")
    parser.add_argument("--unique", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument("--pos_noise", type=float, default=0.004)
    parser.add_argument("--vel_noise", type=float, default=0.0005)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--x_min", type=float, default=-5.0)
    parser.add_argument("--x_max", type=float, default=-2.5)
    parser.add_argument("--y_min", type=float, default=-5.0)
    parser.add_argument("--y_max", type=float, default=-2.5)
    parser.add_argument("--noise_tag", type=str, default="",
                        help="Label to append to filename (e.g. 'nw_2'). Does not change noise values.")
    args = parser.parse_args()

    if args.output is None:
        noise_suffix = f"_{args.noise_tag}" if args.noise_tag else ""
        args.output = f"boids_cache_{args.unique}x{args.repeats}_x{args.x_min}_{args.x_max}_y{args.y_min}_{args.y_max}{noise_suffix}.h5"

    print(f"\n>>> Generating {args.unique} unique × {args.repeats} repeats = {args.unique * args.repeats} trajectories")
    print(f">>> n_boids={args.n_boids}, pos_noise={args.pos_noise}, vel_noise={args.vel_noise}")
    print(f">>> Sampling centers from x=[{args.x_min}, {args.x_max}], y=[{args.y_min}, {args.y_max}]")
    print(f">>> Output: {args.output}\n")

    boids = Boids(n_boids=args.n_boids, pos_noise=args.pos_noise, vel_noise=args.vel_noise)

    # Theoretical max edges: every boid neighbors every other (n*(n-1))
    max_edges = args.n_boids * (args.n_boids - 1)
    n_feat_x = 4   # pos_x, pos_y, vel_x, vel_y
    n_feat_y = 10  # cur_state(4) + next_state(4) + goal(2)
    print(f">>> max_edges={max_edges} (theoretical max), x_feat={n_feat_x}, y_feat={n_feat_y}\n")

    all_centers = []

    with h5py.File(args.output, 'w') as f:
        ds_x    = f.create_dataset('x',     shape=(0, args.n_boids, n_feat_x), maxshape=(None, args.n_boids, n_feat_x), dtype='float32', chunks=(256, args.n_boids, n_feat_x), compression='gzip', compression_opts=4)
        ds_y    = f.create_dataset('y',     shape=(0, args.n_boids, n_feat_y), maxshape=(None, args.n_boids, n_feat_y), dtype='float32', chunks=(256, args.n_boids, n_feat_y), compression='gzip', compression_opts=4)
        ds_arow = f.create_dataset('a_row', shape=(0, max_edges), maxshape=(None, max_edges), dtype='int32', chunks=(256, max_edges), fillvalue=-1, compression='gzip', compression_opts=4)
        ds_acol = f.create_dataset('a_col', shape=(0, max_edges), maxshape=(None, max_edges), dtype='int32', chunks=(256, max_edges), fillvalue=-1, compression='gzip', compression_opts=4)
        ds_alen = f.create_dataset('a_len', shape=(0,), maxshape=(None,), dtype='int32', chunks=(1024,), compression='gzip', compression_opts=4)

        for i in tqdm(range(args.unique), desc="Unique centers"):
            center = np.array([np.random.uniform(args.x_min, args.x_max),
                                np.random.uniform(args.y_min, args.y_max)], dtype=np.float32)
            all_centers.append(center)

            for j in range(args.repeats):
                history = boids.generate_trajectory(random_init=center, save_config=False)
                samples = history_to_samples(history)
                n = len(samples)

                x_buf    = np.zeros((n, args.n_boids, n_feat_x), dtype=np.float32)
                y_buf    = np.zeros((n, args.n_boids, n_feat_y), dtype=np.float32)
                arow_buf = np.full((n, max_edges), -1, dtype=np.int32)
                acol_buf = np.full((n, max_edges), -1, dtype=np.int32)
                alen_buf = np.zeros(n, dtype=np.int32)

                for k, (x, a, y) in enumerate(samples):
                    x_buf[k] = x
                    y_buf[k] = y
                    length = a.nnz
                    arow_buf[k, :length] = a.row
                    acol_buf[k, :length] = a.col
                    alen_buf[k] = length

                cur = ds_x.shape[0]
                for ds, buf in [(ds_x, x_buf), (ds_y, y_buf), (ds_arow, arow_buf), (ds_acol, acol_buf), (ds_alen, alen_buf)]:
                    ds.resize(cur + n, axis=0)
                    ds[cur:cur + n] = buf

                del history, samples, x_buf, y_buf, arow_buf, acol_buf, alen_buf

        centers_arr = np.stack(all_centers, axis=0)
        f.create_dataset('centers', data=centers_arr)
        f.attrs['unique_reps'] = args.unique
        f.attrs['repeats']     = args.repeats
        f.attrs['n_boids']     = args.n_boids
        f.attrs['max_edges']   = max_edges

        total = ds_x.shape[0]

    print(f"\n✅ Saved {total} samples to '{args.output}'")
    print(f"   centers: {centers_arr.shape}")
    print(f"   File size: ~{os.path.getsize(args.output) / 1e6:.1f} MB")

if __name__ == "__main__":
    main()