"""
Standalone visualization script.
Loads a saved GNCA model and boids object, then runs evaluate() to produce the PDF.

Usage:
    python boids/visualize_boids.py [--viz_mode tubular|per_boid]
                                    [--model_path gnca_model]
                                    [--boids_path boids_tr.pkl]
                                    [--max_trajectory_len 1850]
                                    [--n_boids 100]
"""
import argparse
import os
import sys

import joblib
import tensorflow as tf

# Make sure the workspace root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from boids.evaluate_boids import evaluate
from boids.forward import forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_mode", default="tubular", choices=["tubular", "per_boid", "multi_tubular"],
                        help="Visualization style: 'tubular' (mean over all 50), 'per_boid' (one random run, all boids), 'multi_tubular' (10 random separate tubes)")
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
    args = parser.parse_args()

    if args.run_tag:
        model_path = f"gnca_model_{args.run_tag}"
        boids_path = f"boids_tr_{args.run_tag}.pkl"
        traj_cache_path = f"viz_trajectories_{args.run_tag}.npz"
    else:
        model_path = args.model_path
        boids_path = args.boids_path
        traj_cache_path = args.traj_cache_path

    print(f"Loading model from '{model_path}'...")
    model = tf.saved_model.load(model_path)

    print(f"Loading boids from '{boids_path}'...")
    boids_tr = joblib.load(boids_path)

    print(f"Running evaluation with viz_mode='{args.viz_mode}'...")
    evaluate(
        model=model,
        forward=forward,
        max_trajectory_len=args.max_trajectory_len,
        n_boids=args.n_boids,
        use_saved_config=True,
        saved_boids=boids_tr,
        viz_mode=args.viz_mode,
        traj_cache_path=traj_cache_path,
    )


if __name__ == "__main__":
    main()
