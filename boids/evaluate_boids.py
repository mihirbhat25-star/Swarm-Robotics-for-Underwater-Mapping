"""
Evaluates the trained GNCA by comparing it to the true Boids GCA.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import nolds
import numpy as np
import tensorflow as tf
from spektral.data import DisjointLoader
from spektral.layers import ops
from tensorflow.keras.models import load_model
from modules.boids import make_dataset
from modules.boids import Boids

@tf.function(experimental_relax_shapes=True)
def forward(model, x, a, i, training=None):
    """Computes one forward pass of the GNCA"""
    x_pred = model((x, a, i[:, None]), training=training)
    return x_pred

def avg_measure(trajectory, measure_fn, n_boids=None, coord=0, **kwargs):
    n_boids_total = trajectory.shape[-2]
    measures = []
    # Ensure we don't try to sample more boids than exist
    sample_size = min(n_boids if n_boids else 5, n_boids_total)
    for i in np.random.permutation(n_boids_total)[:sample_size]:
        measures.append(measure_fn(trajectory[:, i, coord], **kwargs))

    mn, std = np.mean(measures), np.std(measures)
    print(f"{measure_fn.__name__} {mn} +- {std}")
    return np.array(measures)

def convert_to_tf_sparse(a):
    """Safe conversion from scipy sparse to TF SparseTensor using standard API"""
    # 1. Get the indices where the connections are (row, col)
    # a.row and a.col come from the Scipy COO matrix
    indices = np.stack([a.row, a.col], axis=1)
    
    # 2. Create the SparseTensor
    a_tf = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape
    )
    
    # 3. Always reorder to ensure the sparse indices are in canonical order
    return tf.sparse.reorder(a_tf)

def evaluate(model, forward, max_trajectory_len, n_boids, init_blob=False):
    """
    Minimal evaluation: only autoregressive GNCA trajectory and animation using fixed init.
    """
    np.random.seed(0)

    # Create a stateless Boids instance for neighbor calculation and static info
    boids = Boids(n_boids=n_boids)
    positions, velocities, _ = boids.get_random_init(n_boids)
    x = np.concatenate([positions, velocities], axis=-1)
    goals = boids.goal_positions
    borders = boids.borders

    # Initial adjacency
    a_scipy = boids.get_neighbors(positions)
    def to_tf_sparse(a):
        indices = np.stack([a.row, a.col], axis=1)
        a_tf = tf.SparseTensor(indices=indices, values=a.data.astype(np.float32), dense_shape=a.shape)
        return tf.sparse.reorder(a_tf)
    a = to_tf_sparse(a_scipy)

    # Autoregressive GNCA trajectory
    boid_trajectory_auto = [x.astype(np.float32)]
    for t in range(max_trajectory_len-1):
        x_last = boid_trajectory_auto[-1]
        # Update neighbors based on GNCA's own previous prediction
        a_scipy = boids.get_neighbors(x_last[:, :2])
        a = to_tf_sparse(a_scipy)
        # Forward pass
        # For GNCA, you may need to pass goal info as well if required by your model
        # Here, we assume only (x, a) are needed
        x_next = forward(model, x_last, a, np.zeros((n_boids, 1)), training=False)
        boid_trajectory_auto.append(x_next.numpy())

    boid_trajectory_auto = np.array(boid_trajectory_auto)

    # Save autoregressive trajectory as PDF
    plt.figure(figsize=(8, 6))
    indices = np.random.permutation(n_boids)[:5]
    for i, boid_idx in enumerate(indices):
        label = "GNCA" if i == 0 else None
        plt.plot(*boid_trajectory_auto[:, boid_idx, :2].T, label=label, c="g", lw=2)
    plt.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150, label='Goals')
    plt.title("Autoregressive (Full Flight) Paths")
    plt.legend()
    plt.savefig("boids_auto_fixed.pdf")
    plt.close()

    # Animation
    fig, ax = plt.subplots(figsize=(7, 7))
    writer = FFMpegWriter(fps=20)
    print("🎬 Saving GNCA flight to gnca_boids.mp4...")
    with writer.saving(fig, "gnca_boids.mp4", dpi=100):
        for i in range(len(boid_trajectory_auto)):
            ax.clear()
            pos = boid_trajectory_auto[i][:, :2]
            ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150)
            ax.scatter(pos[:, 0], pos[:, 1], c='lime', s=20, edgecolors='k')
            ax.set_xlim(borders[0], borders[2])
            ax.set_ylim(borders[1], borders[3])
            ax.set_title(f"Step {i}")
            writer.grab_frame()
    print("✅ Done! Check your workspace folder for gnca_boids.mp4")

def evaluate_complexity(model, forward, te_set_size, trajectory_len, n_boids, init_blob=False):
    """
    Runs multiple randomized test trajectories to calculate average SampEn and CorrDim.
    """
    np.random.seed(0)
    all_measures = []
    
    for i in range(te_set_size):
        # We rely on get_random_init() for the 'clump', so we keep init=None
        data_te, boids_te = make_dataset(
            1,
            trajectory_len,
            random_init=True,
            return_boids=True,
            n_boids=n_boids,
            n_jobs=1,
            init=None, 
        )
        loader_te = DisjointLoader(data_te, node_level=True, epochs=1, shuffle=False)

        boid_trajectory_true = []
        boid_trajectory_auto = []
        
        for sample in loader_te:
            inputs, x_next = sample
            
            if len(boid_trajectory_auto) == 0:
                # Synchronize start: use the first prediction as the seed
                x_start_pred = forward(model, *inputs, training=False)
                boid_trajectory_auto.append(x_start_pred)
            else:
                x_last = boid_trajectory_auto[-1]
                # Re-calculate neighbors for the GNN's current position
                a_scipy = boids_te.get_neighbors(x_last[:, :2])
                a = convert_to_tf_sparse(a_scipy)
                
                # Forward pass using GNN's own previous output
                inputs_auto = [x_last, a, inputs[-1]]
                x_next_auto = forward(model, *inputs_auto, training=False)
                boid_trajectory_auto.append(x_next_auto)

            boid_trajectory_true.append(x_next)

        # Convert to numpy for complexity analysis
        traj_true = np.array(boid_trajectory_true)
        traj_auto = np.array(boid_trajectory_auto)

        # Calculate metrics for this specific trajectory
        m_true_samp = avg_measure(traj_true, nolds.sampen)
        m_auto_samp = avg_measure(traj_auto, nolds.sampen)
        m_true_corr = avg_measure(traj_true, nolds.corr_dim, emb_dim=10)
        m_auto_corr = avg_measure(traj_auto, nolds.corr_dim, emb_dim=10)
        
        all_measures.append((m_true_samp, m_auto_samp, m_true_corr, m_auto_corr))

    # Convert to array and calculate final averages/stds
    measures = np.array(all_measures)
    measures_mean = np.mean(measures, (0, -1))
    measures_std = np.std(measures, (0, -1))
    
    print(f"\nFINAL COMPLEXITY STATS OVER {te_set_size} RUNS:")
    print(f"SampEn True: {measures_mean[0]:.6f} +- {measures_std[0]:.6f}")
    print(f"SampEn GNCA: {measures_mean[1]:.6f} +- {measures_std[1]:.6f}")
    print(f"CorrDim True: {measures_mean[2]:.6f} +- {measures_std[2]:.6f}")
    print(f"CorrDim GNCA: {measures_mean[3]:.6f} +- {measures_std[3]:.6f}")
    return measures_mean, measures_std