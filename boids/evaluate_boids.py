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

def evaluate(model, forward, reps_unique, repeat_reps, n_boids, loiter, anim_gen, time_bool):
    """
    Evaluates the GNCA by comparing it to the true Boids math.
    Uses randomized starting clumps to test the model's robustness.
    """
    # Set seed for reproducibility of this specific test run
    np.random.seed(0) 
    
    # Initialize the FFMpegWriter instance
    writer = FFMpegWriter(fps=20)
    
    # Initialize the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(7, 7))

    # 1. Generate the ground truth trajectory
    # By passing init=None, it forces the use of your new clumped get_random_init()
    data_te, boids_te = make_dataset(
        reps_unique=1,
        repeat_reps=1,
        random_init=True,
        fixed_init=False,
        return_boids=True,
        loiter=loiter,
        time_bool=time_bool,
        n_boids=n_boids,
        n_jobs=1,                     
    )
    loader_te = DisjointLoader(data_te, node_level=True, epochs=1, shuffle=False)

    # Ensure goals are explicitly assigned from boids_te
    goals = boids_te.goal_positions

    boid_trajectory_true = []
    boid_trajectory_pred = [] # One-step predictions
    boid_trajectory_auto = [] # Autoregressive (recursive) predictions
    avg_degree_trajectory_true = []
    avg_degree_trajectory_auto = []
    
    for sample in loader_te:
        inputs, x_next = sample
        
        # Compute the one-step prediction (standard GNCA behavior)
        x_next_pred = forward(model, *inputs, training=False)
        avg_degree_trajectory_true.append(np.average(ops.degrees(inputs[1]).numpy()))
        
        if len(boid_trajectory_auto) == 0:
            # SYNC POINT: To compare fairly, the GNCA must start at the 
            # exact same location as the first step of the True trajectory.
            boid_trajectory_auto.append(x_next_pred)
        else:
            x_last = boid_trajectory_auto[-1]
            
            # Update neighbors based on the GNCA's own previous movement
            a_scipy = boids_te.get_neighbors(x_last[:, :2])
            a = convert_to_tf_sparse(a_scipy) # Using the safe TF conversion function
            
            avg_degree_trajectory_auto.append(np.average(ops.degrees(a).numpy()))

            # Recursive step: GNCA uses its OWN previous prediction as input
            # inputs[-1] provides the Goal coordinates which remain constant/external
            inputs_auto = [x_last, a, inputs[-1]]
            x_next_auto = forward(model, *inputs_auto, training=False)
            boid_trajectory_auto.append(x_next_auto)

        boid_trajectory_true.append(x_next)
        boid_trajectory_pred.append(x_next_pred.numpy())

    # Finalize arrays for plotting and metrics
    boid_trajectory_true = np.array(boid_trajectory_true)
    boid_trajectory_pred = np.array(boid_trajectory_pred)
    boid_trajectory_auto = np.array(boid_trajectory_auto)

    if anim_gen:

        local_path = "gnca_boids.mp4"  # Save in the container's workspace
        print(f"🎬 Saving GNCA flight to {local_path}...")

        try:
            with writer.saving(fig, local_path, dpi=100):
                for i in range(len(boid_trajectory_auto)):
                    ax.clear()

                    # Pull positions (Latitude/Longitude)
                    pos = boid_trajectory_auto[i][:, :2]

                    # Draw everything
                    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=150)
                    ax.scatter(pos[:, 0], pos[:, 1], c='lime', s=20, edgecolors='k')

                    # Keep the view consistent
                    ax.set_xlim(boids_te.borders[0], boids_te.borders[2])
                    ax.set_ylim(boids_te.borders[1], boids_te.borders[3])
                    ax.set_title(f"Step {i}")

                    writer.grab_frame()

            print(f"✅ Done! Check your local path for {local_path}")
        except FileNotFoundError as e:
            print("Error: ffmpeg is not installed or the path is invalid.")
            return

    # Evaluation metrics
    # print("\n--- Evaluation Metrics ---")
    # print("True values:")
    # avg_measure(boid_trajectory_true, nolds.sampen)
    # avg_measure(boid_trajectory_true, nolds.corr_dim, emb_dim=10)
    # print("\nAuto (Recursive) values:")
    # avg_measure(boid_trajectory_auto, nolds.sampen)
    # avg_measure(boid_trajectory_auto, nolds.corr_dim, emb_dim=10)

    # --- Plotting 1: One-Step Comparison ---
    plt.figure(figsize=(8, 6))
    indices = np.random.permutation(boid_trajectory_auto.shape[-2])[:5]
    # for i, boid_idx in enumerate(indices):
    #     l_t = "True" if i == 0 else None
    #     l_g = "GNCA" if i == 0 else None
    #     plt.plot(*boid_trajectory_true[:, boid_idx, :2].T, label=l_t, c="k", alpha=0.3, ls='--')
    #     plt.plot(*boid_trajectory_pred[:, boid_idx, :2].T, label=l_g, c="g", lw=2)
    # plt.title("One-step Prediction Paths")
    # plt.legend()
    # plt.savefig(f"boids_pred_{reps_unique}_{repeat_reps}.pdf")

    # --- Plotting 2: Autoregressive Comparison ---
    plt.figure(figsize=(8, 6))
    for i, boid_idx in enumerate(indices):
        l_t = "True" if i == 0 else None
        l_g = "GNCA" if i == 0 else None
        plt.plot(*boid_trajectory_true[:, boid_idx, :2].T, label=l_t, c="k", alpha=0.3, ls='--')
        plt.plot(*boid_trajectory_auto[:, boid_idx, :2].T, label=l_g, c="g", lw=2)
    plt.title("Autoregressive (Full Flight) Paths")
    plt.legend()
    plt.savefig(f"boids_auto_{reps_unique}_{repeat_reps}.pdf")

    # --- Plotting 3: Average Degree (Neighbor Stability) ---
    # plt.figure()
    # plt.plot(avg_degree_trajectory_true, label="True", c="k", alpha=0.5)
    # plt.plot(avg_degree_trajectory_auto, label="GNCA", c="g")
    # plt.ylabel("Average Neighbor Count")
    # plt.xlabel("Step")
    # plt.legend()
    # plt.savefig(f"boids_avg_degree_{reps_unique}_{repeat_reps}.pdf")

    # plt.show()

    return boid_trajectory_true, boid_trajectory_pred, boid_trajectory_auto

def evaluate_complexity(model, forward, te_set_size, trajectory_len, n_boids, init_blob=False):
    """
    Runs multiple randomized test trajectories to calculate average SampEn and CorrDim.
    """
    np.random.seed(0)
    all_measures = []
    
    for i in range(te_set_size):
        # We rely on get_random_init() for the 'clump', so we keep init=None
        data_te, boids_te = make_dataset(
            reps_unique=1,
            repeat_reps=1,
            random_init=True,
            fixed_init=False,
            return_boids=True,
            n_boids=n_boids,
            n_jobs=1,
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