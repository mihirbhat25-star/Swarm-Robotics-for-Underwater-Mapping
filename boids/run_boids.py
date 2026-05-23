"""
Trains the GNCA to imitate the Boids GCA.
"""
import argparse
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from spektral.data import DisjointLoader
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from boids.evaluate_boids import evaluate
from boids.forward import forward
from models.gnn_ca_simple_boids import GNNCASimpleBoids
from modules.boids import make_dataset
from modules.callbacks import ComplexityCallback

# tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def custom_weighted_mse(y_true, y_pred):
    # y_true is [current_state, next_state, current_goal_pos], y_pred is predicted next_state
    n_features = tf.shape(y_pred)[-1]
    current_state = y_true[..., :n_features]
    next_state = y_true[..., n_features:2*n_features]
    # Current goal position is the last 2 columns (broadcast to all nodes, same value per timestep)
    current_goal = y_true[..., 2*n_features:]
    mse = tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)

    # Compute average position and extract current goal (same for all nodes)
    avg_pos = tf.reduce_mean(current_state[..., :2], axis=-2)
    avg_goal = tf.reduce_mean(current_goal, axis=-2)  # all nodes share same goal value
    dist_to_goal = tf.norm(avg_pos - avg_goal, axis=-1)

    # Weight by 2.5 if within 3 units of the current goal, else 1
    goal_weight = tf.where(dist_to_goal < 2.5, 2.5, 1.0)
    return mse * goal_weight

def run(data_tr, data_va):

    model = GNNCASimpleBoids(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    optimizer = Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss=custom_weighted_mse)

    loader_tr = DisjointLoader(data_tr, node_level=True, batch_size=args.batch_size)
    loader_va = DisjointLoader(data_va, node_level=True, batch_size=args.batch_size)

    history = model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=args.epochs, # Modified to use args
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        callbacks=[
            EarlyStopping(
                patience=args.es_patience, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(patience=args.lr_patience, min_delta=1e-8, verbose=1),
            # RESTORED: Complexity Callback
            ComplexityCallback(test_every=args.test_complexity_every),
        ],
    )

    return history, model

####################################################################################
# Configuration
####################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3, type=float, help="Initial LR")
parser.add_argument(
    "--batch_size", default=30, type=int, help="Size of the mini-batches"
)
parser.add_argument(
    "--epochs", default=1000000, type=int, help="Number of training epochs"
)
parser.add_argument(
    "--es_patience", default=80, type=int, help="Patience for early stopping"
)
parser.add_argument(
    "--lr_patience", default=10, type=int, help="Patience for LR annealing"
)
parser.add_argument(
    "--lr_red_factor", default=0.1, type=float, help="Rate for LR annealing"
)
parser.add_argument(
    "--n_boids", default=100, type=int, help="N. of boids in simulation"
)
parser.add_argument(
    "--trajectory_len", default=300, type=int, help="Length of trajectories"
)
parser.add_argument(
    "--tr_set_unique", default=100, type=int, help="N. of unique training trajectories"
)
parser.add_argument(
    "--tr_set_repeats", default=10, type=int, help="N. of repeats for each training trajectory"
)
parser.add_argument(
    "--va_set_size", default=30, type=int, help="N. of valid. trajectories"
)
parser.add_argument(
    "--te_set_size", default=30, type=int, help="N. of test trajectories"
)
parser.add_argument(
    "--test_complexity_every",
    default=-1,
    type=int,
    help="How often to test for complexity (-1 for never)",
)

####################################################################################
# Training
####################################################################################

args = parser.parse_args()

print(f"\n>>> Generating dataset (n_jobs=1)...")
# Generate training set and save the random initial centers used
data_tr, boids_tr = make_dataset(
    unique_reps=args.tr_set_unique, repeat_reps=args.tr_set_repeats, save_config=True, trajectory_len=args.trajectory_len, n_boids=args.n_boids, n_jobs=1, return_boids=True
)

# Save the random initial centers used for training
train_init_centers = boids_tr.rand_configs.copy() if hasattr(boids_tr, 'rand_configs') else None

# For validation, use the same initial centers as training (repeat each once)
def make_val_init_list(centers, n_boids, max_speed):
    # Returns a list of (positions, velocities) tuples for each center
    val_inits = []
    direction = np.array([1.0, 0.0])
    velocity = direction * max_speed
    for center in centers:
        positions = center + 0.325 * np.random.rand(n_boids, 2)
        velocities = np.tile(velocity, (n_boids, 1))
        val_inits.append((positions, velocities))
    return val_inits

if train_init_centers is not None and len(train_init_centers) == args.tr_set_unique:
    # Use the saved centers directly for validation
    val_centers = np.array(train_init_centers)
    # Debug: check if validation centers match training centers
    # if np.allclose(val_centers, train_init_centers):
    #     print("[DEBUG] Validation initial configs MATCH training centers.")
    # else:
    #     print("[DEBUG] Validation initial configs DO NOT match training centers!")
    data_va = make_dataset(
        unique_reps=args.tr_set_unique, repeat_reps=1, save_config=False, trajectory_len=args.trajectory_len, n_boids=args.n_boids, n_jobs=1, random_init=train_init_centers
    )
else:
    # Fallback: old behavior
    data_va = make_dataset(
        unique_reps=args.va_set_size, repeat_reps=1, save_config=False, trajectory_len=args.trajectory_len, n_boids=args.n_boids, n_jobs=1
    )

history, model = run(data_tr, data_va)

run_tag = f"{args.tr_set_unique}x{args.tr_set_repeats}"
model.save(f"gnca_model_{run_tag}", save_format="tf")
joblib.dump(history.history, f"history_{run_tag}.pkl")
joblib.dump(boids_tr, f"boids_tr_{run_tag}.pkl")

####################################################################################
# Evaluation
####################################################################################
max_trajectory_len = 2000
n_boids = 100
init_blob = False
evaluate(model, forward, max_trajectory_len, n_boids, use_saved_config=True, saved_boids=boids_tr, init_blob=init_blob, viz_mode='tubular', traj_cache_path=f"viz_trajectories_{run_tag}.npz")

####################################################################################
# Plot SampEn and Correlation Dimension
####################################################################################
if args.test_complexity_every > 0:
    # This assumes complexities.npz was generated by the callback
    if os.path.exists("complexities.npz"):
        c = np.load("complexities.npz")["complexities"]
        means = c[:, 0, :]
        stds = c[:, 1, :]
        x = np.arange(means.shape[0] - 1) * 10

        plt.figure(figsize=(4.5, 2))
        plt.subplot(121)
        plt.axhline(means[:, 0].mean(), label="True")
        plt.plot(x, means[:-1, 1], " x", label="GNCA")
        plt.xlabel("Epoch")
        plt.ylabel("SampEn")
        plt.xticks(x[::2])
        plt.legend()

        plt.subplot(122)
        plt.axhline(means[:, 2].mean(), label="True")
        plt.plot(x, means[:-1, 3], " x", label="GNCA")
        plt.xlabel("Epoch")
        plt.ylabel("CD")
        plt.xticks(x[::2])
        plt.legend()
        plt.tight_layout()

        plt.savefig("complexities.pdf", bbox_inches="tight")