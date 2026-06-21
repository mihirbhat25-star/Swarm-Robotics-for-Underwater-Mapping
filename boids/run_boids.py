"""
Trains the GNCA to imitate the Boids algorithm.
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
import h5py

# tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_init_centers_from_npz(npz_path):
    """Load start-center configs from an NPZ file (expects a 'centers' array)."""
    if not npz_path:
        return None
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Initial-center npz not found: {npz_path}")

    cache = np.load(npz_path, allow_pickle=True)
    if "centers" not in cache.files:
        raise KeyError(
            f"NPZ '{npz_path}' does not contain a 'centers' key. Available keys: {cache.files}"
        )

    centers = np.array(cache["centers"], dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
        raise ValueError(
            f"Invalid centers shape in '{npz_path}': {centers.shape}. Expected (N, 2) with N > 0."
        )

    print(f"✅ Loaded {len(centers)} initial start centers from {npz_path}")
    return [np.array(c, dtype=np.float32) for c in centers]

def confirm_centers_match_reference(candidate_centers, reference_path):
    """Print a debug confirmation showing whether candidate centers match a reference run."""
    if not candidate_centers:
        print("⚠️ No candidate centers available for reference comparison.")
        return
    if not reference_path or not os.path.exists(reference_path):
        print(f"⚠️ Reference centers file not found for comparison: {reference_path}")
        return

    if reference_path.endswith(".npz"):
        reference_cache = np.load(reference_path, allow_pickle=True)
        if "centers" not in reference_cache.files:
            print(f"⚠️ Reference NPZ '{reference_path}' does not contain centers. Available keys: {reference_cache.files}")
            return
        reference_centers = np.array(reference_cache["centers"], dtype=np.float32)
    else:
        reference_boids = joblib.load(reference_path)
        reference_centers = np.array(getattr(reference_boids, "rand_configs", []), dtype=np.float32)
    candidate_arr = np.array(candidate_centers, dtype=np.float32)

    if reference_centers.size == 0:
        print(f"⚠️ Reference file '{reference_path}' does not contain rand_configs.")
        return

    same_shape = candidate_arr.shape == reference_centers.shape
    same_values = same_shape and np.allclose(candidate_arr, reference_centers)
    max_abs_diff = float(np.max(np.abs(candidate_arr - reference_centers))) if same_shape else float("nan")

    print(
        f"🔎 Center comparison vs '{reference_path}': "
        f"same_shape={same_shape}, same_values={same_values}, max_abs_diff={max_abs_diff:.8f}"
    )

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

    # Weight by the configured factor if within the configured critical distance of the goal.
    goal_weight = tf.where(dist_to_goal < args.critical_distance, args.distance_weight, 1.0) if UPWEIGHT else 1.0
    return mse * goal_weight


def build_run_tag(unique_reps, tr_repeats, upweight, critical_distance, distance_weight, noise_config):
    cd_dw_tag = f"_cd_{critical_distance:g}_dw_{distance_weight:g}" if upweight else ""
    return (
        f"{unique_reps}x{tr_repeats}"
        + ("_newl" if upweight else "_oldl")
        + cd_dw_tag
        + (f"_{noise_config}" if noise_config else "")
    )

def print_run_parameters(args, upweight, noise_config, effective_unique_reps, run_tag):
    print("\n>>> Full run parameters (before training):")
    for key in sorted(vars(args).keys()):
        print(f"  {key}: {getattr(args, key)}")
    print(f"  UPWEIGHT: {upweight}")
    print(f"  noise_config: {noise_config}")
    print(f"  effective_unique_reps: {effective_unique_reps}")
    print(f"  run_tag: {run_tag}")

class BestAfterEpochCallback(tf.keras.callbacks.Callback):
    """Saves weights whenever val_loss improves after min_epoch."""
    def __init__(self, save_path, min_epoch=50):
        super().__init__()
        self.save_path = save_path
        self.min_epoch = min_epoch
        self.best_val_loss = float('inf')
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.min_epoch:
            val_loss = logs.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.model.save_weights(self.save_path)
                print(f"\n💾 New best val_loss after epoch 50: {val_loss:.6f} at epoch {self.best_epoch}")

def run(data_tr, data_va, run_tag):

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

    checkpoint_path = f"saved_models/best_weights_{run_tag}"
    best_after50_cb = BestAfterEpochCallback(save_path=checkpoint_path, min_epoch=50)

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
            best_after50_cb,
        ],
    )

    return history, model, best_after50_cb

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
    "--es_patience", default=12, type=int, help="Patience for early stopping"
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
    "--tr_set_unique", default=50, type=int, help="N. of unique training trajectories"
)
parser.add_argument(
    "--tr_set_repeats", default=20, type=int, help="N. of repeats for each training trajectory"
)
parser.add_argument(
    "--va_set_size", default=10, type=int, help="N. of valid. trajectories"
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
parser.add_argument(
    "--init_centers_npz",
    default="",
    type=str,
    help="Optional NPZ path containing key 'centers' (N,2). If provided, training uses these centers and repeats each by tr_set_repeats.",
)
parser.add_argument(
    "--loss_type",
    default="oldl",
    choices=["oldl", "newl"],
    help="Loss mode: 'oldl' = plain MSE, 'newl' = near-goal upweighted MSE.",
)
parser.add_argument(
    "--critical_distance",
    default=2.5,
    type=float,
    help="Distance to goal within which near-goal upweighting is applied when using loss_type=newl.",
)
parser.add_argument(
    "--distance_weight",
    default=2.5,
    type=float,
    help="Weight multiplier used inside the critical distance when using loss_type=newl.",
)
parser.add_argument(
    "--noise_tag",
    default="nw2",
    type=str,
    help="String suffix to append in run_tag for noise setting (e.g. 'nw2'). Use empty string for no-noise tags.",
)
parser.add_argument(
    "--n_jobs",
    default=1,
    type=int,
    help="Number of parallel jobs for dataset generation (-1 = all cores).",
)
parser.add_argument(
    "--boids_cache",
    default="",
    type=str,
    help="Path to a precomputed boids cache HDF5 file (from generate_boids_cache.py). If set, skips simulation entirely.",
)
parser.add_argument(
    "--timestep_stride",
    default=1,
    type=int,
    help="Load every Nth timestep from cache (default 1 = all). Increase to reduce RAM usage (e.g. 10 = 10x fewer samples).",
)
parser.add_argument(
    "--viz_n_centers",
    default=50,
    type=int,
    help="Number of centers to use for the test/robustness visualization (default: 50).",
)
parser.add_argument(
    "--viz_centers_source",
    default="trained",
    choices=["trained", "unseen_cache", "random"],
    help="Source of centers for the test visualization: 'trained' (from training set), 'unseen_cache' (cache centers not used in training), 'random' (fresh random, default).",
)
parser.add_argument(
    "--viz_mode",
    default="tubular",
    choices=["tubular", "per_boid", "multi_tubular", "individual"],
    help="Visualization mode: 'tubular' (mean + tube), 'per_boid' (one run, all boids), 'multi_tubular' (multiple tubes), 'individual' (ranked PDFs).",
)
parser.add_argument(
    "--viz_n_show",
    default=None,
    type=int,
    help="Number of tubes to show in multi_tubular mode (default: use all training runs).",
)
parser.add_argument(
    "--viz_runs",
    type=int,
    nargs="+",
    default=None,
    help="Specific run indices to plot in multi_tubular mode (e.g., 0 5 10 15).",
)

####################################################################################
# Training
####################################################################################

args = parser.parse_args()

UPWEIGHT = args.loss_type == "newl"
print(
    f"\n>>> Loss config: {args.loss_type} (UPWEIGHT={UPWEIGHT}, critical_distance={args.critical_distance}, distance_weight={args.distance_weight})"
)

noise_config = args.noise_tag
train_init_centers = load_init_centers_from_npz(args.init_centers_npz)
if train_init_centers is not None:
    confirm_centers_match_reference(train_init_centers, args.init_centers_npz)
effective_unique_reps = len(train_init_centers) if train_init_centers is not None else args.tr_set_unique
run_tag = build_run_tag(
    unique_reps=effective_unique_reps,
    tr_repeats=args.tr_set_repeats,
    upweight=UPWEIGHT,
    critical_distance=args.critical_distance,
    distance_weight=args.distance_weight,
    noise_config=noise_config,
)
print_run_parameters(args, UPWEIGHT, noise_config, effective_unique_reps, run_tag)

print(f"\n>>> Generating dataset (n_jobs=1)...")
# If a precomputed cache is provided, load from it; otherwise simulate
boids_cache = args.boids_cache if args.boids_cache else None
if boids_cache:
    with h5py.File(boids_cache, 'r') as _f:
        cache_total_unique = int(_f.attrs["unique_reps"])
    print(f">>> Using precomputed cache: {boids_cache} ({cache_total_unique} total unique centers, training on {effective_unique_reps})")

data_tr, boids_tr = make_dataset(
    unique_reps=effective_unique_reps,
    repeat_reps=args.tr_set_repeats,
    save_config=train_init_centers is None and not boids_cache,
    n_boids=args.n_boids,
    n_jobs=args.n_jobs,
    return_boids=True,
    random_init=train_init_centers if train_init_centers is not None else True,
    boids_cache_npz=boids_cache,
    timestep_stride=args.timestep_stride,
)

if train_init_centers is not None: # What is this for? Preserving the initial centers for evaluation. What does the else go to? It 
    # Preserve reused center list for downstream evaluate(use_saved_config=True)
    boids_tr.rand_configs = [np.array(c, dtype=np.float32) for c in train_init_centers]
# else: boids_tr.rand_configs already populated during make_dataset with save_config=True

data_va = make_dataset(
    unique_reps=args.va_set_size,
    repeat_reps=1,
    save_config=False,
    n_boids=args.n_boids,
    n_jobs=args.n_jobs,
    random_init=True,
)

history, model, best_after50_cb = run(data_tr, data_va, run_tag)

print(f"\n>>> Saving model and history with run_tag='{run_tag}'...")
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_boids_tr", exist_ok=True)
os.makedirs("saved_history", exist_ok=True)
model.save(f"saved_models/gnca_model_{run_tag}", save_format="tf")
joblib.dump(history.history, f"saved_history/history_{run_tag}.pkl")
joblib.dump(boids_tr, f"saved_boids_tr/boids_tr_{run_tag}.pkl")

# Restore best post-epoch-50 weights onto the existing model object
checkpoint_path = f"saved_models/best_weights_{run_tag}"
if best_after50_cb.best_epoch is not None:
    print(f"\n✅ Restoring best post-epoch-50 weights (epoch {best_after50_cb.best_epoch}, val_loss={best_after50_cb.best_val_loss:.6f})...")
    model.load_weights(checkpoint_path)
else:
    print(f"\n⚠️ No post-epoch-50 checkpoint found (training ended before epoch 50). Using early-stopped model.")

####################################################################################
# Evaluation
####################################################################################
max_trajectory_len = 2000
n_boids = 100
init_blob = False

# Assemble test centers for robustness visualization
if args.viz_centers_source == "trained":
    test_centers = boids_tr.rand_configs[:args.viz_n_centers] if boids_tr.rand_configs else None
    print(f">>> Viz test centers: {len(test_centers) if test_centers else 0} trained centers")
elif args.viz_centers_source == "unseen_cache":
    if not boids_tr.unseen_configs:
        print("⚠️ No unseen_cache centers available (requires --boids_cache with more unique centers than --tr_set_unique). Falling back to random.")
        test_centers = None
    else:
        test_centers = boids_tr.unseen_configs[:args.viz_n_centers]
else:  # random
    test_centers = None
    print(f">>> Viz test centers: random (fresh)")

evaluate(
    model,
    forward,
    max_trajectory_len,
    n_boids,
    use_saved_config=True,
    saved_boids=boids_tr,
    init_blob=init_blob,
    viz_mode=args.viz_mode,
    max_viz_runs=effective_unique_reps,
    traj_cache_path=f"saved_trajectory/viz_trajectories_{run_tag}.npz",
    run_tag=run_tag,
    n_show=args.viz_n_show if args.viz_n_show is not None else effective_unique_reps,
    specific_runs=args.viz_runs,
    output_dir=".",
    test_centers=test_centers,
)

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