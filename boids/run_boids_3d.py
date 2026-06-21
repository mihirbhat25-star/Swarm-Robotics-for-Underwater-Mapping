"""
Trains the 3D GNCA to imitate the 3D Boids GCA.
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
from boids.evaluate_boids_3d import evaluate_3d
from boids.forward import forward
from models.gnn_ca_simple_boids_3d import GNNCASimpleBoids3D
from modules.boids_3d import make_dataset_3d
import h5py

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
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError(
            f"Invalid centers shape in '{npz_path}': {centers.shape}. Expected (N, 3) with N > 0."
        )

    print(f"✅ Loaded {len(centers)} initial start centers from {npz_path}")
    return [np.array(c, dtype=np.float32) for c in centers]

def to_tf_sparse(a):
    """Convert scipy sparse matrix to TensorFlow sparse tensor."""
    indices = np.stack([a.row, a.col], axis=1)
    a_tf = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape
    )
    return tf.sparse.reorder(a_tf)


def build_run_tag_3d(unique_reps, tr_repeats, upweight, critical_distance, distance_weight, noise_config):
    """Build run tag for 3D model saving."""
    cd_dw_tag = f"_cd_{critical_distance:g}_dw_{distance_weight:g}" if upweight else ""
    return (
        f"{unique_reps}x{tr_repeats}_3d"
        + ("_newl" if upweight else "_oldl")
        + cd_dw_tag
        + (f"_{noise_config}" if noise_config else "")
    )

def print_run_parameters_3d(args, upweight, critical_distance, distance_weight, noise_config, run_tag):
    """Print training parameters before starting."""
    print("\n" + "="*70)
    print("3D GNCA TRAINING PARAMETERS")
    print("="*70)
    print(f"Run tag:              {run_tag}")
    print(f"Loss type:            {'newl (distance-weighted)' if upweight else 'oldl (standard MSE)'}")
    if upweight:
        print(f"  Critical distance:  {critical_distance}")
        print(f"  Distance weight:    {distance_weight}")
    print(f"Training set:         {args.tr_set_unique} unique × {args.tr_set_repeats} repeats")
    print(f"Validation set:       {args.va_set_size} unique")
    print(f"Boids:                {args.n_boids}")
    print(f"Noise config:         {noise_config or ''}")
    print(f"Learning rate:        {args.lr}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Epochs:               {args.epochs} (early stop patience: {args.es_patience})")
    print(f"Visualization mode:   {args.viz_mode}")
    print("="*70 + "\n")

def custom_weighted_mse_3d(y_true, y_pred):
    """
    Custom weighted MSE loss for 3D GNCA.
    y_true is [current_state (6D), next_state (6D), current_goal (3D)].
    y_pred is predicted next_state (6D).
    """
    n_features = tf.shape(y_pred)[-1]  # Should be 6
    current_state = y_true[..., :n_features]
    next_state = y_true[..., n_features:2*n_features]
    
    mse = tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)

    if UPWEIGHT_NEAR_GOAL:
        avg_pos = tf.reduce_mean(current_state[..., :3], axis=-2)
        current_goal = y_true[..., 2*n_features:]
        avg_goal = tf.reduce_mean(current_goal, axis=-2)
        dist_to_goal = tf.norm(avg_pos - avg_goal, axis=-1)
        goal_weight = tf.where(dist_to_goal < args.critical_distance, args.distance_weight, 1.0)
    else:
        goal_weight = 1.0

    return mse * goal_weight

class BestAfterEpochCallback3D(tf.keras.callbacks.Callback):
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

def run(data_tr, data_va, run_tag=""):
    model = GNNCASimpleBoids3D(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    optimizer = Adam(learning_rate=args.lr)
    
    # Use custom loss with global UPWEIGHT_NEAR_GOAL
    loss_fn = lambda y_true, y_pred: custom_weighted_mse_3d(y_true, y_pred)
    
    model.compile(optimizer=optimizer, loss=loss_fn)

    loader_tr = DisjointLoader(data_tr, node_level=True, batch_size=args.batch_size)
    loader_va = DisjointLoader(data_va, node_level=True, batch_size=args.batch_size)

    checkpoint_path = f"saved_models/best_weights_3d_{run_tag}"
    best_after50_cb = BestAfterEpochCallback3D(save_path=checkpoint_path, min_epoch=50)

    history = model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=args.epochs,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        callbacks=[
            EarlyStopping(patience=args.es_patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=args.lr_patience, min_delta=1e-8, verbose=1),
            best_after50_cb,
        ],
    )

    return history, model, best_after50_cb

####################################################################################
# Configuration
####################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch_size", default=30, type=int)
parser.add_argument("--epochs", default=1000000, type=int)
parser.add_argument("--es_patience", default=12, type=int)
parser.add_argument("--lr_patience", default=10, type=int)
parser.add_argument("--lr_red_factor", default=0.1, type=float)
parser.add_argument("--n_boids", default=100, type=int)
parser.add_argument("--tr_set_unique", default=20, type=int)
parser.add_argument("--tr_set_repeats", default=50, type=int)
parser.add_argument("--va_set_size", default=10, type=int)
parser.add_argument("--te_set_size", default=30, type=int)
parser.add_argument("--loss_type", default="oldl", choices=["oldl", "newl"],
                   help="oldl: standard MSE, newl: distance-weighted MSE")
parser.add_argument("--critical_distance", default=2.5, type=float,
                   help="Critical distance for distance-weighted loss (only used with --loss_type newl)")
parser.add_argument("--distance_weight", default=2.5, type=float,
                   help="Upweighting factor for boids within critical_distance (only used with --loss_type newl)")
parser.add_argument("--viz_mode", default="tubular", choices=["tubular", "per_boid", "multi_tubular", "individual"],
                   help="Visualization mode after training")
parser.add_argument("--noise_tag", default="", type=str,
                   help="String suffix to append in run_tag for noise setting (e.g. 'nw2'). Use empty string for no-noise tags.")
parser.add_argument("--init_centers_npz", default="", type=str,
                   help="Optional NPZ path containing key 'centers' (N,3).")
parser.add_argument("--boids_cache", default="", type=str,
                   help="Path to a precomputed 3D boids cache HDF5 (from generate_boids_cache_3d.py).")
parser.add_argument("--timestep_stride", default=1, type=int,
                   help="Load every Nth timestep from cache (default 1 = all).")
parser.add_argument("--viz_n_centers", default=50, type=int,
                   help="Number of centers to use for the test visualization (default: 50).")
parser.add_argument("--viz_centers_source", default="trained", choices=["trained", "unseen_cache", "random"],
                   help="Source of centers for test visualization.")


####################################################################################
# Training
####################################################################################
args = parser.parse_args()

# Determine if using upweighted loss
UPWEIGHT_NEAR_GOAL = (args.loss_type == "newl")

# Set noise config
noise_config = args.noise_tag

# Load initial centers from NPZ if provided
train_init_centers = load_init_centers_from_npz(args.init_centers_npz)
effective_unique_reps = len(train_init_centers) if train_init_centers is not None else args.tr_set_unique

boids_cache = args.boids_cache if args.boids_cache else None
if boids_cache:
    with h5py.File(boids_cache, 'r') as _f:
        cache_total_unique = int(_f.attrs["unique_reps"])
    print(f">>> Using precomputed 3D cache: {boids_cache} ({cache_total_unique} total unique centers, training on {effective_unique_reps})")

# Build run tag
run_tag = build_run_tag_3d(
    effective_unique_reps, 
    args.tr_set_repeats, 
    UPWEIGHT_NEAR_GOAL,
    args.critical_distance,
    args.distance_weight,
    noise_config
)

# Print parameters
print_run_parameters_3d(args, UPWEIGHT_NEAR_GOAL, args.critical_distance, args.distance_weight, noise_config, run_tag)

print(f"\n>>> Generating 3D dataset...")
data_tr, boids_tr = make_dataset_3d(
    unique_reps=effective_unique_reps,
    repeat_reps=args.tr_set_repeats,
    save_config=train_init_centers is None and not boids_cache,
    n_boids=args.n_boids,
    n_jobs=1,
    return_boids=True,
    random_init=train_init_centers if train_init_centers is not None else True,
    boids_cache_npz=boids_cache,
    timestep_stride=args.timestep_stride,
)

if train_init_centers is not None:
    boids_tr.rand_configs = [np.array(c, dtype=np.float32) for c in train_init_centers]
# else: boids_tr.rand_configs already populated during make_dataset_3d with save_config=True

# Validation uses random centers (robustness testing)
data_va = make_dataset_3d(
    unique_reps=args.va_set_size,
    repeat_reps=1,
    save_config=False,
    n_boids=args.n_boids,
    n_jobs=1,
    random_init=True,
)

history, model, best_after50_cb = run(
    data_tr,
    data_va,
    run_tag=run_tag,
)

# Save model, boids, history
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_boids_tr", exist_ok=True)
os.makedirs("saved_history", exist_ok=True)

model_save_path    = f"saved_models/gnca_model_3d_{run_tag}"
boids_save_path    = f"saved_boids_tr/boids_tr_3d_{run_tag}.pkl"
history_save_path  = f"saved_history/history_3d_{run_tag}.pkl"

model.save(model_save_path, save_format="tf")
print(f"✅ Saved model to {model_save_path}")
joblib.dump(boids_tr, boids_save_path)
print(f"✅ Saved boids_tr to {boids_save_path}")
joblib.dump(history.history, history_save_path)
print(f"✅ Saved training history to {history_save_path}")

# Restore best post-epoch-50 weights onto the existing model object
checkpoint_path = f"saved_models/best_weights_3d_{run_tag}"
if best_after50_cb.best_epoch is not None:
    print(f"\n✅ Restoring best post-epoch-50 weights (epoch {best_after50_cb.best_epoch}, val_loss={best_after50_cb.best_val_loss:.6f})...")
    model.load_weights(checkpoint_path)
else:
    print(f"\n⚠️ No post-epoch-50 checkpoint (training ended before epoch 50). Using early-stopped model.")

####################################################################################
# Evaluation
####################################################################################
max_trajectory_len = 1250

# Assemble test centers for visualization
if args.viz_centers_source == "trained":
    test_centers = boids_tr.rand_configs[:args.viz_n_centers] if boids_tr.rand_configs else None
    print(f">>> Viz test centers: {len(test_centers) if test_centers else 0} trained centers")
elif args.viz_centers_source == "unseen_cache":
    if not boids_tr.unseen_configs:
        print("⚠️ No unseen_cache centers available. Falling back to random.")
        test_centers = None
    else:
        test_centers = boids_tr.unseen_configs[:args.viz_n_centers]
        print(f">>> Viz test centers: {len(test_centers)} unseen cache centers")
else:
    test_centers = None
    print(f">>> Viz test centers: random (fresh)")

evaluate_3d(model, max_trajectory_len, args.n_boids,
            saved_boids=boids_tr,
            run_tag=run_tag, viz_mode=args.viz_mode,
            test_centers=test_centers)
