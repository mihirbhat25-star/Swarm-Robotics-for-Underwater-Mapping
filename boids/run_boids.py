"""
Trains the GNCA to imitate the Boids algorithm.
"""
import argparse
import gc
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
from modules.boids import make_dataset, Boids, load_chunk_from_cache
from modules.callbacks import ComplexityCallback
import h5py
import psutil

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

class _BestInfo:
    """Minimal checkpoint-info object compatible with the post-training restore block."""
    def __init__(self, best_epoch, best_val_loss):
        self.best_epoch    = best_epoch
        self.best_val_loss = best_val_loss


class _HistoryProxy:
    """Wraps a plain dict so it has the same .history attribute as a Keras History."""
    def __init__(self, history_dict):
        self.history = history_dict


def _build_model(lr):
    """Construct and compile a fresh GNNCASimpleBoids. Called once per chunk."""
    m = GNNCASimpleBoids(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    m.compile(optimizer=Adam(learning_rate=lr), loss=custom_weighted_mse, run_eagerly=True)
    return m


def run_chunked(boids_cache_path, flat_traj_indices, boundaries, cache_info, data_va, run_tag):
    """Sequential chunk fine-tuning with val_loss early stopping per chunk.

    For each chunk:
      1. tf.keras.backend.clear_session() — frees ALL TF memory from previous chunk.
      2. Rebuild model fresh and restore weights from previous chunk (numpy arrays).
      3. Load chunk data from HDF5.
      4. Train with val_loss EarlyStopping until this chunk converges.
      5. Save best weights as numpy arrays; track current LR.
      6. Free all Python/TF objects. Repeat.

    clear_session() between chunks prevents the cumulative TF eager-kernel and
    optimizer-state memory growth that causes OOM on long chunk sequences.
    """
    chunk_size     = args.chunk_size
    chunk_patience = args.chunk_patience
    n_boids_c      = cache_info['n_boids']
    stride           = cache_info['timestep_stride']
    near_goal_radius = cache_info.get('near_goal_radius', 1.0)
    cache_repeats    = cache_info.get('cache_repeats', 1)
    centers          = cache_info.get('centers', None)  # (cache_unique, 2) center coords

    os.makedirs("saved_models", exist_ok=True)

    # ── Build quadrant-balanced chunks ───────────────────────────────────
    from boids.generate_boids_cache import QUADRANTS

    def _center_quadrant(flat_idx):
        """Return quadrant index (0-3) for a flat trajectory index."""
        if centers is None:
            return 0
        ci = int(flat_idx) // cache_repeats
        cx, cy = centers[ci]
        for q, (xmn, xmx, ymn, ymx, _) in enumerate(QUADRANTS):
            if xmn <= cx <= xmx and ymn <= cy <= ymx:
                return q
        return 0

    # Group all flat indices by quadrant, then shuffle each bucket
    q_buckets = [[] for _ in range(4)]
    for fi in flat_traj_indices:
        q_buckets[_center_quadrant(fi)].append(fi)
    for q in range(4):
        np.random.shuffle(q_buckets[q])

    # Find non-empty quadrants and interleave from them only
    active_quads = [q for q in range(4) if len(q_buckets[q]) > 0]
    n_per_q   = chunk_size // len(active_quads) if active_quads else chunk_size
    remainder = chunk_size % len(active_quads) if active_quads else 0
    q_ptrs = [0, 0, 0, 0]
    balanced_chunks = []
    while any(q_ptrs[q] < len(q_buckets[q]) for q in active_quads):
        chunk = []
        for i, q in enumerate(active_quads):
            take = n_per_q + (1 if i < remainder else 0)
            end  = min(q_ptrs[q] + take, len(q_buckets[q]))
            chunk.extend(q_buckets[q][q_ptrs[q]:end])
            q_ptrs[q] = end
        if chunk:
            balanced_chunks.append(np.array(chunk, dtype=np.int64))
    n_chunks = len(balanced_chunks)

    # Weight state carried across chunks as plain numpy lists (survive clear_session).
    saved_weights_np = None   # None → chunk 1 starts from random init
    saved_lr         = args.lr
    last_chunk_losses = []
    total_trajs = sum(len(c) for c in balanced_chunks)

    print(f"\n>>> Chunked training: {total_trajs} trajectories, "
          f"chunk_size={chunk_size} ({n_chunks} balanced chunks), "
          f"chunk_patience={chunk_patience} (val_loss early stopping per chunk)")

    for chunk_num, chunk_idxs in enumerate(balanced_chunks, start=1):
        # ── Per-chunk quadrant breakdown (debug) ─────────────────────────
        q_counts = [0, 0, 0, 0]
        for fi in chunk_idxs:
            q_counts[_center_quadrant(fi)] += 1
        q_summary = "  ".join(
            f"{QUADRANTS[q][4]}:{q_counts[q]}" for q in range(4)
        )
        print(f"\n--- Chunk {chunk_num}/{n_chunks}  [{q_summary}] ---")

        # ── 1. Clear ALL TF state from previous chunk ────────────────────
        tf.keras.backend.clear_session()
        gc.collect()

        # ── 2. Rebuild model; restore previous chunk's weights ───────────
        model = _build_model(saved_lr)
        chunk_data = load_chunk_from_cache(
            boids_cache_path, chunk_idxs, boundaries, n_boids_c, stride, near_goal_radius
        )
        loader_tr = DisjointLoader(chunk_data, node_level=True, batch_size=args.batch_size)
        loader_va = DisjointLoader(data_va,    node_level=True, batch_size=args.batch_size)

        if saved_weights_np is not None:
            # Build the model with 1 step, then overwrite with saved weights.
            model.fit(loader_tr.load(), steps_per_epoch=1, epochs=1, verbose=0)
            model.set_weights(saved_weights_np)
            # Recreate loaders — generators were exhausted by the build step.
            loader_tr = DisjointLoader(chunk_data, node_level=True, batch_size=args.batch_size)
            loader_va = DisjointLoader(data_va,    node_level=True, batch_size=args.batch_size)

        # ── 3 & 4. Train until val_loss plateaus ─────────────────────────
        h = model.fit(
            loader_tr.load(),
            steps_per_epoch=loader_tr.steps_per_epoch,
            epochs=10_000,
            validation_data=loader_va.load(),
            validation_steps=loader_va.steps_per_epoch,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=chunk_patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(patience=args.lr_patience, min_delta=1e-8, verbose=1),
            ],
        )
        last_chunk_losses = h.history.get('loss', [])

        # ── 5. Persist weights & LR as numpy (survive clear_session) ─────
        saved_weights_np = model.get_weights()
        saved_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))

        # ── 6. Destroy all TF/Python objects ─────────────────────────────
        del chunk_data, loader_tr, loader_va, h, model
        gc.collect()

        try:
            rss_mb = psutil.Process().memory_info().rss / 1e6
            print(f"  [chunk {chunk_num}/{n_chunks}] RAM after cleanup: {rss_mb:.0f} MB")
        except ImportError:
            print(f"  [chunk {chunk_num}/{n_chunks}] cleanup done (install psutil for RAM stats)")

    # ── Reconstruct final model from saved weights for return ────────────
    tf.keras.backend.clear_session()
    gc.collect()
    final_model = _build_model(saved_lr)
    checkpoint_path = f"saved_models/best_weights_{run_tag}"
    # Build via one val batch then set weights
    loader_va_build = DisjointLoader(data_va, node_level=True, batch_size=args.batch_size)
    final_model.fit(loader_va_build.load(), steps_per_epoch=1, epochs=1, verbose=0)
    del loader_va_build
    final_model.set_weights(saved_weights_np)
    final_model.save_weights(checkpoint_path)   # persist to disk for post-training restore

    print(f"\n✅ Chunked training done ({n_chunks} chunks). "
          f"Final model = last chunk's best weights.")
    return _HistoryProxy({'loss': last_chunk_losses}), final_model, _BestInfo(n_chunks, float('nan'))


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
    best_after50_cb = BestAfterEpochCallback(save_path=checkpoint_path, min_epoch=args.best_after_epoch)

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
    "--best_after_epoch", default=50, type=int,
    help="Save best val_loss checkpoint only after this epoch (default: 50)."
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
    default="",
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
    "--near_goal_radius",
    default=1.0,
    type=float,
    help="When timestep_stride>1, always include timesteps where the flock mean position "
         "is within this distance of the current goal (default: 1.0). "
         "Prevents goal-transition frames from being skipped by the stride. "
         "Set to 0 to disable.",
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
    choices=["trained", "unseen_cache", "random", "fresh_quadrant"],
    help="Source of centers for the test visualization: "
         "'trained' = from training set; "
         "'unseen_cache' = cache centers not used in training; "
         "'random' = fresh random; "
         "'fresh_quadrant' = newly sampled centers per quadrant, confirmed not in training set.",
)
parser.add_argument(
    "--selected_quadrants",
    type=int,
    nargs="+",
    default=None,
    help="Which quadrants to restrict fresh_quadrant sampling to (0=Q1, 1=Q2, 2=Q3, 3=Q4). "
         "Default: all 4. E.g. '--selected_quadrants 2 3' for Q3 and Q4 only.",
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
parser.add_argument(
    "--viz_trained",
    action="store_true",
    default=False,
    help="Also run GNCA inference on training centers and include them in visualization. "
         "Off by default — only test centers are visualized.",
)
parser.add_argument(
    "--chunk_size",
    default=0,
    type=int,
    help="Trajectories per training chunk (0 = disabled, load all at once). "
         "Enable to train on large HDF5 caches without loading them fully into RAM.",
)
parser.add_argument(
    "--chunk_patience",
    default=3,
    type=int,
    help="Per-chunk early-stopping patience (default: 3). Training on each chunk "
         "stops when the chunk's training loss doesn't improve for this many epochs.",
)
parser.add_argument(
    "--train_quadrants",
    type=int,
    nargs="+",
    default=None,
    help="Which quadrants to use for training (0=Q1, 1=Q2, 2=Q3, 3=Q4). "
         "Default: all 4. E.g. '--train_quadrants 2 3' for Q3 and Q4 only. "
         "Only applicable when loading from --boids_cache.",
)
parser.add_argument(
    "--exclusion_size",
    type=float,
    default=0.2,
    help="Buffer radius around the goal triangle for excluding test center sampling. "
         "Should match the exclusion size used when generating the cache (default: 0.2).",
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

print(f"\n>>> Generating dataset...")
boids_cache  = args.boids_cache if args.boids_cache else None
use_chunked  = args.chunk_size > 0 and boids_cache is not None

data_va = make_dataset(
    unique_reps=args.va_set_size,
    repeat_reps=1,
    save_config=False,
    n_boids=args.n_boids,
    n_jobs=args.n_jobs,
    random_init=True,
)

if use_chunked:
    # ------------------------------------------------------------------ #
    # Chunked path: extract metadata without loading all graphs into RAM. #
    # ------------------------------------------------------------------ #
    boids_tr = Boids(n_boids=args.n_boids)
    with h5py.File(boids_cache, 'r') as _f:
        _centers       = _f['centers'][:]
        _cache_unique  = int(_f.attrs['unique_reps'])
        _cache_repeats = int(_f.attrs['repeats'])
        _n_boids_cache = int(_f.attrs['n_boids'])
        _traj_lengths  = _f['traj_lengths'][:] if 'traj_lengths' in _f else None

    _n_train     = min(effective_unique_reps, _cache_unique)
    
    # Filter to selected quadrants first if specified, then select from filtered set
    if args.train_quadrants is not None:
        from boids.generate_boids_cache import QUADRANTS
        # Group all centers by quadrant
        _idxs_by_quad = {q: [] for q in args.train_quadrants}
        for ti in range(_cache_unique):
            cx, cy = _centers[ti]
            for q in args.train_quadrants:
                xmn, xmx, ymn, ymx, _ = QUADRANTS[q]
                if xmn <= cx <= xmx and ymn <= cy <= ymx:
                    _idxs_by_quad[q].append(ti)
                    break
        # Balance selection across quadrants
        n_per_q = _n_train // len(args.train_quadrants)
        remainder = _n_train % len(args.train_quadrants)
        _train_idxs_list = []
        for i, q in enumerate(args.train_quadrants):
            take = n_per_q + (1 if i < remainder else 0)
            take = min(take, len(_idxs_by_quad[q]))
            _train_idxs_list.extend(np.random.choice(_idxs_by_quad[q], size=take, replace=False))
        _train_idxs = np.array(_train_idxs_list, dtype=np.int64)
        _n_train = len(_train_idxs)
        print(f">>> Filtered to quadrants {args.train_quadrants}: {_n_train} unique centers selected")
    else:
        _train_idxs = np.random.choice(_cache_unique, size=_n_train, replace=False)
    _unseen_idxs = np.array([i for i in range(_cache_unique) if i not in set(_train_idxs)])
    boids_tr.rand_configs   = [np.array(_centers[i], dtype=np.float32) for i in _train_idxs]
    boids_tr.unseen_configs = [np.array(_centers[i], dtype=np.float32) for i in _unseen_idxs]

    _n_reps = min(args.tr_set_repeats, _cache_repeats)
    _flat_traj_idxs = np.array([ti * _cache_repeats + rj
                                 for ti in _train_idxs
                                 for rj in range(_n_reps)], dtype=np.int64)

    if _traj_lengths is not None:
        _boundaries = np.concatenate([[0], np.cumsum(_traj_lengths)])
    else:
        _total_samples    = int(h5py.File(boids_cache, 'r')['x'].shape[0])
        _samples_per_traj = _total_samples // (_cache_unique * _cache_repeats)
        _boundaries       = np.arange(_cache_unique * _cache_repeats + 1) * _samples_per_traj

    _cache_info = {'n_boids': _n_boids_cache, 'timestep_stride': args.timestep_stride,
                   'near_goal_radius': args.near_goal_radius,
                   'cache_repeats': _cache_repeats,
                   'centers': _centers}
    print(f">>> Chunked training: cache={boids_cache}, "
          f"{_n_train} unique × {_n_reps} reps = {len(_flat_traj_idxs)} trajectories, "
          f"chunk_size={args.chunk_size}, chunk_patience={args.chunk_patience}")

    history, model, best_after50_cb = run_chunked(
        boids_cache, _flat_traj_idxs, _boundaries, _cache_info, data_va, run_tag
    )

else:
    # ------------------------------------------------------------------ #
    # Standard path: load all training graphs into RAM, then train.      #
    # ------------------------------------------------------------------ #
    if boids_cache:
        with h5py.File(boids_cache, 'r') as _f:
            cache_total_unique = int(_f.attrs["unique_reps"])
        print(f">>> Using precomputed cache: {boids_cache} "
              f"({cache_total_unique} total unique centers, training on {effective_unique_reps})")

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
        near_goal_radius=args.near_goal_radius,
    )

    if train_init_centers is not None:
        # Preserve reused center list for downstream evaluate()
        boids_tr.rand_configs = [np.array(c, dtype=np.float32) for c in train_init_centers]

    history, model, best_after50_cb = run(data_tr, data_va, run_tag)

print(f"\n>>> Saving model and history with run_tag='{run_tag}'...")
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_boids_tr", exist_ok=True)
os.makedirs("saved_history", exist_ok=True)
model.save(f"saved_models/gnca_model_{run_tag}", save_format="tf")
joblib.dump(history.history, f"saved_history/history_{run_tag}.pkl")
joblib.dump(boids_tr, f"saved_boids_tr/boids_tr_{run_tag}.pkl")

# Restore best weights onto the existing model object
checkpoint_path = f"saved_models/best_weights_{run_tag}"
if best_after50_cb.best_epoch is not None:
    print(f"\n✅ Restoring best weights (epoch {best_after50_cb.best_epoch}, val_loss={best_after50_cb.best_val_loss:.6f})...")
    model.load_weights(checkpoint_path)
else:
    print(f"\n⚠️ No best-weights checkpoint found. Using early-stopped model.")

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
        print("⚠️ No unseen_cache centers available. Falling back to random.")
        test_centers = None
    else:
        test_centers = boids_tr.unseen_configs[:args.viz_n_centers]
elif args.viz_centers_source == "fresh_quadrant":
    # Sample brand-new centers per quadrant — not from the cache at all.
    # Verified not to match any training center (continuous space → exact collision ≈ 0).
    from boids.generate_boids_cache import QUADRANTS, build_exclusion_zone
    _excl       = build_exclusion_zone(boids_tr.goal_positions, args.exclusion_size)
    _selected_q = args.selected_quadrants if args.selected_quadrants is not None else [0, 1, 2, 3]
    _n_per_q    = max(1, args.viz_n_centers // len(_selected_q))
    _train_set  = {tuple(np.round(c, 6)) for c in boids_tr.rand_configs}
    _sampler    = Boids(n_boids=args.n_boids)
    test_centers = []
    for _q_idx in _selected_q:
        _xmn, _xmx, _ymn, _ymx, _qlabel = QUADRANTS[_q_idx]
        _q_centers = []
        while len(_q_centers) < _n_per_q:
            _, _, _, _c = _sampler.get_random_init(
                args.n_boids, save_config=False,
                bounds=(_xmn, _xmx, _ymn, _ymx),
                exclusion_zone=_excl,
            )
            if tuple(np.round(_c, 6)) not in _train_set:
                _q_centers.append(_c)
        test_centers.extend(_q_centers)
    print(f">>> Viz test centers: {len(test_centers)} fresh per-quadrant centers "
          f"({_n_per_q} per quadrant, {len(_selected_q)} quadrants), none in training set")
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
    viz_trained=args.viz_trained,
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