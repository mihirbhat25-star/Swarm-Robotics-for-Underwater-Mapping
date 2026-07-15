"""
Trains the 3D GNCA to imitate the 3D Boids GCA.
"""
import argparse
import gc
import os
import sys
import json
import subprocess
import tempfile
import warnings
warnings.filterwarnings('ignore')
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
from modules.boids_3d import make_dataset_3d, load_chunk_from_cache_3d, Boids3D
import h5py
import scipy.sparse as sp
import psutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'

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

def _build_model_3d(lr):
    """Construct and compile a fresh GNNCASimpleBoids3D."""
    m = GNNCASimpleBoids3D(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    loss_fn = lambda y_true, y_pred: custom_weighted_mse_3d(y_true, y_pred)
    m.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn, run_eagerly=True)
    return m


def run_chunked_3d(cache_path, boundaries, n_boids_cache, all_flat_indices,
                   data_va, run_tag, chunk_size, chunk_patience,
                   octant_buckets=None):
    """Chunked training mirroring 2D: get_weights/set_weights + clear_session in-process."""
    checkpoint_path = f"saved_models/best_weights_3d_{run_tag}"
    os.makedirs("saved_models", exist_ok=True)

    stride      = args.timestep_stride
    near_goal_r = args.near_goal_radius

    # ── Build balanced chunks ────────────────────────────────────────────────
    if octant_buckets and len(octant_buckets) > 1:
        active = {o: np.array(idxs) for o, idxs in octant_buckets.items() if len(idxs) > 0}
        n_active = len(active)
        n_per_o  = chunk_size // n_active
        remainder = chunk_size % n_active
        for o in active:
            np.random.shuffle(active[o])
        pointers = {o: 0 for o in active}
        chunks, current = [], []
        while True:
            added = 0
            for i, o in enumerate(active):
                take = n_per_o + (1 if i < remainder else 0)
                p = pointers[o]
                batch = active[o][p:p + take]
                current.extend(batch.tolist())
                pointers[o] += take
                added += len(batch)
            if added == 0:
                break
            if len(current) >= chunk_size:
                chunks.append(np.array(current[:chunk_size]))
                current = current[chunk_size:]
        if current:
            chunks.append(np.array(current))
        n_chunks = len(chunks)
        print(f">>> Chunked training: {n_chunks} balanced chunks ({n_per_o} per octant × {n_active} octants)")
    else:
        n_chunks = max(1, len(all_flat_indices) // chunk_size)
        chunks = np.array_split(all_flat_indices, n_chunks)
        print(f">>> Chunked training: {n_chunks} chunks of ~{chunk_size} trajectories each")

    def _mem_mb():
        return psutil.Process(os.getpid()).memory_info().rss / 1e6

    saved_weights_np = None
    saved_lr = args.lr

    for chunk_idx, chunk_flat in enumerate(chunks):
        print(f"\n{'='*60}")
        print(f"  Chunk {chunk_idx+1}/{n_chunks} | {len(chunk_flat)} trajectories | RAM: {_mem_mb():.0f} MB")
        print(f"{'='*60}")

        tf.keras.backend.clear_session()
        gc.collect()

        model = _build_model_3d(saved_lr)
        data_chunk = load_chunk_from_cache_3d(
            cache_path, chunk_flat, boundaries, n_boids_cache,
            timestep_stride=stride, near_goal_radius=near_goal_r,
        )
        loader_tr = DisjointLoader(data_chunk, node_level=True, batch_size=args.batch_size)
        loader_va = DisjointLoader(data_va,    node_level=True, batch_size=args.batch_size)

        if saved_weights_np is not None:
            model.fit(loader_tr.load(), steps_per_epoch=1, epochs=1, verbose=0)
            model.set_weights(saved_weights_np)
            loader_tr = DisjointLoader(data_chunk, node_level=True, batch_size=args.batch_size)
            loader_va = DisjointLoader(data_va,    node_level=True, batch_size=args.batch_size)

        n_epochs = args.chunk_epochs if args.chunk_epochs is not None else 10_000
        if args.chunk_epochs is not None:
            cbs = []
            val_kwargs = {}
        else:
            cbs = [
                EarlyStopping(monitor='val_loss', patience=chunk_patience,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', patience=args.lr_patience,
                                  min_delta=1e-8, verbose=1),
            ]
            val_kwargs = {'validation_data': loader_va.load(),
                          'validation_steps': loader_va.steps_per_epoch}

        model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch,
                  epochs=n_epochs, callbacks=cbs, **val_kwargs)

        saved_weights_np = model.get_weights()
        saved_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))

        del data_chunk, loader_tr, loader_va, model
        gc.collect()
        print(f"  RAM after cleanup: {_mem_mb():.0f} MB")

    # Reconstruct final model
    tf.keras.backend.clear_session()
    gc.collect()
    final_model = _build_model_3d(saved_lr)
    loader_va_build = DisjointLoader(data_va, node_level=True, batch_size=args.batch_size)
    final_model.fit(loader_va_build.load(), steps_per_epoch=1, epochs=1, verbose=0)
    del loader_va_build
    final_model.set_weights(saved_weights_np)
    final_model.save_weights(checkpoint_path)
    print(f"Saved final weights to {checkpoint_path}")
    return final_model


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
    loss_fn = lambda y_true, y_pred: custom_weighted_mse_3d(y_true, y_pred)
    model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=True)

    loader_tr = DisjointLoader(data_tr, node_level=True, batch_size=args.batch_size)
    loader_va = DisjointLoader(data_va, node_level=True, batch_size=args.batch_size)

    checkpoint_path = f"saved_models/best_weights_3d_{run_tag}"
    best_after50_cb = BestAfterEpochCallback3D(save_path=checkpoint_path, min_epoch=args.best_after_epoch)

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
parser.add_argument("--chunk_epochs", default=None, type=int,
                   help="Fixed epochs per chunk (overrides early stopping). Use for debugging.")
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
parser.add_argument("--critical_distance", default=2.5, type=float)
parser.add_argument("--distance_weight", default=2.5, type=float)
parser.add_argument("--viz_mode", default="tubular", choices=["tubular", "per_boid", "multi_tubular", "individual"])
# ── Chunk worker args (internal use, passed by run_chunked_3d subprocess) ──
parser.add_argument("--_chunk_worker", action="store_true", default=False)
parser.add_argument("--_chunk_indices_file", default=None)
parser.add_argument("--_cache_path", default=None)
parser.add_argument("--_checkpoint_path", default=None)
parser.add_argument("--_n_boids_cache", default=None, type=int)
parser.add_argument("--_boundaries_file", default=None)
parser.add_argument("--_val_indices_file", default=None)
parser.add_argument("--_val_npz_file", default=None)
parser.add_argument("--noise_tag", default="", type=str)
parser.add_argument("--init_centers_npz", default="", type=str)
parser.add_argument("--boids_cache", default="", type=str)
parser.add_argument("--timestep_stride", default=1, type=int)
parser.add_argument("--near_goal_radius", default=1.0, type=float)
parser.add_argument("--chunk_size", default=100, type=int)
parser.add_argument("--chunk_patience", default=15, type=int)
parser.add_argument("--best_after_epoch", default=50, type=int,
                   help="Non-chunked only: save best val_loss checkpoint after this epoch (default: 50).")
parser.add_argument("--train_octants", type=int, nargs="+", default=None)
parser.add_argument("--viz_n_centers", default=50, type=int)
parser.add_argument("--viz_centers_source", default="trained", choices=["trained", "unseen_cache", "random", "fresh_octant"])
parser.add_argument("--skip_trained_viz", action="store_true", default=False,
                   help="Skip computing trajectories from trained centers during evaluation.")


####################################################################################
# Training
####################################################################################
args = parser.parse_args()

# ── Chunk worker mode (subprocess per chunk) ─────────────────────────────────
if args._chunk_worker:
    UPWEIGHT_NEAR_GOAL = (args.loss_type == "newl")
    chunk_flat = np.array(json.load(open(args._chunk_indices_file)))
    boundaries = np.load(args._boundaries_file)
    data_chunk = load_chunk_from_cache_3d(
        args._cache_path, chunk_flat, boundaries, args._n_boids_cache,
        timestep_stride=args.timestep_stride, near_goal_radius=args.near_goal_radius,
    )
    loader_tr = DisjointLoader(data_chunk, node_level=True, batch_size=args.batch_size)

    # Load validation data from serialized npz if provided and not fixed-epoch mode
    use_val = (args.chunk_epochs is None) and (args._val_npz_file is not None)
    if use_val:
        _vd = np.load(args._val_npz_file)
        _vx, _vy, _varow, _vacol, _valen = _vd['x'], _vd['y'], _vd['a_row'], _vd['a_col'], _vd['a_len']
        _nb = int(_vd['n_boids'][0])
        from spektral.data import Graph
        _va_graphs = []
        for k in range(len(_vx)):
            _nnz = int(_valen[k])
            _a = sp.coo_matrix((np.ones(_nnz, dtype=np.float32),
                                (_varow[k, :_nnz], _vacol[k, :_nnz])), shape=(_nb, _nb))
            _va_graphs.append(Graph(x=_vx[k], a=_a, y=_vy[k]))
        from modules.boids_3d import BoidsDataset3D
        data_val = BoidsDataset3D(_va_graphs)
        loader_va = DisjointLoader(data_val, node_level=True, batch_size=args.batch_size)

    model = _build_model_3d(args.lr)
    if os.path.exists(args._checkpoint_path + ".index"):
        model.load_weights(args._checkpoint_path).expect_partial()
    n_epochs = args.chunk_epochs if args.chunk_epochs is not None else args.epochs

    if args.chunk_epochs is not None:
        # Fixed epochs — no early stopping, no validation
        cbs = []
        val_kwargs = {}
    elif use_val:
        cbs = [
            EarlyStopping(monitor='val_loss', patience=args.es_patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=args.lr_patience,
                              min_delta=1e-8, verbose=1),
        ]
        val_kwargs = {'validation_data': loader_va.load(),
                      'validation_steps': loader_va.steps_per_epoch}
    else:
        cbs = [
            EarlyStopping(monitor='loss', patience=args.es_patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='loss', patience=args.lr_patience,
                              min_delta=1e-8, verbose=1),
        ]
        val_kwargs = {}

    model.fit(
        loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=n_epochs, callbacks=cbs, **val_kwargs,
    )
    model.save_weights(args._checkpoint_path)
    sys.exit(0)
# ─────────────────────────────────────────────────────────────────────────────

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

# Validation set (always fresh random)
print(f"\n>>> Generating validation set ({args.va_set_size} centers)...")
data_va = make_dataset_3d(
    unique_reps=args.va_set_size,
    repeat_reps=1,
    save_config=False,
    n_boids=args.n_boids,
    n_jobs=1,
    random_init=True,
)

use_chunked = args.chunk_size > 0 and boids_cache is not None

if use_chunked:
    print(f"\n>>> Chunked training from cache: {boids_cache}")
    with h5py.File(boids_cache, 'r') as _f:
        n_boids_cache   = int(_f.attrs['n_boids'])
        cache_unique    = int(_f.attrs['unique_reps'])
        cache_repeats   = int(_f.attrs['repeats'])
        if 'traj_lengths' in _f:
            boundaries = np.concatenate([[0], np.cumsum(_f['traj_lengths'][:])])
        else:
            total_s = _f['x'].shape[0]
            spt = total_s // (cache_unique * cache_repeats)
            boundaries = np.arange(cache_unique * cache_repeats + 1) * spt
        centers_all = _f['centers'][:]

    # Filter by octants if requested
    if args.train_octants is not None:
        from boids.generate_boids_cache_3d import OCTANTS
        def _center_octant(c):
            for oi, (xmn, xmx, ymn, ymx, zmn, zmx, _) in enumerate(OCTANTS):
                if xmn <= c[0] < xmx and ymn <= c[1] < ymx and zmn <= c[2] < zmx:
                    return oi
            return -1
        keep_mask = np.array([_center_octant(centers_all[i]) in args.train_octants
                              for i in range(cache_unique)])
        train_center_indices = np.where(keep_mask)[0]
        # Subsample to effective_unique_reps if requested
        if effective_unique_reps < len(train_center_indices):
            train_center_indices = np.random.choice(train_center_indices, size=effective_unique_reps, replace=False)
        octant_buckets = {o: [] for o in args.train_octants}
        for ci in train_center_indices:
            oi = _center_octant(centers_all[ci])
            for ri in range(cache_repeats):
                octant_buckets[oi].append(ci * cache_repeats + ri)
    else:
        train_center_indices = np.arange(min(cache_unique, effective_unique_reps))
        octant_buckets = None

    boids_tr = Boids3D(n_boids=args.n_boids)
    boids_tr.rand_configs = [np.array(centers_all[i], dtype=np.float32) for i in train_center_indices]

    all_flat_indices = np.array([
        ci * cache_repeats + ri
        for ci in train_center_indices
        for ri in range(cache_repeats)
    ])
    np.random.shuffle(all_flat_indices)

    model = run_chunked_3d(
        boids_cache, boundaries, n_boids_cache, all_flat_indices,
        data_va, run_tag, args.chunk_size, args.chunk_patience,
        octant_buckets=octant_buckets
    )
    best_after50_cb = None
else:
    print(f"\n>>> Generating 3D training dataset...")
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
    _, model, best_after50_cb = run(data_tr, data_va, run_tag=run_tag)

# Save model, boids, history
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_boids_tr", exist_ok=True)
os.makedirs("saved_history", exist_ok=True)

model_save_path    = f"saved_models/gnca_model_3d_{run_tag}"
boids_save_path    = f"saved_boids_tr/boids_tr_3d_{run_tag}.pkl"
history_save_path  = f"saved_history/history_3d_{run_tag}.pkl"

# Build model weights by running a dummy forward pass before saving
_n = args.n_boids
_x_dummy = tf.zeros((_n, 6), dtype=tf.float32)
_a_dummy = tf.SparseTensor(indices=tf.zeros((0, 2), dtype=tf.int64),
                            values=tf.zeros((0,), dtype=tf.float32),
                            dense_shape=(_n, _n))
_a_dummy = tf.sparse.reorder(_a_dummy)
model([_x_dummy, _a_dummy, tf.constant(0)], training=False)

model.save_weights(model_save_path)
print(f"✅ Saved weights to {model_save_path}")
joblib.dump(boids_tr, boids_save_path)
print(f"✅ Saved boids_tr to {boids_save_path}")

# Restore best post-epoch-50 weights onto the existing model object
checkpoint_path = f"saved_models/best_weights_3d_{run_tag}"
if best_after50_cb is not None and best_after50_cb.best_epoch is not None:
    print(f"\n✅ Restoring best post-epoch-50 weights (epoch {best_after50_cb.best_epoch}, val_loss={best_after50_cb.best_val_loss:.6f})...")
    model.load_weights(checkpoint_path)

####################################################################################
# Evaluation
####################################################################################
max_trajectory_len = 3000

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
elif args.viz_centers_source == "fresh_octant":
    from boids.generate_boids_cache_3d import OCTANTS, build_exclusion_zone_3d
    _boids_tmp = Boids3D(n_boids=args.n_boids)
    _excl = build_exclusion_zone_3d(_boids_tmp.goal_positions, 0.5)
    _octant_list = args.train_octants if args.train_octants is not None else list(range(8))
    _n_per_oct = max(1, args.viz_n_centers // len(_octant_list))
    test_centers = []
    for _oi in _octant_list:
        _xmn, _xmx, _ymn, _ymx, _zmn, _zmx, _ = OCTANTS[_oi]
        _bounds = (_xmn, _xmx, _ymn, _ymx, _zmn, _zmx)
        _q_centers = []
        while len(_q_centers) < _n_per_oct:
            _, _, _, _c = _boids_tmp.get_random_init(args.n_boids, save_config=False,
                                                      bounds=_bounds, exclusion_zone=_excl)
            _q_centers.append(_c)
        test_centers.extend(_q_centers)
    print(f">>> Viz test centers: {len(test_centers)} fresh octant centers ({_n_per_oct} per octant)")
else:
    test_centers = None
    print(f">>> Viz test centers: random (fresh)")

evaluate_3d(model, max_trajectory_len, args.n_boids,
            saved_boids=boids_tr,
            run_tag=run_tag, viz_mode=args.viz_mode,
            test_centers=test_centers,
            skip_trained_viz=args.skip_trained_viz)
