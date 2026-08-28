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
import atexit
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
warnings.filterwarnings('ignore')
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from spektral.data import DisjointLoader
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from boids.forward import forward
from models.gnn_ca_simple_boids_3d import GNNCASimpleBoids3D
from modules.boids_3d import (
    BOIDS_GOAL_POSITIONS_3D,
    FIXED_ORDER_POLICY,
    NEAREST_CCW_POLICY,
    Boids3D,
    load_chunk_from_cache_3d,
    make_dataset_3d,
)
import h5py
import scipy.sparse as sp
import psutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'

physical_devices = tf.config.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

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
    print(f"Multi-GPU:            {args.multi_gpu}")
    print(f"Cloud optimized:      {args.cloud_optimized}")
    print(f"Cloud native data:    {args.cloud_native_dataset}")
    print(f"Cloud BF16:           {args.cloud_bfloat16}")
    print(
        f"Execution:            "
        f"{'eager (debug)' if args.eager_training else 'compiled graph'}; "
        f"steps_per_execution={args.steps_per_execution}"
    )
    print(f"Epochs:               {args.epochs} (early stop patience: {args.es_patience})")
    print(f"Visualization mode:   {args.viz_mode}")
    print("="*70 + "\n")

def custom_weighted_mse_3d(y_true, y_pred):
    """
    Custom weighted MSE loss for 3D GNCA.
    y_true is [current_state (6D), next_state (6D), current_goal (3D), graph_id].
    Weighting uses distance to the nearest waypoint so the departure turn
    remains emphasized immediately after the active-goal label switches.
    y_pred is predicted next_state (6D).
    """
    # BF16 is used only for cloud model compute. Keep target slicing, distance
    # weighting, squared error, and loss reduction in float32 for stability.
    if getattr(args, "cloud_bfloat16", False):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

    n_features = tf.shape(y_pred)[-1]  # Should be 6
    current_state = y_true[..., :n_features]
    next_state = y_true[..., n_features:2*n_features]
    
    mse = tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)

    if UPWEIGHT_NEAR_GOAL:
        graph_ids = tf.cast(y_true[..., 15], tf.int32)
        n_graphs = tf.reduce_max(graph_ids) + 1
        avg_pos = tf.math.unsorted_segment_mean(
            current_state[..., :3], graph_ids, n_graphs
        )
        goals = tf.cast(BOIDS_GOAL_POSITIONS_3D, avg_pos.dtype)
        dist_to_goals = tf.norm(
            avg_pos[:, None, :] - goals[None, :, :],
            axis=-1,
        )
        dist_to_goal = tf.reduce_min(dist_to_goals, axis=1)
        graph_weight = tf.where(
            dist_to_goal < args.critical_distance,
            tf.cast(args.distance_weight, mse.dtype),
            tf.ones_like(dist_to_goal, dtype=mse.dtype),
        )
        goal_weight = tf.gather(graph_weight, graph_ids)
    else:
        goal_weight = 1.0

    return mse * goal_weight


def add_graph_ids_to_targets_3d(inputs, targets):
    """Append Spektral's graph id to each node target for graph-wise loss."""
    graph_ids = tf.cast(inputs[2], targets.dtype)
    targets = tf.concat([targets, graph_ids[:, None]], axis=-1)
    targets = tf.ensure_shape(targets, [None, 16])
    return inputs, targets


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


class BestTrainingStateCallback3D(tf.keras.callbacks.Callback):
    """Checkpoint the complete model and optimizer state at the best epoch."""

    def __init__(self, checkpoint_manager, monitor="val_loss", min_delta=0.0):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.monitor = monitor
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        value = float(value)
        if value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch + 1
            path = self.checkpoint_manager.save()
            print(
                f"\n💾 Saved best complete training state at epoch "
                f"{self.best_epoch}: {self.monitor}={value:.6g} ({path})"
            )


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
    m.compile(
        optimizer=Adam(learning_rate=lr),
        loss=loss_fn,
        run_eagerly=args.eager_training,
        steps_per_execution=args.steps_per_execution,
    )
    return m


def _build_model_variables_3d(model, n_boids):
    """Run a dummy forward pass so Keras variables exist before load_weights."""
    x_dummy = tf.zeros((n_boids, 6), dtype=tf.float32)
    a_dummy = tf.SparseTensor(
        indices=tf.zeros((0, 2), dtype=tf.int64),
        values=tf.zeros((0,), dtype=tf.float32),
        dense_shape=(n_boids, n_boids),
    )
    model([x_dummy, tf.sparse.reorder(a_dummy), tf.constant(0)], training=False)


def _expert_waypoint_policy_3d(args):
    return (
        NEAREST_CCW_POLICY
        if args.expert_goal_order == "nearest_ccw"
        else FIXED_ORDER_POLICY
    )


def make_validation_dataset_3d(args):
    """Generate validation trajectories from the same octants used for training."""
    if args.train_octants is None:
        return make_dataset_3d(
            unique_reps=args.va_set_size,
            repeat_reps=1,
            save_config=False,
            n_boids=args.n_boids,
            n_jobs=1,
            random_init=True,
            perception=args.perception,
            pos_noise=args.expert_pos_noise,
            vel_noise=args.expert_vel_noise,
            waypoint_order_policy=_expert_waypoint_policy_3d(args),
        )

    from boids.generate_boids_cache_3d import OCTANTS, build_exclusion_zone_3d

    sampler = Boids3D(
        n_boids=args.n_boids,
        perception=args.perception,
        pos_noise=args.expert_pos_noise,
        vel_noise=args.expert_vel_noise,
        waypoint_order_policy=_expert_waypoint_policy_3d(args),
    )
    exclusion_zone = build_exclusion_zone_3d(
        sampler.goal_positions, args.goal_exclusion_size
    )
    centers = []
    n_octants = len(args.train_octants)
    base = args.va_set_size // n_octants
    remainder = args.va_set_size % n_octants

    for i, octant in enumerate(args.train_octants):
        xmn, xmx, ymn, ymx, zmn, zmx, _ = OCTANTS[octant]
        take = base + (1 if i < remainder else 0)
        for _ in range(take):
            _, _, _, center = sampler.get_random_init(
                args.n_boids,
                save_config=False,
                bounds=(xmn, xmx, ymn, ymx, zmn, zmx),
                exclusion_zone=exclusion_zone,
            )
            centers.append(np.array(center, dtype=np.float32))

    print(f">>> Validation centers: {len(centers)} fresh centers from train_octants {args.train_octants}")
    return make_dataset_3d(
        unique_reps=len(centers),
        repeat_reps=1,
        save_config=False,
        n_boids=args.n_boids,
        n_jobs=1,
        random_init=centers,
        perception=args.perception,
        pos_noise=args.expert_pos_noise,
        vel_noise=args.expert_vel_noise,
        waypoint_order_policy=_expert_waypoint_policy_3d(args),
    )


def save_dataset_npz_3d(dataset, path, n_boids):
    """Serialize a small Spektral 3D dataset for subprocess validation."""
    graphs = list(dataset.graphs)
    max_edges = max((g.a.nnz for g in graphs), default=0)
    x = np.stack([g.x for g in graphs], axis=0).astype(np.float32)
    y = np.stack([g.y for g in graphs], axis=0).astype(np.float32)
    a_row = np.full((len(graphs), max_edges), -1, dtype=np.int32)
    a_col = np.full((len(graphs), max_edges), -1, dtype=np.int32)
    a_len = np.zeros(len(graphs), dtype=np.int32)

    for i, g in enumerate(graphs):
        a = g.a.tocoo()
        nnz = a.nnz
        a_row[i, :nnz] = a.row
        a_col[i, :nnz] = a.col
        a_len[i] = nnz

    np.savez_compressed(
        path,
        x=x,
        y=y,
        a_row=a_row,
        a_col=a_col,
        a_len=a_len,
        n_boids=np.array([n_boids], dtype=np.int32),
    )


def make_disjoint_batch_3d(
    x_list, y_list, row_list, col_list, len_list, n_boids, compact=False
):
    """Pack per-timestep cached arrays into one disjoint sparse mini-batch."""
    batch_size = len(x_list)
    x = np.concatenate(x_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)

    edge_rows = []
    edge_cols = []
    for b, (rows, cols, nnz) in enumerate(zip(row_list, col_list, len_list)):
        nnz = int(nnz)
        if nnz <= 0:
            continue
        offset = b * n_boids
        edge_rows.append(rows[:nnz].astype(np.int64) + offset)
        edge_cols.append(cols[:nnz].astype(np.int64) + offset)

    if edge_rows:
        indices = np.stack([np.concatenate(edge_rows), np.concatenate(edge_cols)], axis=1)
        # Cache COO entries are row-major sorted. Graph offsets increase with
        # batch order, so concatenation remains SparseTensor-compatible without
        # an expensive global lexsort for every mini-batch.
        if compact:
            return x, y, indices
        values = np.ones(indices.shape[0], dtype=np.float32)
    else:
        indices = np.zeros((0, 2), dtype=np.int64)
        if compact:
            return x, y, indices
        values = np.zeros((0,), dtype=np.float32)

    n_nodes = batch_size * n_boids
    dense_shape = np.array([n_nodes, n_nodes], dtype=np.int64)
    step_i = np.repeat(np.arange(batch_size), n_boids).astype(np.int64)
    return (x, indices, values, dense_shape, step_i), y


def batch_to_tf_sparse_3d(batch):
    """Convert a compact NumPy disjoint batch into model-ready TensorFlow inputs."""
    (x, indices, values, dense_shape, step_i), y = batch
    adj = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    return (x, adj, step_i), y


def _load_adaptive_cached_samples_3d(
    f, start, end, timestep_stride, near_goal_radius
):
    """Load selected states, targets, and their cached current-state graphs."""
    if timestep_stride <= 1 or near_goal_radius <= 0:
        selection = slice(start, end, timestep_stride)
        return (
            f["x"][selection],
            f["y"][selection],
            f["a_row"][selection],
            f["a_col"][selection],
            f["a_len"][selection],
        )

    x_full = f["x"][start:end]
    y_full = f["y"][start:end]
    mean_pos = x_full[:, :, :3].mean(axis=1)
    dist_to_goal = np.min(
        np.linalg.norm(
            mean_pos[:, None, :] - BOIDS_GOAL_POSITIONS_3D[None, :, :],
            axis=-1,
        ),
        axis=1,
    )

    keep = np.zeros(end - start, dtype=bool)
    keep[::timestep_stride] = True
    keep |= dist_to_goal < near_goal_radius
    absolute_indices = np.flatnonzero(keep) + start
    return (
        x_full[keep],
        y_full[keep],
        f["a_row"][absolute_indices],
        f["a_col"][absolute_indices],
        f["a_len"][absolute_indices],
    )


def _validate_cached_adjacency_3d(f, sample_indices, perception):
    """Fail fast if a cache graph is not aligned with x or uses another radius."""
    for sample_idx in sample_indices:
        sample_idx = int(sample_idx)
        x = f["x"][sample_idx]
        nnz = int(f["a_len"][sample_idx])
        cached = set(zip(
            f["a_row"][sample_idx, :nnz].tolist(),
            f["a_col"][sample_idx, :nnz].tolist(),
        ))

        positions = x[:, :3]
        distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1
        )
        neighbors = distances < perception
        np.fill_diagonal(neighbors, False)
        row, col = np.nonzero(neighbors)
        expected = set(zip(row.tolist(), col.tolist()))
        if cached != expected:
            raise ValueError(
                "Cached adjacency does not match the current state at sample "
                f"{sample_idx} for perception={perception}. Use a compatible "
                "cache; training will not silently use a stale graph."
            )
        cached_pairs = np.column_stack((
            f["a_row"][sample_idx, :nnz],
            f["a_col"][sample_idx, :nnz],
        ))
        if nnz > 1:
            order = np.lexsort((cached_pairs[:, 1], cached_pairs[:, 0]))
            if not np.array_equal(order, np.arange(nnz)):
                raise ValueError(
                    f"Cached adjacency at sample {sample_idx} is not row-major "
                    "sorted and cannot use the optimized sparse packing path."
                )


def _serialize_cloud_tfrecord_batch_3d(x, y, indices, n_boids):
    """Serialize one packed graph batch without an intermediate NPZ shard."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    example = tf.train.Example(features=tf.train.Features(feature={
        "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.tobytes()])),
        "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.tobytes()])),
        "indices": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[indices.tobytes()])
        ),
        "n_nodes": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[len(x)])
        ),
        "n_edges": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[len(indices)])
        ),
        "n_boids": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(n_boids)])
        ),
    }))
    return example.SerializeToString()


_CLOUD_TFRECORD_SPEC_3D = {
    "x": tf.io.FixedLenFeature([], tf.string),
    "y": tf.io.FixedLenFeature([], tf.string),
    "indices": tf.io.FixedLenFeature([], tf.string),
    "n_nodes": tf.io.FixedLenFeature([], tf.int64),
    "n_edges": tf.io.FixedLenFeature([], tf.int64),
    "n_boids": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_cloud_tfrecord_batch_3d(serialized):
    """Parse one direct TFRecord batch entirely with TensorFlow operations."""
    record = tf.io.parse_single_example(serialized, _CLOUD_TFRECORD_SPEC_3D)
    n_nodes = record["n_nodes"]
    n_edges = record["n_edges"]
    n_boids = record["n_boids"]

    x = tf.reshape(
        tf.io.decode_raw(record["x"], tf.float32),
        tf.stack([n_nodes, tf.constant(6, dtype=tf.int64)]),
    )
    y = tf.reshape(
        tf.io.decode_raw(record["y"], tf.float32),
        tf.stack([n_nodes, tf.constant(15, dtype=tf.int64)]),
    )
    indices = tf.reshape(
        tf.io.decode_raw(record["indices"], tf.int32),
        tf.stack([n_edges, tf.constant(2, dtype=tf.int64)]),
    )
    indices = tf.cast(indices, tf.int64)
    adjacency = tf.SparseTensor(
        indices=indices,
        values=tf.ones([n_edges], dtype=tf.float32),
        dense_shape=tf.stack([n_nodes, n_nodes]),
    )
    graph_count = n_nodes // n_boids
    graph_ids = tf.repeat(tf.range(graph_count, dtype=tf.int64), n_boids)
    x = tf.ensure_shape(x, [None, 6])
    y = tf.ensure_shape(y, [None, 15])
    graph_ids = tf.ensure_shape(graph_ids, [None])
    return (x, adjacency, graph_ids), y


def dataset_from_cloud_tfrecord_shards_3d(shard_paths, steps):
    """Load packed batches through native C++ TFRecord readers and parsers."""
    if not shard_paths:
        raise ValueError("Cloud TFRecord dataset received no shard paths.")
    dataset = tf.data.TFRecordDataset(
        shard_paths,
        compression_type="",
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        _parse_cloud_tfrecord_batch_3d,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    dataset = dataset.map(
        add_graph_ids_to_targets_3d,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    if steps > 1:
        dataset = dataset.shuffle(
            min(256, int(steps)),
            reshuffle_each_iteration=True,
        )
    options = tf.data.Options()
    options.deterministic = False
    return dataset.with_options(options).repeat().prefetch(tf.data.AUTOTUNE)


def write_cached_chunk_shards_3d(
    cache_paths,
    trajectory_refs,
    boundaries_by_cache,
    n_boids,
    batch_size,
    output_dir,
    timestep_stride=1,
    near_goal_radius=1.0,
    perception=0.1,
    shuffle_buffer_size=4096,
    batches_per_shard=16,
    drop_remainder=False,
    native_tfrecord=False,
):
    """Write shuffled disjoint batches into compact multi-batch shards.

    Trajectory order is randomized first, then individual timesteps pass through
    a bounded shuffle buffer before packing.  This prevents the optimizer from
    receiving thousands of adjacent states from one trajectory in sequence.
    Cached current-state adjacency is reused instead of rebuilding an O(N^2)
    distance matrix for every sample. Sparse values and shapes are reconstructed
    while loading, and int32 indices are stored on disk to reduce temporary data.
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_paths = []
    shard_batches = []
    record_writer = None
    record_batches_in_shard = 0
    total_batches = 0
    x_batch, y_batch, row_batch, col_batch, len_batch = [], [], [], [], []
    shuffle_buffer = []
    shuffle_buffer_size = max(batch_size, int(shuffle_buffer_size))
    batches_per_shard = max(1, int(batches_per_shard))
    rng = np.random.default_rng()

    def write_record_batch(x, y, indices):
        nonlocal record_writer, record_batches_in_shard
        if record_writer is None:
            path = os.path.join(
                output_dir, f"shard_{len(shard_paths):05d}.tfrecord"
            )
            record_writer = tf.io.TFRecordWriter(path)
            shard_paths.append(path)
            record_batches_in_shard = 0
        record_writer.write(
            _serialize_cloud_tfrecord_batch_3d(x, y, indices, n_boids)
        )
        record_batches_in_shard += 1
        if record_batches_in_shard >= batches_per_shard:
            record_writer.close()
            record_writer = None
            record_batches_in_shard = 0

    def flush_shard():
        nonlocal shard_batches
        if not shard_batches:
            return

        node_lengths = np.asarray(
            [len(batch[0]) for batch in shard_batches], dtype=np.int64
        )
        edge_lengths = np.asarray(
            [len(batch[2]) for batch in shard_batches], dtype=np.int64
        )
        node_offsets = np.concatenate([
            np.zeros(1, dtype=np.int64), np.cumsum(node_lengths)
        ])
        edge_offsets = np.concatenate([
            np.zeros(1, dtype=np.int64), np.cumsum(edge_lengths)
        ])
        path = os.path.join(output_dir, f"shard_{len(shard_paths):05d}.npz")
        np.savez(
            path,
            x=np.concatenate([batch[0] for batch in shard_batches], axis=0),
            y=np.concatenate([batch[1] for batch in shard_batches], axis=0),
            indices=np.concatenate(
                [batch[2] for batch in shard_batches], axis=0
            ).astype(np.int32, copy=False),
            node_offsets=node_offsets,
            edge_offsets=edge_offsets,
            n_boids=np.asarray(n_boids, dtype=np.int32),
        )
        shard_paths.append(path)
        shard_batches = []

    def flush_batch():
        nonlocal total_batches, x_batch, y_batch, row_batch, col_batch, len_batch
        x, y, indices = make_disjoint_batch_3d(
            x_batch,
            y_batch,
            row_batch,
            col_batch,
            len_batch,
            n_boids,
            compact=True,
        )
        if native_tfrecord:
            write_record_batch(x, y, indices)
        else:
            shard_batches.append((x, y, indices))
        total_batches += 1
        x_batch, y_batch, row_batch, col_batch, len_batch = [], [], [], [], []
        if not native_tfrecord and len(shard_batches) >= batches_per_shard:
            flush_shard()

    def emit_sample(sample):
        x, y, row, col = sample
        x_batch.append(x)
        y_batch.append(y)
        row_batch.append(row)
        col_batch.append(col)
        len_batch.append(len(row))
        if len(x_batch) == batch_size:
            flush_batch()

    def emit_random_buffered_sample():
        sample_idx = int(rng.integers(len(shuffle_buffer)))
        sample = shuffle_buffer[sample_idx]
        shuffle_buffer[sample_idx] = shuffle_buffer[-1]
        shuffle_buffer.pop()
        emit_sample(sample)

    with ExitStack() as stack:
        caches = [stack.enter_context(h5py.File(path, "r")) for path in cache_paths]
        refs = np.asarray(trajectory_refs, dtype=np.int64).reshape(-1, 2)
        for cache_idx, f in enumerate(caches):
            cache_refs = refs[refs[:, 0] == cache_idx, 1]
            if len(cache_refs) == 0:
                continue
            boundaries = boundaries_by_cache[cache_idx]
            probe_refs = cache_refs[np.linspace(
                0, len(cache_refs) - 1, num=min(3, len(cache_refs)), dtype=int
            )]
            probe_samples = [int(boundaries[int(ref)]) for ref in probe_refs]
            _validate_cached_adjacency_3d(f, probe_samples, perception)

        trajectory_order = rng.permutation(refs)
        for cache_idx, flat_idx in trajectory_order:
            f = caches[int(cache_idx)]
            boundaries = boundaries_by_cache[int(cache_idx)]
            start = int(boundaries[int(flat_idx)])
            end = int(boundaries[int(flat_idx) + 1])
            x_c, y_c, row_c, col_c, len_c = _load_adaptive_cached_samples_3d(
                f, start, end, timestep_stride, near_goal_radius
            )
            # Randomize individual timesteps before they enter the cross-
            # trajectory buffer; adjacent expert states must not remain an
            # optimizer batch merely because they share a trajectory.
            for k in rng.permutation(len(x_c)):
                nnz = int(len_c[k])
                row = row_c[k, :nnz]
                col = col_c[k, :nnz]
                # Copy the sample so buffered views do not keep an entire
                # trajectory-sized HDF5 slice alive in RAM.
                shuffle_buffer.append((
                    x_c[k].copy(),
                    y_c[k].copy(),
                    row.copy(),
                    col.copy(),
                ))
                if len(shuffle_buffer) >= shuffle_buffer_size:
                    emit_random_buffered_sample()
            del x_c, y_c, row_c, col_c, len_c

    while shuffle_buffer:
        emit_random_buffered_sample()

    if x_batch and not drop_remainder:
        flush_batch()
    if native_tfrecord:
        if record_writer is not None:
            record_writer.close()
    else:
        flush_shard()
    return shard_paths, max(1, total_batches)


def dataset_from_batch_shards_3d(shard_paths):
    """Create a repeatable dataset while opening each shard only once per epoch."""
    def generator():
        # from_generator is restarted by repeat(), so this is a new random
        # shard/batch order without retaining the full chunk in RAM.
        for path in np.random.permutation(shard_paths):
            with np.load(path) as shard:
                # Materialize each uncompressed NPZ member once. Re-indexing an
                # NpzFile member inside the batch loop would reopen the ZIP
                # member and reload the entire shard for every mini-batch.
                x_all = shard["x"]
                y_all = shard["y"]
                indices_all = shard["indices"]
                node_offsets = shard["node_offsets"]
                edge_offsets = shard["edge_offsets"]
                batch_order = np.random.permutation(len(node_offsets) - 1)
                n_boids = int(shard["n_boids"])
                for batch_idx in batch_order:
                    node_start = int(node_offsets[batch_idx])
                    node_end = int(node_offsets[batch_idx + 1])
                    edge_start = int(edge_offsets[batch_idx])
                    edge_end = int(edge_offsets[batch_idx + 1])
                    n_nodes = node_end - node_start
                    graph_count = n_nodes // n_boids
                    yield batch_to_tf_sparse_3d((
                        (
                            x_all[node_start:node_end],
                            indices_all[edge_start:edge_end].astype(
                                np.int64, copy=False
                            ),
                            np.ones(edge_end - edge_start, dtype=np.float32),
                            np.asarray([n_nodes, n_nodes], dtype=np.int64),
                            np.repeat(
                                np.arange(graph_count, dtype=np.int64), n_boids
                            ),
                        ),
                        y_all[node_start:node_end],
                    ))

    output_signature = (
        (
            tf.TensorSpec(shape=(None, 6), dtype=tf.float32),
            tf.SparseTensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
        tf.TensorSpec(shape=(None, 15), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    ).repeat()


def materialize_cloud_native_dataset_3d(
    dataset,
    steps,
    output_dir,
    shuffle_batches,
):
    """Serialize one finite epoch and reload it through native tf.data I/O.

    Materialization pays the Python generator cost once per chunk. Training
    epochs subsequently read TensorFlow's native saved-dataset representation,
    so no Python callback is required for each sparse graph batch.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("Native cloud dataset requires at least one batch.")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    finite_dataset = dataset.take(steps)
    save_t0 = time.time()
    print(
        f">>> Cloud native dataset: materializing {steps} batches to "
        f"{output_dir}...",
        flush=True,
    )
    finite_dataset.save(output_dir, compression=None)

    def reader_func(shard_datasets):
        # Shuffle/interleave native shards in TensorFlow rather than opening
        # NPZ members and yielding batches from Python during every epoch.
        return shard_datasets.shuffle(1024).interleave(
            lambda shard: shard,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    native_dataset = tf.data.Dataset.load(
        output_dir,
        element_spec=finite_dataset.element_spec,
        compression=None,
        reader_func=reader_func,
    )
    if shuffle_batches and steps > 1:
        # The examples inside every packed batch were already timestep-shuffled
        # by the unchanged packer. A bounded native shuffle varies batch order
        # across epochs without buffering an entire 70+ GB chunk in RAM.
        native_dataset = native_dataset.shuffle(
            min(256, steps),
            reshuffle_each_iteration=True,
        )
    native_dataset = native_dataset.repeat().prefetch(tf.data.AUTOTUNE)
    print(
        f">>> Cloud native dataset ready in {time.time() - save_t0:.1f}s.",
        flush=True,
    )
    return native_dataset


def _proportional_chunk_allocations_3d(total_counts, chunk_size):
    """Allocate every chunk proportionally while preserving exact octant totals."""
    remaining = {int(o): int(count) for o, count in total_counts.items()}
    allocations = []
    while sum(remaining.values()) > 0:
        current_size = min(int(chunk_size), sum(remaining.values()))
        total_remaining = sum(remaining.values())
        exact = {
            o: current_size * count / total_remaining
            for o, count in remaining.items()
        }
        allocation = {
            o: min(remaining[o], int(np.floor(exact[o])))
            for o in remaining
        }
        unassigned = current_size - sum(allocation.values())
        while unassigned > 0:
            candidates = [
                o for o in remaining if allocation[o] < remaining[o]
            ]
            chosen = max(
                candidates,
                key=lambda o: (exact[o] - allocation[o], remaining[o]),
            )
            allocation[chosen] += 1
            unassigned -= 1
        allocation = {o: count for o, count in allocation.items() if count}
        allocations.append(allocation)
        for o, count in allocation.items():
            remaining[o] -= count
    return allocations


def _build_chunk_worker_command_3d(
    chunk_idx,
    chunk_file,
    cache_paths_file,
    checkpoint_path,
    n_boids_cache,
    boundaries_file,
    val_npz_file,
    training_state_dir,
    chunk_patience,
):
    """Build the isolated worker command shared by cached and generated chunks."""
    cmd = [
        sys.executable,
        "-m",
        "boids.run_boids_3d",
        "--_chunk_worker",
        "--_chunk_idx", str(chunk_idx),
        "--_chunk_indices_file", chunk_file,
        "--_cache_paths_file", cache_paths_file,
        "--_checkpoint_path", checkpoint_path,
        "--_n_boids_cache", str(n_boids_cache),
        "--_boundaries_file", boundaries_file,
        "--_val_npz_file", val_npz_file,
        "--_training_state_dir", training_state_dir,
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--epochs", "10000",
        "--es_patience", str(chunk_patience),
        "--lr_patience", str(args.lr_patience),
        "--lr_red_factor", str(args.lr_red_factor),
        "--min_lr", str(args.min_lr),
        "--n_boids", str(args.n_boids),
        "--loss_type", args.loss_type,
        "--critical_distance", str(args.critical_distance),
        "--distance_weight", str(args.distance_weight),
        "--timestep_stride", str(args.timestep_stride),
        "--near_goal_radius", str(args.near_goal_radius),
        "--perception", str(args.perception),
        "--shuffle_buffer_size", str(args.shuffle_buffer_size),
        "--packed_shard_batches", str(args.packed_shard_batches),
        "--steps_per_execution", str(args.steps_per_execution),
        "--expert_goal_order", args.expert_goal_order,
        "--goal_exclusion_size", str(args.goal_exclusion_size),
        "--expert_pos_noise", str(args.expert_pos_noise),
        "--expert_vel_noise", str(args.expert_vel_noise),
    ]
    if args.eager_training:
        cmd.append("--eager_training")
    if args.multi_gpu:
        cmd.append("--multi_gpu")
    if args.cloud_optimized:
        cmd.append("--cloud_optimized")
    if args.cloud_native_dataset:
        cmd.append("--cloud_native_dataset")
    if args.cloud_bfloat16:
        cmd.append("--cloud_bfloat16")
    if args.chunk_epochs is not None:
        cmd.extend(["--chunk_epochs", str(args.chunk_epochs)])
    if args.init_weights:
        cmd.extend(["--init_weights", args.init_weights])
    return cmd


def _generate_chunk_cache_shards_3d(chunk_idx, octant_counts, output_dir):
    """Generate one temporary cache shard per CPU task for a training chunk."""
    total = sum(octant_counts.values())
    workers = args.generation_workers
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 2)
    workers = max(1, min(workers, total))
    target_task_size = max(1, int(np.ceil(total / workers)))

    task_specs = []
    for octant, count in sorted(octant_counts.items()):
        n_tasks = max(1, int(np.ceil(count / target_task_size)))
        task_counts = [len(part) for part in np.array_split(np.arange(count), n_tasks)]
        for task_count in task_counts:
            task_specs.append({"octant": int(octant), "count": int(task_count)})

    def run_task(task_idx, spec):
        seed = int(args.generation_seed + chunk_idx * 1_000_003 + task_idx)
        path = os.path.join(output_dir, f"generated_{task_idx:04d}.h5")
        cmd = [
            sys.executable,
            "-m",
            "boids.generate_boids_cache_3d",
            "--unique", str(spec["count"]),
            "--repeats", "1",
            "--n_boids", str(args.n_boids),
            "--sample_mode", "octant",
            "--octants", str(spec["octant"]),
            "--goal_exclusion_size", str(args.goal_exclusion_size),
            "--pos_noise", str(args.expert_pos_noise),
            "--vel_noise", str(args.expert_vel_noise),
            "--perception", str(args.perception),
            "--goal_order", args.expert_goal_order,
            "--seed", str(seed),
            "--output", path,
            "--skip_centers_plot",
            "--quiet",
        ]
        env = os.environ.copy()
        # Expert simulation is CPU-only. Hiding the GPU prevents dozens of
        # generator processes from each creating a TensorFlow CUDA context.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = "1"
        started = time.time()
        result = subprocess.run(
            cmd, env=env, text=True, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Expert-generation task {task_idx} failed for octant "
                f"{spec['octant']} (exit {result.returncode}).\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return {
            **spec,
            "seed": seed,
            "path": path,
            "seconds": time.time() - started,
        }

    print(
        f">>> Generating {total} expert trajectories across {len(task_specs)} "
        f"tasks with up to {workers} concurrent CPU workers...",
        flush=True,
    )
    completed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_task, task_idx, spec): task_idx
            for task_idx, spec in enumerate(task_specs)
        }
        for future in as_completed(futures):
            record = future.result()
            completed.append(record)
            print(
                f"  Generated {record['count']} octant-{record['octant']} "
                f"trajectories in {record['seconds']:.1f}s",
                flush=True,
            )

    completed.sort(key=lambda record: record["path"])
    cache_paths = [record["path"] for record in completed]
    boundaries_by_cache = []
    trajectory_refs = []
    centers = []
    center_octants = []
    for cache_idx, record in enumerate(completed):
        with h5py.File(record["path"], "r") as cache_file:
            if int(cache_file.attrs["n_boids"]) != args.n_boids:
                raise ValueError("Generated cache n_boids does not match training.")
            lengths = cache_file["traj_lengths"][:]
            boundaries_by_cache.append(np.concatenate([
                np.zeros(1, dtype=np.int64),
                np.cumsum(lengths, dtype=np.int64),
            ]))
            shard_centers = cache_file["centers"][:].astype(np.float32)
        trajectory_refs.extend((cache_idx, i) for i in range(len(lengths)))
        centers.append(shard_centers)
        center_octants.extend([record["octant"]] * len(shard_centers))

    refs = np.asarray(trajectory_refs, dtype=np.int64)
    np.random.shuffle(refs)
    return (
        cache_paths,
        boundaries_by_cache,
        refs,
        np.concatenate(centers, axis=0),
        np.asarray(center_octants, dtype=np.int8),
        completed,
    )


def run_chunked_3d(cache_paths, boundaries_by_cache, n_boids_cache,
                   all_trajectory_refs,
                   run_tag, chunk_size, chunk_patience,
                   octant_buckets=None):
    """Train balanced chunks in isolated workers while carrying model state and LR."""
    checkpoint_path = f"saved_models/best_weights_3d_{run_tag}"
    os.makedirs("saved_models", exist_ok=True)

    # ── Build balanced chunks ────────────────────────────────────────────────
    if octant_buckets and len(octant_buckets) > 1:
        active = {o: np.array(idxs) for o, idxs in octant_buckets.items() if len(idxs) > 0}
        for o in active:
            np.random.shuffle(active[o])
        pointers = {o: 0 for o in active}
        chunks, chunk_mixes = [], []
        while True:
            remaining = {o: len(active[o]) - pointers[o] for o in active}
            total_remaining = sum(remaining.values())
            if total_remaining == 0:
                break

            current_size = min(chunk_size, total_remaining)
            exact = {
                o: current_size * remaining[o] / total_remaining
                for o in active
            }
            allocation = {
                o: min(remaining[o], int(np.floor(exact[o])))
                for o in active
            }
            unassigned = current_size - sum(allocation.values())
            while unassigned > 0:
                candidates = [
                    o for o in active
                    if allocation[o] < remaining[o]
                ]
                if not candidates:
                    raise RuntimeError("Could not allocate a complete replay chunk.")
                chosen = max(
                    candidates,
                    key=lambda o: (exact[o] - allocation[o], remaining[o]),
                )
                allocation[chosen] += 1
                unassigned -= 1

            current = []
            for o in active:
                take = allocation[o]
                start = pointers[o]
                current.extend(active[o][start:start + take].tolist())
                pointers[o] += take
            np.random.shuffle(current)
            chunks.append(np.array(current, dtype=np.int64))
            chunk_mixes.append({o: allocation[o] for o in active})
        n_chunks = len(chunks)
        requested_mix = {o: len(active[o]) for o in active}
        print(
            f">>> Chunked training: {n_chunks} proportional balanced chunks; "
            f"requested octant totals {requested_mix}"
        )
    else:
        n_chunks = max(1, len(all_trajectory_refs) // chunk_size)
        chunks = np.array_split(all_trajectory_refs, n_chunks)
        chunk_mixes = None
        print(f">>> Chunked training: {n_chunks} chunks of ~{chunk_size} trajectories each")

    def _mem_mb():
        return psutil.Process(os.getpid()).memory_info().rss / 1e6

    with tempfile.TemporaryDirectory(prefix="gnca_3d_chunks_") as tmpdir:
        boundaries_file = os.path.join(tmpdir, "boundaries.npz")
        cache_paths_file = os.path.join(tmpdir, "cache_paths.json")
        training_state_dir = os.path.join(tmpdir, "training_state")
        np.savez(
            boundaries_file,
            **{
                f"cache_{cache_idx}": boundaries
                for cache_idx, boundaries in enumerate(boundaries_by_cache)
            },
        )
        with open(cache_paths_file, "w", encoding="utf-8") as f:
            json.dump([os.path.abspath(path) for path in cache_paths], f)

        for chunk_idx, chunk_flat in enumerate(chunks):
            chunk_file = os.path.join(tmpdir, f"chunk_{chunk_idx:04d}.json")
            val_npz_file = os.path.join(tmpdir, f"validation_{chunk_idx:04d}.npz")
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(np.asarray(chunk_flat, dtype=np.int64).tolist(), f)

            print(f"\n{'='*60}")
            print(f"  Chunk {chunk_idx+1}/{n_chunks} | {len(chunk_flat)} trajectories | RAM: {_mem_mb():.0f} MB")
            print(f"{'='*60}")
            if chunk_mixes is not None:
                print(f">>> Chunk octant mix: {chunk_mixes[chunk_idx]}")
            print(f">>> Generating fresh validation set for chunk {chunk_idx+1}/{n_chunks} ({args.va_set_size} centers)...")
            data_va_chunk = make_validation_dataset_3d(args)
            save_dataset_npz_3d(data_va_chunk, val_npz_file, n_boids_cache)
            del data_va_chunk
            gc.collect()

            cmd = _build_chunk_worker_command_3d(
                chunk_idx,
                chunk_file,
                cache_paths_file,
                checkpoint_path,
                n_boids_cache,
                boundaries_file,
                val_npz_file,
                training_state_dir,
                chunk_patience,
            )

            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Chunk worker {chunk_idx + 1}/{n_chunks} failed with "
                    f"exit code {result.returncode}"
                )
            if os.path.exists(val_npz_file):
                os.remove(val_npz_file)
            gc.collect()
            print(f"  Parent RAM after chunk process exited: {_mem_mb():.0f} MB")

    final_model = _build_model_3d(args.lr)
    _build_model_variables_3d(final_model, n_boids_cache)
    final_model.load_weights(checkpoint_path).expect_partial()
    final_model.save_weights(checkpoint_path)
    print(f"Saved final weights to {checkpoint_path}")
    return final_model


def run_generated_chunked_3d(run_tag, requested_counts, chunk_size, chunk_patience):
    """Generate, train, and delete one balanced expert chunk at a time."""
    checkpoint_path = f"saved_models/best_weights_3d_{run_tag}"
    os.makedirs("saved_models", exist_ok=True)
    allocations = _proportional_chunk_allocations_3d(
        requested_counts, chunk_size
    )
    n_chunks = len(allocations)
    print(
        f">>> On-the-fly chunked training: {n_chunks} chunks; "
        f"requested octant totals {requested_counts}"
    )

    temp_root = args.on_the_fly_tmp_dir or None
    if temp_root is not None:
        os.makedirs(temp_root, exist_ok=True)
    all_centers = []
    all_center_octants = []
    manifest_tasks = []

    with tempfile.TemporaryDirectory(
        prefix="gnca_3d_onthefly_", dir=temp_root
    ) as tmpdir:
        training_state_dir = os.path.join(tmpdir, "training_state")
        for chunk_idx, chunk_mix in enumerate(allocations):
            print(f"\n{'='*60}")
            print(
                f"  Chunk {chunk_idx + 1}/{n_chunks} | "
                f"{sum(chunk_mix.values())} generated trajectories"
            )
            print(f"{'='*60}")
            print(f">>> Chunk octant mix: {chunk_mix}")

            with tempfile.TemporaryDirectory(
                prefix=f"chunk_{chunk_idx:04d}_", dir=tmpdir
            ) as chunk_dir:
                generation_t0 = time.time()
                (
                    cache_paths,
                    boundaries_by_cache,
                    chunk_refs,
                    chunk_centers,
                    center_octants,
                    task_records,
                ) = _generate_chunk_cache_shards_3d(
                    chunk_idx, chunk_mix, chunk_dir
                )
                print(
                    f">>> Expert chunk ready in "
                    f"{time.time() - generation_t0:.1f}s; beginning packing/training."
                )
                all_centers.append(chunk_centers)
                all_center_octants.append(center_octants)
                for record in task_records:
                    manifest_tasks.append({
                        "chunk": chunk_idx,
                        "octant": record["octant"],
                        "count": record["count"],
                        "seed": record["seed"],
                    })

                boundaries_file = os.path.join(chunk_dir, "boundaries.npz")
                cache_paths_file = os.path.join(chunk_dir, "cache_paths.json")
                chunk_file = os.path.join(chunk_dir, "chunk.json")
                val_npz_file = os.path.join(chunk_dir, "validation.npz")
                np.savez(
                    boundaries_file,
                    **{
                        f"cache_{cache_idx}": boundaries
                        for cache_idx, boundaries in enumerate(boundaries_by_cache)
                    },
                )
                with open(cache_paths_file, "w", encoding="utf-8") as f:
                    json.dump([os.path.abspath(path) for path in cache_paths], f)
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(chunk_refs.tolist(), f)

                print(
                    f">>> Generating fresh validation set for chunk "
                    f"{chunk_idx + 1}/{n_chunks} ({args.va_set_size} centers)..."
                )
                data_va_chunk = make_validation_dataset_3d(args)
                save_dataset_npz_3d(data_va_chunk, val_npz_file, args.n_boids)
                del data_va_chunk
                gc.collect()

                cmd = _build_chunk_worker_command_3d(
                    chunk_idx,
                    chunk_file,
                    cache_paths_file,
                    checkpoint_path,
                    args.n_boids,
                    boundaries_file,
                    val_npz_file,
                    training_state_dir,
                    chunk_patience,
                )
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Generated chunk worker {chunk_idx + 1}/{n_chunks} "
                        f"failed with exit code {result.returncode}"
                    )
            gc.collect()
            print(
                f">>> Deleted temporary expert caches for chunk "
                f"{chunk_idx + 1}/{n_chunks}."
            )

    final_model = _build_model_3d(args.lr)
    _build_model_variables_3d(final_model, args.n_boids)
    final_model.load_weights(checkpoint_path).expect_partial()
    final_model.save_weights(checkpoint_path)
    print(f"Saved final weights to {checkpoint_path}")
    manifest = {
        "mode": "on_the_fly",
        "generation_seed": args.generation_seed,
        "generation_workers": args.generation_workers,
        "goal_order": args.expert_goal_order,
        "goal_exclusion_size": args.goal_exclusion_size,
        "pos_noise": args.expert_pos_noise,
        "vel_noise": args.expert_vel_noise,
        "perception": args.perception,
        "octant_unique_counts": requested_counts,
        "tasks": manifest_tasks,
    }
    return (
        final_model,
        np.concatenate(all_centers, axis=0),
        np.concatenate(all_center_octants, axis=0),
        manifest,
    )


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
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        run_eagerly=args.eager_training,
        steps_per_execution=args.steps_per_execution,
    )
    if args.init_weights:
        _build_model_variables_3d(model, args.n_boids)
        model.load_weights(args.init_weights).expect_partial()
        print(f">>> Initialized model from weights: {args.init_weights}")

    loader_tr = DisjointLoader(data_tr, node_level=True, batch_size=args.batch_size)
    loader_va = DisjointLoader(data_va, node_level=True, batch_size=args.batch_size)

    checkpoint_path = f"saved_models/best_weights_3d_{run_tag}"
    best_after50_cb = BestAfterEpochCallback3D(save_path=checkpoint_path, min_epoch=args.best_after_epoch)

    history = model.fit(
        loader_tr.load().map(add_graph_ids_to_targets_3d),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=args.epochs,
        validation_data=loader_va.load().map(add_graph_ids_to_targets_3d),
        validation_steps=loader_va.steps_per_epoch,
        callbacks=[
            EarlyStopping(patience=args.es_patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=args.lr_patience, factor=args.lr_red_factor,
                              min_lr=args.min_lr, min_delta=1e-8, verbose=1),
            best_after50_cb,
        ],
    )

    return history, model, best_after50_cb


def _distribute_complete_graph_batches_3d(strategy, local_batches):
    """Place one already-disjoint, graph-complete batch on each replica."""
    return strategy.experimental_distribute_values_from_function(
        lambda context: local_batches[context.replica_id_in_sync_group]
    )


def fit_mirrored_worker_3d(
    model,
    strategy,
    train_data,
    train_local_steps,
    n_epochs,
    callbacks,
    global_node_count,
    val_data=None,
    val_local_steps=0,
):
    """Synchronous graph-safe data parallelism for prepacked sparse batches.

    Each replica receives a complete local block-diagonal adjacency matrix.
    Gradients are normalized across all nodes and synchronously all-reduced by
    the mirrored optimizer, preserving the single-GPU global-batch objective.
    """
    replicas = strategy.num_replicas_in_sync
    train_steps = train_local_steps // replicas
    if train_steps < 1:
        raise ValueError(
            f"Need at least {replicas} local train batches for {replicas} GPUs."
        )
    dropped_train = train_local_steps - train_steps * replicas
    if dropped_train:
        print(
            f">>> Multi-GPU: dropping {dropped_train} incomplete local "
            "batch(es) per epoch to keep replicas synchronized."
        )

    val_steps = val_local_steps // replicas if val_data is not None else 0
    if val_data is not None and val_steps < 1:
        raise ValueError(
            f"Need at least {replicas} local validation batches for "
            f"{replicas} GPUs."
        )

    @tf.function(reduce_retracing=True)
    def distributed_train_step(distributed_batch):
        def replica_step(batch):
            inputs, targets = batch
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                per_node_loss = custom_weighted_mse_3d(targets, predictions)
                local_loss_sum = tf.reduce_sum(per_node_loss)
                local_node_count = tf.cast(tf.size(per_node_loss), tf.float32)
                loss = local_loss_sum / tf.cast(global_node_count, tf.float32)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            return local_loss_sum, local_node_count

        loss_sums, node_counts = strategy.run(
            replica_step, args=(distributed_batch,)
        )
        total_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, loss_sums, axis=None
        )
        total_nodes = strategy.reduce(
            tf.distribute.ReduceOp.SUM, node_counts, axis=None
        )
        return total_loss, total_nodes

    @tf.function(reduce_retracing=True)
    def distributed_validation_step(distributed_batch):
        def replica_step(batch):
            inputs, targets = batch
            predictions = model(inputs, training=False)
            per_node_loss = custom_weighted_mse_3d(targets, predictions)
            return (
                tf.reduce_sum(per_node_loss),
                tf.cast(tf.size(per_node_loss), tf.float32),
            )

        loss_sums, node_counts = strategy.run(
            replica_step, args=(distributed_batch,)
        )
        return (
            strategy.reduce(tf.distribute.ReduceOp.SUM, loss_sums, axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, node_counts, axis=None),
        )

    callback_list = tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=False,
        add_progbar=False,
        model=model,
        epochs=n_epochs,
        steps=train_steps,
        verbose=1,
        metrics=["loss"] + (["val_loss"] if val_data is not None else []),
    )
    model.stop_training = False
    callback_list.on_train_begin()
    train_iterator = iter(train_data)
    try:
        for epoch in range(n_epochs):
            if model.stop_training:
                break
            epoch_t0 = time.time()
            callback_list.on_epoch_begin(epoch)
            epoch_loss_sum = 0.0
            epoch_node_count = 0.0

            for step in range(train_steps):
                callback_list.on_train_batch_begin(step)
                local_batches = [next(train_iterator) for _ in range(replicas)]
                distributed_batch = _distribute_complete_graph_batches_3d(
                    strategy, local_batches
                )
                loss_sum, node_count = distributed_train_step(distributed_batch)
                epoch_loss_sum += float(loss_sum.numpy())
                epoch_node_count += float(node_count.numpy())
                callback_list.on_train_batch_end(step)

            logs = {"loss": epoch_loss_sum / max(epoch_node_count, 1.0)}
            if val_data is not None:
                val_iterator = iter(val_data)
                val_loss_sum = 0.0
                val_node_count = 0.0
                for _ in range(val_steps):
                    local_batches = [
                        next(val_iterator) for _ in range(replicas)
                    ]
                    distributed_batch = _distribute_complete_graph_batches_3d(
                        strategy, local_batches
                    )
                    loss_sum, node_count = distributed_validation_step(
                        distributed_batch
                    )
                    val_loss_sum += float(loss_sum.numpy())
                    val_node_count += float(node_count.numpy())
                logs["val_loss"] = val_loss_sum / max(val_node_count, 1.0)

            callback_list.on_epoch_end(epoch, logs)
            learning_rate = float(
                tf.keras.backend.get_value(model.optimizer.learning_rate)
            )
            summary = (
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"{time.time() - epoch_t0:.1f}s - loss: {logs['loss']:.6g}"
            )
            if "val_loss" in logs:
                summary += f" - val_loss: {logs['val_loss']:.6g}"
            summary += f" - lr: {learning_rate:.6g}"
            print(summary, flush=True)
    finally:
        callback_list.on_train_end()

    return train_steps


def fit_cloud_mirrored_worker_3d(
    model,
    strategy,
    train_data,
    train_local_steps,
    n_epochs,
    callbacks,
    global_node_count,
    steps_per_execution,
    val_data=None,
    val_local_steps=0,
):
    """Cloud-only distributed loop with compiled multi-step execution.

    The input datasets already contain complete per-replica disjoint graph
    batches. ``distribute_datasets_from_function`` dequeues one such batch for
    every replica without splitting its node or adjacency tensors. Multiple
    optimizer steps then execute inside one ``tf.function`` call, avoiding the
    Python dispatch and device-to-host synchronization performed per step by
    the compatibility multi-GPU loop above.
    """
    replicas = strategy.num_replicas_in_sync
    train_steps = train_local_steps // replicas
    if train_steps < 1:
        raise ValueError(
            f"Need at least {replicas} local train batches for {replicas} GPUs."
        )
    dropped_train = train_local_steps - train_steps * replicas
    if dropped_train:
        print(
            f">>> Cloud optimized: dropping {dropped_train} incomplete local "
            "batch(es) per epoch to keep replicas synchronized."
        )

    val_steps = val_local_steps // replicas if val_data is not None else 0
    if val_data is not None and val_steps < 1:
        raise ValueError(
            f"Need at least {replicas} local validation batches for "
            f"{replicas} GPUs."
        )

    # The source datasets are already batched at the per-replica size. This API
    # distributes whole dataset elements instead of slicing a disjoint graph's
    # node dimension as experimental_distribute_dataset would do.
    distributed_train_data = strategy.distribute_datasets_from_function(
        lambda _input_context: train_data
    )
    distributed_val_data = None
    if val_data is not None:
        distributed_val_data = strategy.distribute_datasets_from_function(
            lambda _input_context: val_data
        )

    def replica_train_step(batch):
        inputs, targets = batch
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            per_node_loss = custom_weighted_mse_3d(targets, predictions)
            local_loss_sum = tf.reduce_sum(per_node_loss)
            local_node_count = tf.cast(tf.size(per_node_loss), tf.float32)
            loss = local_loss_sum / tf.cast(global_node_count, tf.float32)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return local_loss_sum, local_node_count

    def replica_validation_step(batch):
        inputs, targets = batch
        predictions = model(inputs, training=False)
        per_node_loss = custom_weighted_mse_3d(targets, predictions)
        return (
            tf.reduce_sum(per_node_loss),
            tf.cast(tf.size(per_node_loss), tf.float32),
        )

    @tf.function(reduce_retracing=True)
    def distributed_train_block(iterator, block_steps):
        total_loss = tf.constant(0.0, dtype=tf.float32)
        total_nodes = tf.constant(0.0, dtype=tf.float32)
        for _ in tf.range(block_steps):
            loss_sums, node_counts = strategy.run(
                replica_train_step, args=(next(iterator),)
            )
            total_loss += strategy.reduce(
                tf.distribute.ReduceOp.SUM, loss_sums, axis=None
            )
            total_nodes += strategy.reduce(
                tf.distribute.ReduceOp.SUM, node_counts, axis=None
            )
        return total_loss, total_nodes

    @tf.function(reduce_retracing=True)
    def distributed_validation_block(iterator, block_steps):
        total_loss = tf.constant(0.0, dtype=tf.float32)
        total_nodes = tf.constant(0.0, dtype=tf.float32)
        for _ in tf.range(block_steps):
            loss_sums, node_counts = strategy.run(
                replica_validation_step, args=(next(iterator),)
            )
            total_loss += strategy.reduce(
                tf.distribute.ReduceOp.SUM, loss_sums, axis=None
            )
            total_nodes += strategy.reduce(
                tf.distribute.ReduceOp.SUM, node_counts, axis=None
            )
        return total_loss, total_nodes

    callback_list = tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=False,
        add_progbar=False,
        model=model,
        epochs=n_epochs,
        steps=train_steps,
        verbose=1,
        metrics=["loss"] + (["val_loss"] if val_data is not None else []),
    )
    execution_steps = max(1, min(int(steps_per_execution), train_steps))
    print(
        f">>> Cloud optimized execution: {replicas} replicas, "
        f"{train_steps} global steps/epoch, "
        f"{execution_steps} compiled steps/dispatch.",
        flush=True,
    )

    model.stop_training = False
    callback_list.on_train_begin()
    train_iterator = iter(distributed_train_data)
    try:
        for epoch in range(n_epochs):
            if model.stop_training:
                break
            epoch_t0 = time.time()
            callback_list.on_epoch_begin(epoch)
            epoch_loss_sum = 0.0
            epoch_node_count = 0.0

            for block_start in range(0, train_steps, execution_steps):
                block_steps = min(execution_steps, train_steps - block_start)
                loss_sum, node_count = distributed_train_block(
                    train_iterator, tf.constant(block_steps, dtype=tf.int32)
                )
                # One host synchronization per compiled block, never per step.
                epoch_loss_sum += float(loss_sum.numpy())
                epoch_node_count += float(node_count.numpy())

            logs = {"loss": epoch_loss_sum / max(epoch_node_count, 1.0)}
            if distributed_val_data is not None:
                val_iterator = iter(distributed_val_data)
                val_loss_sum = 0.0
                val_node_count = 0.0
                val_execution_steps = max(1, min(execution_steps, val_steps))
                for block_start in range(0, val_steps, val_execution_steps):
                    block_steps = min(
                        val_execution_steps, val_steps - block_start
                    )
                    loss_sum, node_count = distributed_validation_block(
                        val_iterator,
                        tf.constant(block_steps, dtype=tf.int32),
                    )
                    val_loss_sum += float(loss_sum.numpy())
                    val_node_count += float(node_count.numpy())
                logs["val_loss"] = val_loss_sum / max(val_node_count, 1.0)

            callback_list.on_epoch_end(epoch, logs)
            learning_rate = float(
                tf.keras.backend.get_value(model.optimizer.learning_rate)
            )
            summary = (
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"{time.time() - epoch_t0:.1f}s - loss: {logs['loss']:.6g}"
            )
            if "val_loss" in logs:
                summary += f" - val_loss: {logs['val_loss']:.6g}"
            summary += f" - lr: {learning_rate:.6g}"
            print(summary, flush=True)
    finally:
        callback_list.on_train_end()

    return train_steps

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
parser.add_argument("--lr_patience", default=5, type=int,
                   help="Epochs without improvement before reducing LR; keep below chunk_patience.")
parser.add_argument("--lr_red_factor", default=0.1, type=float)
parser.add_argument("--min_lr", default=1e-6, type=float,
                   help="Minimum learning-rate floor for ReduceLROnPlateau.")
parser.add_argument("--n_boids", default=100, type=int)
parser.add_argument("--tr_set_unique", default=20, type=int)
parser.add_argument(
    "--tr_set_repeats",
    default=50,
    type=int,
    help=(
        "Training exposures per selected center. If this exceeds the cache's "
        "repeat count, cached trajectories are cycled for reinforcement."
    ),
)
parser.add_argument("--va_set_size", default=10, type=int)
parser.add_argument("--te_set_size", default=30, type=int)
parser.add_argument("--loss_type", default="oldl", choices=["oldl", "newl"],
                   help="oldl: standard MSE, newl: distance-weighted MSE")
parser.add_argument("--critical_distance", default=2.5, type=float)
parser.add_argument("--distance_weight", default=2.5, type=float)
parser.add_argument("--viz_mode", default="tubular", choices=["tubular", "per_boid", "multi_tubular", "individual"])
# ── Chunk worker args (internal use, passed by run_chunked_3d subprocess) ──
parser.add_argument("--_chunk_worker", action="store_true", default=False)
parser.add_argument("--_chunk_idx", default=0, type=int)
parser.add_argument("--_chunk_indices_file", default=None)
parser.add_argument("--_cache_paths_file", default=None)
parser.add_argument("--_checkpoint_path", default=None)
parser.add_argument("--_n_boids_cache", default=None, type=int)
parser.add_argument("--_boundaries_file", default=None)
parser.add_argument("--_val_indices_file", default=None)
parser.add_argument("--_val_npz_file", default=None)
parser.add_argument("--_training_state_dir", default=None)
parser.add_argument("--noise_tag", default="", type=str)
parser.add_argument("--init_centers_npz", default="", type=str)
parser.add_argument(
    "--boids_cache",
    nargs="+",
    default=None,
    help="One or more compatible HDF5 caches used directly during chunked training.",
)
parser.add_argument(
    "--generate_on_the_fly",
    action="store_true",
    help=(
        "Generate each expert chunk in parallel into temporary local HDF5 "
        "shards, train it, then delete it. Cannot be combined with --boids_cache."
    ),
)
parser.add_argument(
    "--generation_workers",
    default=0,
    type=int,
    help="Concurrent expert-generation subprocesses; 0 uses CPU count minus two.",
)
parser.add_argument("--generation_seed", default=0, type=int)
parser.add_argument(
    "--on_the_fly_tmp_dir",
    default="",
    help="Temporary generation root. Empty uses the OS temporary directory (/tmp).",
)
parser.add_argument(
    "--expert_goal_order",
    choices=["nearest_ccw", "fixed"],
    default="nearest_ccw",
)
parser.add_argument("--goal_exclusion_size", default=0.5, type=float)
parser.add_argument("--expert_pos_noise", default=0.0, type=float)
parser.add_argument("--expert_vel_noise", default=0.0, type=float)
parser.add_argument("--init_weights", default="", type=str,
                   help="Optional checkpoint prefix to initialize training from.")
parser.add_argument("--timestep_stride", default=1, type=int)
parser.add_argument("--near_goal_radius", default=1.0, type=float)
parser.add_argument("--perception", default=0.1, type=float,
                   help="Neighbor radius required of cached adjacency matrices.")
parser.add_argument("--shuffle_buffer_size", default=4096, type=int,
                   help="Number of individual timesteps mixed before packed batching.")
parser.add_argument("--packed_shard_batches", default=16, type=int,
                   help="Number of packed mini-batches stored in each temporary shard.")
parser.add_argument("--steps_per_execution", default=10, type=int,
                   help="Compiled optimizer steps run per Keras/Python dispatch.")
parser.add_argument("--eager_training", action="store_true",
                   help="Debug training eagerly instead of using the faster compiled graph.")
parser.add_argument(
    "--multi_gpu",
    action="store_true",
    help=(
        "Use all visible GPUs with graph-safe synchronous MirroredStrategy "
        "training. --batch_size remains the global batch size."
    ),
)
parser.add_argument(
    "--cloud_optimized",
    action="store_true",
    help=(
        "Opt-in cloud multi-GPU execution path: distribute complete graph "
        "batches through tf.data and compile multiple optimizer steps per "
        "Python dispatch. Requires --multi_gpu. Local/default paths are unchanged."
    ),
)
parser.add_argument(
    "--cloud_native_dataset",
    action="store_true",
    help=(
        "Cloud-only: write packed batches directly as TFRecord shards and "
        "train through native parallel tf.data readers without NPZ or a "
        "per-batch Python generator. Requires --cloud_optimized and --multi_gpu."
    ),
)
parser.add_argument(
    "--cloud_bfloat16",
    action="store_true",
    help=(
        "Cloud-only: use mixed_bfloat16 model compute with float32 variables, "
        "optimizer state, targets, weighting, and loss. Requires "
        "--cloud_optimized and --multi_gpu."
    ),
)
parser.add_argument("--chunk_size", default=100, type=int)
parser.add_argument("--chunk_patience", default=15, type=int)
parser.add_argument("--best_after_epoch", default=50, type=int,
                   help="Non-chunked only: save best val_loss checkpoint after this epoch (default: 50).")
parser.add_argument("--train_octants", type=int, nargs="+", default=None)
parser.add_argument(
    "--octant_unique_counts",
    type=int,
    nargs="+",
    default=None,
    help=(
        "Optional unique-center count for each --train_octants entry. "
        "Use this for focus-octant training with balanced replay."
    ),
)
parser.add_argument("--viz_n_centers", default=50, type=int)
parser.add_argument("--viz_centers_source", default="trained", choices=["trained", "unseen_cache", "random", "fresh_octant"])
parser.add_argument("--skip_trained_viz", action="store_true", default=False,
                   help="Skip computing trajectories from trained centers during evaluation.")
parser.add_argument("--skip_evaluation", action="store_true", default=False,
                   help="Skip post-training evaluation/visualization.")
parser.add_argument("--eval_centers_per_octant", default=10, type=int,
                   help="Fresh unseen centers sampled per evaluated octant after training.")
parser.add_argument("--eval_output_dir", default="", type=str,
                   help="Post-training inference directory. Defaults to inference_3d_<run_tag>.")
parser.add_argument("--eval_max_steps", default=3000, type=int)
parser.add_argument("--eval_success_threshold", default=0.5, type=float)
parser.add_argument("--eval_max_success_r", default=2.0, type=float)
parser.add_argument("--eval_seed", default=None, type=int)


####################################################################################
# Training
####################################################################################
args = parser.parse_args()

if args.tr_set_repeats < 1:
    raise ValueError("--tr_set_repeats must be at least 1.")
if args.generate_on_the_fly and args.boids_cache:
    raise ValueError("--generate_on_the_fly cannot be combined with --boids_cache.")
if args.generate_on_the_fly and args.tr_set_repeats != 1:
    raise ValueError(
        "On-the-fly generation currently requires --tr_set_repeats 1; "
        "every generated trajectory is unique."
    )
if args.generate_on_the_fly and args.train_octants is None:
    raise ValueError("--generate_on_the_fly requires explicit --train_octants.")
if args.generate_on_the_fly and args.chunk_size <= 0:
    raise ValueError("--generate_on_the_fly requires --chunk_size greater than 0.")
if args.generate_on_the_fly and args.init_centers_npz:
    raise ValueError(
        "--generate_on_the_fly does not currently combine with --init_centers_npz."
    )
if args.generation_workers < 0:
    raise ValueError("--generation_workers must be nonnegative.")
if args.train_octants is not None and len(set(args.train_octants)) != len(args.train_octants):
    raise ValueError("--train_octants must not contain duplicates.")
if args.octant_unique_counts is not None:
    if args.train_octants is None:
        raise ValueError("--octant_unique_counts requires --train_octants.")
    if len(args.octant_unique_counts) != len(args.train_octants):
        raise ValueError(
            "--octant_unique_counts must provide one count per --train_octants entry."
        )
    if any(count < 1 for count in args.octant_unique_counts):
        raise ValueError("Every --octant_unique_counts value must be positive.")
    if sum(args.octant_unique_counts) != args.tr_set_unique:
        raise ValueError(
            "The sum of --octant_unique_counts must equal --tr_set_unique."
        )
if args.min_lr <= 0:
    raise ValueError("--min_lr must be greater than 0.")
if args.min_lr > args.lr:
    raise ValueError("--min_lr cannot exceed --lr.")
if args.steps_per_execution < 1:
    raise ValueError("--steps_per_execution must be at least 1.")
if args.cloud_optimized and not args.multi_gpu:
    raise ValueError("--cloud_optimized requires --multi_gpu.")
if args.cloud_optimized and args.eager_training:
    raise ValueError("--cloud_optimized cannot be combined with --eager_training.")
if args.cloud_native_dataset and not args.cloud_optimized:
    raise ValueError("--cloud_native_dataset requires --cloud_optimized.")
if args.cloud_bfloat16 and not args.cloud_optimized:
    raise ValueError("--cloud_bfloat16 requires --cloud_optimized.")
if args.cloud_bfloat16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    policy = tf.keras.mixed_precision.global_policy()
    print(
        f">>> Cloud BF16 policy enabled: compute={policy.compute_dtype}, "
        f"variables={policy.variable_dtype}; loss=float32.",
        flush=True,
    )
if args.multi_gpu:
    if len(physical_devices) < 2:
        raise ValueError(
            f"--multi_gpu requires at least 2 visible GPUs; found "
            f"{len(physical_devices)}."
        )
    if args.batch_size % len(physical_devices) != 0:
        raise ValueError(
            "--batch_size is the global batch size in multi-GPU mode and must "
            f"be divisible by {len(physical_devices)} visible GPUs."
        )
    # The parent process validates the external data source, then launches each
    # isolated worker with an internal cache manifest rather than repeating
    # --generate_on_the_fly/--boids_cache.  Do not reject that valid worker mode.
    if not args._chunk_worker and (
        args.chunk_size <= 0
        or not (args.generate_on_the_fly or args.boids_cache)
    ):
        raise ValueError(
            "--multi_gpu currently requires chunked training from either "
            "--generate_on_the_fly or --boids_cache."
        )

if (
    not args._chunk_worker
    and args.chunk_size > 0
    and args.chunk_epochs is None
    and args.lr_patience >= args.chunk_patience
):
    raise ValueError(
        "--lr_patience must be smaller than --chunk_patience so the reduced "
        "learning rate is used before early stopping."
    )

# ── Chunk worker mode (subprocess per chunk) ─────────────────────────────────
if args._chunk_worker:
    UPWEIGHT_NEAR_GOAL = (args.loss_type == "newl")
    setup_t0 = time.time()
    with open(args._chunk_indices_file, encoding="utf-8") as f:
        chunk_refs = np.asarray(json.load(f), dtype=np.int64).reshape(-1, 2)
    with open(args._cache_paths_file, encoding="utf-8") as f:
        worker_cache_paths = json.load(f)
    with np.load(args._boundaries_file) as boundary_data:
        boundaries_by_cache = [
            boundary_data[f"cache_{cache_idx}"]
            for cache_idx in range(len(worker_cache_paths))
        ]

    replica_count = len(physical_devices) if args.multi_gpu else 1
    per_replica_batch_size = args.batch_size // replica_count

    worker_tmpdir = tempfile.mkdtemp(prefix="gnca_3d_worker_batches_")
    atexit.register(shutil.rmtree, worker_tmpdir, ignore_errors=True)
    print(
        f">>> Worker setup: {len(chunk_refs)} trajectories from "
        f"{len(worker_cache_paths)} cache(s), "
        f"global_batch_size={args.batch_size}, "
        f"per_replica_batch_size={per_replica_batch_size}, "
        f"replicas={replica_count}, stride={args.timestep_stride}, "
        f"near_goal_radius={args.near_goal_radius}",
        flush=True,
    )
    shard_format = "direct TFRecord" if args.cloud_native_dataset else "compact NPZ"
    print(
        f">>> Worker setup: writing {shard_format} train batch shards...",
        flush=True,
    )
    train_shard_paths, train_steps = write_cached_chunk_shards_3d(
        worker_cache_paths, chunk_refs, boundaries_by_cache, args._n_boids_cache,
        per_replica_batch_size,
        os.path.join(worker_tmpdir, "train_batches"),
        timestep_stride=args.timestep_stride, near_goal_radius=args.near_goal_radius,
        perception=args.perception, shuffle_buffer_size=args.shuffle_buffer_size,
        batches_per_shard=args.packed_shard_batches,
        drop_remainder=args.multi_gpu,
        native_tfrecord=args.cloud_native_dataset,
    )
    print(
        f">>> Worker setup: wrote {train_steps} train batches in "
        f"{len(train_shard_paths)} compact shards in "
        f"{time.time() - setup_t0:.1f}s",
        flush=True,
    )
    if args.cloud_native_dataset:
        train_data = dataset_from_cloud_tfrecord_shards_3d(
            train_shard_paths,
            train_steps,
        )
        print(
            ">>> Cloud native dataset: direct TFRecord loader ready; "
            "no NPZ conversion or Dataset.save pass required.",
            flush=True,
        )
    else:
        train_data = dataset_from_batch_shards_3d(train_shard_paths).map(
            add_graph_ids_to_targets_3d,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

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
        loader_va = DisjointLoader(
            data_val,
            node_level=True,
            batch_size=per_replica_batch_size,
        )

    if args._training_state_dir is None:
        raise RuntimeError("Chunk worker requires --_training_state_dir")

    strategy = None
    if args.multi_gpu:
        strategy = tf.distribute.MirroredStrategy(
            devices=[device.name for device in tf.config.list_logical_devices("GPU")]
        )
        print(
            f">>> Multi-GPU strategy initialized with "
            f"{strategy.num_replicas_in_sync} replicas."
        )
        with strategy.scope():
            model = _build_model_3d(args.lr)
            _build_model_variables_3d(model, args._n_boids_cache)
    else:
        model = _build_model_3d(args.lr)
        _build_model_variables_3d(model, args._n_boids_cache)
    training_checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=model.optimizer,
    )
    training_state_manager = tf.train.CheckpointManager(
        training_checkpoint,
        args._training_state_dir,
        max_to_keep=1,
    )

    if args._chunk_idx > 0:
        if training_state_manager.latest_checkpoint is None:
            raise RuntimeError(
                f"No complete training-state checkpoint available for chunk "
                f"{args._chunk_idx + 1}"
            )
        training_checkpoint.restore(
            training_state_manager.latest_checkpoint
        ).expect_partial()
        restored_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        restored_step = int(tf.keras.backend.get_value(model.optimizer.iterations))
        print(
            f">>> Worker restored model + Adam state from "
            f"{training_state_manager.latest_checkpoint} "
            f"(lr={restored_lr:g}, optimizer_step={restored_step})"
        )
    elif args.init_weights:
        model.load_weights(args.init_weights).expect_partial()
        print(f">>> Worker initialized from seed weights: {args.init_weights}")
    elif args._chunk_idx == 0 and os.path.exists(args._checkpoint_path + ".index"):
        print(f">>> Worker starting fresh for chunk 1; ignoring stale checkpoint: {args._checkpoint_path}")
    n_epochs = args.chunk_epochs if args.chunk_epochs is not None else args.epochs

    if args.chunk_epochs is not None:
        # Fixed epochs — no early stopping, no validation
        cbs = []
        val_data = None
        val_kwargs = {}
    elif use_val:
        best_state_cb = BestTrainingStateCallback3D(
            training_state_manager,
            monitor="val_loss",
        )
        cbs = [
            best_state_cb,
            ReduceLROnPlateau(monitor='val_loss', patience=args.lr_patience,
                              factor=args.lr_red_factor, min_lr=args.min_lr,
                              min_delta=1e-8, verbose=1),
            EarlyStopping(monitor='val_loss', patience=args.es_patience,
                          restore_best_weights=False, verbose=1),
        ]
        val_data = loader_va.load().map(add_graph_ids_to_targets_3d)
        if args.cloud_native_dataset:
            val_data = materialize_cloud_native_dataset_3d(
                val_data,
                loader_va.steps_per_epoch,
                os.path.join(worker_tmpdir, "native_validation_dataset"),
                shuffle_batches=False,
            )
        val_kwargs = {
            'validation_data': val_data,
            'validation_steps': loader_va.steps_per_epoch,
        }
    else:
        best_state_cb = BestTrainingStateCallback3D(
            training_state_manager,
            monitor="loss",
        )
        cbs = [
            best_state_cb,
            ReduceLROnPlateau(monitor='loss', patience=args.lr_patience,
                              factor=args.lr_red_factor, min_lr=args.min_lr,
                              min_delta=1e-8, verbose=1),
            EarlyStopping(monitor='loss', patience=args.es_patience,
                          restore_best_weights=False, verbose=1),
        ]
        val_data = None
        val_kwargs = {}

    if args.multi_gpu:
        mirrored_fit = (
            fit_cloud_mirrored_worker_3d
            if args.cloud_optimized
            else fit_mirrored_worker_3d
        )
        mirrored_kwargs = {}
        if args.cloud_optimized:
            mirrored_kwargs["steps_per_execution"] = args.steps_per_execution
        mirrored_fit(
            model,
            strategy,
            train_data,
            train_steps,
            n_epochs,
            cbs,
            global_node_count=args.batch_size * args._n_boids_cache,
            val_data=val_data,
            val_local_steps=(loader_va.steps_per_epoch if use_val else 0),
            **mirrored_kwargs,
        )
    else:
        model.fit(
            train_data, steps_per_epoch=train_steps,
            epochs=n_epochs, callbacks=cbs, **val_kwargs,
        )

    if args.chunk_epochs is not None:
        training_state_manager.save()
    elif best_state_cb.best_epoch is None:
        raise RuntimeError("Chunk did not produce a checkpointable training metric")
    else:
        training_checkpoint.restore(
            training_state_manager.latest_checkpoint
        ).expect_partial()
        print(
            f">>> Restored best complete state from epoch "
            f"{best_state_cb.best_epoch} ({best_state_cb.monitor}={best_state_cb.best:.6g})"
        )

    model.save_weights(args._checkpoint_path)
    final_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    final_step = int(tf.keras.backend.get_value(model.optimizer.iterations))
    print(
        f">>> Worker checkpoint: lr={final_lr:g}, optimizer_step={final_step}",
        flush=True,
    )
    shutil.rmtree(worker_tmpdir, ignore_errors=True)
    sys.exit(0)
# ─────────────────────────────────────────────────────────────────────────────

# Determine if using upweighted loss
UPWEIGHT_NEAR_GOAL = (args.loss_type == "newl")

# Set noise config
noise_config = args.noise_tag

# Load initial centers from NPZ if provided
train_init_centers = load_init_centers_from_npz(args.init_centers_npz)
effective_unique_reps = len(train_init_centers) if train_init_centers is not None else args.tr_set_unique

boids_caches = args.boids_cache or []
if boids_caches:
    cache_total_unique = 0
    for cache_path in boids_caches:
        with h5py.File(cache_path, "r") as cache_file:
            cache_total_unique += int(cache_file.attrs["unique_reps"])
    print(
        f">>> Using {len(boids_caches)} precomputed 3D cache(s): "
        f"{cache_total_unique} total unique centers available, "
        f"training on {effective_unique_reps}"
    )

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

use_cached_chunked = args.chunk_size > 0 and bool(boids_caches)
use_generated_chunked = args.chunk_size > 0 and args.generate_on_the_fly
use_chunked = use_cached_chunked or use_generated_chunked
data_va = None
generation_manifest = None

if not use_chunked and len(boids_caches) > 1:
    raise ValueError("Multiple --boids_cache files currently require chunked training.")

if not use_chunked:
    # Validation set
    print(f"\n>>> Generating validation set ({args.va_set_size} centers)...")
    data_va = make_validation_dataset_3d(args)

if use_generated_chunked:
    if args.octant_unique_counts is not None:
        requested_counts = dict(
            zip(args.train_octants, args.octant_unique_counts)
        )
    else:
        n_per_o = effective_unique_reps // len(args.train_octants)
        remainder = effective_unique_reps % len(args.train_octants)
        requested_counts = {
            o: n_per_o + (1 if i < remainder else 0)
            for i, o in enumerate(args.train_octants)
        }
    print(
        f"\n>>> Generating expert data one temporary chunk at a time: "
        f"{requested_counts}"
    )
    (
        model,
        generated_centers,
        generated_center_octants,
        generation_manifest,
    ) = run_generated_chunked_3d(
        run_tag,
        requested_counts,
        args.chunk_size,
        args.chunk_patience,
    )
    boids_tr = Boids3D(
        n_boids=args.n_boids,
        perception=args.perception,
        pos_noise=args.expert_pos_noise,
        vel_noise=args.expert_vel_noise,
        waypoint_order_policy=_expert_waypoint_policy_3d(args),
    )
    boids_tr.rand_configs = [
        np.asarray(center, dtype=np.float32) for center in generated_centers
    ]
    generation_manifest["center_octants"] = generated_center_octants.tolist()
    best_after50_cb = None
elif use_cached_chunked:
    print(f"\n>>> Chunked training directly from caches: {boids_caches}")
    boundaries_by_cache = []
    cache_repeats_by_cache = []
    center_cache_ids = []
    center_local_indices = []
    center_arrays = []
    n_boids_cache = None

    for cache_idx, cache_path in enumerate(boids_caches):
        with h5py.File(cache_path, "r") as cache_file:
            this_n_boids = int(cache_file.attrs["n_boids"])
            cache_unique = int(cache_file.attrs["unique_reps"])
            cache_repeats = int(cache_file.attrs["repeats"])
            if cache_repeats < 1:
                raise ValueError(
                    f"Cache '{cache_path}' must contain at least one trajectory per center."
                )
            if n_boids_cache is None:
                n_boids_cache = this_n_boids
            elif this_n_boids != n_boids_cache:
                raise ValueError(
                    f"Cache '{cache_path}' has {this_n_boids} boids; "
                    f"expected {n_boids_cache}."
                )

            if "traj_lengths" in cache_file:
                boundaries = np.concatenate(
                    [[0], np.cumsum(cache_file["traj_lengths"][:], dtype=np.int64)]
                )
            else:
                total_samples = cache_file["x"].shape[0]
                samples_per_trajectory = total_samples // (cache_unique * cache_repeats)
                boundaries = (
                    np.arange(cache_unique * cache_repeats + 1, dtype=np.int64)
                    * samples_per_trajectory
                )

            centers = cache_file["centers"][:]
            if len(centers) != cache_unique:
                raise ValueError(
                    f"Cache '{cache_path}' center count does not match unique_reps."
                )

        boundaries_by_cache.append(boundaries)
        cache_repeats_by_cache.append(cache_repeats)
        center_arrays.append(centers)
        center_cache_ids.extend([cache_idx] * cache_unique)
        center_local_indices.extend(range(cache_unique))

        if args.tr_set_repeats > cache_repeats:
            print(
                f">>> Training-time reinforcement for '{cache_path}': requesting "
                f"{args.tr_set_repeats} exposures from {cache_repeats} cached "
                "trajectory/trajectories per center; cached trajectories will be cycled."
            )

    centers_all = np.concatenate(center_arrays, axis=0)
    center_cache_ids = np.asarray(center_cache_ids, dtype=np.int64)
    center_local_indices = np.asarray(center_local_indices, dtype=np.int64)
    cache_unique_total = len(centers_all)

    # Filter by octants if requested
    if args.train_octants is not None:
        from boids.generate_boids_cache_3d import OCTANTS
        def _center_octant(c):
            for oi, (xmn, xmx, ymn, ymx, zmn, zmx, _) in enumerate(OCTANTS):
                if xmn <= c[0] < xmx and ymn <= c[1] < ymx and zmn <= c[2] < zmx:
                    return oi
            return -1
        center_buckets = {o: [] for o in args.train_octants}
        for ci in range(cache_unique_total):
            oi = _center_octant(centers_all[ci])
            if oi in center_buckets:
                center_buckets[oi].append(ci)

        if args.octant_unique_counts is not None:
            requested_counts = dict(zip(args.train_octants, args.octant_unique_counts))
        else:
            n_per_o = effective_unique_reps // len(args.train_octants)
            remainder = effective_unique_reps % len(args.train_octants)
            requested_counts = {
                o: n_per_o + (1 if i < remainder else 0)
                for i, o in enumerate(args.train_octants)
            }
        selected_center_indices = []
        selected_counts = {}
        for i, o in enumerate(args.train_octants):
            bucket = np.array(center_buckets[o], dtype=np.int64)
            np.random.shuffle(bucket)
            take = requested_counts[o]
            if take > len(bucket):
                raise ValueError(
                    f"Requested {take} unique centers from octant {o}, "
                    f"but cache only has {len(bucket)}."
                )
            picked = bucket[:take]
            selected_center_indices.extend(picked.tolist())
            selected_counts[o] = len(picked)

        train_center_indices = np.array(selected_center_indices, dtype=np.int64)
        np.random.shuffle(train_center_indices)
        print(f">>> Balanced unique center selection by octant: {selected_counts}")

        octant_buckets = {o: [] for o in args.train_octants}
        for ci in train_center_indices:
            oi = _center_octant(centers_all[ci])
            cache_idx = int(center_cache_ids[ci])
            local_center_idx = int(center_local_indices[ci])
            cache_repeats = cache_repeats_by_cache[cache_idx]
            for exposure_idx in range(args.tr_set_repeats):
                cached_repeat_idx = exposure_idx % cache_repeats
                octant_buckets[oi].append(
                    (cache_idx, local_center_idx * cache_repeats + cached_repeat_idx)
                )
    else:
        train_center_indices = np.arange(min(cache_unique_total, effective_unique_reps))
        octant_buckets = None

    boids_tr = Boids3D(n_boids=args.n_boids)
    boids_tr.rand_configs = [np.array(centers_all[i], dtype=np.float32) for i in train_center_indices]

    all_trajectory_refs = []
    for ci in train_center_indices:
        cache_idx = int(center_cache_ids[ci])
        local_center_idx = int(center_local_indices[ci])
        cache_repeats = cache_repeats_by_cache[cache_idx]
        for exposure_idx in range(args.tr_set_repeats):
            all_trajectory_refs.append(
                (
                    cache_idx,
                    local_center_idx * cache_repeats
                    + (exposure_idx % cache_repeats),
                )
            )
    all_trajectory_refs = np.asarray(all_trajectory_refs, dtype=np.int64)
    np.random.shuffle(all_trajectory_refs)

    model = run_chunked_3d(
        boids_caches, boundaries_by_cache, n_boids_cache, all_trajectory_refs,
        run_tag, args.chunk_size, args.chunk_patience,
        octant_buckets=octant_buckets
    )
    best_after50_cb = None
else:
    print(f"\n>>> Generating 3D training dataset...")
    data_tr, boids_tr = make_dataset_3d(
        unique_reps=effective_unique_reps,
        repeat_reps=args.tr_set_repeats,
        save_config=train_init_centers is None and not boids_caches,
        n_boids=args.n_boids,
        n_jobs=1,
        return_boids=True,
        random_init=train_init_centers if train_init_centers is not None else True,
        boids_cache_npz=boids_caches[0] if boids_caches else None,
        timestep_stride=args.timestep_stride,
        perception=args.perception,
        pos_noise=args.expert_pos_noise,
        vel_noise=args.expert_vel_noise,
        waypoint_order_policy=_expert_waypoint_policy_3d(args),
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
if generation_manifest is not None:
    generation_manifest_path = (
        f"saved_history/generation_manifest_3d_{run_tag}.json"
    )
    with open(generation_manifest_path, "w", encoding="utf-8") as f:
        json.dump(generation_manifest, f, indent=2, sort_keys=True)
    print(f"Saved generation manifest to {generation_manifest_path}")

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

if args.skip_evaluation:
    print(">>> Skipping all post-training inference and visualization.")
    sys.exit(0)

####################################################################################
# Evaluation
####################################################################################
eval_octants = args.train_octants if args.train_octants is not None else list(range(8))
eval_output_dir = args.eval_output_dir or f"inference_3d_{run_tag}"
eval_cmd = [
    sys.executable,
    "-m",
    "boids.run_inference_3d",
    "--run_tag",
    run_tag,
    "--octants",
    *[str(octant) for octant in eval_octants],
    "--centers_per_octant",
    str(args.eval_centers_per_octant),
    "--n_boids",
    str(args.n_boids),
    "--max_steps",
    str(args.eval_max_steps),
    "--success_threshold",
    str(args.eval_success_threshold),
    "--max_success_r",
    str(args.eval_max_success_r),
    "--output_dir",
    eval_output_dir,
    "--save_multi",
    "--save_individual",
]
if args.eval_seed is not None:
    eval_cmd.extend(["--seed", str(args.eval_seed)])

print(
    f">>> Running post-training inference on {args.eval_centers_per_octant} "
    f"fresh unseen centers per octant {eval_octants}."
)
print(f">>> Inference outputs: {eval_output_dir}")
subprocess.run(eval_cmd, check=True)
sys.exit(0)
