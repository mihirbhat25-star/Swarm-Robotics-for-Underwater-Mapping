"""Train the 3D GNCA to imitate the 3D Boids GCA.

This remains the experiment entry point. Computational data movement is
selected with ``--backend local`` or ``--backend cloud`` and implemented in
the :mod:`runtime` package; model and experiment semantics stay here.
"""
import gc
import os

# Configure native libraries before TensorFlow/ABSL are imported.  These
# settings affect only log verbosity, not numerical execution.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import sys
import json
import subprocess
import tempfile
import atexit
import shutil
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from spektral.data import DisjointLoader
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from models.gnn_ca_simple_boids_3d import GNNCASimpleBoids3D
from modules.boids_3d import (
    BOIDS_GOAL_POSITIONS_3D,
    FIXED_ORDER_POLICY,
    NEAREST_CCW_POLICY,
    Boids3D,
    make_dataset_3d,
)
from runtime import CLOUD_BACKEND, LOCAL_BACKEND
from runtime.cli_3d import build_parser, validate_args
from runtime.cloud_3d import (
    dataset_from_trajectories as cloud_dataset_from_trajectories,
    fit_distributed as fit_cloud_distributed,
    generate_compact_chunk as generate_cloud_chunk,
)
from runtime.chunking import proportional_chunk_allocations
from runtime.local_3d import (
    dataset_from_shards as local_dataset_from_shards,
    load_validation_dataset,
    save_validation_dataset,
    write_batch_shards as write_local_batch_shards,
)
import h5py
import psutil

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
    if args.run_tag_unique is not None:
        print(f"Run-tag unique count: {args.run_tag_unique} (naming only)")
    print(f"Validation set:       {args.va_set_size} unique")
    print(f"Boids:                {args.n_boids}")
    print(f"Noise config:         {noise_config or ''}")
    print(f"Learning rate:        {args.lr}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Training backend:     {args.backend}")
    if args.backend == CLOUD_BACKEND:
        print(f"Visible GPUs:         {len(physical_devices)}")
    print(
        f"Execution:            "
        f"{'eager (debug)' if args.eager_training else 'compiled graph'}; "
        f"steps_per_execution={args.steps_per_execution}"
    )
    print(
        f"Epochs:               {args.epochs} "
        f"(early stop patience: {args.es_patience}, "
        f"min_delta: {args.early_stopping_min_delta:g})"
    )
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
    ram_chunk_mix_file=None,
    ram_centers_output_file=None,
):
    """Build the isolated worker command shared by cached and generated chunks."""
    cmd = [
        sys.executable,
        "-m",
        "boids.run_boids_3d",
        "--_chunk_worker",
        "--_chunk_idx", str(chunk_idx),
        "--_checkpoint_path", checkpoint_path,
        "--_n_boids_cache", str(n_boids_cache),
        "--_training_state_dir", training_state_dir,
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--es_patience", str(chunk_patience),
        "--early_stopping_min_delta", str(args.early_stopping_min_delta),
        "--lr_patience", str(args.lr_patience),
        "--lr_red_factor", str(args.lr_red_factor),
        "--min_lr", str(args.min_lr),
        "--n_boids", str(args.n_boids),
        "--tr_set_unique", str(args.tr_set_unique),
        "--tr_set_repeats", str(args.tr_set_repeats),
        "--loss_type", args.loss_type,
        "--critical_distance", str(args.critical_distance),
        "--distance_weight", str(args.distance_weight),
        "--timestep_stride", str(args.timestep_stride),
        "--near_goal_radius", str(args.near_goal_radius),
        "--perception", str(args.perception),
        "--shuffle_buffer_size", str(args.shuffle_buffer_size),
        "--packed_shard_batches", str(args.packed_shard_batches),
        "--steps_per_execution", str(args.steps_per_execution),
        "--backend", args.backend,
        "--expert_goal_order", args.expert_goal_order,
        "--goal_exclusion_size", str(args.goal_exclusion_size),
        "--expert_pos_noise", str(args.expert_pos_noise),
        "--expert_vel_noise", str(args.expert_vel_noise),
    ]
    if ram_chunk_mix_file is not None:
        cmd.extend([
            "--_ram_chunk_mix_file", ram_chunk_mix_file,
            "--_ram_centers_output_file", ram_centers_output_file,
            "--generation_workers", str(args.generation_workers),
            "--generation_seed", str(args.generation_seed),
            "--va_set_size", str(args.va_set_size),
            "--train_octants", *[str(o) for o in args.train_octants],
        ])
    else:
        cmd.extend([
            "--_chunk_indices_file", chunk_file,
            "--_cache_paths_file", cache_paths_file,
            "--_boundaries_file", boundaries_file,
            "--_val_npz_file", val_npz_file,
        ])
    if args.eager_training:
        cmd.append("--eager_training")
    if args.chunk_epochs is not None:
        cmd.extend(["--chunk_epochs", str(args.chunk_epochs)])
    if args.init_weights:
        cmd.extend(["--init_weights", args.init_weights])
    return cmd


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
        requested_mix = {o: len(active[o]) for o in active}
        allocations = proportional_chunk_allocations(requested_mix, chunk_size)
        for allocation in allocations:
            current = []
            for o, take in allocation.items():
                start = pointers[o]
                current.extend(active[o][start:start + take].tolist())
                pointers[o] += take
            np.random.shuffle(current)
            chunks.append(np.array(current, dtype=np.int64))
            chunk_mixes.append(allocation)
        n_chunks = len(chunks)
        print(
            f">>> Chunked training: {n_chunks} proportional balanced chunks; "
            f"requested octant totals {requested_mix}"
        )
    else:
        chunks = [
            all_trajectory_refs[start:start + chunk_size]
            for start in range(0, len(all_trajectory_refs), chunk_size)
        ]
        n_chunks = len(chunks)
        chunk_mixes = None
        print(
            f">>> Chunked training: {n_chunks} chunks of at most "
            f"{chunk_size} trajectories each"
        )

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
            save_validation_dataset(data_va_chunk, val_npz_file, n_boids_cache)
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


def run_cloud_chunked_3d(
    run_tag, requested_counts, chunk_size, chunk_patience
):
    """Run generated chunks entirely in each worker's host RAM."""
    checkpoint_path = f"saved_models/best_weights_3d_{run_tag}"
    os.makedirs("saved_models", exist_ok=True)
    allocations = proportional_chunk_allocations(
        requested_counts, chunk_size
    )
    print(
        f">>> Cloud RAM chunked training: {len(allocations)} chunks; "
        f"requested octant totals {requested_counts}",
        flush=True,
    )

    all_centers = []
    all_center_octants = []
    manifest_tasks = []
    with tempfile.TemporaryDirectory(prefix="gnca_3d_ram_control_") as tmpdir:
        training_state_dir = os.path.join(tmpdir, "training_state")
        for chunk_idx, chunk_mix in enumerate(allocations):
            print(f"\n{'='*60}")
            print(
                f"  Chunk {chunk_idx + 1}/{len(allocations)} | "
                f"{sum(chunk_mix.values())} RAM-generated trajectories"
            )
            print(f"{'='*60}")
            print(f">>> Chunk octant mix: {chunk_mix}")

            mix_file = os.path.join(tmpdir, f"mix_{chunk_idx:04d}.json")
            centers_file = os.path.join(
                tmpdir, f"centers_{chunk_idx:04d}.npz"
            )
            with open(mix_file, "w", encoding="utf-8") as file_handle:
                json.dump(chunk_mix, file_handle)

            cmd = _build_chunk_worker_command_3d(
                chunk_idx,
                None,
                None,
                checkpoint_path,
                args.n_boids,
                None,
                None,
                training_state_dir,
                chunk_patience,
                ram_chunk_mix_file=mix_file,
                ram_centers_output_file=centers_file,
            )
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Cloud RAM chunk worker {chunk_idx + 1}/"
                    f"{len(allocations)} failed with exit code "
                    f"{result.returncode}"
                )
            with np.load(centers_file) as center_data:
                all_centers.append(center_data["centers"].copy())
                all_center_octants.append(
                    center_data["center_octants"].copy()
                )
            for octant, count in chunk_mix.items():
                manifest_tasks.append({
                    "chunk": chunk_idx,
                    "octant": int(octant),
                    "count": int(count),
                })
            gc.collect()
            print(
                f">>> Cloud RAM worker exited; all trajectory and packed-batch "
                f"memory for chunk {chunk_idx + 1} was released.",
                flush=True,
            )

    final_model = _build_model_3d(args.lr)
    _build_model_variables_3d(final_model, args.n_boids)
    final_model.load_weights(checkpoint_path).expect_partial()
    final_model.save_weights(checkpoint_path)
    print(f"Saved final weights to {checkpoint_path}")
    manifest = {
        "mode": "cloud_ram",
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
            EarlyStopping(
                patience=args.es_patience,
                min_delta=args.early_stopping_min_delta,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(patience=args.lr_patience, factor=args.lr_red_factor,
                              min_lr=args.min_lr, min_delta=1e-8, verbose=1),
            best_after50_cb,
        ],
    )

    return history, model, best_after50_cb


####################################################################################
# Training entry point
####################################################################################
args = validate_args(build_parser().parse_args(), len(physical_devices))


if args._chunk_worker:
    UPWEIGHT_NEAR_GOAL = (args.loss_type == "newl")
    setup_t0 = time.time()
    is_cloud = args.backend == CLOUD_BACKEND
    replica_count = len(physical_devices) if is_cloud else 1
    per_replica_batch_size = args.batch_size // replica_count
    worker_tmpdir = None
    loader_va = None
    val_data = None
    val_steps = 0

    if is_cloud:
        if args._ram_chunk_mix_file is None:
            raise RuntimeError("Cloud RAM worker requires a chunk-mix file.")
        with open(args._ram_chunk_mix_file, encoding="utf-8") as file_handle:
            chunk_mix = {
                int(octant): int(count)
                for octant, count in json.load(file_handle).items()
            }
        (
            compact_trajectories,
            chunk_centers,
            chunk_center_octants,
            _,
        ) = generate_cloud_chunk(
            args, args._chunk_idx, chunk_mix, _expert_waypoint_policy_3d(args)
        )
        if args._ram_centers_output_file is None:
            raise RuntimeError("Cloud RAM worker requires a centers output file.")
        np.savez(
            args._ram_centers_output_file,
            centers=chunk_centers,
            center_octants=chunk_center_octants,
        )
        train_data, train_steps = cloud_dataset_from_trajectories(
            compact_trajectories,
            per_replica_batch_size,
            args._n_boids_cache,
            args.timestep_stride,
            args.near_goal_radius,
            seed=args.generation_seed + args._chunk_idx,
            shuffle_batches=True,
        )
        del compact_trajectories
        gc.collect()

        use_val = args.chunk_epochs is None
        if use_val:
            n_octants = len(args.train_octants)
            base = args.va_set_size // n_octants
            remainder = args.va_set_size % n_octants
            validation_mix = {
                int(octant): base + (1 if idx < remainder else 0)
                for idx, octant in enumerate(args.train_octants)
            }
            validation_mix = {
                octant: count
                for octant, count in validation_mix.items()
                if count > 0
            }
            validation_seed_chunk = args._chunk_idx + 10_000
            validation_trajectories, _, _, _ = generate_cloud_chunk(
                args,
                validation_seed_chunk,
                validation_mix,
                _expert_waypoint_policy_3d(args),
            )
            val_data, val_steps = cloud_dataset_from_trajectories(
                validation_trajectories,
                per_replica_batch_size,
                args._n_boids_cache,
                args.timestep_stride,
                args.near_goal_radius,
                seed=(
                    args.generation_seed
                    + 1_000_000_000
                    + args._chunk_idx
                ),
                shuffle_batches=False,
            )
            del validation_trajectories
            gc.collect()
        print(
            f">>> Cloud worker setup complete in "
            f"{time.time() - setup_t0:.1f}s; training begins now.",
            flush=True,
        )
    else:
        with open(args._chunk_indices_file, encoding="utf-8") as file_handle:
            chunk_refs = np.asarray(
                json.load(file_handle), dtype=np.int64
            ).reshape(-1, 2)
        with open(args._cache_paths_file, encoding="utf-8") as file_handle:
            worker_cache_paths = json.load(file_handle)
        with np.load(args._boundaries_file) as boundary_data:
            boundaries_by_cache = [
                boundary_data[f"cache_{cache_idx}"]
                for cache_idx in range(len(worker_cache_paths))
            ]

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
        print(
            ">>> Local worker: writing compact NPZ train shards...",
            flush=True,
        )
        train_shard_paths, train_steps = write_local_batch_shards(
            worker_cache_paths,
            chunk_refs,
            boundaries_by_cache,
            args._n_boids_cache,
            per_replica_batch_size,
            os.path.join(worker_tmpdir, "train_batches"),
            timestep_stride=args.timestep_stride,
            near_goal_radius=args.near_goal_radius,
            perception=args.perception,
            shuffle_buffer_size=args.shuffle_buffer_size,
            batches_per_shard=args.packed_shard_batches,
        )
        print(
            f">>> Worker setup: wrote {train_steps} train batches in "
            f"{len(train_shard_paths)} compact shards in "
            f"{time.time() - setup_t0:.1f}s",
            flush=True,
        )
        train_data = local_dataset_from_shards(train_shard_paths).map(
            add_graph_ids_to_targets_3d,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

        use_val = (
            args.chunk_epochs is None and args._val_npz_file is not None
        )
        if use_val:
            data_val = load_validation_dataset(args._val_npz_file)
            loader_va = DisjointLoader(
                data_val,
                node_level=True,
                batch_size=per_replica_batch_size,
            )
            val_steps = loader_va.steps_per_epoch

    if args._training_state_dir is None:
        raise RuntimeError("Chunk worker requires --_training_state_dir")

    strategy = None
    if is_cloud:
        strategy = tf.distribute.MirroredStrategy(
            devices=[device.name for device in tf.config.list_logical_devices("GPU")]
        )
        print(
            f">>> Cloud strategy initialized with "
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
            EarlyStopping(
                monitor='val_loss',
                patience=args.es_patience,
                min_delta=args.early_stopping_min_delta,
                restore_best_weights=False,
                verbose=1,
            ),
        ]
        if not is_cloud:
            val_data = loader_va.load().map(add_graph_ids_to_targets_3d)
        val_kwargs = {
            'validation_data': val_data,
            'validation_steps': val_steps,
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
            EarlyStopping(
                monitor='loss',
                patience=args.es_patience,
                min_delta=args.early_stopping_min_delta,
                restore_best_weights=False,
                verbose=1,
            ),
        ]
        val_data = None
        val_kwargs = {}

    if is_cloud:
        fit_cloud_distributed(
            model,
            strategy,
            train_data,
            train_steps,
            n_epochs,
            cbs,
            global_node_count=args.batch_size * args._n_boids_cache,
            steps_per_execution=args.steps_per_execution,
            loss_fn=custom_weighted_mse_3d,
            val_data=val_data,
            val_local_steps=(val_steps if use_val else 0),
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
    if worker_tmpdir is not None:
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
    args.run_tag_unique or effective_unique_reps,
    args.tr_set_repeats,
    UPWEIGHT_NEAR_GOAL,
    args.critical_distance,
    args.distance_weight,
    noise_config
)

# Print parameters
print_run_parameters_3d(args, UPWEIGHT_NEAR_GOAL, args.critical_distance, args.distance_weight, noise_config, run_tag)

use_cloud_chunked = args.backend == CLOUD_BACKEND
use_cached_chunked = (
    args.backend == LOCAL_BACKEND and args.chunk_size > 0 and bool(boids_caches)
)
use_chunked = use_cached_chunked or use_cloud_chunked
data_va = None
generation_manifest = None

if not use_chunked and len(boids_caches) > 1:
    raise ValueError("Multiple --boids_cache files currently require chunked training.")

if not use_chunked:
    # Validation set
    print(f"\n>>> Generating validation set ({args.va_set_size} centers)...")
    data_va = make_validation_dataset_3d(args)

if use_cloud_chunked:
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
        f"\n>>> Generating expert data one chunk at a time "
        f"in worker RAM: {requested_counts}"
    )
    (
        model,
        generated_centers,
        generated_center_octants,
        generation_manifest,
    ) = run_cloud_chunked_3d(
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
    "evaluation.run_inference_3d",
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
