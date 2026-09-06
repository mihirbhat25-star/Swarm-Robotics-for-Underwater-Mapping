"""Cacheless, goal-conditioned 3D training on the compiled cloud backend."""

import gc
import json
import os
import sys
import tempfile
import time

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models.gnn_ca_goal_conditioned_boids_3d import (
    GoalConditionedGNNCABoids3D,
)
from runtime.chunking import proportional_chunk_allocations
from runtime.cloud_3d import fit_distributed
from runtime.cloud_bitpacked_3d import dataset_from_bitpacked_trajectories
from runtime.subprocess_utils import run_with_filtered_stderr
from runtime.timing import (
    TIMEOUT_EXIT_CODE,
    append_timing_event,
    deadline_reached,
    write_timing_report,
)


PHYSICAL_FEATURES = 6
CURRENT_GOAL_SLICE = slice(12, 15)
PREVIOUS_GOAL_SLICE = slice(15, 18)
PREVIOUS_GOAL_MAX_DISTANCE_INDEX = 18
TRANSITION_TARGET_FEATURES = 19


def online_goal_run_tag_3d(args):
    """Build an unambiguous checkpoint tag for the online 3D task."""
    loss_tag = (
        f"newl_cd_{args.critical_distance:g}_dw_{args.distance_weight:g}"
        if args.loss_type == "newl"
        else "oldl"
    )
    suffix = f"_{args.noise_tag}" if args.noise_tag else ""
    return f"online3d2goal_{args.tr_set_unique}x1_{loss_tag}{suffix}"


def make_online_goal_loss_3d(args):
    """Return transition-aware state imitation loss for random 3D goals."""
    upweight = args.loss_type == "newl"
    critical_distance = float(args.critical_distance)
    distance_weight = float(args.distance_weight)

    def loss(y_true, y_pred):
        graph_ids = tf.cast(
            y_true[..., TRANSITION_TARGET_FEATURES], tf.int32
        )
        current_state = y_true[..., :PHYSICAL_FEATURES]
        next_state = y_true[..., PHYSICAL_FEATURES : 2 * PHYSICAL_FEATURES]
        node_mse = tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)
        if not upweight:
            return node_mse

        graph_count = tf.reduce_max(graph_ids) + 1
        centroid = tf.math.unsorted_segment_mean(
            current_state[..., :3], graph_ids, graph_count
        )
        active_goal = tf.math.unsorted_segment_mean(
            y_true[..., CURRENT_GOAL_SLICE], graph_ids, graph_count
        )
        previous_goal = tf.math.unsorted_segment_mean(
            y_true[..., PREVIOUS_GOAL_SLICE], graph_ids, graph_count
        )
        max_previous_distance = tf.math.unsorted_segment_max(
            y_true[..., PREVIOUS_GOAL_MAX_DISTANCE_INDEX],
            graph_ids,
            graph_count,
        )
        active_distance = tf.norm(centroid - active_goal, axis=-1)
        previous_distance = tf.norm(centroid - previous_goal, axis=-1)
        near_active = active_distance < critical_distance
        near_eligible_previous = (
            (max_previous_distance >= 0.0)
            & (max_previous_distance < critical_distance)
            & (previous_distance < critical_distance)
        )
        graph_weight = tf.where(
            near_active | near_eligible_previous,
            tf.cast(distance_weight, node_mse.dtype),
            tf.ones_like(active_distance, dtype=node_mse.dtype),
        )
        return node_mse * tf.gather(graph_weight, graph_ids)

    return loss


def build_online_goal_model_3d(args, learning_rate):
    """Construct the 3D analogue of the established conditioned 2D GNCA."""
    model = GoalConditionedGNNCABoids3D(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model


def build_online_goal_variables_3d(model, n_boids):
    x = tf.zeros((n_boids, 9), dtype=tf.float32)
    adjacency = tf.SparseTensor(
        indices=tf.zeros((0, 2), dtype=tf.int64),
        values=tf.zeros((0,), dtype=tf.float32),
        dense_shape=(n_boids, n_boids),
    )
    model([x, tf.sparse.reorder(adjacency), tf.constant(0)], training=False)


class _BestCompleteState3D(tf.keras.callbacks.Callback):
    """Save model and Adam state together whenever validation improves."""

    def __init__(self, manager, min_delta):
        super().__init__()
        self.manager = manager
        self.min_delta = float(min_delta)
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        value = float((logs or {}).get("val_loss", np.inf))
        if value < self.best - self.min_delta:
            self.best = value
            path = self.manager.save()
            print(
                f">>> Saved best complete online-3D state at epoch "
                f"{epoch + 1}: val_loss={value:.6g} ({path})",
                flush=True,
            )


def _validation_mix(octants, count):
    base, remainder = divmod(int(count), len(octants))
    return {
        int(octant): base + (1 if index < remainder else 0)
        for index, octant in enumerate(octants)
        if base + (1 if index < remainder else 0) > 0
    }


def _worker(
    args,
    chunk_index,
    chunk_mix,
    state_dir,
    result_path,
    run_tag,
):
    """Train one isolated chunk; process exit releases all generated data."""
    from runtime.cloud_compiled_3d import generate_compiled_chunk

    np.random.seed(args.generation_seed + chunk_index)
    tf.random.set_seed(args.generation_seed + chunk_index)
    wall_deadline = float(getattr(args, "_online_wall_deadline", 0.0))
    timing_events = str(getattr(args, "_online_timing_events", ""))

    phase_started = time.perf_counter()
    trajectories, centers, center_octants, generation_records = (
        generate_compiled_chunk(
            args,
            chunk_index,
            chunk_mix,
            task="online_goals",
        )
    )
    append_timing_event(
        timing_events,
        "expert_trajectory_generation",
        time.perf_counter() - phase_started,
        chunk=chunk_index,
        split="train",
        trajectories=sum(chunk_mix.values()),
    )
    if deadline_reached(wall_deadline):
        raise SystemExit(TIMEOUT_EXIT_CODE)

    validation_mix = _validation_mix(args.train_octants, args.va_set_size)
    phase_started = time.perf_counter()
    validation_trajectories, _, _, validation_records = generate_compiled_chunk(
        args,
        chunk_index + 10_000,
        validation_mix,
        task="online_goals",
        validation=True,
    )
    append_timing_event(
        timing_events,
        "expert_trajectory_generation",
        time.perf_counter() - phase_started,
        chunk=chunk_index,
        split="validation",
        trajectories=sum(validation_mix.values()),
    )
    if deadline_reached(wall_deadline):
        raise SystemExit(TIMEOUT_EXIT_CODE)

    strategy = tf.distribute.MirroredStrategy()
    replicas = strategy.num_replicas_in_sync
    if args.batch_size % replicas:
        raise ValueError(
            f"--batch_size {args.batch_size} must be divisible by "
            f"{replicas} GPUs."
        )
    per_replica_batch = args.batch_size // replicas

    phase_started = time.perf_counter()
    train_data, train_steps = dataset_from_bitpacked_trajectories(
        trajectories,
        per_replica_batch,
        args.n_boids,
        args.timestep_stride,
        args.near_goal_radius,
        args.generation_seed + chunk_index,
        shuffle_batches=True,
        goal_conditioned=True,
    )
    append_timing_event(
        timing_events,
        "graph_and_batch_packing",
        time.perf_counter() - phase_started,
        chunk=chunk_index,
        split="train",
        batches=train_steps,
    )
    phase_started = time.perf_counter()
    validation_data, validation_steps = dataset_from_bitpacked_trajectories(
        validation_trajectories,
        per_replica_batch,
        args.n_boids,
        args.timestep_stride,
        args.near_goal_radius,
        args.generation_seed + 1_000_000_000 + chunk_index,
        shuffle_batches=False,
        goal_conditioned=True,
    )
    append_timing_event(
        timing_events,
        "graph_and_batch_packing",
        time.perf_counter() - phase_started,
        chunk=chunk_index,
        split="validation",
        batches=validation_steps,
    )
    del trajectories, validation_trajectories
    gc.collect()

    with strategy.scope():
        model = build_online_goal_model_3d(args, args.lr)
        build_online_goal_variables_3d(model, args.n_boids)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        manager = tf.train.CheckpointManager(
            checkpoint, state_dir, max_to_keep=1
        )
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(
                f">>> Restored model + Adam state: "
                f"{manager.latest_checkpoint}",
                flush=True,
            )
        elif args.init_weights:
            restore_status = model.load_weights(args.init_weights)
            restore_status.assert_existing_objects_matched()
            print(f">>> Initialized online-3D model from: {args.init_weights}")

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=args.lr_patience,
            factor=args.lr_red_factor,
            min_lr=args.min_lr,
            min_delta=args.early_stopping_min_delta,
            verbose=1,
        ),
        _BestCompleteState3D(manager, args.early_stopping_min_delta),
        EarlyStopping(
            monitor="val_loss",
            patience=args.chunk_patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=False,
            verbose=1,
        ),
    ]
    training_started = time.perf_counter()
    history = fit_distributed(
        model,
        strategy,
        train_data,
        train_steps,
        args.epochs,
        callbacks,
        args.batch_size * args.n_boids,
        args.steps_per_execution,
        make_online_goal_loss_3d(args),
        validation_data,
        validation_steps,
        wall_deadline=wall_deadline,
    )
    append_timing_event(
        timing_events,
        "training",
        time.perf_counter() - training_started,
        chunk=chunk_index,
        epochs=history["epochs_completed"],
        timed_out=history["timed_out"],
    )
    if not manager.latest_checkpoint:
        raise RuntimeError(
            "Online 3D worker did not save a validation checkpoint."
        )
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    os.makedirs("saved_models", exist_ok=True)
    weights_path = f"saved_models/best_weights_{run_tag}"
    model.save_weights(weights_path)
    saved_model_path = f"saved_models/gnca_model_{run_tag}"
    model.save(saved_model_path, save_format="tf")
    result = {
        "chunk": chunk_index,
        "chunk_mix": {str(key): value for key, value in chunk_mix.items()},
        "trajectory_count": int(sum(chunk_mix.values())),
        "centers": centers.tolist(),
        "center_octants": center_octants.tolist(),
        "train_generation": generation_records,
        "validation_generation": validation_records,
        "history": history,
        "weights_path": weights_path,
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)

    del train_data, validation_data
    tf.keras.backend.clear_session()
    gc.collect()
    if history["timed_out"]:
        raise SystemExit(TIMEOUT_EXIT_CODE)


def validate_online_cloud_args_3d(args):
    """Validate the additional invariants of online 3D training."""
    if args.backend != "cloud" or args.cloud_data_mode != "compiled":
        raise ValueError(
            "Online 3D requires --backend cloud --cloud_data_mode compiled."
        )
    if args.boids_cache:
        raise ValueError("Online 3D is cacheless; omit --boids_cache.")
    if args.tr_set_repeats != 1:
        raise ValueError("Online 3D requires --tr_set_repeats 1.")
    if args.goal_waypoints_per_episode != 2:
        raise ValueError("Online 3D requires exactly two waypoints per episode.")
    if not args.skip_evaluation:
        raise ValueError(
            "Online 3D post-training evaluation is not yet routed through the "
            "fixed-waypoint evaluator; pass --skip_evaluation for training."
        )


def train_online_goal_cloud_3d(args):
    """Train online goal navigation over balanced 3D start octants."""
    validate_online_cloud_args_3d(args)
    run_tag = online_goal_run_tag_3d(args)
    if args.octant_unique_counts is not None:
        requested_counts = dict(
            zip(args.train_octants, args.octant_unique_counts)
        )
    else:
        base, remainder = divmod(args.tr_set_unique, len(args.train_octants))
        requested_counts = {
            int(octant): base + (1 if index < remainder else 0)
            for index, octant in enumerate(args.train_octants)
        }
    allocations = proportional_chunk_allocations(
        requested_counts, args.chunk_size
    )

    started = time.time()
    report_path = args.timing_report
    if args.wall_time_limit_hours and not report_path:
        report_path = f"results/timing_{run_tag}.json"
    report_path = os.path.abspath(report_path) if report_path else ""
    events_path = f"{report_path}.events.jsonl" if report_path else ""
    if events_path and os.path.exists(events_path):
        os.remove(events_path)
    wall_deadline = (
        started + args.wall_time_limit_hours * 3600.0
        if args.wall_time_limit_hours
        else 0.0
    )
    args._online_wall_deadline = wall_deadline
    args._online_timing_events = events_path

    print(
        f">>> Online-goal 3D cloud training: {args.tr_set_unique} fresh "
        f"episodes, exactly two random goals/episode, {len(allocations)} "
        f"isolated chunks, start totals={requested_counts}.",
        flush=True,
    )
    print(
        ">>> Data mode: compiled Numba expert + lazy bit-packed adjacency; "
        "no training cache or temporary graph shards.",
        flush=True,
    )

    all_results = []
    status = "completed"
    with tempfile.TemporaryDirectory(prefix="gnca_3d_online_compiled_") as tmpdir:
        state_dir = os.path.join(tmpdir, "training_state")
        os.makedirs(state_dir)
        args_file = os.path.join(tmpdir, "args.pkl")
        joblib.dump(args, args_file)
        for chunk_index, chunk_mix in enumerate(allocations):
            if deadline_reached(wall_deadline):
                status = "timed_out"
                break
            result_path = os.path.join(
                tmpdir, f"chunk_{chunk_index:04d}.json"
            )
            mix_path = os.path.join(tmpdir, f"mix_{chunk_index:04d}.json")
            with open(mix_path, "w", encoding="utf-8") as handle:
                json.dump(chunk_mix, handle)
            print(
                f"\n{'=' * 68}\n  Online 3D chunk "
                f"{chunk_index + 1}/{len(allocations)} | "
                f"{sum(chunk_mix.values())} episodes | mix={chunk_mix}\n"
                f"{'=' * 68}",
                flush=True,
            )
            returncode = run_with_filtered_stderr(
                [
                    sys.executable,
                    "-m",
                    "runtime.online_goal_3d_worker",
                    "--args_file",
                    args_file,
                    "--chunk_index",
                    str(chunk_index),
                    "--chunk_mix_file",
                    mix_path,
                    "--state_dir",
                    state_dir,
                    "--result_path",
                    result_path,
                    "--run_tag",
                    run_tag,
                ],
                ignored_fragments=(
                    ("ptx85", "not a recognized feature"),
                    ("successful NUMA node read", "negative value"),
                    ("Unable to register cuFFT factory",),
                    ("Unable to register cuDNN factory",),
                    ("Unable to register cuBLAS factory",),
                ),
            )
            if returncode == TIMEOUT_EXIT_CODE:
                status = "timed_out"
                break
            if returncode != 0:
                status = "failed"
                if report_path:
                    write_timing_report(
                        events_path,
                        report_path,
                        time.time() - started,
                        status,
                    )
                raise RuntimeError(
                    f"Online 3D chunk {chunk_index + 1}/"
                    f"{len(allocations)} failed with exit code {returncode}."
                )
            with open(result_path, encoding="utf-8") as handle:
                all_results.append(json.load(handle))

    if report_path:
        write_timing_report(
            events_path,
            report_path,
            time.time() - started,
            status,
        )
    if status == "timed_out":
        raise SystemExit(TIMEOUT_EXIT_CODE)

    os.makedirs("saved_history", exist_ok=True)
    os.makedirs("saved_boids_tr", exist_ok=True)
    joblib.dump(
        {"task": "online_goals_3d", "chunks": all_results},
        f"saved_history/history_{run_tag}.pkl",
    )
    manifest = {
        "task": "online_goals_3d",
        "run_tag": run_tag,
        "training_episodes": args.tr_set_unique,
        "waypoints_per_episode": args.goal_waypoints_per_episode,
        "cacheless": True,
        "backend": "cloud",
        "cloud_data_mode": "compiled",
        "goal_bounds": list(args.goal_bounds),
        "start_bounds": list(args.start_bounds),
        "goal_min_distance": args.goal_min_distance,
        "goal_arrival_radius": args.goal_arrival_radius,
        "perception": args.perception,
        "start_octant_counts": requested_counts,
        "chunks": all_results,
    }
    with open(
        f"saved_boids_tr/online_goal_manifest_{run_tag}.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nSaved online-3D weights: saved_models/best_weights_{run_tag}")
    print(f"Saved online-3D model: saved_models/gnca_model_{run_tag}")
    print("No training-data cache or temporary graph dataset was created.")
    return run_tag
