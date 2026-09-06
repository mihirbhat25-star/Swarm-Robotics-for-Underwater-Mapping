"""Cloud backend: in-memory expert data and synchronous multi-GPU training."""

import os
import time

import joblib
import numpy as np
import tensorflow as tf

from runtime.cloud_data_3d import simulate_compact_task
from runtime.timing import deadline_reached
from modules.boids_3d import BOIDS_GOAL_POSITIONS_3D


def generate_compact_chunk(args, chunk_idx, octant_counts, waypoint_policy):
    """Generate one expert chunk across CPU workers and retain it in RAM."""
    if args.cloud_data_mode == "compiled":
        from runtime.cloud_compiled_3d import generate_compiled_chunk

        return generate_compiled_chunk(
            args,
            chunk_idx,
            octant_counts,
            task="fixed_waypoints",
            waypoint_policy=waypoint_policy,
        )

    total = sum(octant_counts.values())
    workers = args.generation_workers
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 2)
    workers = max(1, min(workers, total))
    task_configs = []
    task_idx = 0
    for octant, count in sorted(octant_counts.items()):
        # Single-trajectory tasks dynamically balance variable trajectory times.
        for _ in range(count):
            task_configs.append({
                "octant": int(octant),
                "count": 1,
                "seed": int((
                    args.generation_seed
                    + chunk_idx * 1_000_003
                    + task_idx
                ) % (2**32 - 1)),
                "n_boids": int(args.n_boids),
                "perception": float(args.perception),
                "pos_noise": float(args.expert_pos_noise),
                "vel_noise": float(args.expert_vel_noise),
                "waypoint_order_policy": waypoint_policy,
                "goal_exclusion_size": float(args.goal_exclusion_size),
            })
            task_idx += 1

    print(
        f">>> Cloud: generating {total} trajectories across "
        f"{len(task_configs)} tasks with {workers} CPU workers...",
        flush=True,
    )
    generation_t0 = time.time()
    previous_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    try:
        task_results = joblib.Parallel(
            n_jobs=workers,
            backend="loky",
            batch_size=1,
            pre_dispatch=workers,
            max_nbytes=None,
        )(
            joblib.delayed(simulate_compact_task)(config)
            for config in task_configs
        )
    finally:
        if previous_cuda_visibility is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visibility

    trajectories = []
    centers = []
    center_octants = []
    task_records = []
    for result in task_results:
        trajectories.extend(result["trajectories"])
        centers.append(result["centers"])
        center_octants.extend([result["octant"]] * result["count"])
        task_records.append({
            "octant": result["octant"],
            "count": result["count"],
            "seed": result["seed"],
            "seconds": result["seconds"],
        })

    print(
        f">>> Cloud: expert trajectories ready in "
        f"{time.time() - generation_t0:.1f}s; no temporary dataset was written.",
        flush=True,
    )
    return (
        trajectories,
        np.concatenate(centers, axis=0),
        np.asarray(center_octants, dtype=np.int8),
        task_records,
    )


def _selected_sample_refs(trajectories, timestep_stride, near_goal_radius):
    trajectory_ids = []
    timestep_ids = []
    for trajectory_idx, trajectory in enumerate(trajectories):
        n_steps = len(trajectory["x"])
        if timestep_stride <= 1 or near_goal_radius <= 0:
            selected = np.arange(0, n_steps, timestep_stride, dtype=np.int32)
        else:
            mean_pos = trajectory["x"][..., :3].mean(axis=1)
            distance = np.min(
                np.linalg.norm(
                    mean_pos[:, None, :]
                    - BOIDS_GOAL_POSITIONS_3D[None, :, :],
                    axis=-1,
                ),
                axis=1,
            )
            keep = np.zeros(n_steps, dtype=bool)
            keep[::timestep_stride] = True
            keep |= distance < near_goal_radius
            selected = np.flatnonzero(keep).astype(np.int32, copy=False)
        trajectory_ids.append(
            np.full(len(selected), trajectory_idx, dtype=np.int32)
        )
        timestep_ids.append(selected)
    return np.concatenate(trajectory_ids), np.concatenate(timestep_ids)


def _to_model_inputs(x, y, indices, n_boids):
    indices = tf.cast(indices, tf.int64)
    n_nodes = tf.shape(x, out_type=tf.int64)[0]
    adjacency = tf.SparseTensor(
        indices=indices,
        values=tf.ones([tf.shape(indices)[0]], dtype=tf.float32),
        dense_shape=tf.stack([n_nodes, n_nodes]),
    )
    graph_count = n_nodes // tf.cast(n_boids, tf.int64)
    graph_ids = tf.repeat(
        tf.range(graph_count, dtype=tf.int64), tf.cast(n_boids, tf.int64)
    )
    targets = tf.concat([y, tf.cast(graph_ids[:, None], y.dtype)], axis=-1)
    return (
        (
            tf.ensure_shape(x, [None, 6]),
            adjacency,
            tf.ensure_shape(graph_ids, [None]),
        ),
        tf.ensure_shape(targets, [None, 16]),
    )


def dataset_from_trajectories(
    trajectories,
    batch_size,
    n_boids,
    timestep_stride,
    near_goal_radius,
    seed,
    *,
    shuffle_batches=True,
):
    """Pack a chunk once in host RAM and return a native ``tf.data`` stream."""
    pack_t0 = time.time()
    trajectory_ids, timestep_ids = _selected_sample_refs(
        trajectories, timestep_stride, near_goal_radius
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trajectory_ids))
    usable = (len(order) // batch_size) * batch_size
    trajectory_ids = trajectory_ids[order[:usable]]
    timestep_ids = timestep_ids[order[:usable]]
    n_batches = len(trajectory_ids) // batch_size
    if n_batches < 1:
        raise ValueError("Cloud backend produced no complete training batches.")

    n_nodes = batch_size * n_boids
    x_batches = np.empty((n_batches, n_nodes, 6), dtype=np.float32)
    y_batches = np.empty((n_batches, n_nodes, 15), dtype=np.float32)
    x_samples = x_batches.reshape(-1, n_boids, 6)
    y_samples = y_batches.reshape(-1, n_boids, 15)

    grouped_positions = np.argsort(trajectory_ids, kind="stable")
    trajectory_counts = np.bincount(
        trajectory_ids, minlength=len(trajectories)
    )
    trajectory_splits = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(trajectory_counts, dtype=np.int64),
    ))
    for trajectory_idx, trajectory in enumerate(trajectories):
        start = int(trajectory_splits[trajectory_idx])
        stop = int(trajectory_splits[trajectory_idx + 1])
        if start == stop:
            continue
        sample_positions = grouped_positions[start:stop]
        selected_steps = timestep_ids[sample_positions]
        x_samples[sample_positions] = trajectory["x"][selected_steps]
        y_samples[sample_positions] = trajectory["y"][selected_steps]

    edge_batches = []
    edge_lengths = np.zeros(n_batches, dtype=np.int64)
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        stop = start + batch_size
        edge_parts = []
        for graph_idx, sample_idx in enumerate(range(start, stop)):
            trajectory = trajectories[int(trajectory_ids[sample_idx])]
            timestep = int(timestep_ids[sample_idx])
            edge_start = int(trajectory["edge_offsets"][timestep])
            edge_stop = int(trajectory["edge_offsets"][timestep + 1])
            if edge_stop > edge_start:
                edge_parts.append(
                    trajectory["edge_values"][edge_start:edge_stop]
                    + np.int32(graph_idx * n_boids)
                )
        edges = (
            np.concatenate(edge_parts, axis=0)
            if edge_parts
            else np.zeros((0, 2), dtype=np.int32)
        )
        edge_batches.append(edges)
        edge_lengths[batch_idx] = len(edges)

    edge_values = (
        np.concatenate(edge_batches, axis=0)
        if edge_lengths.sum() > 0
        else np.zeros((0, 2), dtype=np.int32)
    )
    edge_splits = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(edge_lengths, dtype=np.int64),
    ))
    with tf.device("/CPU:0"):
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor(x_batches),
            tf.convert_to_tensor(y_batches),
            tf.RaggedTensor.from_row_splits(
                tf.convert_to_tensor(edge_values),
                tf.convert_to_tensor(edge_splits),
                validate=False,
            ),
        ))
    dataset = dataset.map(
        lambda x, y, indices: _to_model_inputs(x, y, indices, n_boids),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    if shuffle_batches and n_batches > 1:
        dataset = dataset.shuffle(
            min(256, n_batches), seed=seed, reshuffle_each_iteration=True
        )
    options = tf.data.Options()
    options.deterministic = False
    dataset = dataset.with_options(options).repeat().prefetch(tf.data.AUTOTUNE)
    resident_gib = (
        x_batches.nbytes + y_batches.nbytes + edge_values.nbytes
    ) / (1024**3)
    print(
        f">>> Cloud: packed {n_batches} batches in "
        f"{time.time() - pack_t0:.1f}s ({resident_gib:.1f} GiB resident).",
        flush=True,
    )
    return dataset, n_batches


def fit_distributed(
    model,
    strategy,
    train_data,
    train_local_steps,
    n_epochs,
    callbacks,
    global_node_count,
    steps_per_execution,
    loss_fn,
    val_data=None,
    val_local_steps=0,
    wall_deadline=0.0,
):
    """Train complete graph batches synchronously on all visible GPUs."""
    replicas = strategy.num_replicas_in_sync
    train_steps = train_local_steps // replicas
    if train_steps < 1:
        raise ValueError(
            f"Need at least {replicas} train batches for {replicas} GPUs."
        )
    dropped_train = train_local_steps - train_steps * replicas
    if dropped_train:
        print(
            f">>> Cloud: dropping {dropped_train} incomplete batch(es) per "
            "epoch to keep replicas synchronized."
        )
    val_steps = val_local_steps // replicas if val_data is not None else 0
    if val_data is not None and val_steps < 1:
        raise ValueError(
            f"Need at least {replicas} validation batches for {replicas} GPUs."
        )

    distributed_train = strategy.distribute_datasets_from_function(
        lambda _context: train_data
    )
    distributed_val = (
        strategy.distribute_datasets_from_function(lambda _context: val_data)
        if val_data is not None
        else None
    )

    def replica_train_step(batch):
        inputs, targets = batch
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            per_node_loss = loss_fn(targets, predictions)
            local_loss_sum = tf.reduce_sum(per_node_loss)
            local_node_count = tf.cast(tf.size(per_node_loss), tf.float32)
            loss = local_loss_sum / tf.cast(global_node_count, tf.float32)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return local_loss_sum, local_node_count

    def replica_validation_step(batch):
        inputs, targets = batch
        per_node_loss = loss_fn(targets, model(inputs, training=False))
        return tf.reduce_sum(per_node_loss), tf.cast(
            tf.size(per_node_loss), tf.float32
        )

    @tf.function(reduce_retracing=True)
    def train_block(iterator, block_steps):
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
    def validation_block(iterator, block_steps):
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
        f">>> Cloud execution: {replicas} replicas, "
        f"{train_steps} global steps/epoch, "
        f"{execution_steps} compiled steps/dispatch.",
        flush=True,
    )

    model.stop_training = False
    callback_list.on_train_begin()
    train_iterator = iter(distributed_train)
    timed_out = False
    epochs_completed = 0
    try:
        for epoch in range(n_epochs):
            if model.stop_training or deadline_reached(wall_deadline):
                timed_out = deadline_reached(wall_deadline)
                break
            epoch_t0 = time.time()
            callback_list.on_epoch_begin(epoch)
            epoch_loss_sum = 0.0
            epoch_node_count = 0.0
            for block_start in range(0, train_steps, execution_steps):
                block_steps = min(execution_steps, train_steps - block_start)
                loss_sum, node_count = train_block(
                    train_iterator, tf.constant(block_steps, dtype=tf.int32)
                )
                epoch_loss_sum += float(loss_sum.numpy())
                epoch_node_count += float(node_count.numpy())
                if deadline_reached(wall_deadline):
                    timed_out = True
                    break

            if timed_out:
                break

            logs = {"loss": epoch_loss_sum / max(epoch_node_count, 1.0)}
            if distributed_val is not None:
                val_iterator = iter(distributed_val)
                val_loss_sum = 0.0
                val_node_count = 0.0
                val_execution_steps = max(1, min(execution_steps, val_steps))
                for block_start in range(0, val_steps, val_execution_steps):
                    block_steps = min(
                        val_execution_steps, val_steps - block_start
                    )
                    loss_sum, node_count = validation_block(
                        val_iterator, tf.constant(block_steps, dtype=tf.int32)
                    )
                    val_loss_sum += float(loss_sum.numpy())
                    val_node_count += float(node_count.numpy())
                    if deadline_reached(wall_deadline):
                        timed_out = True
                        break
                if timed_out:
                    break
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
            print(f"{summary} - lr: {learning_rate:.6g}", flush=True)
            epochs_completed += 1
    finally:
        callback_list.on_train_end()

    return {
        "steps_per_epoch": train_steps,
        "epochs_completed": epochs_completed,
        "timed_out": timed_out,
    }
