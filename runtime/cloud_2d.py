"""RAM-only data generation and distributed training primitives for 2D."""

import os
import time

import joblib
import numpy as np
import tensorflow as tf

from modules.boids import goal_transition_proximity
from runtime.cloud_data_2d import simulate_online_goal_task


def generate_online_goal_chunk(args, chunk_index, trajectory_count, *, validation=False):
    """Generate fresh expert episodes across CPU workers without a cache."""
    workers = int(args.generation_workers)
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 2)
    workers = min(workers, int(trajectory_count))
    seed_offset = 50_000_021 if validation else 0
    x_min, x_max, y_min, y_max = map(float, args.start_bounds)
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    quadrant_bounds = (
        (x_mid, x_max, y_mid, y_max),
        (x_min, x_mid, y_mid, y_max),
        (x_min, x_mid, y_min, y_mid),
        (x_mid, x_max, y_min, y_mid),
    )
    tasks = []
    for task_index in range(int(trajectory_count)):
        start_quadrant = task_index % 4
        tasks.append({
            "seed": int((
                args.seed
                + seed_offset
                + chunk_index * 1_000_003
                + task_index
            ) % (2**32 - 1)),
            "n_boids": args.n_boids,
            "perception": args.perception,
            "pos_noise": args.expert_pos_noise,
            "vel_noise": args.expert_vel_noise,
            "goal_bounds": args.goal_bounds,
            "start_bounds": quadrant_bounds[start_quadrant],
            "start_quadrant": start_quadrant,
            "goal_min_distance": args.goal_min_distance,
            "goal_arrival_radius": args.goal_arrival_radius,
            "expert_max_steps": args.expert_max_steps,
        })

    label = "validation" if validation else "training"
    quadrant_counts = {
        quadrant: sum(task["start_quadrant"] == quadrant for task in tasks)
        for quadrant in range(4)
    }
    print(
        f">>> Cloud 2D: generating {trajectory_count} {label} episodes "
        f"across {workers} CPU workers; start counts={quadrant_counts}...",
        flush=True,
    )
    started = time.time()
    previous_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    try:
        results = joblib.Parallel(
            n_jobs=workers,
            backend="loky",
            batch_size=1,
            pre_dispatch=workers,
            max_nbytes=None,
        )(
            joblib.delayed(simulate_online_goal_task)(task) for task in tasks
        )
    finally:
        if previous_cuda_visibility is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visibility

    print(
        f">>> Cloud 2D: {label} episodes ready in "
        f"{time.time() - started:.1f}s; no HDF5/TFRecord was written.",
        flush=True,
    )
    return (
        [result["trajectory"] for result in results],
        np.stack([result["center"] for result in results]),
        [
            {
                "seed": result["seed"],
                "seconds": result["seconds"],
                "start_quadrant": result["start_quadrant"],
            }
            for result in results
        ],
    )


def _sample_references(trajectories, timestep_stride, near_goal_radius):
    trajectory_ids = []
    timestep_ids = []
    for trajectory_index, trajectory in enumerate(trajectories):
        n_steps = len(trajectory["x"])
        keep = np.zeros(n_steps, dtype=bool)
        keep[::timestep_stride] = True
        if near_goal_radius > 0 and timestep_stride > 1:
            keep |= goal_transition_proximity(
                trajectory["x"], trajectory["y"], near_goal_radius
            )
        selected = np.flatnonzero(keep)
        trajectory_ids.append(
            np.full(len(selected), trajectory_index, dtype=np.int32)
        )
        timestep_ids.append(selected.astype(np.int32, copy=False))
    return np.concatenate(trajectory_ids), np.concatenate(timestep_ids)


def _to_model_inputs(x, y, indices, n_boids):
    indices = tf.cast(indices, tf.int64)
    node_count = tf.shape(x, out_type=tf.int64)[0]
    adjacency = tf.SparseTensor(
        indices=indices,
        values=tf.ones([tf.shape(indices)[0]], dtype=tf.float32),
        dense_shape=tf.stack([node_count, node_count]),
    )
    graph_count = node_count // tf.cast(n_boids, tf.int64)
    graph_ids = tf.repeat(
        tf.range(graph_count, dtype=tf.int64), tf.cast(n_boids, tf.int64)
    )
    targets = tf.concat((y, tf.cast(graph_ids[:, None], y.dtype)), axis=-1)
    return (
        (
            tf.ensure_shape(x, [None, 6]),
            adjacency,
            tf.ensure_shape(graph_ids, [None]),
        ),
        tf.ensure_shape(targets, [None, 14]),
    )


def dataset_from_trajectories(
    trajectories,
    batch_size,
    n_boids,
    timestep_stride,
    near_goal_radius,
    seed,
    *,
    shuffle_batches,
):
    """Pack expert timesteps once in host RAM as a native tf.data dataset."""
    started = time.time()
    trajectory_ids, timestep_ids = _sample_references(
        trajectories, timestep_stride, near_goal_radius
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trajectory_ids))
    usable = (len(order) // batch_size) * batch_size
    if usable == 0:
        raise ValueError("Not enough expert timesteps for one complete batch.")
    trajectory_ids = trajectory_ids[order[:usable]]
    timestep_ids = timestep_ids[order[:usable]]
    n_batches = usable // batch_size
    n_nodes = batch_size * n_boids
    x_batches = np.empty((n_batches, n_nodes, 6), dtype=np.float32)
    y_batches = np.empty((n_batches, n_nodes, 13), dtype=np.float32)
    x_samples = x_batches.reshape(-1, n_boids, 6)
    y_samples = y_batches.reshape(-1, n_boids, 13)
    for sample_index, (trajectory_index, timestep) in enumerate(
        zip(trajectory_ids, timestep_ids)
    ):
        trajectory = trajectories[int(trajectory_index)]
        x_samples[sample_index] = trajectory["x"][int(timestep)]
        y_samples[sample_index] = trajectory["y"][int(timestep)]

    edge_batches = []
    edge_lengths = np.zeros(n_batches, dtype=np.int64)
    for batch_index in range(n_batches):
        parts = []
        sample_start = batch_index * batch_size
        for graph_index in range(batch_size):
            sample_index = sample_start + graph_index
            trajectory = trajectories[int(trajectory_ids[sample_index])]
            timestep = int(timestep_ids[sample_index])
            edge_start = int(trajectory["edge_offsets"][timestep])
            edge_stop = int(trajectory["edge_offsets"][timestep + 1])
            if edge_stop > edge_start:
                parts.append(
                    trajectory["edge_values"][edge_start:edge_stop]
                    + np.int32(graph_index * n_boids)
                )
        edges = np.concatenate(parts) if parts else np.zeros((0, 2), np.int32)
        edge_batches.append(edges)
        edge_lengths[batch_index] = len(edges)
    edge_values = (
        np.concatenate(edge_batches)
        if edge_lengths.sum()
        else np.zeros((0, 2), dtype=np.int32)
    )
    edge_splits = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(edge_lengths, dtype=np.int64))
    )

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
        lambda x, y, edges: _to_model_inputs(x, y, edges, n_boids),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    if shuffle_batches and n_batches > 1:
        dataset = dataset.shuffle(
            min(512, n_batches), seed=seed, reshuffle_each_iteration=True
        )
    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options).repeat().prefetch(tf.data.AUTOTUNE)
    resident_gib = (x_batches.nbytes + y_batches.nbytes + edge_values.nbytes) / 2**30
    print(
        f">>> Cloud 2D: packed {n_batches} batches in "
        f"{time.time() - started:.1f}s ({resident_gib:.2f} GiB resident).",
        flush=True,
    )
    return dataset, n_batches


def fit_distributed(
    model,
    strategy,
    train_data,
    train_local_steps,
    epochs,
    callbacks,
    global_node_count,
    steps_per_execution,
    loss_fn,
    val_data,
    val_local_steps,
):
    """Synchronous graph-safe training over every visible GPU."""
    replicas = strategy.num_replicas_in_sync
    train_steps = train_local_steps // replicas
    val_steps = val_local_steps // replicas
    if train_steps < 1 or val_steps < 1:
        raise ValueError(
            f"Need at least {replicas} train and validation batches for "
            f"{replicas} replicas."
        )
    distributed_train = strategy.distribute_datasets_from_function(
        lambda _context: train_data
    )
    distributed_val = strategy.distribute_datasets_from_function(
        lambda _context: val_data
    )

    def train_step(batch):
        inputs, targets = batch
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            node_losses = loss_fn(targets, predictions)
            loss_sum = tf.reduce_sum(node_losses)
            loss = loss_sum / tf.cast(global_node_count, tf.float32)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_sum, tf.cast(tf.size(node_losses), tf.float32)

    def validation_step(batch):
        inputs, targets = batch
        node_losses = loss_fn(targets, model(inputs, training=False))
        return tf.reduce_sum(node_losses), tf.cast(tf.size(node_losses), tf.float32)

    @tf.function
    def train_block(iterator, block_steps):
        loss_sum = tf.constant(0.0)
        node_count = tf.constant(0.0)
        for _ in tf.range(block_steps):
            replica_loss, replica_nodes = strategy.run(
                train_step, args=(next(iterator),)
            )
            loss_sum += strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, None)
            node_count += strategy.reduce(tf.distribute.ReduceOp.SUM, replica_nodes, None)
        return loss_sum, node_count

    @tf.function
    def validation_block(iterator, block_steps):
        loss_sum = tf.constant(0.0)
        node_count = tf.constant(0.0)
        for _ in tf.range(block_steps):
            replica_loss, replica_nodes = strategy.run(
                validation_step, args=(next(iterator),)
            )
            loss_sum += strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, None)
            node_count += strategy.reduce(tf.distribute.ReduceOp.SUM, replica_nodes, None)
        return loss_sum, node_count

    callback_list = tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=False,
        add_progbar=False,
        model=model,
        epochs=epochs,
        steps=train_steps,
        verbose=1,
        metrics=["loss", "val_loss"],
    )
    dispatch = max(1, min(int(steps_per_execution), train_steps))
    print(
        f">>> Cloud 2D execution: {replicas} GPUs, {train_steps} global "
        f"steps/epoch, {dispatch} compiled steps/dispatch.",
        flush=True,
    )
    train_iterator = iter(distributed_train)
    model.stop_training = False
    callback_list.on_train_begin()
    history = {"loss": [], "val_loss": [], "lr": []}
    try:
        for epoch in range(int(epochs)):
            if model.stop_training:
                break
            started = time.time()
            callback_list.on_epoch_begin(epoch)
            loss_sum = node_count = 0.0
            for block_start in range(0, train_steps, dispatch):
                block_size = min(dispatch, train_steps - block_start)
                block_loss, block_nodes = train_block(
                    train_iterator, tf.constant(block_size, tf.int32)
                )
                loss_sum += float(block_loss.numpy())
                node_count += float(block_nodes.numpy())
            val_iterator = iter(distributed_val)
            val_loss_sum = val_node_count = 0.0
            for block_start in range(0, val_steps, dispatch):
                block_size = min(dispatch, val_steps - block_start)
                block_loss, block_nodes = validation_block(
                    val_iterator, tf.constant(block_size, tf.int32)
                )
                val_loss_sum += float(block_loss.numpy())
                val_node_count += float(block_nodes.numpy())
            logs = {
                "loss": loss_sum / max(node_count, 1.0),
                "val_loss": val_loss_sum / max(val_node_count, 1.0),
            }
            callback_list.on_epoch_end(epoch, logs)
            lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
            history["loss"].append(logs["loss"])
            history["val_loss"].append(logs["val_loss"])
            history["lr"].append(lr)
            print(
                f"Epoch {epoch + 1}/{epochs} - {time.time() - started:.1f}s - "
                f"loss: {logs['loss']:.6g} - val_loss: "
                f"{logs['val_loss']:.6g} - lr: {lr:.6g}",
                flush=True,
            )
    finally:
        callback_list.on_train_end()
    return history
