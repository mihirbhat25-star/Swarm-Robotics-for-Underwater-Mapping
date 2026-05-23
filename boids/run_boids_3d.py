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

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def custom_weighted_mse_3d(y_true, y_pred):
    # y_true is [current_state (6), next_state (6)], y_pred is predicted next_state (6)
    n_features = tf.shape(y_pred)[-1]
    current_state = y_true[..., :n_features]
    next_state = y_true[..., n_features:]
    mse = tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)

    # Compute average 3D position from current_state (first 3 columns)
    avg_pos = tf.reduce_mean(current_state[..., :3], axis=-2)
    goal = tf.constant([3.0, 3.0, 3.0], dtype=avg_pos.dtype)
    dist_to_goal = tf.norm(avg_pos - goal, axis=-1)

    # Weight loss by 2 if within 2 units of the goal, else 1
    goal_weight = tf.where(dist_to_goal < 2.0, 2.0, 1.0)
    return mse * goal_weight


def run(data_tr, data_va):
    model = GNNCASimpleBoids3D(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    optimizer = Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss="mse")

    loader_tr = DisjointLoader(data_tr, node_level=True, batch_size=args.batch_size)
    loader_va = DisjointLoader(data_va, node_level=True, batch_size=args.batch_size)

    history = model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=args.epochs,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        callbacks=[
            EarlyStopping(patience=args.es_patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=args.lr_patience, min_delta=1e-8, verbose=1),
        ],
    )

    return history, model


####################################################################################
# Configuration
####################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch_size", default=30, type=int)
parser.add_argument("--epochs", default=1000000, type=int)
parser.add_argument("--es_patience", default=200, type=int)
parser.add_argument("--lr_patience", default=10, type=int)
parser.add_argument("--lr_red_factor", default=0.1, type=float)
parser.add_argument("--n_boids", default=100, type=int)
parser.add_argument("--trajectory_len", default=300, type=int)
parser.add_argument("--tr_set_unique", default=20, type=int)
parser.add_argument("--tr_set_repeats", default=50, type=int)
parser.add_argument("--va_set_size", default=30, type=int)
parser.add_argument("--te_set_size", default=30, type=int)

####################################################################################
# Training
####################################################################################
args = parser.parse_args()
print(f"\n>>> Generating 3D dataset...")
data_tr, boids_tr = make_dataset_3d(
    unique_reps=args.tr_set_unique,
    repeat_reps=args.tr_set_repeats,
    save_config=True,
    trajectory_len=args.trajectory_len,
    n_boids=args.n_boids,
    n_jobs=1,
    return_boids=True,
)

data_va = make_dataset_3d(
    unique_reps=args.va_set_size,
    repeat_reps=1,
    save_config=False,
    trajectory_len=args.trajectory_len,
    n_boids=args.n_boids,
    n_jobs=1,
)

history, model = run(data_tr, data_va)

model.save("gnca_model_3d", save_format="tf")
joblib.dump(history.history, "history_3d.pkl")

####################################################################################
# Evaluation
####################################################################################
max_trajectory_len = 1850
evaluate_3d(model, forward, max_trajectory_len, args.n_boids, use_saved_config=True, saved_boids=boids_tr)
