"""
Trains the GNCA to imitate the Boids GCA.
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

# RESTORED: Import evaluate
from boids.evaluate_boids import evaluate
from boids.forward import forward
from models.gnn_ca_simple_boids import GNNCASimpleBoids
from modules.boids import make_dataset
# RESTORED: Import Callback
from modules.callbacks import ComplexityCallback

# --- NEW CALLBACK TO PRINT PATIENCE ---
class PrintESPatience(tf.keras.callbacks.Callback):
    def __init__(self, es_callback):
        super().__init__()
        self.es_callback = es_callback

    def on_epoch_end(self, epoch, logs=None):
        # patience is the total allowed, wait is the current counter of non-improvement
        remaining = self.es_callback.patience - self.es_callback.wait
        print(f" — EarlyStopping Patience: {remaining}/{self.es_callback.patience}")

# tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def run(data_tr, data_va):

    model = GNNCASimpleBoids(
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

    # Define EarlyStopping explicitly to pass it to the helper
    es = EarlyStopping(
        patience=args.es_patience, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=args.epochs,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        callbacks=[
            es,
            PrintESPatience(es),
            ReduceLROnPlateau(patience=args.lr_patience, min_delta=1e-8, verbose=1),
            # RESTORED: Complexity Callback
            ComplexityCallback(test_every=args.test_complexity_every),
        ],
    )

    return history, model

####################################################################################
# Configuration
####################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3, type=float, help="Initial LR")
parser.add_argument("--anim_gen", default=False, type=bool, help="Whether to generate animations")
parser.add_argument("--time_bool", default=True, type=bool, help="Whether to include time as a feature")
parser.add_argument("--loiter_bool", default=True, type=bool, help="Whether boids should loiter")
parser.add_argument(
    "--batch_size", default=30, type=int, help="Size of the mini-batches"
)
parser.add_argument(
    "--epochs", default=1000000, type=int, help="Number of training epochs"
)
parser.add_argument(
    "--es_patience", default=200, type=int, help="Patience for early stopping"
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
    "--tr_set_size", default=300, type=int, help="N. of training trajectories"
)
parser.add_argument(
    "--tr_set_unique", default=300, type=int, help="N. of unique training trajectories"
)
# parser.add_argument(
#     "--va_set_unique", default=30, type=int, help="N. of unique validation trajectories"
# )
# parser.add_argument(
#     "--te_set_unique", default=30, type=int, help="N. of unique test trajectories"
# )
parser.add_argument(
    "--test_complexity_every",
    default=-1,
    type=int,
    help="How often to test for complexity (-1 for never)",
)

####################################################################################
# Training
####################################################################################

args = parser.parse_args()
print(f"\n>>> Generating dataset (n_jobs=1)...")
data_tr = make_dataset(
    reps_unique=args.tr_set_unique, repeat_reps=args.tr_set_size // args.tr_set_unique, random_init=True, fixed_init=False, n_boids=args.n_boids, n_jobs=1, loiter=args.loiter_bool, time_bool=args.time_bool
)
print("Train dataset size:", len(data_tr.graphs))

data_va = make_dataset(
    reps_unique=30, repeat_reps=1, random_init=True, fixed_init=False, n_boids=args.n_boids, n_jobs=1, loiter=args.loiter_bool, time_bool=args.time_bool
)
print("Validation dataset size:", len(data_va.graphs))

print(f"\n>>> Training GNCA model... with config: {args}")
history, model = run(data_tr, data_va)

model.save("gnca_model", save_format="tf")
joblib.dump(history.history, "history.pkl")

####################################################################################
# Evaluation
####################################################################################
evaluate(model, forward, repeat_reps=args.tr_set_size // args.tr_set_unique, reps_unique=args.tr_set_unique, n_boids=args.n_boids, loiter=args.loiter_bool, anim_gen=args.anim_gen, time_bool=args.time_bool)