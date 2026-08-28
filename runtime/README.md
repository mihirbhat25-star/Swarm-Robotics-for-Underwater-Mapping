# Runtime backends

This package contains computational plumbing for the existing experiment
scripts. It does not define Boids dynamics, GNCA layers, losses, or experiment
semantics.

The supported 3D modes are:

- `local`: retain the original in-memory Spektral path for small experiments,
  or stream a precomputed HDF5 cache through bounded NPZ shards in isolated,
  single-GPU chunk workers. Worker exit releases TensorFlow and host memory
  after every cache chunk.
- `cloud`: call `Boids3D` across CPU workers, retain compact trajectories and
  packed graph batches in host RAM, and train synchronously on all visible
  GPUs. No temporary HDF5 or TFRecord dataset is created.

`boids/run_boids_3d.py` remains the public training entry point. Select a mode
with `--backend local` or `--backend cloud`.

The former collection of overlapping cloud switches has intentionally been
replaced by this single backend choice. Existing local commands continue to
use `local` by default; cloud commands should pass `--backend cloud`.

Module responsibilities:

- `local_2d.py`: local 2D cache packing and bounded batch-file streaming.
- `local_3d.py`: local cache validation, timestep shuffling, and shard I/O.
- `cloud_3d.py`: cloud RAM packing and distributed execution.
- `cloud_data_3d.py`: TensorFlow-free adapter that calls `Boids3D` in CPU
  workers and compacts its returned histories.
- `cli_3d.py`: runtime CLI and argument validation.
- `chunking.py`: backend-independent chunk allocation.
