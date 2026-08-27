"""Merge compatible 3D Boids HDF5 caches without loading them into RAM.

Large timestep datasets are copied in bounded blocks. The destination is first
written with a ``.partial`` suffix and is renamed only after its structure and
trajectory boundaries have been verified.
"""

import argparse
import os
from contextlib import ExitStack

import h5py
import numpy as np
from tqdm import tqdm


SAMPLE_DATASETS = ("x", "y", "a_row", "a_col", "a_len")
METADATA_DATASETS = ("centers", "traj_lengths")
REQUIRED_ATTRIBUTES = ("repeats", "n_boids", "max_edges", "sample_mode")


def _attribute_value(value):
    """Convert HDF5 scalar attributes into directly comparable Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_sources(files, paths):
    """Validate schemas and return aggregate cache dimensions."""
    reference = files[0]
    required_datasets = SAMPLE_DATASETS + METADATA_DATASETS

    for path, cache in zip(paths, files):
        missing_datasets = [name for name in required_datasets if name not in cache]
        missing_attrs = [name for name in REQUIRED_ATTRIBUTES if name not in cache.attrs]
        if "unique_reps" not in cache.attrs:
            missing_attrs.append("unique_reps")
        if missing_datasets or missing_attrs:
            raise ValueError(
                f"Cache '{path}' is incomplete: missing datasets={missing_datasets}, "
                f"missing attributes={missing_attrs}"
            )

        n_samples = int(cache["x"].shape[0])
        for name in SAMPLE_DATASETS:
            dataset = cache[name]
            reference_dataset = reference[name]
            if int(dataset.shape[0]) != n_samples:
                raise ValueError(
                    f"Cache '{path}' dataset '{name}' has {dataset.shape[0]} samples; "
                    f"expected {n_samples}."
                )
            if dataset.shape[1:] != reference_dataset.shape[1:]:
                raise ValueError(
                    f"Dataset '{name}' trailing shape mismatch in '{path}': "
                    f"{dataset.shape[1:]} != {reference_dataset.shape[1:]}"
                )
            if dataset.dtype != reference_dataset.dtype:
                raise ValueError(
                    f"Dataset '{name}' dtype mismatch in '{path}': "
                    f"{dataset.dtype} != {reference_dataset.dtype}"
                )

        for name in REQUIRED_ATTRIBUTES:
            actual = _attribute_value(cache.attrs[name])
            expected = _attribute_value(reference.attrs[name])
            if actual != expected:
                raise ValueError(
                    f"Attribute '{name}' mismatch in '{path}': {actual!r} != {expected!r}"
                )

        if "goal_exclusion_size" in reference.attrs or "goal_exclusion_size" in cache.attrs:
            if "goal_exclusion_size" not in reference.attrs or "goal_exclusion_size" not in cache.attrs:
                raise ValueError(f"Cache '{path}' has incompatible goal-exclusion metadata.")
            actual = float(cache.attrs["goal_exclusion_size"])
            expected = float(reference.attrs["goal_exclusion_size"])
            if not np.isclose(actual, expected):
                raise ValueError(
                    f"goal_exclusion_size mismatch in '{path}': {actual} != {expected}"
                )

        unique_reps = int(cache.attrs["unique_reps"])
        repeats = int(cache.attrs["repeats"])
        if int(cache["centers"].shape[0]) != unique_reps:
            raise ValueError(
                f"Cache '{path}' has {cache['centers'].shape[0]} centers but "
                f"unique_reps={unique_reps}."
            )
        expected_trajectories = unique_reps * repeats
        if int(cache["traj_lengths"].shape[0]) != expected_trajectories:
            raise ValueError(
                f"Cache '{path}' has {cache['traj_lengths'].shape[0]} trajectory lengths; "
                f"expected {expected_trajectories}."
            )
        if int(np.sum(cache["traj_lengths"], dtype=np.int64)) != n_samples:
            raise ValueError(
                f"Cache '{path}' trajectory lengths do not sum to its {n_samples} samples."
            )

    return {
        "samples": sum(int(cache["x"].shape[0]) for cache in files),
        "unique_reps": sum(int(cache.attrs["unique_reps"]) for cache in files),
        "trajectories": sum(int(cache["traj_lengths"].shape[0]) for cache in files),
    }


def _creation_options(dataset):
    """Preserve the source dataset's storage/filter configuration."""
    options = {}
    if dataset.chunks is not None:
        options["chunks"] = dataset.chunks
    if dataset.compression is not None:
        options["compression"] = dataset.compression
        options["compression_opts"] = dataset.compression_opts
    if dataset.shuffle:
        options["shuffle"] = True
    if dataset.fletcher32:
        options["fletcher32"] = True
    if dataset.scaleoffset is not None:
        options["scaleoffset"] = dataset.scaleoffset
    if dataset.fillvalue is not None:
        options["fillvalue"] = dataset.fillvalue
    return options


def _copy_sample_dataset(files, destination, name, total_samples, block_size):
    source_template = files[0][name]
    shape = (total_samples,) + source_template.shape[1:]
    maxshape = (None,) + source_template.shape[1:]
    output = destination.create_dataset(
        name,
        shape=shape,
        maxshape=maxshape,
        dtype=source_template.dtype,
        **_creation_options(source_template),
    )

    destination_offset = 0
    progress = tqdm(total=total_samples, unit="samples", desc=f"Copying {name}")
    for cache in files:
        source = cache[name]
        for start in range(0, source.shape[0], block_size):
            stop = min(start + block_size, source.shape[0])
            count = stop - start
            output[destination_offset:destination_offset + count] = source[start:stop]
            destination_offset += count
            progress.update(count)
    progress.close()


def merge_caches(input_paths, output_path, block_size, overwrite=False, validate_only=False):
    output_path = os.path.abspath(output_path)
    partial_path = output_path + ".partial"

    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(path, "r")) for path in input_paths]
        totals = _validate_sources(files, input_paths)

        print(
            f"Validated {len(files)} caches: {totals['unique_reps']} unique centers, "
            f"{totals['trajectories']} trajectories, {totals['samples']} samples."
        )
        if validate_only:
            return

        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(
                f"Output already exists: '{output_path}'. Use --overwrite to replace it."
            )
        if os.path.exists(partial_path):
            if not overwrite:
                raise FileExistsError(
                    f"Partial output already exists: '{partial_path}'. Remove it or use --overwrite."
                )
            os.remove(partial_path)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        reference = files[0]
        with h5py.File(partial_path, "w") as destination:
            for name in SAMPLE_DATASETS:
                _copy_sample_dataset(
                    files, destination, name, totals["samples"], block_size
                )

            centers = np.concatenate([cache["centers"][:] for cache in files], axis=0)
            trajectory_lengths = np.concatenate(
                [cache["traj_lengths"][:] for cache in files], axis=0
            )
            destination.create_dataset("centers", data=centers)
            destination.create_dataset("traj_lengths", data=trajectory_lengths)

            for name, value in reference.attrs.items():
                destination.attrs[name] = value
            destination.attrs["unique_reps"] = totals["unique_reps"]
            destination.flush()

        with h5py.File(partial_path, "r") as merged:
            merged_totals = _validate_sources([merged], [partial_path])
            if merged_totals["samples"] != totals["samples"]:
                raise RuntimeError("Merged cache sample-count verification failed.")
            if merged_totals["unique_reps"] != totals["unique_reps"]:
                raise RuntimeError("Merged cache center-count verification failed.")

    if overwrite:
        os.replace(partial_path, output_path)
    else:
        os.rename(partial_path, output_path)
    size_gib = os.path.getsize(output_path) / (1024 ** 3)
    print(f"Merged cache saved to: {output_path}")
    print(f"File size: {size_gib:.2f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Source HDF5 caches")
    parser.add_argument("--output", required=True, help="Merged HDF5 cache path")
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Number of timesteps copied per block (default: 256)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate source compatibility without writing an output cache",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("--inputs requires at least two cache files")
    if args.block_size <= 0:
        parser.error("--block_size must be positive")

    merge_caches(
        input_paths=[os.path.abspath(path) for path in args.inputs],
        output_path=args.output,
        block_size=args.block_size,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    main()
