"""Command-line interface and validation for 3D GNCA training."""

import argparse

from runtime import CLOUD_BACKEND, LOCAL_BACKEND, SUPPORTED_BACKENDS


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the 3D GNCA with a local or cloud data backend."
    )
    parser.add_argument(
        "--task",
        choices=("fixed_waypoints", "online_goals"),
        default="fixed_waypoints",
        help=(
            "Experiment semantics. fixed_waypoints preserves the established "
            "3D experiment; online_goals trains a goal-conditioned 3D GNCA."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default=LOCAL_BACKEND,
        help=(
            "'local' streams precomputed HDF5 caches through isolated workers; "
            "'cloud' generates experts in RAM and uses all visible GPUs."
        ),
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--lr", default=1e-3, type=float)
    optimization.add_argument("--batch_size", default=30, type=int)
    optimization.add_argument("--epochs", default=1_000_000, type=int)
    optimization.add_argument("--chunk_epochs", default=None, type=int)
    optimization.add_argument("--es_patience", default=12, type=int)
    optimization.add_argument("--early_stopping_min_delta", default=1e-8, type=float)
    optimization.add_argument("--lr_patience", default=5, type=int)
    optimization.add_argument("--lr_red_factor", default=0.1, type=float)
    optimization.add_argument("--min_lr", default=1e-6, type=float)
    optimization.add_argument("--steps_per_execution", default=10, type=int)
    optimization.add_argument("--eager_training", action="store_true")

    data = parser.add_argument_group("experiment data")
    data.add_argument("--n_boids", default=100, type=int)
    data.add_argument("--tr_set_unique", default=20, type=int)
    data.add_argument("--run_tag_unique", default=None, type=int)
    data.add_argument("--tr_set_repeats", default=50, type=int)
    data.add_argument("--va_set_size", default=10, type=int)
    data.add_argument("--te_set_size", default=30, type=int)
    data.add_argument("--boids_cache", nargs="+", default=None)
    data.add_argument("--init_centers_npz", default="", type=str)
    data.add_argument("--train_octants", type=int, nargs="+", default=None)
    data.add_argument("--octant_unique_counts", type=int, nargs="+", default=None)
    data.add_argument("--timestep_stride", default=1, type=int)
    data.add_argument("--near_goal_radius", default=1.0, type=float)
    data.add_argument("--perception", default=0.1, type=float)
    data.add_argument(
        "--expert_goal_order",
        choices=["nearest_ccw", "fixed"],
        default="nearest_ccw",
    )
    data.add_argument("--goal_exclusion_size", default=0.5, type=float)
    data.add_argument("--expert_pos_noise", default=0.0, type=float)
    data.add_argument("--expert_vel_noise", default=0.0, type=float)
    data.add_argument(
        "--goal_waypoints_per_episode",
        default=2,
        type=int,
        help="Online-goal training uses exactly two waypoint terminations.",
    )
    data.add_argument(
        "--goal_bounds",
        nargs=6,
        type=float,
        default=[-4.5, 4.5, -4.5, 4.5, -4.5, 4.5],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    data.add_argument(
        "--start_bounds",
        nargs=6,
        type=float,
        default=[-5.0, 5.0, -5.0, 5.0, -5.0, 5.0],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    data.add_argument("--goal_min_distance", default=2.5, type=float)
    data.add_argument("--goal_arrival_radius", default=0.5, type=float)
    data.add_argument("--expert_max_steps", default=10_000, type=int)

    loss = parser.add_argument_group("loss")
    loss.add_argument("--loss_type", default="oldl", choices=["oldl", "newl"])
    loss.add_argument("--critical_distance", default=2.5, type=float)
    loss.add_argument("--distance_weight", default=2.5, type=float)

    chunks = parser.add_argument_group("chunking")
    chunks.add_argument("--chunk_size", default=100, type=int)
    chunks.add_argument("--chunk_patience", default=15, type=int)
    chunks.add_argument("--shuffle_buffer_size", default=4096, type=int)
    chunks.add_argument("--packed_shard_batches", default=16, type=int)
    chunks.add_argument("--generation_workers", default=0, type=int)
    chunks.add_argument("--generation_seed", default=0, type=int)
    chunks.add_argument("--init_weights", default="", type=str)

    profiling = parser.add_argument_group("profiling")
    profiling.add_argument(
        "--timing_report",
        default="",
        type=str,
        help=(
            "Write a pipeline timing allocation report as JSON. Cloud workers "
            "record expert generation, graph/batch packing, and training."
        ),
    )
    parser.add_argument(
        "--cloud_data_mode",
        choices=("legacy", "compiled"),
        default="legacy",
        help=(
            "Cloud-only expert/data implementation. legacy preserves the "
            "existing SciPy/COO path; compiled uses Numba rollouts and "
            "bit-packed adjacency without changing the GNCA architecture."
        ),
    )
    profiling.add_argument(
        "--wall_time_limit_hours",
        default=0.0,
        type=float,
        help=(
            "Stop cloud chunked training cleanly after this many wall-clock "
            "hours; 0 disables the limit."
        ),
    )

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument(
        "--viz_mode",
        default="tubular",
        choices=["tubular", "per_boid", "multi_tubular", "individual"],
    )
    evaluation.add_argument("--best_after_epoch", default=50, type=int)
    evaluation.add_argument("--viz_n_centers", default=50, type=int)
    evaluation.add_argument(
        "--viz_centers_source",
        default="trained",
        choices=["trained", "unseen_cache", "random", "fresh_octant"],
    )
    evaluation.add_argument("--skip_trained_viz", action="store_true")
    evaluation.add_argument("--skip_evaluation", action="store_true")
    evaluation.add_argument("--eval_centers_per_octant", default=10, type=int)
    evaluation.add_argument("--eval_output_dir", default="", type=str)
    evaluation.add_argument("--eval_max_steps", default=3000, type=int)
    evaluation.add_argument("--eval_success_threshold", default=0.5, type=float)
    evaluation.add_argument("--eval_max_success_r", default=2.0, type=float)
    evaluation.add_argument("--eval_online_goal_count", default=5, type=int)
    evaluation.add_argument("--eval_max_tube_radius", default=1.0, type=float)
    evaluation.add_argument("--eval_seed", default=None, type=int)
    evaluation.add_argument("--noise_tag", default="", type=str)

    # Subprocess protocol. These are intentionally absent from public help.
    internal = argparse.SUPPRESS
    parser.add_argument("--_chunk_worker", action="store_true", help=internal)
    parser.add_argument("--_chunk_idx", default=0, type=int, help=internal)
    parser.add_argument("--_chunk_indices_file", default=None, help=internal)
    parser.add_argument("--_cache_paths_file", default=None, help=internal)
    parser.add_argument("--_checkpoint_path", default=None, help=internal)
    parser.add_argument("--_n_boids_cache", default=None, type=int, help=internal)
    parser.add_argument("--_boundaries_file", default=None, help=internal)
    parser.add_argument("--_val_npz_file", default=None, help=internal)
    parser.add_argument("--_training_state_dir", default=None, help=internal)
    parser.add_argument("--_ram_chunk_mix_file", default=None, help=internal)
    parser.add_argument("--_ram_centers_output_file", default=None, help=internal)
    parser.add_argument("--_timing_events_file", default=None, help=internal)
    parser.add_argument("--_wall_deadline", default=0.0, type=float, help=internal)
    return parser


def validate_args(args, visible_gpu_count):
    if args.n_boids < 1:
        raise ValueError("--n_boids must be at least 1.")
    if args.tr_set_unique < 1:
        raise ValueError("--tr_set_unique must be at least 1.")
    if args.tr_set_repeats < 1:
        raise ValueError("--tr_set_repeats must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.timestep_stride < 1:
        raise ValueError("--timestep_stride must be at least 1.")
    if args.va_set_size < 1 and args.chunk_epochs is None:
        raise ValueError("--va_set_size must be at least 1 with validation.")
    if args.chunk_patience < 1:
        raise ValueError("--chunk_patience must be at least 1.")
    if args.lr_patience < 0:
        raise ValueError("--lr_patience cannot be negative.")
    if args.run_tag_unique is not None and args.run_tag_unique < 1:
        raise ValueError("--run_tag_unique must be at least 1.")
    if args.generation_workers < 0:
        raise ValueError("--generation_workers must be nonnegative.")
    if args.wall_time_limit_hours < 0:
        raise ValueError("--wall_time_limit_hours must be nonnegative.")
    if args.train_octants is not None:
        if len(set(args.train_octants)) != len(args.train_octants):
            raise ValueError("--train_octants must not contain duplicates.")
        if any(octant not in range(8) for octant in args.train_octants):
            raise ValueError("--train_octants values must be between 0 and 7.")
    if args.octant_unique_counts is not None:
        if args.train_octants is None:
            raise ValueError("--octant_unique_counts requires --train_octants.")
        if len(args.octant_unique_counts) != len(args.train_octants):
            raise ValueError(
                "--octant_unique_counts needs one count per train octant."
            )
        if any(count < 1 for count in args.octant_unique_counts):
            raise ValueError("Every --octant_unique_counts value must be positive.")
        if sum(args.octant_unique_counts) != args.tr_set_unique:
            raise ValueError(
                "The sum of --octant_unique_counts must equal --tr_set_unique."
            )
    if args.min_lr <= 0 or args.min_lr > args.lr:
        raise ValueError("--min_lr must be positive and cannot exceed --lr.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early_stopping_min_delta must be nonnegative.")
    if args.steps_per_execution < 1:
        raise ValueError("--steps_per_execution must be at least 1.")
    if args.cloud_data_mode == "compiled" and args.backend != CLOUD_BACKEND:
        raise ValueError("--cloud_data_mode compiled requires --backend cloud.")
    if (
        not args._chunk_worker
        and args.chunk_size > 0
        and args.chunk_epochs is None
        and args.lr_patience >= args.chunk_patience
    ):
        raise ValueError(
            "--lr_patience must be smaller than --chunk_patience."
        )

    if args.backend == CLOUD_BACKEND:
        if args.boids_cache:
            raise ValueError("Cloud mode generates data in RAM; omit --boids_cache.")
        if args.tr_set_repeats != 1:
            raise ValueError("Cloud mode requires --tr_set_repeats 1.")
        if args.train_octants is None:
            raise ValueError("Cloud mode requires explicit --train_octants.")
        if args.chunk_size <= 0:
            raise ValueError("Cloud mode requires --chunk_size greater than 0.")
        if args.init_centers_npz:
            raise ValueError("Cloud mode does not combine with --init_centers_npz.")
        if args.eager_training:
            raise ValueError("Cloud mode requires compiled graph execution.")
        if visible_gpu_count < 1:
            raise ValueError("Cloud mode requires at least one visible GPU.")
        if args.batch_size % visible_gpu_count:
            raise ValueError(
                "Cloud --batch_size is global and must be divisible by the "
                f"{visible_gpu_count} visible GPUs."
            )

    if args.task == "online_goals":
        if args.backend != CLOUD_BACKEND:
            raise ValueError("The online_goals task requires --backend cloud.")
        if args.cloud_data_mode != "compiled":
            raise ValueError(
                "The 3D online_goals task requires --cloud_data_mode compiled."
            )
        if args.goal_waypoints_per_episode != 2:
            raise ValueError(
                "The online-goal experiment requires exactly two waypoint "
                "terminations per training episode."
            )
        for name, bounds in (
            ("goal_bounds", args.goal_bounds),
            ("start_bounds", args.start_bounds),
        ):
            if len(bounds) != 6 or any(
                bounds[index] >= bounds[index + 1]
                for index in (0, 2, 4)
            ):
                raise ValueError(
                    f"--{name} must contain three increasing min/max pairs."
                )
        if args.goal_min_distance < 0:
            raise ValueError("--goal_min_distance cannot be negative.")
        if args.goal_arrival_radius <= 0:
            raise ValueError("--goal_arrival_radius must be positive.")
        if args.expert_max_steps < 1:
            raise ValueError("--expert_max_steps must be positive.")
        if args.eval_online_goal_count < 1:
            raise ValueError("--eval_online_goal_count must be positive.")
        if args.eval_max_tube_radius <= 0:
            raise ValueError("--eval_max_tube_radius must be positive.")

    return args
