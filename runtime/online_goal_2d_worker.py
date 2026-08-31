"""Private subprocess entry point for one online-goal cloud chunk."""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib

from runtime.online_goal_2d import _worker


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--args_file", required=True)
    parser.add_argument("--chunk_index", required=True, type=int)
    parser.add_argument("--trajectory_count", required=True, type=int)
    parser.add_argument("--state_dir", required=True)
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--run_tag", required=True)
    cli = parser.parse_args()
    args = joblib.load(cli.args_file)
    _worker(
        args,
        cli.chunk_index,
        cli.trajectory_count,
        cli.state_dir,
        cli.result_path,
        cli.run_tag,
    )


if __name__ == "__main__":
    main()
