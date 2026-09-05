"""Subprocess helpers that preserve diagnostics while filtering known noise."""

from __future__ import annotations

import os
import subprocess
import sys


def run_with_filtered_stderr(command, ignored_fragments):
    """Run a command and hide stderr lines matching every fragment in a rule."""
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    process = subprocess.Popen(
        command,
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stderr is not None
    for line in process.stderr:
        should_ignore = any(
            all(fragment in line for fragment in rule)
            for rule in ignored_fragments
        )
        if should_ignore:
            continue
        sys.stderr.write(line)
        sys.stderr.flush()
    return process.wait()
