#!/usr/bin/env python3
#
# Generate example datasets: 125 samples per task (REL-A1 through REL-A7).
# Run from this directory: python create_example_datasets.py
#

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tasks import get_valid_tasks


def main() -> int:
    base = Path(__file__).resolve().parent
    out_base = base / "example_datasets"
    out_base.mkdir(parents=True, exist_ok=True)
    num_samples = 125
    gridsize = 3
    maxval = 1000
    seed = 42

    for task in get_valid_tasks():
        out_dir = out_base / task
        cmd = [
            sys.executable,
            str(base / "create_dataset.py"),
            "--task",
            task,
            "--gridsize",
            str(gridsize),
            "--num_samples",
            str(num_samples),
            "--maxval",
            str(maxval),
            "--output_dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--generate",
        ]
        ret = subprocess.run(cmd, cwd=base)
        if ret.returncode != 0:
            return ret.returncode
    print(f"Created 7 example datasets in {out_base} (125 samples each).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
