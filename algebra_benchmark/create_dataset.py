#!/usr/bin/env python3
#
# Unified CLI to create algebra_benchmark datasets (numerical RPM/RPT only).
# Creates output dir, manifest YAML, and placeholder dataset JSON.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from algebra_benchmark.config_schema import build_data_config, validate_config
from algebra_benchmark.generators import generate_dataset
from algebra_benchmark.tasks import build_config, RULE_TO_TASK

# Default dataset filename written under output_dir
DATASET_FILENAME = "dataset.json"
MANIFEST_FILENAME = "config.yml"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create algebra_benchmark dataset (numerical RPM/RPT): manifest + placeholder data.",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples in the dataset (default: 100).",
    )
    p.add_argument(
        "--gridsize",
        type=int,
        default=3,
        help="Matrix/tensor size (e.g. 3, 9, 15, 30). Default: 3.",
    )
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task: REL-A1 … REL-A7, or rule name (constant, progression, distribute-three, permutation, arithmetic, 4-spatiotemporal, 5-spatiotemporal, neighborhoodsum).",
    )
    p.add_argument(
        "--maxval",
        type=int,
        default=1000,
        help="Max value for numeric entries (default: 1000).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for manifest and dataset file (default: data).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (optional; used when --generate is set).",
    )
    p.add_argument(
        "--generate",
        action="store_true",
        help="Generate numerical samples (implemented for constant/REL-A1; other tasks write placeholder).",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")
    if args.gridsize < 1:
        raise ValueError("--gridsize must be >= 1")
    if args.maxval < 1:
        raise ValueError("--maxval must be >= 1")
    # Validate task by building config (will raise if invalid)
    build_config(args.task, args.gridsize, args.maxval)


def _write_manifest(output_dir: Path, args: argparse.Namespace, config_str: str) -> Path:
    """Write minimal dataset config YAML (schema: path, config, gridsize, nattr, nshow, ntest). Returns path to manifest file."""
    tid = args.task if args.task.startswith("REL-") else RULE_TO_TASK[args.task]
    is_irpt = tid in {"REL-A5", "REL-A6", "REL-A7"}
    nattr = 1 if is_irpt else 3

    data_section = build_data_config(
        path=str(output_dir),
        config=config_str,
        gridsize=args.gridsize,
        nattr=nattr,
        nshow=3,
        ntest=args.num_samples,
        task=args.task,
        num_samples=args.num_samples,
        maxval=args.maxval,
    )
    validate_config(data_section, strict=True)
    data = {"data": data_section}

    try:
        import yaml
    except ImportError:
        manifest_path = output_dir / "config.json"
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)
        return manifest_path

    manifest_path = output_dir / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return manifest_path


def _write_dataset(output_dir: Path, samples: list) -> Path:
    """Write dataset JSON (list of samples). Returns path to file."""
    path = output_dir / DATASET_FILENAME
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    return path


def main() -> int:
    args = _parse_args()
    _validate_args(args)

    config_str = build_config(args.task, args.gridsize, args.maxval)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = _write_manifest(output_dir, args, config_str)

    if args.generate:
        samples = generate_dataset(
            args.task,
            args.num_samples,
            args.gridsize,
            args.maxval,
            args.seed,
        )
        dataset_path = _write_dataset(output_dir, samples)
        print(f"Created manifest: {manifest_path}")
        print(f"Created dataset: {dataset_path} ({len(samples)} samples)")
    else:
        dataset_path = _write_dataset(output_dir, [])
        print(f"Created manifest: {manifest_path}")
        print(f"Created placeholder dataset: {dataset_path} (0 samples; use --generate to fill)")
    print(f"Config: {config_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
