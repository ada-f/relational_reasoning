#
# Load dataset config and numerical dataset for algebra_benchmark.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from config_schema import validate_config
from format import load_sample_from_dataset


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load dataset config from a YAML or JSON file.
    Expects top-level key "data" with path, config, gridsize, nattr, nshow, ntest.
    Returns the full dict (with "data" key).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = path.read_text()
    if path.suffix in (".yml", ".yaml"):
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML config; install pyyaml")
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict) or "data" not in data:
        raise ValueError("Config must be a dict with top-level key 'data'")
    validate_config(data["data"], strict=False)
    return data


def load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    """
    Load dataset from JSON file. Expects a list of samples or a dict keyed by sample ID.
    Returns a list of sample dicts (panels, target, choices).
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        keys = sorted(data.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
        return [data[k] for k in keys]
    raise ValueError("Dataset must be a JSON list or dict")


def load_config_and_dataset(config_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Load config and dataset. The dataset file is expected at config_path's
    parent / dataset.json (create_dataset writes config and dataset in the same dir).
    """
    cfg = load_config(config_path)
    config_dir = Path(config_path).resolve().parent
    dataset_file = config_dir / "dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    samples = load_dataset(dataset_file)
    return cfg, samples
