#
# Numerical RPM/RPT data format for algebra_benchmark.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

from typing import Any, Union

# ---------------------------------------------------------------------------
# Schema: one sample = context panels + target index + answer choices
# ---------------------------------------------------------------------------

# A single panel is a numerical matrix (2D) or tensor (3D), represented as
# nested lists of numbers for JSON serialization.
# - Matrix (RPM): list of rows, each row list of numbers (e.g. 3x3 → 3 lists of 3 numbers).
# - Tensor (RPT): list of slices, each slice a 2D matrix (list of list of numbers).
Panel = Union[list[list[float]], list[list[list[float]]]]


def _is_numeric_panel(obj: Any, allow_ints: bool = True) -> bool:
    """Check that obj is a (possibly nested) list of numbers."""
    if isinstance(obj, (int, float)):
        return allow_ints or isinstance(obj, float)
    if isinstance(obj, list):
        return len(obj) > 0 and all(_is_numeric_panel(x, allow_ints) for x in obj)
    return False


def validate_sample(sample: dict[str, Any], n_choices: int = 8) -> None:
    """
    Validate a single sample dict. Raises ValueError if invalid.

    Expects:
        panels : list of panels (context; typically 8 for 3x3 RPM).
        target : int, index of the correct answer in choices (0 <= target < len(choices)).
        choices : list of panels (candidate answers; typically 8).
    """
    if not isinstance(sample, dict):
        raise ValueError("Sample must be a dict")
    if "panels" not in sample:
        raise ValueError("Sample must have 'panels'")
    if "target" not in sample:
        raise ValueError("Sample must have 'target'")
    if "choices" not in sample:
        raise ValueError("Sample must have 'choices'")

    panels = sample["panels"]
    target = sample["target"]
    choices = sample["choices"]

    if not isinstance(panels, list) or len(panels) == 0:
        raise ValueError("'panels' must be a non-empty list")
    if not isinstance(choices, list) or len(choices) != n_choices:
        raise ValueError(f"'choices' must be a list of length {n_choices}")
    if not isinstance(target, int) or target < 0 or target >= len(choices):
        raise ValueError(f"'target' must be an int in [0, {len(choices)-1}]")

    for i, p in enumerate(panels):
        if not _is_numeric_panel(p):
            raise ValueError(f"panels[{i}] must be a numerical matrix/tensor (nested list of numbers)")
    for i, c in enumerate(choices):
        if not _is_numeric_panel(c):
            raise ValueError(f"choices[{i}] must be a numerical matrix/tensor (nested list of numbers)")


# ---------------------------------------------------------------------------
# JSON file layout for a dataset
# ---------------------------------------------------------------------------

# A dataset file is a JSON object keyed by sample ID (string or int), each value a sample dict:
#
#   {
#     "0": { "panels": [...], "target": 2, "choices": [...] },
#     "1": { "panels": [...], "target": 5, "choices": [...] },
#     ...
#   }
#
# Or a JSON array of samples (order = sample index):
#
#   [
#     { "panels": [...], "target": 2, "choices": [...] },
#     ...
#   ]


def load_sample_from_dataset(data: Union[dict, list], index: int) -> dict[str, Any]:
    """
    Get one sample by index from a dataset loaded from JSON.

    data : either a dict (id -> sample) or a list of samples.
    index : integer index (for list, direct index; for dict, key str(index)).
    """
    if isinstance(data, list):
        if index < 0 or index >= len(data):
            raise IndexError(f"Index {index} out of range [0, {len(data)-1}]")
        return data[index]
    if isinstance(data, dict):
        key = str(index)
        if key not in data:
            raise KeyError(f"Sample index '{key}' not in dataset")
        return data[key]
    raise TypeError("Dataset must be a dict or list")


# ---------------------------------------------------------------------------
# Example (for documentation and tests)
# ---------------------------------------------------------------------------

_ONE_3X3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

EXAMPLE_SAMPLE: dict[str, Any] = {
    "panels": [_ONE_3X3] * 8,  # 8 context panels (3x3 each)
    "target": 0,
    "choices": [_ONE_3X3] * 4 + [[[0.0] * 3] * 3] * 4,  # 8 candidate answer panels
}


if __name__ == "__main__":
    validate_sample(EXAMPLE_SAMPLE)
    assert load_sample_from_dataset([EXAMPLE_SAMPLE], 0) == EXAMPLE_SAMPLE
    assert load_sample_from_dataset({"0": EXAMPLE_SAMPLE}, 0) == EXAMPLE_SAMPLE
    print("format.py: validation and load_sample_from_dataset checks passed.")
