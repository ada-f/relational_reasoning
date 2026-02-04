#
# Minimal dataset config schema for algebra_benchmark.
# No nconf, uncertainty, permutation, ood, or visual-only fields.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Schema: required and optional keys under top-level "data"
# ---------------------------------------------------------------------------

# Required keys for the benchmark (loaders expect these)
REQUIRED_KEYS = frozenset({"path", "config", "gridsize", "nattr", "nshow", "ntest"})

# Optional keys the CLI may write (task, num_samples, maxval)
OPTIONAL_KEYS = frozenset({"task", "num_samples", "maxval"})

# Keys we explicitly do NOT use (confounder/noise/visual)
EXCLUDED_KEYS = frozenset({
    "nconf",
    "uncertainty",
    "maxval_uncert",
    "permutation",
    "ood_set_attr",
    "ood_set_rule",
    "offset",
    "angle",
    "scaling",
    "dataset",  # optional: we use path + config instead of dataset name
})


def validate_config(cfg: dict[str, Any], *, strict: bool = True) -> None:
    """
    Validate a dataset config (the inner "data" dict). Raises ValueError if invalid.

    Parameters
    ----------
    cfg : dict
        The "data" section of the config (flat key-value).
    strict : bool
        If True, reject any key in EXCLUDED_KEYS. If False, only check required keys.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict")

    missing = REQUIRED_KEYS - set(cfg)
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    for key in REQUIRED_KEYS:
        val = cfg[key]
        if key == "path" and not isinstance(val, str):
            raise ValueError("config.data.path must be a string")
        if key == "config" and not isinstance(val, str):
            raise ValueError("config.data.config must be a string")
        if key in ("gridsize", "nattr", "nshow", "ntest") and not isinstance(val, int):
            raise ValueError(f"config.data.{key} must be an int")
        if key in ("gridsize", "nattr", "nshow", "ntest") and val < 1:
            raise ValueError(f"config.data.{key} must be >= 1")

    if strict:
        disallowed = set(cfg) & EXCLUDED_KEYS
        if disallowed:
            raise ValueError(f"Config must not contain: {sorted(disallowed)}")


def build_data_config(
    path: str,
    config: str,
    gridsize: int,
    nattr: int,
    nshow: int,
    ntest: int,
    *,
    task: str | None = None,
    num_samples: int | None = None,
    maxval: int | None = None,
) -> dict[str, Any]:
    """
    Build the "data" section of a dataset config dict using only schema-allowed keys.
    """
    out: dict[str, Any] = {
        "path": path,
        "config": config,
        "gridsize": gridsize,
        "nattr": nattr,
        "nshow": nshow,
        "ntest": ntest,
    }
    if task is not None:
        out["task"] = task
    if num_samples is not None:
        out["num_samples"] = num_samples
    if maxval is not None:
        out["maxval"] = maxval
    return out


if __name__ == "__main__":
    cfg = build_data_config(
        path="/tmp/out",
        config="center_singleConstant_shuffle_n_3_maxval_1000",
        gridsize=3,
        nattr=3,
        nshow=3,
        ntest=10,
        task="REL-A1",
        num_samples=10,
        maxval=1000,
    )
    validate_config(cfg, strict=True)
    # Reject excluded key
    bad = {**cfg, "nconf": 0}
    try:
        validate_config(bad, strict=True)
    except ValueError as e:
        assert "nconf" in str(e) or "must not contain" in str(e).lower()
    print("config_schema.py: build_data_config and validate_config checks passed.")
