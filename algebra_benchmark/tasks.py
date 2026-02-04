#
# Task mapping and config string builder for algebra_benchmark.
# Numerical RPM/RPT only; no visual inputs.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Task ID (REL-A1 … REL-A7) → ground rule name
# ---------------------------------------------------------------------------

TASK_TO_RULE = {
    "REL-A1": "constant",
    "REL-A2": "progression",
    "REL-A3": "distribute-three",  # or "permutation"
    "REL-A4": "arithmetic",
    "REL-A5": "4-spatiotemporal",
    "REL-A6": "5-spatiotemporal",
    "REL-A7": "neighborhoodsum",
}

# Inverse: rule name → canonical task ID (for rules that map from one task)
RULE_TO_TASK = {
    "constant": "REL-A1",
    "progression": "REL-A2",
    "distribute-three": "REL-A3",
    "permutation": "REL-A3",
    "arithmetic": "REL-A4",
    "4-spatiotemporal": "REL-A5",
    "5-spatiotemporal": "REL-A6",
    "neighborhoodsum": "REL-A7",
}

# Matrix (iravenx) rules use center_single + RuleName + _shuffle_n_<n>_maxval_<v>
# Rule name as it appears in config string (iravenx)
_IRAVENX_RULE_SUFFIX = {
    "constant": "Constant",
    "progression": "Progression",
    "distribute-three": "Distribute_Three",
    "permutation": "",  # center_single_shuffle_n_3_maxval_1000 (no rule in middle)
    "arithmetic": "Arithmetic",
}

# Tensor (irpt) rules use rpt_<Rule>_n_<n>_maxval_<v>_...
_IRPT_RULE_NAME = {
    "4-spatiotemporal": "SpatioTemporal4",
    "5-spatiotemporal": "SpatioTemporal5",
    "neighborhoodsum": "NeighborhoodSum",
}

# Which tasks are matrix (iravenx) vs tensor (irpt)
_MATRIX_TASKS = {"REL-A1", "REL-A2", "REL-A3", "REL-A4"}
_IRPT_TASKS = {"REL-A5", "REL-A6", "REL-A7"}


def _is_matrix_task(task: str) -> bool:
    """True if task uses iravenx (matrix) config format."""
    tid = task if task.startswith("REL-") else RULE_TO_TASK.get(task)
    return tid in _MATRIX_TASKS if tid else False


def _is_irpt_task(task: str) -> bool:
    """True if task uses irpt (tensor) config format."""
    tid = task if task.startswith("REL-") else RULE_TO_TASK.get(task)
    return tid in _IRPT_TASKS if tid else False


def _normalize_task(task: str) -> str:
    """Return canonical task ID (REL-Ax) from task ID or rule name."""
    if task.startswith("REL-"):
        if task not in TASK_TO_RULE:
            raise ValueError(f"Unknown task ID: {task}. Known: {list(TASK_TO_RULE)}")
        return task
    if task in RULE_TO_TASK:
        return RULE_TO_TASK[task]
    raise ValueError(f"Unknown task/rule: {task}. Known tasks: {list(TASK_TO_RULE)}; rules: {list(RULE_TO_TASK)}")


def _get_rule_name(task: str) -> str:
    """Return ground rule name for a task (REL-Ax or rule name)."""
    tid = _normalize_task(task)
    return TASK_TO_RULE[tid]


def build_config(
    task: str,
    gridsize: int,
    maxval: int = 1000,
    *,
    # irpt-only (optional)
    p: Optional[int] = None,
    depth: Optional[int] = None,
    conn: Optional[int] = None,
    **irpt_kw: Any,
) -> str:
    """
    Build the config string for a given task, gridsize, and maxval.

    For matrix tasks (REL-A1 … REL-A4): uses iravenx format
        center_single[RuleName]_shuffle_n_<gridsize>_maxval_<maxval>
    For tensor tasks (REL-A5 … REL-A7): uses irpt format
        rpt_<Rule>_n_<gridsize>_maxval_<maxval>_[p_<p>_depth_<depth>|conn_<conn>_p<p>]

    Parameters
    ----------
    task : str
        Task ID (e.g. REL-A1) or rule name (e.g. constant, arithmetic).
    gridsize : int
        Matrix/tensor size (e.g. 3, 9, 15, 30).
    maxval : int
        Max value for numeric entries (default 1000).
    p, depth, conn : optional
        Used for irpt configs. p/depth for SpatioTemporal; conn/p for NeighborhoodSum.
    **irpt_kw
        Additional irpt params (e.g. depth=3, conn=6).

    Returns
    -------
    str
        Config string suitable for dataset config / loader.
    """
    tid = _normalize_task(task)
    # If user passed a rule name (e.g. "permutation"), use it for suffix; else use canonical rule for task ID
    rule = task if task in RULE_TO_TASK else TASK_TO_RULE[tid]

    if tid in _MATRIX_TASKS:
        suffix = _IRAVENX_RULE_SUFFIX.get(rule)
        if suffix is None:
            raise ValueError(f"Matrix task {task} has no iravenx rule mapping: {rule}")
        mid = f"{suffix}_" if suffix else "_"  # always need _ before "shuffle"
        return f"center_single{mid}shuffle_n_{gridsize}_maxval_{maxval}"

    if tid in _IRPT_TASKS:
        rpt_name = _IRPT_RULE_NAME.get(rule)
        if rpt_name is None:
            raise ValueError(f"Tensor task {task} has no irpt rule mapping: {rule}")
        base = f"rpt_{rpt_name}_n_{gridsize}_maxval_{maxval}"
        if rule == "4-spatiotemporal":
            _p = p if p is not None else irpt_kw.get("p", 11)
            _depth = depth if depth is not None else irpt_kw.get("depth", 3)
            return f"{base}_p_{_p}_depth_{_depth}"
        if rule == "5-spatiotemporal":
            _p = p if p is not None else irpt_kw.get("p", 11)
            _depth = depth if depth is not None else irpt_kw.get("depth", 3)
            return f"{base}_p_{_p}_depth_{_depth}"
        if rule == "neighborhoodsum":
            _conn = conn if conn is not None else irpt_kw.get("conn", 6)
            _p = p if p is not None else irpt_kw.get("p", 17)
            return f"{base}_conn_{_conn}_p{_p}"
        return base

    raise ValueError(f"Task not in matrix or irpt set: {tid}")


def get_valid_tasks() -> list[str]:
    """Return list of valid task IDs (REL-A1 … REL-A7)."""
    return list(TASK_TO_RULE)


def get_valid_rules() -> list[str]:
    """Return list of valid rule names (including permutation)."""
    return list(RULE_TO_TASK)


if __name__ == "__main__":
    # Sanity-check build_config for a few tasks
    assert build_config("REL-A1", 3, 1000) == "center_singleConstant_shuffle_n_3_maxval_1000"
    assert build_config("REL-A4", 9, 1000) == "center_singleArithmetic_shuffle_n_9_maxval_1000"
    assert build_config("REL-A3", 3, 1000) == "center_singleDistribute_Three_shuffle_n_3_maxval_1000"
    assert build_config("permutation", 3, 1000) == "center_single_shuffle_n_3_maxval_1000"
    assert build_config("REL-A5", 3, 10, p=11, depth=3) == "rpt_SpatioTemporal4_n_3_maxval_10_p_11_depth_3"
    assert build_config("REL-A7", 3, 10, conn=6, p=17) == "rpt_NeighborhoodSum_n_3_maxval_10_conn_6_p17"
    print("tasks.py: all build_config checks passed.")
