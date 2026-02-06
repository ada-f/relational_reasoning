#
# Numerical RPM/RPT sample generators for algebra_benchmark.
# Rule generators inspired by I-RAVEN-X–style generation logic;
# adapted to numerical-only panels (list of matrices) with no confounders.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

import random
from typing import Any

from algebra_benchmark.format import validate_sample
from algebra_benchmark.tasks import RULE_TO_TASK, TASK_TO_RULE

N_CHOICES = 8
N_CONTEXT_PANELS = 8


def _make_matrix(gridsize: int, maxval: int, rng: random.Random) -> list[list[float]]:
    """Return a gridsize x gridsize matrix of floats in [0, maxval]."""
    return [
        [rng.uniform(0, maxval) for _ in range(gridsize)]
        for _ in range(gridsize)
    ]


def _copy_matrix(M: list[list[float]]) -> list[list[float]]:
    """Deep copy a 2D matrix (list of lists)."""
    return [row[:] for row in M]


# ---------------------------------------------------------------------------
# Progression (REL-A2): nine panels form a progression; delta in {-2,-1,1,2}
# ---------------------------------------------------------------------------
def generate_progression_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
) -> dict[str, Any]:
    """
    Generate one sample for the progression rule (REL-A2).
    Nine matrices M0..M8 form a progression along the grid; context = [M0..M7], answer = M8.
    Delta is in {-2, -1, 1, 2} (I-RAVEN-X–style progression).
    """
    delta = rng.choice([-2, -1, 1, 2])
    n = gridsize
    # One progression matrix gives values for one panel; we need 9 panels with values
    # M_k[row][col] = base[row] + (k * n + col) * delta, clamped to [0, maxval]
    matrices: list[list[list[float]]] = []
    base = [
        float(rng.randint(0, max(0, maxval - 9 * n * abs(delta))))
        for _ in range(n)
    ]
    for k in range(9):
        M: list[list[float]] = []
        for row in range(n):
            M.append([])
            for col in range(n):
                val = base[row] + (k * n + col) * delta
                M[-1].append(float(max(0, min(maxval, int(val)))))
        matrices.append(M)
    panels = [_copy_matrix(matrices[i]) for i in range(8)]
    answer = _copy_matrix(matrices[8])
    target = rng.randint(0, N_CHOICES - 1)
    choices = [None] * N_CHOICES
    choices[target] = answer
    for i in range(N_CHOICES):
        if choices[i] is None:
            choices[i] = _make_matrix(gridsize, maxval, rng)
    sample = {"panels": panels, "target": target, "choices": choices}
    validate_sample(sample, n_choices=N_CHOICES)
    return sample


# ---------------------------------------------------------------------------
# Distribute-three (REL-A3): first row = n distinct values, each row = roll(prev, ±1)
# ---------------------------------------------------------------------------
def _non_repeating_list(n: int, maxval: int, rng: random.Random) -> list[int]:
    """List of n distinct random integers in [0, maxval]."""
    out: list[int] = []
    while len(out) < n:
        r = rng.randint(0, maxval)
        if r not in out:
            out.append(r)
    return out


def _distribute_three_matrix(
    n: int,
    maxval: int,
    rng: random.Random,
) -> list[list[float]]:
    """One n×n matrix: first row = n distinct values; each next row = roll(prev, ±1)."""
    row0 = _non_repeating_list(n, maxval, rng)
    delta = rng.choice([-1, 1])

    def roll(row: list[int], d: int) -> list[int]:
        return [row[(i - d) % n] for i in range(n)]

    context: list[list[float]] = [[float(x) for x in row0]]
    for _ in range(n - 1):
        row0 = roll(row0, delta)
        context.append([float(x) for x in row0])
    return context


def generate_distribute_three_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
) -> dict[str, Any]:
    """
    Generate one sample for the distribute-three rule (REL-A3).
    All 8 context panels and the answer are the same matrix (cyclic row permutations).
    I-RAVEN-X–style distribute-three (cyclic row permutations).
    """
    M = _distribute_three_matrix(gridsize, maxval, rng)
    panels = [_copy_matrix(M) for _ in range(N_CONTEXT_PANELS)]
    correct = _copy_matrix(M)
    target = rng.randint(0, N_CHOICES - 1)
    choices = [None] * N_CHOICES
    choices[target] = correct
    for i in range(N_CHOICES):
        if choices[i] is None:
            choices[i] = _make_matrix(gridsize, maxval, rng)
    sample = {"panels": panels, "target": target, "choices": choices}
    validate_sample(sample, n_choices=N_CHOICES)
    return sample


# ---------------------------------------------------------------------------
# Arithmetic (REL-A4): each row has (a, b, a+b) or similar
# ---------------------------------------------------------------------------
def _arithmetic_matrix(
    n: int,
    maxval: int,
    rng: random.Random,
) -> list[list[float]]:
    """One n×n matrix where each row has (a, b, a+b) in first three cells; rest filled."""
    M: list[list[float]] = []
    half = max(1, maxval // 2)
    for _ in range(n):
        a = float(rng.randint(0, half))
        b = float(rng.randint(0, half))
        c = min(maxval, a + b)
        row = [a, b, c]
        while len(row) < n:
            row.append(float(rng.randint(0, maxval)))
        M.append(row[:n])
    return M


def generate_arithmetic_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
) -> dict[str, Any]:
    """
    Generate one sample for the arithmetic rule (REL-A4).
    All 8 context panels and the answer are the same matrix (rows satisfy a+b=c).
    I-RAVEN-X–style arithmetic (a, b, a+b per row).
    """
    M = _arithmetic_matrix(gridsize, maxval, rng)
    panels = [_copy_matrix(M) for _ in range(N_CONTEXT_PANELS)]
    correct = _copy_matrix(M)
    target = rng.randint(0, N_CHOICES - 1)
    choices = [None] * N_CHOICES
    choices[target] = correct
    for i in range(N_CHOICES):
        if choices[i] is None:
            choices[i] = _make_matrix(gridsize, maxval, rng)
    sample = {"panels": panels, "target": target, "choices": choices}
    validate_sample(sample, n_choices=N_CHOICES)
    return sample


def generate_constant_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
) -> dict[str, Any]:
    """
    Generate one sample for the constant rule (REL-A1).

    All 8 context panels are the same matrix M. The correct answer is M;
    the 7 distractors are random matrices (same shape). Choices are shuffled
    so the correct answer is not always at index 0.
    """
    M = _make_matrix(gridsize, maxval, rng)
    panels = [_copy_matrix(M) for _ in range(N_CONTEXT_PANELS)]
    correct = _copy_matrix(M)
    target = rng.randint(0, N_CHOICES - 1)
    choices = [None] * N_CHOICES
    choices[target] = correct
    for i in range(N_CHOICES):
        if choices[i] is None:
            choices[i] = _make_matrix(gridsize, maxval, rng)
    sample = {"panels": panels, "target": target, "choices": choices}
    validate_sample(sample, n_choices=N_CHOICES)
    return sample


def generate_sample(
    task: str,
    gridsize: int,
    maxval: int,
    rng: random.Random,
) -> dict[str, Any]:
    """
    Generate one sample for the given task. For tasks without a generator,
    returns a placeholder sample.
    """
    tid = task if task.startswith("REL-") else RULE_TO_TASK.get(task)
    rule = TASK_TO_RULE.get(tid, task) if tid else task

    if rule == "constant" or tid == "REL-A1":
        return generate_constant_sample(gridsize, maxval, rng)
    if rule == "progression" or tid == "REL-A2":
        return generate_progression_sample(gridsize, maxval, rng)
    if rule in ("distribute-three", "permutation") or tid == "REL-A3":
        return generate_distribute_three_sample(gridsize, maxval, rng)
    if rule == "arithmetic" or tid == "REL-A4":
        return generate_arithmetic_sample(gridsize, maxval, rng)
    return generate_placeholder_sample(gridsize, maxval, rng)


def generate_placeholder_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
) -> dict[str, Any]:
    """
    Generate one placeholder sample (valid structure, simple numeric content).
    Used for tasks that do not yet have a rule-specific generator.
    """
    M = _make_matrix(gridsize, maxval, rng)
    panels = [_copy_matrix(M) for _ in range(N_CONTEXT_PANELS)]
    target = rng.randint(0, N_CHOICES - 1)
    choices = [None] * N_CHOICES
    choices[target] = _copy_matrix(M)
    for i in range(N_CHOICES):
        if choices[i] is None:
            choices[i] = _make_matrix(gridsize, maxval, rng)
    sample = {"panels": panels, "target": target, "choices": choices}
    validate_sample(sample, n_choices=N_CHOICES)
    return sample


def generate_dataset(
    task: str,
    num_samples: int,
    gridsize: int,
    maxval: int = 1000,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate a list of samples for the given task. Uses seed for reproducibility.
    Tasks without a rule-specific generator get placeholder samples.
    """
    rng = random.Random(seed)
    return [
        generate_sample(task, gridsize, maxval, rng)
        for _ in range(num_samples)
    ]


if __name__ == "__main__":
    rng = random.Random(42)
    for task_id, name in [("REL-A1", "constant"), ("REL-A2", "progression"), ("REL-A3", "distribute-three"), ("REL-A4", "arithmetic")]:
        s = generate_sample(task_id, 3, 1000, rng)
        assert len(s["panels"]) == 8 and len(s["choices"]) == 8
        assert 0 <= s["target"] < 8
    data = generate_dataset("REL-A1", 5, 3, 1000, seed=42)
    assert len(data) == 5
    print("generators.py: constant, progression, distribute-three, arithmetic checks passed.")
