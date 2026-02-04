#
# Minimal prediction helpers for algebra_benchmark evaluation.
# Parse model output to get answer index (0-7); majority vote; guard answer.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

import re
from collections import Counter


def text2num(
    text: str,
    n_attr: int,
    n_return: int = 1,
    answer_queue: str = "",
) -> list[int]:
    """
    Extract answer index (0-7) from model output text.
    Expects a single number 1-8 (1-based) or 0-7 (0-based); returns 0-based index.
    n_attr and n_return are kept for compatibility with parent API; we return
    a list of length n_return (repeating the parsed index if n_return > 1).
    """
    if not text or not text.strip():
        return [0] * max(1, n_return)
    # Look for "Answer 3", "Answer #3", "3", "# 3", etc.
    text = text.strip()
    # Prefer explicit "Answer N" or "Answer #N"
    m = re.search(r"Answer\s*#?\s*(\d+)", text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        idx = (num - 1) if num >= 1 and num <= 8 else 0
        idx = max(0, min(7, idx))
        return [idx] * max(1, n_return)
    # Single number on its own line or at end
    m = re.search(r"\b([1-8])\b", text)
    if m:
        num = int(m.group(1))
        idx = num - 1
        return [idx] * max(1, n_return)
    m = re.search(r"\b([0-7])\b", text)
    if m:
        idx = int(m.group(1))
        return [idx] * max(1, n_return)
    return [0] * max(1, n_return)


def majority_vote(predictions: list[int]) -> int:
    """Return the most common value; tie-break by first occurrence."""
    if not predictions:
        return 0
    counts = Counter(predictions)
    best = max(counts.keys(), key=lambda k: (counts[k], -predictions.index(k)))
    return best


def guard_answer(x: int | list[int]) -> int | list[int]:
    """Clamp answer index to 0-7. Accept int or list of ints."""
    if isinstance(x, list):
        return [max(0, min(7, int(v))) for v in x]
    return max(0, min(7, int(x)))
