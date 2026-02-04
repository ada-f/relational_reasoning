#
# Numerical RPM/RPT: build context and answer_choices text from a sample.
# No visual inputs; panels and choices are numerical matrices/tensors only.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

from typing import Any


def _format_panel(panel: Any) -> str:
    """Format a single panel (2D or 3D list of numbers) as a string for the prompt."""
    if isinstance(panel, list) and len(panel) > 0:
        if isinstance(panel[0], (int, float)):
            return str(panel)
        if isinstance(panel[0], list):
            return "\n".join(str(row) for row in panel)
        # 3D: list of 2D slices
        return "\n---\n".join(_format_panel(slice_) for slice_ in panel)
    return str(panel)


def sample_to_context(sample: dict[str, Any]) -> str:
    """
    Build the context string for the prompt: the 8 context panels.
    Format: "Panel 0:\n<matrix>\nPanel 1:\n<matrix>\n..."
    """
    panels = sample["panels"]
    parts = []
    for i, p in enumerate(panels):
        parts.append(f"Panel {i}:\n{_format_panel(p)}")
    return "\n\n".join(parts)


def sample_to_answer_choices(sample: dict[str, Any]) -> str:
    """
    Build the answer-choices string for the prompt: the 8 candidate panels.
    Format: "Answer 1: <matrix>\nAnswer 2: <matrix>\n..."
    """
    choices = sample["choices"]
    parts = []
    for i, c in enumerate(choices):
        parts.append(f"Answer {i + 1}: {_format_panel(c)}")
    return "\n".join(parts)


def build_query(
    sample: dict[str, Any],
    *,
    prefix: str = "Only return the missing panel index (1-8)!\n",
    incontext: str = "",
) -> str:
    """
    Build the full query string: prefix + optional in-context examples + context + answer set.
    The model is expected to return a single number 1-8 (1-based answer index).
    """
    context = sample_to_context(sample)
    answer_choices = sample_to_answer_choices(sample)
    out = prefix
    if incontext:
        out += incontext
    out += context
    out += "\n\nAnswer set:\n" + answer_choices
    return out


def get_choices(sample: dict[str, Any]) -> list[Any]:
    """Return the list of 8 choice panels (for scoring: choice_array[pred_idx])."""
    return sample["choices"]
