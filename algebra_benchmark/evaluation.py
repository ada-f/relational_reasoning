"""
Evaluation functions for algebra benchmarks.

Provides evaluate_response function that takes question, answer, and response
and returns a score dictionary for REL-A1 through REL-A7 tasks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .solver_pred import guard_answer, majority_vote, text2num


def evaluate_response(
    question: str,
    answer: Dict[str, Any],
    response: str,
    task: Optional[str] = None,
    *,
    n_attr: int = 1,
    n_return: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate a model response for an algebra benchmark question.
    
    Args:
        question: The question text
        answer: Answer dict containing "target" (int, 0-based index of correct answer)
        response: Model response text
        task: Optional task identifier (REL-A1 through REL-A7)
        n_attr: Number of attributes (for compatibility with text2num, default 1)
        n_return: Number of return values (for compatibility with text2num, default 1)
        
    Returns:
        Dictionary with evaluation results:
        - correct: bool
        - pred: int (predicted answer index, 0-7)
        - gold: int (correct answer index, 0-7)
    """
    gold_target = answer.get("target")
    
    if gold_target is None:
        return {
            "correct": False,
            "pred": None,
            "gold": None,
            "error": "Missing target in answer",
        }
    
    if not isinstance(gold_target, int):
        return {
            "correct": False,
            "pred": None,
            "gold": gold_target,
            "error": f"Target must be an integer, got {type(gold_target)}",
        }
    
    # Extract predictions from response
    preds = text2num(response, n_attr=n_attr, n_return=n_return)
    
    # Get majority vote prediction
    pred_idx = majority_vote(preds)
    
    # Guard answer (clamp to 0-7)
    pred_idx = guard_answer(pred_idx)
    
    # Ensure pred_idx is an int (guard_answer might return list)
    if isinstance(pred_idx, list):
        pred_idx = pred_idx[0] if pred_idx else 0
    
    # Check if correct
    correct = (pred_idx == gold_target)
    
    return {
        "correct": correct,
        "pred": pred_idx,
        "gold": gold_target,
    }
