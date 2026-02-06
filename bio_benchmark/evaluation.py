"""
Evaluation functions for biology benchmarks.

Provides evaluate_response function that takes question, answer, and response
and returns a score dictionary for REL-B1 (homoplasy detection).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def parse_llm_response(response: str, num_expected_taxa: Optional[int] = None) -> Tuple[Optional[bool], List[str]]:
    """
    Parse the LLM response to extract yes/no answer and identified taxa.
    
    Args:
        response: Model response text
        num_expected_taxa: Optional number of expected taxa (for validation)
        
    Returns:
        Tuple of (said_yes: bool or None, identified_taxa: list of str)
        said_yes is None if parsing failed
    """
    response_lower = response.lower()
    
    # Check for yes/no
    has_yes = 'yes' in response_lower
    has_no = 'no' in response_lower
    
    if has_yes and has_no:
        said_yes = None
    elif has_yes:
        said_yes = True
    elif has_no:
        said_yes = False
    else:
        said_yes = None
    
    # Extract taxa - look for patterns like "taxon_1", "taxon_2", etc.
    taxa_pattern = r'taxon[_\s]?(\d+)'
    identified_taxa = re.findall(taxa_pattern, response_lower)
    
    # If no taxon_X pattern found, try to find standalone numbers
    if not identified_taxa:
        identified_taxa = re.findall(r'\b(\d+)\b', response_lower)
    
    return said_yes, identified_taxa


def calculate_taxa_metrics(ground_truth_taxa: List[Any], predicted_taxa: List[str]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for taxa identification.
    
    Args:
        ground_truth_taxa: List of ground truth taxa (can be int or str)
        predicted_taxa: List of predicted taxa (strings)
        
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    if not ground_truth_taxa:
        return {'precision': -1, 'recall': -1, 'f1': -1}
    
    # Normalize taxa names for comparison
    gt_set = set(str(t).lower() for t in ground_truth_taxa)
    pred_set = set(str(t).lower() for t in predicted_taxa)
    
    overlap = gt_set.intersection(pred_set)
    
    recall = len(overlap) / len(gt_set)
    
    if len(pred_set) == 0:
        precision = 0.0
    else:
        precision = len(overlap) / len(pred_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_response(question: str, answer: Dict[str, Any], response: str, task: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate a model response for a biology benchmark question (REL-B1).
    
    Args:
        question: The question text
        answer: Answer dict containing:
            - "label": "yes" or "no"
            - "taxa": List of integers (taxa IDs involved in homoplasy, empty if label is "no")
        response: Model response text
        task: Optional task identifier (defaults to REL-B1)
        
    Returns:
        Dictionary with evaluation results:
        - correct: bool (True if label matches and taxa match if label is "yes")
        - pred_label: str or None ("yes", "no", or None if parsing failed)
        - gold_label: str ("yes" or "no")
        - pred_taxa: list[str] (extracted taxa from response)
        - gold_taxa: list[int] (ground truth taxa)
        - precision: float (taxa precision, -1 if gold_taxa is empty)
        - recall: float (taxa recall, -1 if gold_taxa is empty)
        - f1: float (taxa F1, -1 if gold_taxa is empty)
    """
    gold_label = answer.get("label")
    gold_taxa = answer.get("taxa", [])
    
    if gold_label is None:
        return {
            "correct": False,
            "pred_label": None,
            "gold_label": None,
            "pred_taxa": [],
            "gold_taxa": gold_taxa,
            "precision": -1,
            "recall": -1,
            "f1": -1,
            "error": "Missing label in answer",
        }
    
    # Normalize gold label to lowercase
    gold_label_lower = gold_label.lower()
    if gold_label_lower not in ("yes", "no"):
        return {
            "correct": False,
            "pred_label": None,
            "gold_label": gold_label,
            "pred_taxa": [],
            "gold_taxa": gold_taxa,
            "precision": -1,
            "recall": -1,
            "f1": -1,
            "error": f"Invalid label in answer: {gold_label}",
        }
    
    # Parse response
    said_yes, identified_taxa = parse_llm_response(response)
    
    # Determine predicted label
    if said_yes is None:
        pred_label = None
    elif said_yes:
        pred_label = "yes"
    else:
        pred_label = "no"
    
    # Check if label matches
    label_correct = (pred_label is not None) and (pred_label == gold_label_lower)
    
    # Calculate taxa metrics
    if gold_label_lower == "yes":
        # If label is "yes", check taxa
        metrics = calculate_taxa_metrics(gold_taxa, identified_taxa)
        # For "yes" answers, both label and taxa must match
        # We consider it correct if label matches and there's some overlap in taxa
        # (exact match would require metrics['f1'] == 1.0, but we'll be lenient)
        correct = label_correct and (metrics['f1'] > 0 or len(gold_taxa) == 0)
    else:
        # If label is "no", taxa should be empty
        metrics = {'precision': -1, 'recall': -1, 'f1': -1}
        # For "no" answers, only label needs to match (taxa should be empty)
        correct = label_correct and len(identified_taxa) == 0
    
    return {
        "correct": correct,
        "pred_label": pred_label,
        "gold_label": gold_label_lower,
        "pred_taxa": identified_taxa,
        "gold_taxa": gold_taxa,
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1": metrics['f1'],
    }
