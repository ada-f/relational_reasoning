#!/usr/bin/env python3
"""
Test script for build_benchmark.py
Generates 3 questions of each type (REL-A1, REL-A2, REL-A3, REL-A4) and validates format.

To run this test:
    From parent directory: python -m algebra_benchmark.test_build_benchmark
    Or: cd .. && python -m algebra_benchmark.test_build_benchmark
"""

import json
import random
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# Ensure parent directory is in path for package imports
parent_dir = Path(__file__).parent.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import from package
try:
    from algebra_benchmark.build_benchmark import convert_to_unified_format
    from algebra_benchmark.generators import generate_dataset
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("\nPlease run this test from the parent directory using:")
    print("  python -m algebra_benchmark.test_build_benchmark")
    print("\nOr ensure you're in the relational_reasoning directory and run:")
    print("  python -m algebra_benchmark.test_build_benchmark")
    sys.exit(1)


def validate_unified_format(record, expected_task):
    """Validate that a record matches the unified format specification."""
    errors = []
    
    # Required fields
    required_fields = ["id", "domain", "task", "question", "answer"]
    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    
    # Check domain
    if record.get("domain") != "algebra":
        errors.append(f"Expected domain='algebra', got '{record.get('domain')}'")
    
    # Check task
    if record.get("task") != expected_task:
        errors.append(f"Expected task='{expected_task}', got '{record.get('task')}'")
    
    # Check answer structure
    answer = record.get("answer", {})
    if not isinstance(answer, dict):
        errors.append("Answer must be a dictionary")
    
    # Check required answer fields
    if "target" not in answer:
        errors.append("Answer must contain 'target' field")
    
    target = answer.get("target")
    if not isinstance(target, int):
        errors.append("Target must be an integer")
    elif target < 0 or target >= 8:
        errors.append(f"Target must be in range [0, 7], got {target}")
    
    # Check metadata
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append("Metadata must be a dictionary")
    
    if "panels" not in metadata:
        errors.append("Metadata must contain 'panels' field")
    if "choices" not in metadata:
        errors.append("Metadata must contain 'choices' field")
    
    # Check question format
    question = record.get("question", "")
    if not question:
        errors.append("Question must not be empty")
    if "Panel" not in question and "Answer" not in question:
        errors.append("Question should contain 'Panel' and 'Answer' sections")
    
    return errors


def test_generate_samples():
    """Generate 3 samples of each task type and validate format."""
    print("=" * 60)
    print("Testing Algebra Benchmark Generation")
    print("=" * 60)
    
    tasks = ["REL-A1", "REL-A2", "REL-A3", "REL-A4"]
    results = {task: [] for task in tasks}
    
    random.seed(42)
    
    for task in tasks:
        print(f"\n[TEST] Generating 3 {task} samples...")
        
        try:
            # Generate 3 samples for this task
            samples = generate_dataset(
                task,
                3,
                gridsize=3,
                maxval=1000,
                seed=42,
            )
            
            if len(samples) < 3:
                print(f"  [WARN] Only generated {len(samples)} samples, expected 3")
            
            # Convert and validate each sample
            for i, sample in enumerate(samples[:3]):  # Take first 3
                try:
                    unified = convert_to_unified_format(sample, task, i)
                    
                    # Validate
                    errors = validate_unified_format(unified, task)
                    if errors:
                        print(f"  [ERROR] Sample {i} validation failed:")
                        for error in errors:
                            print(f"    - {error}")
                    else:
                        print(f"  [OK] Sample {i} validated successfully")
                        results[task].append(unified)
                        
                except Exception as e:
                    print(f"  [ERROR] Failed to convert {task} sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"  [ERROR] Failed to generate {task} samples: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for task in tasks:
        print(f"{task}: {len(results[task])}/3 samples generated and validated")
    
    # Write test output to temporary file
    with NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for task_samples in results.values():
            for sample in task_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"\n[INFO] Test output written to: {f.name}")
    
    # Check if all tests passed
    all_passed = all(len(samples) == 3 for samples in results.values())
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(test_generate_samples())
