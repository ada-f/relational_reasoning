#!/usr/bin/env python3
"""
Test script for build_benchmark.py
Generates 3 'yes' and 3 'no' examples and validates format.

To run this test:
    From parent directory: python -m bio_benchmark.test_build_benchmark
    Or: cd .. && python -m bio_benchmark.test_build_benchmark
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
    from bio_benchmark.build_benchmark import (
        convert_to_unified_format,
        generate_negative_example,
        convert_taxon_names_to_indices,
    )
    from bio_benchmark.prompt_generation import generate_homoplasy_llm_dataset
    from bio_benchmark.random_tree import RandomTree
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("\nPlease run this test from the parent directory using:")
    print("  python -m bio_benchmark.test_build_benchmark")
    print("\nOr ensure you're in the relational_reasoning directory and run:")
    print("  python -m bio_benchmark.test_build_benchmark")
    sys.exit(1)


def validate_unified_format(record):
    """Validate that a record matches the unified format specification."""
    errors = []
    
    # Required fields
    required_fields = ["id", "domain", "task", "question", "answer"]
    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    
    # Check domain
    if record.get("domain") != "biology":
        errors.append(f"Expected domain='biology', got '{record.get('domain')}'")
    
    # Check task
    if record.get("task") != "REL-B1":
        errors.append(f"Expected task='REL-B1', got '{record.get('task')}'")
    
    # Check answer structure
    answer = record.get("answer", {})
    if not isinstance(answer, dict):
        errors.append("Answer must be a dictionary")
    
    # Check required answer fields
    if "label" not in answer:
        errors.append("Answer must contain 'label' field")
    if "taxa" not in answer:
        errors.append("Answer must contain 'taxa' field")
    
    # Validate label
    label = answer.get("label")
    if label not in ["yes", "no"]:
        errors.append(f"Label must be 'yes' or 'no', got '{label}'")
    
    # Validate taxa
    taxa = answer.get("taxa", [])
    if not isinstance(taxa, list):
        errors.append("Taxa must be a list")
    elif not all(isinstance(t, int) for t in taxa):
        errors.append("All taxa must be integers")
    
    # Validation rules: if label="yes", taxa must be non-empty; if label="no", taxa must be empty
    if label == "yes" and not taxa:
        errors.append("If label is 'yes', taxa must be non-empty")
    if label == "no" and taxa:
        errors.append("If label is 'no', taxa must be empty")
    
    return errors


def test_generate_samples():
    """Generate 3 'yes' and 3 'no' samples and validate format."""
    print("=" * 60)
    print("Testing Biology Benchmark Generation")
    print("=" * 60)
    
    rng = random.Random(42)
    results = {
        "yes": [],
        "no": [],
    }
    
    # Generate 'yes' examples
    print("\n[TEST] Generating 3 'yes' examples...")
    for i in range(3):
        try:
            tree = RandomTree(30)  # Smaller tree for testing
            random_seed = rng.randint(0, int(1e9))
            
            dataset = generate_homoplasy_llm_dataset(
                newick_tree=str(tree),
                seq_len=200,  # Shorter for testing
                n_convergent_sites=3,
                random_seed=random_seed,
                n_convergent_taxa=3,
                min_taxa_distance=2,
                length_convergent_block=5,
                simplified_prompt=False,
            )
            
            # Get all taxa from alignment
            all_taxa = sorted(list(dataset["alignment"].keys()))
            
            # Use first example from dataset
            if dataset["examples"]:
                example = dataset["examples"][0]
                example["convergent_taxa"] = dataset["convergent_taxa"]
                example["tree"] = dataset["tree"]
                example["alignment"] = dataset["alignment"]
                
                # Convert to unified format
                unified = convert_to_unified_format(example, i, all_taxa)
                
                # Validate
                errors = validate_unified_format(unified)
                if errors:
                    print(f"  [ERROR] Sample {i} validation failed:")
                    for error in errors:
                        print(f"    - {error}")
                else:
                    print(f"  [OK] Sample {i} validated successfully")
                    results["yes"].append(unified)
            else:
                print(f"  [ERROR] No examples generated for sample {i}")
                
        except Exception as e:
            print(f"  [ERROR] Failed to generate 'yes' sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate 'no' examples
    print("\n[TEST] Generating 3 'no' examples...")
    for i in range(3):
        try:
            tree = RandomTree(30)
            random_seed = rng.randint(0, int(1e9))
            
            example = generate_negative_example(
                newick_tree=str(tree),
                seq_len=200,
                random_seed=random_seed,
            )
            
            all_taxa = sorted(list(example["alignment"].keys()))
            
            # Convert to unified format
            unified = convert_to_unified_format(example, i + 3, all_taxa)
            
            # Validate
            errors = validate_unified_format(unified)
            if errors:
                print(f"  [ERROR] Sample {i} validation failed:")
                for error in errors:
                    print(f"    - {error}")
            else:
                print(f"  [OK] Sample {i} validated successfully")
                results["no"].append(unified)
                
        except Exception as e:
            print(f"  [ERROR] Failed to generate 'no' sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"'yes' examples: {len(results['yes'])}/3 generated and validated")
    print(f"'no' examples: {len(results['no'])}/3 generated and validated")
    
    # Write test output to temporary file
    with NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for sample in results["yes"] + results["no"]:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"\n[INFO] Test output written to: {f.name}")
    
    # Check if all tests passed
    all_passed = len(results["yes"]) == 3 and len(results["no"]) == 3
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(test_generate_samples())
