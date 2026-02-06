#!/usr/bin/env python3
"""
Build biology benchmark dataset in unified JSONL format.
Generates REL-B1 homoplasy detection questions.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any

from .random_tree import RandomTree
from .prompt_generation import (
    generate_homoplasy_llm_dataset,
    simulate_alignment_with_pyvolve,
    make_homoplasy_question,
)


def convert_taxon_names_to_indices(taxon_names: List[str], all_taxa: List[str]) -> List[int]:
    """Convert taxon names to their indices in the alignment."""
    taxa_map = {name: idx for idx, name in enumerate(all_taxa)}
    return [taxa_map[name] for name in taxon_names if name in taxa_map]


def generate_negative_example(
    newick_tree: str,
    seq_len: int = 300,
    random_seed: int = None,
    temp_seqfile: str = "simulated_alignment_neg.fasta",
) -> Dict[str, Any]:
    """
    Generate a negative example (no homoplasy) by simulating an alignment
    without injecting convergent blocks.
    """
    if random_seed is None:
        random_seed = random.randint(0, int(1e9))
    
    alignment, newick_str = simulate_alignment_with_pyvolve(
        newick_tree=newick_tree,
        seq_len=seq_len,
        seqfile=temp_seqfile,
    )
    
    question = make_homoplasy_question(newick_str, alignment, simplified_prompt=False)
    
    return {
        "tree": newick_str,
        "alignment": alignment,
        "question": question,
        "label": "no",
        "convergent_taxa": [],
    }


def convert_to_unified_format(
    example: Dict[str, Any],
    index: int,
    all_taxa: List[str],
) -> Dict[str, Any]:
    """Convert a biology example to unified JSONL format."""
    label = example.get("label", "no").lower()
    convergent_taxa_names = example.get("convergent_taxa", [])
    
    # Convert taxon names to indices
    if label == "yes" and convergent_taxa_names:
        taxa_indices = convert_taxon_names_to_indices(convergent_taxa_names, all_taxa)
    else:
        taxa_indices = []
    
    # Validate: if label is "yes", taxa must not be empty
    if label == "yes" and not taxa_indices:
        raise ValueError(f"Answer is 'yes' but taxa list is empty for record {index}")
    
    # Ensure taxa is empty list if label is "no"
    if label == "no":
        taxa_indices = []
    
    unified = {
        "id": f"biology_REL-B1_{index:05d}",
        "domain": "biology",
        "task": "REL-B1",
        "question": example.get("question", ""),
        "answer": {
            "label": label,
            "taxa": taxa_indices,
        },
        "metadata": {},
    }
    
    return unified


def main():
    ap = argparse.ArgumentParser(description="Build biology benchmark in unified JSONL format")
    ap.add_argument("--out", type=str, default="REL/biology/REL-B1.jsonl", help="Output JSONL file path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--num_yes", type=int, default=100, help="Number of 'yes' examples to generate")
    ap.add_argument("--num_no", type=int, default=100, help="Number of 'no' examples to generate")
    ap.add_argument("--seq_len", type=int, default=300, help="Sequence length")
    ap.add_argument("--num_leaves", type=int, default=50, help="Number of leaves in tree")
    ap.add_argument("--n_convergent_sites", type=int, default=5, help="Number of convergent blocks for 'yes' examples")
    ap.add_argument("--n_convergent_taxa", type=int, default=2, help="Number of convergent taxa for 'yes' examples")
    ap.add_argument("--length_convergent_block", type=int, default=5, help="Length of convergent blocks")
    ap.add_argument("--min_taxa_distance", type=int, default=3, help="Minimum distance between convergent taxa")
    
    args = ap.parse_args()
    rng = random.Random(args.seed)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    index = 0
    
    print(f"[INFO] Generating {args.num_yes} 'yes' examples and {args.num_no} 'no' examples...")
    
    # Generate 'yes' examples (with homoplasy)
    print(f"[INFO] Generating 'yes' examples...")
    yes_generated = 0
    attempts = 0
    max_attempts = args.num_yes * 10
    
    while yes_generated < args.num_yes and attempts < max_attempts:
        attempts += 1
        try:
            tree = RandomTree(args.num_leaves)
            random_seed = rng.randint(0, int(1e9))
            
            dataset = generate_homoplasy_llm_dataset(
                newick_tree=str(tree),
                seq_len=args.seq_len,
                n_convergent_sites=args.n_convergent_sites,
                random_seed=random_seed,
                n_convergent_taxa=args.n_convergent_taxa,
                min_taxa_distance=args.min_taxa_distance,
                length_convergent_block=args.length_convergent_block,
                simplified_prompt=False,
            )
            
            # Get all taxa from alignment
            all_taxa = sorted(list(dataset["alignment"].keys()))
            
            # Process each example from the dataset
            for example in dataset["examples"]:
                if yes_generated >= args.num_yes:
                    break
                
                # Add convergent_taxa to example
                example["convergent_taxa"] = dataset["convergent_taxa"]
                example["tree"] = dataset["tree"]
                example["alignment"] = dataset["alignment"]
                examples.append((example, all_taxa))
                yes_generated += 1
                
        except (ValueError, RuntimeError) as e:
            if attempts % 10 == 0:
                print(f"  [WARN] Attempt {attempts}: {e}")
            continue
    
    print(f"[INFO] Generated {yes_generated} 'yes' examples")
    
    # Generate 'no' examples (without homoplasy)
    print(f"[INFO] Generating 'no' examples...")
    no_generated = 0
    attempts = 0
    max_attempts = args.num_no * 10
    
    while no_generated < args.num_no and attempts < max_attempts:
        attempts += 1
        try:
            tree = RandomTree(args.num_leaves)
            random_seed = rng.randint(0, int(1e9))
            
            example = generate_negative_example(
                newick_tree=str(tree),
                seq_len=args.seq_len,
                random_seed=random_seed,
            )
            
            all_taxa = sorted(list(example["alignment"].keys()))
            examples.append((example, all_taxa))
            no_generated += 1
            
        except (ValueError, RuntimeError) as e:
            if attempts % 10 == 0:
                print(f"  [WARN] Attempt {attempts}: {e}")
            continue
    
    print(f"[INFO] Generated {no_generated} 'no' examples")
    
    # Shuffle examples
    rng.shuffle(examples)
    
    # Convert to unified format and write
    print(f"[INFO] Converting to unified format and writing to {out_path}...")
    count = 0
    errors = 0
    
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for example, all_taxa in examples:
            try:
                unified = convert_to_unified_format(example, index, all_taxa)
                f_out.write(json.dumps(unified, ensure_ascii=False) + '\n')
                count += 1
                index += 1
            except Exception as e:
                print(f"  [ERROR] Failed to convert example {index}: {e}")
                errors += 1
                continue
    
    print(f"[INFO] Wrote {count} records to {out_path}")
    if errors > 0:
        print(f"[WARN] {errors} records failed to convert")


if __name__ == "__main__":
    main()
