#!/usr/bin/env python3
"""
Update Q2 prompts in the dataset to be more explicit about checking molecular formulas.
Keeps all molecules and other task prompts unchanged.
"""
import json
from pathlib import Path
from typing import Sequence


def _format_smiles_list(smiles: Sequence[str]) -> str:
    lines = []
    for i, s in enumerate(smiles, start=1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def build_q2_prompt_v2(smiles: Sequence[str]) -> str:
    """
    More explicit Q2 prompt that emphasizes checking molecular formulas.
    """
    return (
        "Check if ALL molecules in this list are constitutional isomers of each other.\n\n"
        "Constitutional isomers are molecules that:\n"
        "1. Have the SAME molecular formula (e.g., all C4H8O)\n"
        "2. Have DIFFERENT connectivity (atom arrangements)\n\n"
        "SMILES:\n"
        f"{_format_smiles_list(smiles)}\n\n"
        "Answer <Yes> if ALL molecules have the same molecular formula.\n"
        "Answer <No> if ANY molecule has a different molecular formula.\n\n"
        "Return exactly one of:\n"
        "<Yes>\n"
        "or\n"
        "<No>\n"
        "No explanation."
    )


def main():
    input_path = Path("data/dataset_v2.jsonl")
    output_path = Path("data/dataset_v3.jsonl")

    print(f"Reading dataset from: {input_path}")

    updated_count = 0
    total_count = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            instance = json.loads(line)
            total_count += 1

            # Only update Q2 prompts
            if instance['task'] == 'q2_isomer_set_yes_no':
                # Regenerate prompt with new version
                new_prompt = build_q2_prompt_v2(instance['molecules'])
                instance['prompt'] = new_prompt
                updated_count += 1

            # Write to output (Q2 with new prompt, others unchanged)
            fout.write(json.dumps(instance) + '\n')

    print(f"Total instances: {total_count}")
    print(f"Updated Q2 prompts: {updated_count}")
    print(f"Saved to: {output_path}")

    # Show example of old vs new prompt
    print("\n" + "=" * 80)
    print("EXAMPLE PROMPT COMPARISON")
    print("=" * 80)

    # Get first Q2 instance from each file
    with open(input_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            if instance['task'] == 'q2_isomer_set_yes_no':
                print("\nOLD PROMPT:")
                print(instance['prompt'])
                break

    with open(output_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            if instance['task'] == 'q2_isomer_set_yes_no':
                print("\nNEW PROMPT:")
                print(instance['prompt'])
                break


if __name__ == '__main__':
    main()
