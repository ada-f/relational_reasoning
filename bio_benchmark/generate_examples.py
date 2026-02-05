import json
import random
import math
from pathlib import Path
from random_tree import RandomTree
from prompt_generation import generate_homoplasy_llm_dataset
from tqdm import tqdm

output_folder = Path("examples/")


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def sample_block_len_for_ratio(seq_length: int, rmin: float, rmax: float) -> int:
    """
    Sample an integer block length L such that L/seq_length in [rmin, rmax].
    """
    lo = max(1, math.ceil(rmin * seq_length))
    hi = min(seq_length, math.floor(rmax * seq_length))
    if lo > hi:
        raise ValueError(
            f"No feasible block length for seq_length={seq_length} with ratio range [{rmin}, {rmax}]. "
            f"Computed bounds lo={lo}, hi={hi}."
        )
    return random.randint(lo, hi)


def generate_examples_for_category(
    category_name: str,
    variable_name: str,
    variable_values: list,
    num_examples_per_value: int = 100,
    default_num_leaves: int = 50,
    default_seq_length: int = 1000,
    default_length_convergent_block: int = 100,
    default_num_taxa: int = 2,
    motif_ratio_range: tuple[float, float] | None = (0.10, 0.15),
    enforce_ratio_for_all_modes: bool = True,
):
    """
    Generate examples for a specific category, varying one parameter.

    If motif_ratio_range is not None, enforce:
        length_convergent_block / seq_length in [rmin, rmax].

    Behavior:
      - If variable_name == 'seq_length': sample a valid length_convergent_block PER EXAMPLE.
      - Otherwise: keep the chosen length_convergent_block unless enforce_ratio_for_all_modes=True,
                   in which case resample length_convergent_block to satisfy the ratio.
    """
    category_dir = output_folder / category_name
    category_dir.mkdir(parents=True, exist_ok=True)

    rmin, rmax = motif_ratio_range if motif_ratio_range is not None else (None, None)

    for value in variable_values:
        # Set parameters based on which variable we're varying
        if variable_name == "num_taxa":
            num_taxa = value
            num_leaves = default_num_leaves
            seq_length = default_seq_length
            length_convergent_block = default_length_convergent_block
        elif variable_name == "num_leaves":
            #num_taxa = default_num_taxa
            num_taxa = random.choice([3, 4, 5, 10, 15, 20, 25])  # keep small for homoplasy
            num_leaves = value
            seq_length = random.choice([500, 600, 700, 800, 900])
            length_convergent_block = default_length_convergent_block
        elif variable_name == "seq_length":
            num_taxa = random.choice([2, 3, 4, 5, 10, 15, 20, 25])  # keep small for homoplasy
            num_leaves = default_num_leaves
            seq_length = value
            length_convergent_block = default_length_convergent_block  # will be overridden per-example if ratio enforced
        elif variable_name == "length_convergent_block":
            num_taxa = default_num_taxa
            num_leaves = default_num_leaves
            seq_length = default_seq_length
            length_convergent_block = value
        else:
            raise ValueError(f"Unknown variable name: {variable_name}")

        output_file = category_dir / f"{variable_name}_{value}.jsonl"
        if output_file.exists():
            output_file.unlink()

        print(f"Generating {num_examples_per_value} examples for {variable_name}={value}...")
        generated = 0
        attempts = 0
        max_attempts = num_examples_per_value * 20  # a bit higher since we may reject/resample

        pbar = tqdm(total=num_examples_per_value, desc=f"{variable_name}={value}")

        while generated < num_examples_per_value and attempts < max_attempts:
            attempts += 1
            try:
                tree = RandomTree(num_leaves)
                random_seed = random.randint(0, int(1e9))

                # Choose a block length that respects the ratio constraint
                if motif_ratio_range is not None:
                    if variable_name == "seq_length" or variable_name == 'num_leaves':
                        # enforce ratio by sampling per example
                        length_convergent_block_use = sample_block_len_for_ratio(seq_length, rmin, rmax)
                    else:
                        # either keep fixed (and reject if out of range) or resample to enforce
                        ratio = length_convergent_block / seq_length
                        if rmin <= ratio <= rmax:
                            length_convergent_block_use = length_convergent_block
                        else:
                            if not enforce_ratio_for_all_modes:
                                # reject this attempt
                                continue
                            length_convergent_block_use = sample_block_len_for_ratio(seq_length, rmin, rmax)
                else:
                    length_convergent_block_use = length_convergent_block

                num_taxa = random.choice([3, 4, 5, 10, 15, 20, 25])  # keep small for homoplasy
                dataset = generate_homoplasy_llm_dataset(
                    newick_tree=str(tree),
                    seq_len=seq_length,
                    n_convergent_sites=1,
                    n_control_sites=0,
                    random_seed=random_seed,
                    length_convergent_block=length_convergent_block_use,
                    n_convergent_taxa=num_taxa,
                    simplified_prompt=False,
                )

                for example in dataset["examples"]:
                    if generated >= num_examples_per_value:
                        break

                    record = {
                        "example_id": generated,
                        "num_leaves": num_leaves,
                        "seq_length": seq_length,
                        "num_convergent_taxa": num_taxa,
                        "length_convergent_block": length_convergent_block_use,
                        "motif_ratio": length_convergent_block_use / seq_length,
                        "random_seed": random_seed,
                        "tree": dataset["tree"],
                        "alignment": dataset["alignment"],
                        "question": example["question"],
                        "ground_truth_taxa": example["metadata"].get("convergent_taxa", []),
                        "label": example["label"],
                        "metadata": example["metadata"],
                    }

                    append_jsonl(output_file, record)
                    generated += 1
                    pbar.update(1)

            except ValueError:
                continue

        pbar.close()
        print(f"  Generated {generated} examples (took {attempts} attempts)")


def main():
    output_folder.mkdir(parents=True, exist_ok=True)

    generate_examples_for_category(
        category_name="vary_num_leaves_ratio_fixed",
        variable_name="num_leaves",
        variable_values=[20, 30, 40],
        num_examples_per_value=100,
        motif_ratio_range=(0.05, 0.30),  
    )

    print("\n" + "=" * 60)
    print("DONE! All examples saved to 'examples/' folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
