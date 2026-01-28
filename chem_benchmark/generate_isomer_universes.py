#!/usr/bin/env python3
"""
Generate constitutional isomer universes using surge.
Filters formulas to keep only those with ≤100 isomers.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def count_isomers_with_surge(formula: str, surge_path: str = "./surge-linux-v1.0") -> Optional[Tuple[int, int]]:
    """
    Use surge -u to count isomers for a given formula.
    Returns (count, DBE) or None if surge fails.
    """
    try:
        result = subprocess.run(
            [surge_path, "-u", formula],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Parse output like: "C4OH8  H=8 C=4 O=1  nv=5 edges=4-5 DBE=1 maxd=4 maxc=4"
        # and ">Z generated 8 -> 18 -> 26 in 0.00 sec"
        # Note: surge writes to stderr, not stdout
        output = result.stderr.strip()

        # Extract DBE
        dbe_match = re.search(r'DBE=(\d+)', output)
        if not dbe_match:
            return None
        dbe = int(dbe_match.group(1))

        # Extract final count from ">Z generated ... -> COUNT in"
        count_match = re.search(r'>Z generated.*-> (\d+) in', output)
        if not count_match:
            return None
        count = int(count_match.group(1))

        return (count, dbe)

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
        print(f"  Error counting {formula}: {e}")
        return None


def generate_smiles_with_surge(formula: str, surge_path: str = "./surge-linux-v1.0") -> List[str]:
    """
    Use surge -S to generate all SMILES for a given formula.
    """
    try:
        result = subprocess.run(
            [surge_path, "-S", formula],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse SMILES output
        # Surge outputs:
        # Line 1: Formula info (e.g., "C3OH8  H=8 C=3 O=1  nv=4 edges=3-3 DBE=0...")
        # Line 2: Generation stats (e.g., ">Z wrote 2 -> 3 -> 3 in 0.00 sec")
        # Lines 3+: SMILES strings (one per line)

        smiles_list = []
        # Combine stdout and stderr since surge may use either
        output_text = (result.stdout + result.stderr).strip()

        lines = output_text.split('\n')
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip the formula/header lines (contain "=" followed by digit, or start with ">")
            if re.search(r'=\d', line) or line.startswith('>'):
                continue
            # If line starts with typical SMILES elements (C, O, N, S, Br, Cl, F, I, P, B)
            # and doesn't contain spaces (SMILES don't have spaces)
            if ' ' not in line and line and line[0] in 'CONSPBFIH':
                smiles_list.append(line)

        return smiles_list

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"  Error generating SMILES for {formula}: {e}")
        return []


def generate_candidate_formulas() -> List[str]:
    """
    Generate a diverse list of candidate molecular formulas.
    Includes various combinations of C, H, N, O, S, F, Cl, Br.
    Focuses on higher DBE (more unsaturated/interesting molecules).
    """
    formulas = []

    # CxHyOz compounds - full range including highly unsaturated
    for c in range(3, 10):
        for o in range(1, min(c, 4)):
            # Start from very low H (high DBE) to saturated
            min_h = max(2, c - o)  # Allow highly unsaturated molecules
            max_h = min(2*c + 2*o + 2, 22)
            for h in range(min_h, max_h):
                formulas.append(f"C{c}H{h}O{o}")

    # CxHyNz compounds - including imines, nitriles, etc
    for c in range(3, 9):
        for n in range(1, min(c, 5)):
            min_h = max(1, c - n)  # Very unsaturated (e.g., nitriles)
            max_h = min(2*c + n + 4, 22)
            for h in range(min_h, max_h):
                formulas.append(f"C{c}H{h}N{n}")

    # CxHyOzNw compounds (amino acids, amides, unsaturated)
    for c in range(3, 8):
        for o in range(1, min(c, 3)):
            for n in range(1, min(c, 3)):
                min_h = max(2, c - o - n)  # Highly unsaturated
                max_h = min(2*c + 2*o + n + 2, 20)
                for h in range(min_h, max_h):
                    formulas.append(f"C{c}H{h}O{o}N{n}")

    # CxHyS compounds (thiols, thioethers, thiocarbonyls)
    for c in range(3, 9):
        for s in range(1, min(c, 3)):
            min_h = max(2, c - s)  # Unsaturated sulfur compounds
            max_h = min(2*c + 2*s + 2, 20)
            for h in range(min_h, max_h):
                formulas.append(f"C{c}H{h}S{s}")

    # CxHyOzS compounds
    for c in range(3, 6):
        for o in range(1, 3):
            for s in range(1, 2):
                for h in range(6, min(2*c + 2*o + 2*s + 2, 16)):
                    formulas.append(f"C{c}H{h}O{o}S{s}")

    # Halogenated compounds - single halogen
    for c in range(3, 6):
        for hal_elem in ["F", "Cl", "Br"]:
            for hal_count in range(1, 4):
                for h in range(3, min(2*c - hal_count + 2, 14)):
                    if hal_count == 1:
                        formulas.append(f"C{c}H{h}{hal_elem}")
                    else:
                        formulas.append(f"C{c}H{h}{hal_elem}{hal_count}")

    # Halogenated with oxygen
    for c in range(3, 5):
        for o in range(1, 2):
            for hal_elem in ["F", "Cl"]:
                for h in range(3, min(2*c + 2*o - 1, 12)):
                    formulas.append(f"C{c}H{h}O{o}{hal_elem}")

    # CxHyNzSw compounds
    for c in range(3, 5):
        for n in range(1, 2):
            for s in range(1, 2):
                for h in range(5, min(2*c + n + 2*s + 2, 14)):
                    formulas.append(f"C{c}H{h}N{n}S{s}")

    # Mixed with multiple heteroatoms
    for c in range(4, 6):
        for n in range(1, 2):
            for o in range(1, 2):
                for f in range(1, 3):
                    for h in range(3, min(2*c + 2*o + n - f + 2, 12)):
                        formulas.append(f"C{c}H{h}O{o}N{n}F{f}")

    return list(set(formulas))  # Remove duplicates


def main():
    surge_path = "./surge-linux-v1.0"
    max_isomers = 100
    output_path = Path("data/isomer_universes.json")

    print(f"Generating candidate formulas...")
    candidate_formulas = generate_candidate_formulas()
    print(f"Generated {len(candidate_formulas)} candidate formulas\n")

    print(f"Filtering formulas (keeping those with ≤{max_isomers} isomers)...")
    valid_formulas = {}

    for i, formula in enumerate(candidate_formulas, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(candidate_formulas)}...")

        result = count_isomers_with_surge(formula, surge_path)
        if result is None:
            continue

        count, dbe = result
        if 5 <= count <= max_isomers:  # At least 5 for usefulness
            valid_formulas[formula] = {"count": count, "dbe": dbe}
            if count >= 25:  # Good for n=20 sampling
                print(f"  ✓ {formula}: {count} isomers (DBE={dbe})")

    print(f"\nFound {len(valid_formulas)} valid formulas")
    print(f"\nGenerating SMILES for all valid formulas...")

    isomer_universes = {}
    for i, (formula, metadata) in enumerate(sorted(valid_formulas.items()), 1):
        print(f"  [{i}/{len(valid_formulas)}] Generating {formula} ({metadata['count']} isomers, DBE={metadata['dbe']})...")

        smiles_list = generate_smiles_with_surge(formula, surge_path)

        if len(smiles_list) != metadata['count']:
            print(f"    ⚠ Expected {metadata['count']} but got {len(smiles_list)} SMILES")

        isomer_universes[formula] = {
            "smiles": smiles_list,
            "count": len(smiles_list),
            "dbe": metadata['dbe']
        }

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(isomer_universes, f, indent=2)

    print(f"\n✓ Saved {len(isomer_universes)} isomer universes to {output_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total formulas: {len(isomer_universes)}")
    total_isomers = sum(u['count'] for u in isomer_universes.values())
    print(f"  Total isomers: {total_isomers}")

    # Show formulas suitable for different n values
    for min_size in [5, 10, 20, 30, 50]:
        suitable = [f for f, u in isomer_universes.items() if u['count'] >= min_size + 5]
        print(f"  Formulas with ≥{min_size+5} isomers (suitable for n={min_size}): {len(suitable)}")


if __name__ == "__main__":
    main()
