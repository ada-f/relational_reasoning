from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdFMCS

from .molecule_bank import BankIndex, MoleculeRecord
from .solvers import solve_q1_largest_common_motif, solve_q2_is_constitutional_isomer_set, solve_q3_missing_isomers
from .rdkit_utils import canonical_smiles, canonical_smiles_from_smiles, mol_from_smiles, mol_formula


@dataclass(frozen=True)
class BenchmarkInstance:
    id: str
    task: str
    n_molecules: int
    molecules: List[str]
    prompt: str
    answer: Dict[str, Any]
    metadata: Dict[str, Any]


def _format_smiles_list(smiles: Sequence[str]) -> str:
    lines = []
    for i, s in enumerate(smiles, start=1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def verify_motif_in_all_molecules(motif_smiles: str, molecules: Sequence[str]) -> bool:
    """
    Verify that the motif is actually a substructure of all molecules.
    Returns True if valid, False otherwise.
    """
    motif_mol = mol_from_smiles(motif_smiles)
    if motif_mol is None:
        return False

    for mol_smiles in molecules:
        mol = mol_from_smiles(mol_smiles)
        if mol is None:
            return False

        # Check if motif is a substructure
        if not mol.HasSubstructMatch(motif_mol):
            return False

    return True


def build_q1_prompt(smiles: Sequence[str]) -> str:
    # Mirrors your Q1 description: "largest continuous common chemical motif" and answer as <smiles>.
    return (
        "Given the following list of SMILES, what is the largest *connected* common chemical motif "
        "(maximum common substructure) present in every molecule?\n"
        "Rules:\n"
        "- The motif must be a single connected fragment.\n"
        "- Do NOT tautomerize molecules.\n"
        "- Ignore stereochemistry unless it is explicitly encoded and required.\n\n"
        "SMILES:\n"
        f"{_format_smiles_list(smiles)}\n\n"
        "Return your final answer as a single SMILES wrapped exactly like:\n"
        "<smiles>YOUR_SMILES_HERE</smiles>\n"
        "No explanation."
    )


def build_q2_prompt(smiles: Sequence[str]) -> str:
    return (
        "Is this list of molecules a set of *constitutional isomers* (same molecular formula, different connectivity)?\n\n"
        "SMILES:\n"
        f"{_format_smiles_list(smiles)}\n\n"
        "Return exactly one of:\n"
        "<Yes>\n"
        "or\n"
        "<No>\n"
        "No explanation."
    )


def build_q3_prompt(given: Sequence[str]) -> str:
    return (
        "Given the following list of constitutional isomers, complete the set by identifying the missing constitutional isomers.\n\n"
        "Given SMILES:\n"
        f"{_format_smiles_list(given)}\n\n"
        "Return the missing molecules as SMILES, one per line, each wrapped exactly like:\n"
        "<smiles>YOUR_SMILES_HERE</smiles>\n"
        "No explanation."
    )


def generate_q1a_instance(
    *,
    instance_id: str,
    bank_index: BankIndex,
    n_molecules: int,
    rng,
    min_mcs_atoms: int = 8,
    max_attempts: int = 200,
    mcs_timeout_s: int = 15,
) -> BenchmarkInstance:
    """
    Q1a: Similarity-based sampling from ChEMBL molecules.
    Uses Tanimoto similarity (0.35-0.90) to find related molecules.

    Suitable for smaller n (5-20) where diverse molecules can still share meaningful motifs.
    """
    # Adjust parameters for larger molecule sets
    adaptive_timeout = max(mcs_timeout_s, n_molecules * 2)

    # Scale min_mcs_atoms for larger n
    if n_molecules <= 10:
        adaptive_min_atoms = min_mcs_atoms
    elif n_molecules <= 20:
        adaptive_min_atoms = min_mcs_atoms
    else:
        adaptive_min_atoms = min_mcs_atoms

    # More attempts for larger n
    adaptive_max_attempts = max_attempts if n_molecules <= 10 else max_attempts * 2

    for attempt in range(adaptive_max_attempts):
        group = bank_index.sample_similar_group(n_molecules, rng=rng, force_scaffold_sampling=False)
        mcs = solve_q1_largest_common_motif(group, timeout_s=adaptive_timeout)
        if mcs is None:
            continue
        if mcs.num_atoms < adaptive_min_atoms:
            continue

        # Verify that the motif is actually present in all molecules
        motif_smiles = mcs.motif_smiles
        num_atoms = mcs.num_atoms
        num_bonds = mcs.num_bonds
        corrected = False

        if not verify_motif_in_all_molecules(motif_smiles, group):
            continue
           
        # Check if (possibly corrected) motif still meets minimum size
        if num_atoms < adaptive_min_atoms:
            continue

        prompt = build_q1_prompt(group)
        return BenchmarkInstance(
            id=instance_id,
            task="q1a_largest_common_motif_chembl",
            n_molecules=n_molecules,
            molecules=list(group),
            prompt=prompt,
            answer={"smiles": motif_smiles},
            metadata={
                "mcs_num_atoms": num_atoms,
                "mcs_num_bonds": num_bonds,
                "attempt": attempt,
                "sampling_strategy": "similarity",
                "adaptive_min_atoms": adaptive_min_atoms,
                "corrected": corrected,
            },
        )
    raise RuntimeError(f"Failed to generate a valid Q1a instance after {adaptive_max_attempts} attempts (min_mcs_atoms={adaptive_min_atoms})")


def generate_q1b_instance(
    *,
    instance_id: str,
    bank_index: BankIndex,
    n_molecules: int,
    rng,
    min_mcs_atoms: int = 6,
    max_attempts: int = 200,
    mcs_timeout_s: int = 15,
) -> BenchmarkInstance:
    """
    Q1b: Scaffold-based sampling from ChEMBL molecules.
    Samples molecules sharing a common Murcko scaffold.

    Suitable for larger n (20-50) where scaffold ensures meaningful shared core.
    Default min_mcs_atoms=6 because scaffolds typically give 5-10 atom cores.
    """
    # Adjust parameters for larger molecule sets
    if n_molecules <= 20:
        adaptive_timeout = max(30, n_molecules * 2)
    else:
        adaptive_timeout = max(60, n_molecules * 3)

    # Scale min_mcs_atoms: scaffold core size varies by family
    if n_molecules <= 20:
        adaptive_min_atoms = min_mcs_atoms  # Expect 6+ atoms
    elif n_molecules <= 35:
        adaptive_min_atoms = max(5, min_mcs_atoms - 1)
    else:  # n > 35
        adaptive_min_atoms = 5  # Large families may have smaller cores

    # More attempts for larger n
    if n_molecules <= 20:
        adaptive_max_attempts = max_attempts
    else:
        adaptive_max_attempts = max_attempts * 2

    for attempt in range(adaptive_max_attempts):
        group = bank_index.sample_similar_group(n_molecules, rng=rng, force_scaffold_sampling=True)
        mcs = solve_q1_largest_common_motif(group, timeout_s=adaptive_timeout)
        if mcs is None:
            continue
        if mcs.num_atoms < adaptive_min_atoms:
            continue

        # Verify that the motif is actually present in all molecules
        motif_smiles = mcs.motif_smiles
        num_atoms = mcs.num_atoms
        num_bonds = mcs.num_bonds
        corrected = False

        if not verify_motif_in_all_molecules(motif_smiles, group):
            continue

        # Check if (possibly corrected) motif still meets minimum size
        if num_atoms < adaptive_min_atoms:
            continue

        prompt = build_q1_prompt(group)
        return BenchmarkInstance(
            id=instance_id,
            task="q1b_largest_common_motif_scaffold",
            n_molecules=n_molecules,
            molecules=list(group),
            prompt=prompt,
            answer={"smiles": motif_smiles},
            metadata={
                "mcs_num_atoms": num_atoms,
                "mcs_num_bonds": num_bonds,
                "attempt": attempt,
                "sampling_strategy": "scaffold",
                "adaptive_min_atoms": adaptive_min_atoms,
                "corrected": corrected,
            },
        )
    raise RuntimeError(f"Failed to generate a valid Q1b instance after {adaptive_max_attempts} attempts (min_mcs_atoms={adaptive_min_atoms})")


def generate_q2_instance(
    *,
    instance_id: str,
    universe_by_formula: Dict[str, List[str]],
    n_molecules: int,
    rng,
    want_yes: bool = True,
) -> BenchmarkInstance:
    """
    Generates a Q2 instance. For 'Yes': sample N from one formula universe.
    For 'No': sample N-1 from one formula universe and 1 from a different formula universe.
    """
    formulas = [f for f, u in universe_by_formula.items() if len(u) >= n_molecules]
    if not formulas:
        raise ValueError(f"No formula universe has >= {n_molecules} isomers")

    if want_yes:
        f = formulas[rng.randrange(len(formulas))]
        u = universe_by_formula[f]
        chosen = rng.sample(u, n_molecules)
        label = solve_q2_is_constitutional_isomer_set(chosen)
        assert label == "Yes", f"Failed to solve Q2 instance {instance_id} for formula {f}"
        # Should be Yes; if not, fallback to strict label from solver
        if label is None:
            label = "No"
        prompt = build_q2_prompt(chosen)
        return BenchmarkInstance(
            id=instance_id,
            task="q2_isomer_set_yes_no",
            n_molecules=n_molecules,
            molecules=chosen,
            prompt=prompt,
            answer={"label": label},
            metadata={"formula": f, "constructed_label": "Yes"},
        )

    # No case: mix formulas but keep N the same
    # Pick a base formula for N-1, and a different formula for 1 molecule
    f1 = formulas[rng.randrange(len(formulas))]
    u1 = universe_by_formula[f1]
    base = rng.sample(u1, n_molecules - 1)  # sample WITHOUT replacement
    base_set = set(base)

    other_formulas = [f for f in universe_by_formula.keys() if f != f1 and len(universe_by_formula[f]) >= 1]
    if not other_formulas:
        # If only one formula exists, sample one more from the same formula
        # (without replacement) to get n_molecules total, ensuring all different SMILES
        available = [mol for mol in u1 if mol not in base_set]
        if available:
            chosen = base + [available[rng.randrange(len(available))]]
        else:
            # Fallback: if we've exhausted the universe, duplicate (shouldn't happen with buffer)
            raise ValueError(f"Formula {f1} universe exhausted when trying to generate 'No' case")
    else:
        # Sample from a different formula, ensuring no duplicate SMILES
        f2 = other_formulas[rng.randrange(len(other_formulas))]
        u2 = universe_by_formula[f2]
        # Try to find a molecule from u2 that's not already in base
        available_u2 = [mol for mol in u2 if mol not in base_set]
        if available_u2:
            chosen = base + [available_u2[rng.randrange(len(available_u2))]]
        else:
            # Unlikely case where all molecules in u2 are already in base
            # Just pick any molecule from u2 (will create duplicate)
            chosen = base + [u2[rng.randrange(len(u2))]]

    # Shuffle
    rng.shuffle(chosen)

    # Verify no duplicates
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"Generated Q2 'No' instance has duplicate SMILES: {chosen}")

    label = solve_q2_is_constitutional_isomer_set(chosen) or "No"
    prompt = build_q2_prompt(chosen)
    return BenchmarkInstance(
        id=instance_id,
        task="q2_isomer_set_yes_no",
        n_molecules=n_molecules,
        molecules=chosen,
        prompt=prompt,
        answer={"label": label},
        metadata={"constructed_label": "No", "base_formula": f1},
    )


def generate_q3_instance(
    *,
    instance_id: str,
    universe_by_formula: Dict[str, List[str]],
    n_molecules: int,
    rng,
    max_universe_size: int = 100,
    min_universe_size: int = 8,
) -> BenchmarkInstance:
    """
    Q3 requires a "complete set" universe. We choose a formula universe of manageable size,
    then provide n_molecules as the *given* subset and expect the remaining as the answer.
    """
    # Allow reasonable buffer above n_molecules for larger n values
    # For n=50, this allows universes up to 100, giving ~50 missing molecules to identify
    effective_max_size = max(max_universe_size, n_molecules + max(10, n_molecules // 2))
    
    candidate_formulas = [
        f for f, u in universe_by_formula.items()
        if (min_universe_size <= len(u) <= effective_max_size) and (len(u) > n_molecules)
    ]
    if not candidate_formulas:
        raise ValueError(
            f"No universe found with size in [{min_universe_size},{effective_max_size}] and > given size {n_molecules}"
        )

    f = candidate_formulas[rng.randrange(len(candidate_formulas))]
    universe = universe_by_formula[f]

    given = rng.sample(universe, n_molecules)
    missing = solve_q3_missing_isomers(given, universe)
    if missing is None:
        raise RuntimeError("Failed to compute missing isomers for Q3")

    prompt = build_q3_prompt(given)
    return BenchmarkInstance(
        id=instance_id,
        task="q3_missing_isomers",
        n_molecules=n_molecules,
        molecules=given,
        prompt=prompt,
        answer={"missing_smiles": missing},
        metadata={"formula": f, "universe_size": len(universe), "given_size": len(given)},
    )
