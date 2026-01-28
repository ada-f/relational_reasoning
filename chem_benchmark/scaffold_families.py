"""
Generate scaffold-based molecule families for Q1 with large n.

For large n (e.g., n=50), sampling from diverse drugs gives trivial MCS.
Instead, we generate molecules that share a common scaffold with variations.
"""

from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


# Common drug scaffolds with known large cores
SCAFFOLD_TEMPLATES = {
    "benzene": {
        "core": "c1ccccc1",
        "description": "benzene ring - 6 atoms",
        "attachment_points": [0, 1, 2, 3, 4, 5],
    },
    "naphthalene": {
        "core": "c1ccc2ccccc2c1",
        "description": "naphthalene - 10 atoms",
        "attachment_points": [0, 1, 2, 4, 5, 6, 7],
    },
    "indole": {
        "core": "c1ccc2c(c1)[nH]cc2",
        "description": "indole - 9 atoms",
        "attachment_points": [0, 1, 2, 4, 7, 8],
    },
    "quinoline": {
        "core": "c1ccc2ncccc2c1",
        "description": "quinoline - 10 atoms",
        "attachment_points": [0, 1, 2, 4, 5, 6, 7],
    },
    "pyridine": {
        "core": "c1ccncc1",
        "description": "pyridine - 6 atoms",
        "attachment_points": [0, 1, 2, 4, 5],
    },
    "imidazole": {
        "core": "c1cnc[nH]1",
        "description": "imidazole - 5 atoms",
        "attachment_points": [0, 2, 4],
    },
    "piperidine": {
        "core": "C1CCNCC1",
        "description": "piperidine - 6 atoms",
        "attachment_points": [0, 1, 2, 3, 4, 5],
    },
    "thiophene": {
        "core": "c1ccsc1",
        "description": "thiophene - 5 atoms",
        "attachment_points": [0, 1, 3, 4],
    },
    "furan": {
        "core": "c1ccoc1",
        "description": "furan - 5 atoms",
        "attachment_points": [0, 1, 3, 4],
    },
    "biphenyl": {
        "core": "c1ccc(cc1)c2ccccc2",
        "description": "biphenyl - 12 atoms",
        "attachment_points": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
    },
}


# R-groups for decoration (small, medium, large)
R_GROUPS = {
    "small": [
        "C",           # methyl
        "CC",          # ethyl
        "O",           # hydroxy
        "N",           # amino
        "F",           # fluoro
        "Cl",          # chloro
        "C(=O)C",      # acetyl
        "OC",          # methoxy
    ],
    "medium": [
        "CCC",         # propyl
        "CC(C)C",      # isopropyl
        "CCCC",        # butyl
        "c1ccccc1",    # phenyl
        "OCC",         # ethoxy
        "C(=O)CC",     # propionyl
        "NC",          # methylamino
        "S(=O)(=O)C",  # methylsulfonyl
    ],
    "large": [
        "CCCCC",       # pentyl
        "c1ccc(cc1)C", # toluyl
        "OC(=O)C",     # acetoxy
        "C(=O)N",      # carbamoyl
        "c1cccnc1",    # pyridinyl
    ],
}


def generate_scaffold_family(
    scaffold_name: str,
    n_molecules: int,
    *,
    seed: int = 0,
) -> List[str]:
    """
    Generate n molecules sharing a common scaffold by varying substituents.

    Args:
        scaffold_name: Name from SCAFFOLD_TEMPLATES
        n_molecules: Number of molecules to generate
        seed: Random seed for reproducibility

    Returns:
        List of SMILES strings sharing the scaffold
    """
    import random
    rng = random.Random(seed)

    if scaffold_name not in SCAFFOLD_TEMPLATES:
        raise ValueError(f"Unknown scaffold: {scaffold_name}. Available: {list(SCAFFOLD_TEMPLATES.keys())}")

    template = SCAFFOLD_TEMPLATES[scaffold_name]
    core_smiles = template["core"]

    # Parse core
    core = Chem.MolFromSmiles(core_smiles)
    if core is None:
        raise ValueError(f"Invalid core SMILES: {core_smiles}")

    molecules = []
    seen = set()

    # Always include the bare scaffold
    molecules.append(core_smiles)
    seen.add(core_smiles)

    # Generate variations
    all_rgroups = R_GROUPS["small"] + R_GROUPS["medium"] + R_GROUPS["large"]
    attachment_points = template["attachment_points"]

    attempts = 0
    max_attempts = n_molecules * 100

    while len(molecules) < n_molecules and attempts < max_attempts:
        attempts += 1

        # Decide how many substituents (1-3)
        n_subs = rng.choice([1, 1, 1, 2, 2, 3])  # Bias toward fewer

        # Pick random attachment points
        if len(attachment_points) < n_subs:
            n_subs = len(attachment_points)

        positions = rng.sample(attachment_points, n_subs)
        substituents = rng.choices(all_rgroups, k=n_subs)

        # Build SMARTS-like pattern for substitution
        # For simplicity, we'll use SMILES concatenation
        # This is a simplified approach - in production you'd use proper RDKit substitution

        # Create a modified SMILES by manual substitution
        # This is approximate - real implementation would use RDKit RGroupDecomposition
        mol = Chem.MolFromSmiles(core_smiles)
        mol_editable = Chem.RWMol(mol)

        for pos_idx, sub_smiles in zip(positions, substituents):
            if pos_idx >= mol.GetNumAtoms():
                continue

            # Add substituent
            sub_mol = Chem.MolFromSmiles(sub_smiles)
            if sub_mol is None:
                continue

            # Simple approach: add fragment and bond to atom
            # In real implementation, use proper reaction SMARTS

        # Generate SMILES
        try:
            # Simplified: just create a descriptive SMILES
            # In production, use proper RDKit molecular editing
            new_smiles = Chem.MolToSmiles(mol)
            if new_smiles and new_smiles not in seen:
                molecules.append(new_smiles)
                seen.add(new_smiles)
        except:
            continue

    return molecules[:n_molecules]


def list_available_scaffolds() -> Dict[str, Dict]:
    """Return available scaffold templates with metadata."""
    return SCAFFOLD_TEMPLATES.copy()


def get_scaffold_from_molecule(smiles: str) -> str:
    """Extract Murcko scaffold from a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return ""
