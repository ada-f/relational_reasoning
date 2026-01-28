from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


# Common organic-ish elements (H, C, N, O, F, P, S, Cl, Br, I)
DEFAULT_ALLOWED_ATOMIC_NUMS: Set[int] = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES into an RDKit Mol. Returns None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return mol
    except Exception:
        return None


def canonical_smiles(mol: Chem.Mol, *, isomeric: bool = False) -> str:
    """Canonical SMILES string from a Mol."""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def canonical_smiles_from_smiles(smiles: str, *, isomeric: bool = False) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return canonical_smiles(mol, isomeric=isomeric)


def mol_formula(mol: Chem.Mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)


def heavy_atom_count(mol: Chem.Mol) -> int:
    return int(mol.GetNumHeavyAtoms())


def mol_wt(mol: Chem.Mol) -> float:
    return float(Descriptors.MolWt(mol))


def contains_only_allowed_elements(
    mol: Chem.Mol,
    *,
    allowed_atomic_nums: Set[int] = DEFAULT_ALLOWED_ATOMIC_NUMS,
) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in allowed_atomic_nums:
            return False
    return True


def is_single_component_smiles(smiles: str) -> bool:
    # Multi-component SMILES have '.' separators
    return "." not in smiles


def morgan_fp(mol: Chem.Mol, *, radius: int = 2, nbits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto(a, b) -> float:
    return float(DataStructs.TanimotoSimilarity(a, b))


def are_isomorphic_smiles(a: str, b: str) -> bool:
    """Graph-isomorphism-ish check by mutual substructure match (ignores stereochem)."""
    ma = mol_from_smiles(a)
    mb = mol_from_smiles(b)
    if ma is None or mb is None:
        return False
    # Ignore stereochem by stripping to non-isomeric canonical forms
    ca = canonical_smiles(ma, isomeric=False)
    cb = canonical_smiles(mb, isomeric=False)
    ma2 = mol_from_smiles(ca)
    mb2 = mol_from_smiles(cb)
    if ma2 is None or mb2 is None:
        return False
    return ma2.HasSubstructMatch(mb2) and mb2.HasSubstructMatch(ma2)


def pick_heavy_atom_bin(hac: int, *, bin_width: int = 5) -> Tuple[int, int]:
    lo = (hac // bin_width) * bin_width
    hi = lo + bin_width - 1
    return lo, hi
