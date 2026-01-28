from __future__ import annotations

import json
import time
import urllib.parse
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from rdkit import Chem, DataStructs
from rdkit.SimDivFilters import rdSimDivPickers

from .rdkit_utils import (
    DEFAULT_ALLOWED_ATOMIC_NUMS,
    canonical_smiles,
    contains_only_allowed_elements,
    heavy_atom_count,
    is_single_component_smiles,
    mol_from_smiles,
    mol_wt,
    morgan_fp,
    pick_heavy_atom_bin,
    tanimoto,
)


@dataclass(frozen=True)
class MoleculeRecord:
    source: str
    source_id: str
    name: str
    smiles: str
    heavy_atoms: int
    mw: float


def _chembl_next_url(base_data_url: str, next_url: str) -> str:
    # ChEMBL sometimes returns relative URLs. Normalize.
    if next_url.startswith("http://") or next_url.startswith("https://"):
        return next_url
    # If next_url starts with /, it's an absolute path on the same domain
    if next_url.startswith("/"):
        # Extract base domain from base_data_url (e.g., https://www.ebi.ac.uk)
        parsed = urllib.parse.urlparse(base_data_url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"
        return urllib.parse.urljoin(base_domain, next_url)
    # Otherwise, it's a relative URL
    return urllib.parse.urljoin(base_data_url.rstrip("/") + "/", next_url)


def fetch_chembl_max_phase_smiles(
    *,
    max_records: int = 500,
    page_size: int = 100,
    base_data_url: str = "https://www.ebi.ac.uk/chembl/api/data",
    timeout_s: int = 45,
    sleep_s: float = 0.1,
) -> List[Dict]:
    """
    Fetch molecules from ChEMBL REST API, filtering by max_phase.
    Returns raw JSON dicts for each molecule.
    """
    session = requests.Session()
    out: List[Dict] = []

    # Common ChEMBL endpoint shape:
    #   https://www.ebi.ac.uk/chembl/api/data/molecule?format=json&max_phase=4&limit=100&offset=0
    url = f"{base_data_url.rstrip('/')}/molecule"
    params = {"format": "json", "max_phase__gte": 1, "max_phase__lte": 4, "limit": page_size, "offset": 0}

    while True:
        r = session.get(url, params=params, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()

        molecules = data.get("molecules") or data.get("molecule") or []
        if not isinstance(molecules, list):
            molecules = []

        out.extend(molecules)
        if len(out) >= max_records:
            out = out[:max_records]
            break

        page_meta = data.get("page_meta") or {}
        next_url = page_meta.get("next")
        if not next_url:
            break

        url = _chembl_next_url(base_data_url, next_url)
        params = None  # next already encodes query
        time.sleep(sleep_s)

    return out


def clean_chembl_records(
    raw_molecules: Sequence[Dict],
    *,
    allowed_atomic_nums=DEFAULT_ALLOWED_ATOMIC_NUMS,
    min_heavy_atoms: int = 15,
    max_heavy_atoms: int = 60,
    require_single_component: bool = True,
) -> List[MoleculeRecord]:
    out: List[MoleculeRecord] = []

    for m in raw_molecules:
        chembl_id = str(m.get("molecule_chembl_id") or "").strip()
        name = str(m.get("pref_name") or m.get("molecule_name") or chembl_id or "unknown").strip()

        structs = m.get("molecule_structures") or {}
        smiles = (structs.get("canonical_smiles") or structs.get("smiles") or "").strip()
        if not smiles:
            continue

        if require_single_component and not is_single_component_smiles(smiles):
            continue

        mol = mol_from_smiles(smiles)
        if mol is None:
            continue

        if not contains_only_allowed_elements(mol, allowed_atomic_nums=allowed_atomic_nums):
            continue

        hac = heavy_atom_count(mol)
        if hac < min_heavy_atoms or hac > max_heavy_atoms:
            continue

        mw = mol_wt(mol)
        can = canonical_smiles(mol, isomeric=False)

        out.append(
            MoleculeRecord(
                source="chembl",
                source_id=chembl_id,
                name=name,
                smiles=can,
                heavy_atoms=hac,
                mw=mw,
            )
        )

    # De-duplicate by SMILES
    seen = set()
    dedup: List[MoleculeRecord] = []
    for rec in out:
        if rec.smiles in seen:
            continue
        seen.add(rec.smiles)
        dedup.append(rec)

    return dedup


def select_diverse_subset_maxmin(
    records: Sequence[MoleculeRecord],
    n: int,
    *,
    seed: int = 0,
    fp_radius: int = 2,
    fp_nbits: int = 2048,
) -> List[MoleculeRecord]:
    """
    MaxMinPicker on Tanimoto distance to pick a diverse subset.
    """
    if len(records) <= n:
        return list(records)

    mols = []
    fps = []
    for r in records:
        mol = mol_from_smiles(r.smiles)
        if mol is None:
            continue
        mols.append(mol)
        fps.append(morgan_fp(mol, radius=fp_radius, nbits=fp_nbits))

    if len(fps) <= n:
        return list(records)[:n]

    picker = rdSimDivPickers.MaxMinPicker()
    # LazyBitVectorPick works directly with fingerprints and uses Tanimoto distance
    picks = list(picker.LazyBitVectorPick(fps, len(fps), n, (), seed))
    picks_set = set(picks)

    # picks indexes correspond to mols/fps which are aligned with the "mols" list.
    # But "records" list is aligned to mol parsing success? We built mols/fps by parsing records in order,
    # so it's aligned with records that parsed. We'll reconstruct that mapping:
    parsed_records: List[MoleculeRecord] = []
    for r in records:
        if mol_from_smiles(r.smiles) is not None:
            parsed_records.append(r)

    diverse = [parsed_records[i] for i in picks if i < len(parsed_records)]
    # If any mismatch, truncate/pad safely
    return diverse[:n]


def save_bank(path: Path, records: Sequence[MoleculeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in records]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_bank(path: Path) -> List[MoleculeRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[MoleculeRecord] = []
    for row in data:
        out.append(
            MoleculeRecord(
                source=row["source"],
                source_id=row["source_id"],
                name=row["name"],
                smiles=row["smiles"],
                heavy_atoms=int(row["heavy_atoms"]),
                mw=float(row["mw"]),
            )
        )
    return out


class BankIndex:
    """
    Convenience structure to sample molecules with similar size and nontrivial similarity.
    """

    def __init__(self, records: Sequence[MoleculeRecord], *, bin_width: int = 5):
        from rdkit.Chem.Scaffolds import MurckoScaffold

        self.records = list(records)
        self.bin_width = bin_width

        self.mols = []
        self.fps = []
        self.hac = []
        self.scaffolds = []  # Murcko scaffolds for each molecule
        self.generic_scaffolds = []  # Generic scaffolds (with attachment points)

        for r in self.records:
            mol = mol_from_smiles(r.smiles)
            if mol is None:
                continue
            self.mols.append(mol)
            self.fps.append(morgan_fp(mol))
            self.hac.append(r.heavy_atoms)

            # Extract Murcko scaffold and generic scaffold
            try:
                scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
                self.scaffolds.append(scaffold_smiles)

                # Also extract generic scaffold (includes attachment points)
                generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
                generic_smiles = Chem.MolToSmiles(generic_mol)
                self.generic_scaffolds.append(generic_smiles)
            except:
                self.scaffolds.append("")  # Empty if extraction fails
                self.generic_scaffolds.append("")

        # Bin indices by heavy atom count
        self.bin_to_indices: Dict[Tuple[int, int], List[int]] = {}
        for i, hac in enumerate(self.hac):
            b = pick_heavy_atom_bin(hac, bin_width=bin_width)
            self.bin_to_indices.setdefault(b, []).append(i)

        # Group indices by scaffold (for large n sampling)
        # Strategy: Use generic scaffolds to group molecules
        # Generic scaffolds preserve attachment point information, making groups more meaningful
        # For example, benzene with different substitution patterns will have different generic scaffolds
        self.scaffold_to_indices: Dict[str, List[int]] = {}
        for i, (scaffold, generic) in enumerate(zip(self.scaffolds, self.generic_scaffolds)):
            if not generic:
                continue

            # Use generic scaffold for grouping
            # This groups molecules by both their core structure AND substitution pattern
            # Example: "c1ccc(*)cc1" (mono-substituted benzene) vs "c1cc(*)c(*)cc1" (di-substituted)
            # This provides more structural diversity than bare Murcko scaffolds
            self.scaffold_to_indices.setdefault(generic, []).append(i)

    def sample_similar_group(
        self,
        n: int,
        *,
        rng,
        min_similarity: float = 0.35,
        max_similarity: float = 0.90,
        max_tries: int = 200,
        force_scaffold_sampling: bool = False,
    ) -> List[str]:
        """
        Pick N molecules using similarity-based or scaffold-based sampling.

        Args:
            force_scaffold_sampling: If True, use scaffold-based sampling.
                                    If False, use similarity-based sampling.

        Returns canonical SMILES.
        """
        # SCAFFOLD-BASED SAMPLING
        if force_scaffold_sampling:
            return self._sample_from_scaffold(n, rng=rng, max_tries=max_tries)

        # SIMILARITY-BASED SAMPLING (original method for small n)
        bins = [b for b, idxs in self.bin_to_indices.items() if len(idxs) >= n]
        if not bins:
            raise ValueError(f"No heavy-atom bin has >= {n} molecules")

        for _ in range(max_tries):
            b = bins[rng.randrange(len(bins))]
            idxs = self.bin_to_indices[b]
            seed_i = idxs[rng.randrange(len(idxs))]

            # rank neighbors by similarity within the bin
            sims = []
            fp_seed = self.fps[seed_i]
            for j in idxs:
                if j == seed_i:
                    continue
                sim = float(DataStructs.TanimotoSimilarity(fp_seed, self.fps[j]))
                sims.append((sim, j))
            sims.sort(reverse=True)

            # filter to similarity band
            candidates = [j for sim, j in sims if (sim >= min_similarity and sim <= max_similarity)]
            if len(candidates) < (n - 1):
                # fallback: take top nearest even if outside band
                candidates = [j for _, j in sims]

            if len(candidates) < (n - 1):
                continue

            chosen = [seed_i] + candidates[: (n - 1)]
            smiles = [self.records[i].smiles for i in chosen]
            return smiles

        # If we fail to find within similarity band, fall back to random in some bin
        b = bins[rng.randrange(len(bins))]
        idxs = self.bin_to_indices[b]
        chosen = rng.sample(idxs, n)
        return [self.records[i].smiles for i in chosen]

    def _sample_from_scaffold(self, n: int, *, rng, max_tries: int = 200) -> List[str]:
        """
        Sample n molecules sharing a common scaffold with diversity.

        Strategy: Sample from MULTIPLE related scaffold families for all n.
        - All sampled molecules share the same bare Murcko scaffold (core structure)
        - But come from different generic scaffold families (different substitution patterns)
        - This maintains diversity while ensuring a meaningful common core motif

        Benefits:
        - Maintains diversity across all n values
        - Avoids collapse to simple structures like benzene
        - Ensures interesting MCS (core + partial decorations)
        """
        # Always use multiple scaffold families approach
        return self._sample_from_multiple_scaffolds(n, rng, max_tries)

    def _sample_from_single_scaffold(self, n: int, rng, max_tries: int = 200) -> List[str]:
        """Sample all n molecules from a single scaffold family."""
        valid_scaffolds = [(scaf, idxs) for scaf, idxs in self.scaffold_to_indices.items() if len(idxs) >= n]

        if not valid_scaffolds:
            max_family_size = max(len(idxs) for idxs in self.scaffold_to_indices.values()) if self.scaffold_to_indices else 0
            num_families = len(self.scaffold_to_indices)
            raise ValueError(
                f"No scaffold family has >= {n} molecules.\n"
                f"Available scaffold families: {num_families}\n"
                f"Largest family size: {max_family_size}\n"
            )

        # Prefer scaffolds with more molecules for better diversity
        valid_scaffolds.sort(key=lambda x: len(x[1]), reverse=True)

        # Try top scaffolds
        for _, idxs in valid_scaffolds[:max_tries]:
            if len(idxs) >= n:
                chosen_idxs = rng.sample(idxs, n)
                smiles = [self.records[i].smiles for i in chosen_idxs]
                return smiles

        # Fallback: use the largest scaffold family
        _, idxs = valid_scaffolds[0]
        chosen_idxs = rng.sample(idxs, min(n, len(idxs)))
        smiles = [self.records[i].smiles for i in chosen_idxs]
        return smiles

    def _sample_from_multiple_scaffolds(self, n: int, rng, max_tries: int = 200) -> List[str]:
        """
        Sample n molecules from MULTIPLE scaffold families that share a common core.
        This maintains diversity for large n while ensuring a meaningful common motif.
        """
        # Group scaffolds by their base structure (use bare Murcko scaffold)
        from rdkit.Chem.Scaffolds import MurckoScaffold

        # Map bare scaffold -> list of (generic_scaffold, indices)
        bare_to_families = defaultdict(list)
        for generic_scaf, idxs in self.scaffold_to_indices.items():
            # Get the bare scaffold (without attachment points)
            for idx in idxs[:1]:  # Just check first molecule
                mol = self.mols[idx]
                try:
                    bare_scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                    bare_scaffold = Chem.MolToSmiles(bare_scaffold_mol)
                    bare_to_families[bare_scaffold].append((generic_scaf, idxs))
                    break
                except:
                    pass

        # Find bare scaffolds that have enough TOTAL molecules across families
        valid_bare_scaffolds = []
        for bare_scaf, families in bare_to_families.items():
            total_mols = sum(len(idxs) for _, idxs in families)
            if total_mols >= n:
                valid_bare_scaffolds.append((bare_scaf, families, total_mols))

        if not valid_bare_scaffolds:
            # Fallback to single scaffold approach
            return self._sample_from_single_scaffold(n, rng, max_tries)

        # Sort by total molecules (prefer those with more diversity)
        valid_bare_scaffolds.sort(key=lambda x: x[2], reverse=True)

        # Try to sample from multiple families
        for bare_scaf, families, total_mols in valid_bare_scaffolds[:max_tries]:
            # Skip if bare scaffold is too simple (single benzene ring with ≤6 atoms)
            bare_mol = Chem.MolFromSmiles(bare_scaf)
            if bare_mol:
                num_atoms = bare_mol.GetNumHeavyAtoms()
                num_rings = bare_mol.GetRingInfo().NumRings()
                # Skip single ring with ≤6 atoms (benzene, pyridine, etc.)
                if num_rings == 1 and num_atoms <= 6:
                    continue

            # Sample from multiple families within this bare scaffold group
            # Distribute n across families proportionally
            chosen_idxs = []

            # Sort families by size to prioritize larger ones
            sorted_families = sorted(families, key=lambda x: len(x[1]), reverse=True)

            # Calculate how many to sample from each family
            molecules_per_family = []
            remaining = n
            for i, (gen_scaf, idxs) in enumerate(sorted_families):
                if i == len(sorted_families) - 1:
                    # Last family gets the remainder
                    molecules_per_family.append((gen_scaf, idxs, remaining))
                else:
                    # Sample proportionally, but at least 1 from each non-empty family
                    proportion = len(idxs) / total_mols
                    count = max(1, int(n * proportion))
                    count = min(count, len(idxs), remaining)
                    molecules_per_family.append((gen_scaf, idxs, count))
                    remaining -= count
                    if remaining <= 0:
                        break

            # Sample from each family
            for gen_scaf, idxs, count in molecules_per_family:
                if count > 0 and count <= len(idxs):
                    sampled = rng.sample(idxs, count)
                    chosen_idxs.extend(sampled)

            # Check if we got enough molecules
            if len(chosen_idxs) >= n:
                # Shuffle and trim to exactly n
                rng.shuffle(chosen_idxs)
                chosen_idxs = chosen_idxs[:n]
                smiles = [self.records[i].smiles for i in chosen_idxs]
                return smiles

        # Fallback to single scaffold if multi-scaffold approach fails
        return self._sample_from_single_scaffold(n, rng, max_tries)
