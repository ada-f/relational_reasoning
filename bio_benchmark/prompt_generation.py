#!/usr/bin/env python
"""
Simulate alignments with controlled homoplasy and generate LLM Q/A examples.

- Uses Pyvolve to simulate nucleotide sequences on a given tree.
- Injects homoplasy by forcing convergent states at selected columns
  in a subset of taxa.
- Produces LLM-style question/label pairs.

Dependencies:
    pip install pyvolve biopython
"""

import os
import json
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pyvolve
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree
from .random_tree import RandomTree


# ---------------------------------------------------------------------
# Core simulation utilities
# ---------------------------------------------------------------------

def simulate_alignment_with_pyvolve(
    newick_tree: str,
    seq_len: int,
    seqfile: str = "simulated_alignment.fasta",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Simulate a nucleotide alignment on a given Newick tree using Pyvolve.

    Args:
        newick_tree: Newick tree string OR path to a Newick file.
        seq_len: sequence length (number of columns).
        seqfile: output FASTA file for Pyvolve (temporary).
        model_kwargs: optional dict of model parameters for Pyvolve.Model.

    Returns:
        alignment: dict {taxon_name: sequence_str}
    """
    # If it's a path, read the file; otherwise treat as literal Newick
    if os.path.exists(newick_tree):
        tree = pyvolve.read_tree(file=newick_tree)
        with open(newick_tree, "r") as f:
            newick_str = f.read().strip()
    else:
        tree = pyvolve.read_tree(tree=newick_tree)
        newick_str = newick_tree.strip()

    if model_kwargs is None:
        model_kwargs = {}  # default JC69-like

    model = pyvolve.Model("nucleotide", model_kwargs)
    partition = pyvolve.Partition(models=model, size=seq_len)

    # Use temporary file if seqfile is a simple name (no directory), otherwise use provided path
    # This avoids conflicts when multiple processes run simultaneously
    seqfile_dir = os.path.dirname(seqfile)
    use_temp = not os.path.isabs(seqfile) and (not seqfile_dir or seqfile_dir == '.')
    temp_file_path = None
    
    if use_temp:
        # Create a temporary file path
        temp_file_handle = tempfile.NamedTemporaryFile(
            mode='w', suffix='.fasta', delete=False
        )
        temp_file_path = temp_file_handle.name
        temp_file_handle.close()
        seqfile_str = temp_file_path
    else:
        # Convert to absolute path to avoid working directory issues
        seqfile_path = Path(seqfile).resolve()
        seqfile_path.parent.mkdir(parents=True, exist_ok=True)
        seqfile_str = str(seqfile_path)
    
    # Ensure file doesn't exist - Pyvolve should create it
    if os.path.exists(seqfile_str):
        os.unlink(seqfile_str)

    # Create evolver and run simulation
    try:
        # Try creating evolver without seqfile first, then pass it when calling
        try:
            evolver = pyvolve.Evolver(tree=tree, partitions=partition)
            evolver(seqfile=seqfile_str)
        except (TypeError, Exception):
            # If that fails, try with seqfile in constructor
            evolver = pyvolve.Evolver(
                tree=tree, 
                partitions=partition, 
                seqfile=seqfile_str
            )
            evolver()
    except Exception as e:
        # Clean up temp file if we created one
        if use_temp and temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise RuntimeError(f"Pyvolve simulation failed: {e}") from e

    # Verify the file was created
    if not os.path.exists(seqfile_str):
        # Check if Pyvolve wrote to current working directory
        filename_only = os.path.basename(seqfile_str)
        cwd_file = os.path.join(os.getcwd(), filename_only)
        if os.path.exists(cwd_file):
            seqfile_str = cwd_file
        else:
            # Clean up temp file if we created one
            if use_temp and temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            raise RuntimeError(f"Pyvolve failed to create output file: {seqfile_str}")

    # Load the alignment back into memory as {taxon: sequence}
    alignment: Dict[str, str] = {}
    try:
        for record in SeqIO.parse(seqfile_str, "fasta"):
            alignment[record.id] = str(record.seq)
    except Exception as parse_error:
        raise RuntimeError(f"Failed to parse FASTA file from Pyvolve: {parse_error}") from parse_error
    finally:
        # Clean up temp file after reading
        if use_temp and temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

    # Validate that alignment is not empty
    if not alignment:
        raise RuntimeError(f"Pyvolve created empty alignment from file: {seqfile_str}")

    return alignment, newick_str

def choose_disjoint_clades(
    newick_str: str,
    n_groups: int,
    min_clade_size: int = 2,
    max_clade_size: int = 8,
    min_group_distance_edges: int = 4,
    rng: Optional[random.Random] = None,
) -> List[List[str]]:
    """
    Pick n_groups disjoint clades (each clade is exactly the leaf set of some internal node),
    and ensure clade nodes are pairwise far apart in topology.

    Returns: list of groups, each a list of taxon names.
    """
    if rng is None:
        rng = random.Random()

    t = Tree(newick_str, format=1)

    # Candidate internal nodes whose leaf sets are the desired size
    candidates: List[Tuple[Any, frozenset]] = []
    for node in t.traverse("postorder"):
        if node.is_leaf():
            continue
        leaf_names = [l.name for l in node.get_leaves()]
        if min_clade_size <= len(leaf_names) <= max_clade_size:
            candidates.append((node, frozenset(leaf_names)))

    if len(candidates) < n_groups:
        raise ValueError(
            f"Not enough candidate clades in size range [{min_clade_size},{max_clade_size}] "
            f"to pick {n_groups} disjoint groups."
        )

    rng.shuffle(candidates)

    selected_nodes: List[Any] = []
    selected_leafsets: List[frozenset] = []

    for node, leafset in candidates:
        # disjointness
        if any(len(leafset & s) > 0 for s in selected_leafsets):
            continue

        # distance between clade nodes (topological)
        if any(node.get_distance(prev, topology_only=True) < min_group_distance_edges for prev in selected_nodes):
            continue

        selected_nodes.append(node)
        selected_leafsets.append(leafset)

        if len(selected_nodes) == n_groups:
            break

    if len(selected_nodes) < n_groups:
        raise ValueError(
            f"Could only select {len(selected_nodes)} disjoint clades "
            f"with min_group_distance_edges={min_group_distance_edges}."
        )

    groups = [sorted(list(s)) for s in selected_leafsets]

    # sanity: verify each group is monophyletic leaf set
    for g in groups:
        mrca = t.get_common_ancestor(g)
        if set(l.name for l in mrca.get_leaves()) != set(g):
            raise RuntimeError("Internal error: selected group is not exactly a clade leaf-set.")

    return groups

def inject_convergent_blocks_tree_aware_groups(
    alignment: Dict[str, str],
    newick_str: str,
    n_blocks: int,
    taxa_groups: List[List[str]],
    length_convergent_block: int = 5,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Inject convergent blocks: all taxa in ALL groups share the same motif in each injected block.
    Each group is intended to be a monophyletic clade (arity = len(taxa_groups)).
    """
    if rng is None:
        rng = random.Random()

    all_taxa = list(alignment.keys())
    seq_len = len(next(iter(alignment.values())))
    seq_arrays: Dict[str, List[str]] = {t: list(seq) for t, seq in alignment.items()}

    # validate
    if len(taxa_groups) < 1:
        raise ValueError("taxa_groups must contain at least 1 group.")
    flat = [x for g in taxa_groups for x in g]
    if len(set(flat)) != len(flat):
        raise ValueError("taxa_groups must be disjoint (no taxon appears in multiple groups).")
    for taxon in flat:
        if taxon not in all_taxa:
            raise ValueError(f"Taxon '{taxon}' not found in alignment.")

    if length_convergent_block > seq_len:
        raise ValueError("length_convergent_block cannot exceed alignment length.")

    possible_starts = list(range(0, seq_len - length_convergent_block + 1))
    rng.shuffle(possible_starts)
    selected_starts = possible_starts[:n_blocks]

    bases = ["A", "C", "G", "T"]
    blocks_metadata: List[Dict[str, Any]] = []

    taxa_union = sorted(set(flat))
    arity = len(taxa_groups)

    for start in selected_starts:
        end = start + length_convergent_block - 1
        motif = "".join(rng.choice(bases) for _ in range(length_convergent_block))

        # Apply same motif to everyone in the convergent union
        for taxon in taxa_union:
            seq_arrays[taxon][start : start + length_convergent_block] = list(motif)

        blocks_metadata.append(
            {
                "block_start": start,
                "block_end": end,
                "motif": motif,
                "arity": arity,
                "convergent_groups": [
                    {"group_id": i + 1, "taxa": g} for i, g in enumerate(taxa_groups)
                ],
                "convergent_taxa_union": taxa_union,
            }
        )

    new_alignment = {t: "".join(chars) for t, chars in seq_arrays.items()}
    return new_alignment, blocks_metadata



def choose_distant_taxa(
    newick_str: str,
    n_taxa: int,
    min_distance_edges: int = 3,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Choose n_taxa that are mutually distant from each other in the tree.

    Uses a greedy approach: start with a random taxon, then iteratively
    add the taxon that maximizes the minimum distance to all already-selected taxa.

    Args:
        newick_str: tree in Newick format.
        n_taxa: number of taxa to select.
        min_distance_edges: minimum topological distance between any pair of selected taxa.
        rng: optional random.Random instance for reproducibility.

    Returns:
        List of taxon names.
    """
    if rng is None:
        rng = random.Random()

    t = Tree(newick_str, format=1)
    leaves = t.get_leaves()
    
    if len(leaves) < n_taxa:
        raise ValueError(f"Tree has only {len(leaves)} leaves, cannot select {n_taxa} taxa.")

    # Precompute pairwise distances
    leaf_names = [leaf.name for leaf in leaves]
    leaf_lookup = {leaf.name: leaf for leaf in leaves}
    
    # Start with a random taxon
    selected = [rng.choice(leaf_names)]
    remaining = set(leaf_names) - set(selected)

    while len(selected) < n_taxa and remaining:
        best_candidate = None
        best_min_dist = -1

        for candidate in remaining:
            # Compute minimum distance from candidate to all selected taxa
            min_dist = min(
                leaf_lookup[candidate].get_distance(leaf_lookup[s], topology_only=True)
                for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate

        if best_candidate is None or best_min_dist < min_distance_edges:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    if len(selected) < n_taxa:
        raise ValueError(
            f"Could only find {len(selected)} taxa with pairwise distance >= {min_distance_edges} edges. "
            f"Requested {n_taxa}."
        )

    return selected


def choose_two_taxa(
    newick_str: str,
    min_distance_edges: int = 4,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str]:
    """
    Choose two distantly related taxa from a tree.

    Args:
        newick_str: tree in Newick format.
        min_distance_edges: minimum topological distance between the two taxa.
        rng: optional random.Random instance for reproducibility.

    Returns:
        (taxon1, taxon2): names of the two selected taxa.
    """
    if rng is None:
        rng = random.Random()

    t = Tree(newick_str, format=1)

    # Get all leaf nodes
    leaves = t.get_leaves()
    if len(leaves) < 2:
        raise ValueError("Tree must have at least 2 leaves.")

    # Build all candidate pairs that are sufficiently far apart
    valid_pairs = []
    for i, leaf1 in enumerate(leaves):
        for leaf2 in leaves[i+1:]:
            dist = leaf1.get_distance(leaf2, topology_only=True)
            if dist >= min_distance_edges:
                valid_pairs.append((dist, leaf1.name, leaf2.name))

    if not valid_pairs:
        raise ValueError(
            f"Could not find two taxa with distance >= {min_distance_edges} edges."
        )

    # Prefer the farthest pair; tie-break randomly
    max_dist = max(p[0] for p in valid_pairs)
    far_pairs = [p for p in valid_pairs if p[0] == max_dist]
    rng.shuffle(far_pairs)
    _, taxon1, taxon2 = far_pairs[0]

    return taxon1, taxon2


def inject_convergent_blocks_tree_aware(
    alignment: Dict[str, str],
    newick_str: str,
    n_blocks: int,
    taxa_list: List[str],
    length_convergent_block: int = 5,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Inject convergent homoplasy by forcing a given list of taxa
    to share the same nucleotide motif at selected blocks.

    Args:
        alignment: dict {taxon_name: sequence_str}
        newick_str: tree in Newick format
        n_blocks: number of convergent blocks to inject
        taxa_list: list of taxa names to inject convergent blocks into
        length_convergent_block: length of each convergent block
        rng: optional random.Random instance for reproducibility

    Returns:
        new_alignment: modified alignment dict
        blocks_metadata: list of dicts with block info
    """
    if rng is None:
        rng = random.Random()

    # Validate alignment is not empty
    if not alignment:
        raise ValueError("Alignment is empty; cannot inject convergent blocks.")

    all_taxa = list(alignment.keys())
    if not all_taxa:
        raise ValueError("Alignment has no taxa; cannot inject convergent blocks.")

    # Get sequence length safely
    try:
        seq_len = len(next(iter(alignment.values())))
    except StopIteration:
        raise ValueError("Alignment is empty; cannot inject convergent blocks.")

    seq_arrays: Dict[str, List[str]] = {t: list(seq) for t, seq in alignment.items()}

    # Validate that all taxa in taxa_list are in the alignment
    for taxon in taxa_list:
        if taxon not in all_taxa:
            raise ValueError(f"Taxon '{taxon}' not found in alignment.")

    if len(taxa_list) < 2:
        raise ValueError("taxa_list must contain at least 2 taxa.")

    if length_convergent_block > seq_len:
        raise ValueError("length_convergent_block cannot exceed alignment length.")

    possible_starts = list(range(0, seq_len - length_convergent_block + 1))
    rng.shuffle(possible_starts)
    selected_starts = possible_starts[:n_blocks]

    bases = ["A", "C", "G", "T"]
    blocks_metadata: List[Dict[str, Any]] = []

    for start in selected_starts:
        end = start + length_convergent_block - 1

        convergent_sites: List[int] = []

        for pos in range(start, start + length_convergent_block):
            motif_base = rng.choice(bases)
            # Force same nucleotide at this site for all taxa in the list
            for taxon in taxa_list:
                seq_arrays[taxon][pos] = motif_base
            convergent_sites.append(pos)

        blocks_metadata.append(
            {
                "block_start": start,
                "block_end": end,
                "convergent_sites": convergent_sites,
                "convergent_taxa": taxa_list,
            }
        )

    new_alignment = {t: "".join(chars) for t, chars in seq_arrays.items()}
    return new_alignment, blocks_metadata


def inject_convergent_homoplasy(
    alignment: Dict[str, str],
    n_convergent_sites: int,
    taxa_per_site: int = 3,
    rng: Optional[random.Random] = None,
    length_convergent_block: int = 5,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Inject explicit convergent homoplasy into a baseline alignment by
    forcing the same nucleotide at selected sites in a subset of taxa.

    Args:
        alignment: dict {taxon_name: sequence_str}, all same length.
        n_convergent_sites: how many columns to modify.
        taxa_per_site: how many taxa to force into the convergent state.
        rng: optional random.Random instance for reproducibility.

    Returns:
        new_alignment: modified alignment dict.
        convergent_metadata: list of dicts, one per convergent site:
            {
              "site_index": int (0-based),
              "target_base": str,
              "taxa_involved": [str, ...]
            }
    """
    if rng is None:
        rng = random.Random()

    taxa = list(alignment.keys())
    if not taxa:
        raise ValueError("Alignment is empty; cannot inject homoplasy.")

    seq_len = len(next(iter(alignment.values())))
    for seq in alignment.values():
        if len(seq) != seq_len:
            raise ValueError("All sequences must have the same length.")

    # Work with mutable lists of chars
    seq_arrays: Dict[str, List[str]] = {t: list(seq) for t, seq in alignment.items()}

    possible_sites = list(range(seq_len))
    rng.shuffle(possible_sites)
    selected_sites = possible_sites[:n_convergent_sites]

    bases = ["A", "C", "G", "T"]
    convergent_metadata: List[Dict[str, Any]] = []

    for site in selected_sites:
        # Choose taxa to converge
        chosen_taxa = rng.sample(taxa, min(taxa_per_site, len(taxa)))
        target_motif = ''.join(rng.choices(bases, k=length_convergent_block))

        for t in chosen_taxa:
            seq_arrays[t][site:site+length_convergent_block] = list(target_motif)

        convergent_metadata.append(
            {
                "site_index": site,
                "target_base": target_motif,
                "taxa_involved": chosen_taxa,
            }
        )

    # Convert back to strings
    new_alignment = {t: "".join(chars) for t, chars in seq_arrays.items()}
    return new_alignment, convergent_metadata


def format_alignment_fasta(alignment: Dict[str, str]) -> str:
    """
    Format alignment as a FASTA-style string for inclusion in prompts.
    """
    lines: List[str] = []
    for taxon, seq in alignment.items():
        lines.append(f">{taxon}")
        lines.append(seq)
    return "\n".join(lines)


def make_homoplasy_question(
    newick_str: str,
    alignment: Dict[str, str],
    simplified_prompt: bool = False,
) -> str:
    """
    Build a natural-language question for an LLM about homoplasy at a given site.

    Positions in the prompt are 1-based; site_index is 0-based.
    """
    alignment_str = format_alignment_fasta(alignment)

    question = f"""
        Homoplasy refers to structured convergence:
        pairs or groups of distantly related taxa that repeatedly share the same nucleotide motifs
        across many independent alignment columns more often than expected, while other taxa
        with similar overall sequences do not share those nucleotide motifs as consistently.

        Your job is to examine the entire alignment and provided tree and decide whether such structured
        homoplasy is likely to be present and which taxa are involved.

        Alignment (FASTA; positions indexed from 1):
        {alignment_str}

        Tree (Newick):
        {newick_str}

        Return your answer as: Yes/No and if Yes, list the taxa involved. Nothing else.

    """

    return question


def build_llm_examples(
    newick_str: str,
    alignment: Dict[str, str],
    convergent_metadata: List[Dict[str, Any]],
    simplified_prompt: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build a list of LLM-style examples with question + label.

    Returns:
        list of dicts like:
        {
          "question": str,
          "label": "yes" or "no",
          "site_index": int,
          "metadata": {...}
        }
    """
    examples: List[Dict[str, Any]] = []

    # Positive examples: injected convergent sites
    for meta in convergent_metadata:
        q = make_homoplasy_question(newick_str, alignment, simplified_prompt=simplified_prompt)
        examples.append(
            {
                "question": q,
                "tree":newick_str,
                "label": "yes",
                "metadata": {
                    "type": "convergent_site",
                    **meta,
                },
            }
        )

    return examples

def generate_homoplasy_llm_dataset(
    newick_tree: str,
    seq_len: int = 300,
    n_convergent_sites: int = 5,
    random_seed: int = 42,
    n_convergent_taxa: int = 2,
    min_taxa_distance: int = 3,
    length_convergent_block: int = 5,
    temp_seqfile: str = "simulated_alignment.fasta",
    simplified_prompt: bool = False,
) -> Dict[str, Any]:
    """
    High-level convenience function that:
      1) Simulates a baseline alignment on the given tree.
      2) Samples n_convergent_taxa that are mutually distant in the tree.
      3) Injects controlled convergent homoplasy into those taxa.
      4) Builds LLM question/label examples.

    Args:
        newick_tree: Newick string OR path to Newick file.
        seq_len: alignment length.
        n_convergent_sites: number of homoplastic blocks to inject.
        random_seed: RNG seed for reproducibility.
        n_convergent_taxa: number of taxa to inject convergent blocks into.
        min_taxa_distance: minimum topological distance between any pair of convergent taxa.
        length_convergent_block: length of each convergent block.
        temp_seqfile: temporary FASTA file for Pyvolve output.
        simplified_prompt: whether to use simplified prompt format.

    Returns:
        result dict with keys:
            - "tree": Newick string
            - "alignment": {taxon: sequence}
            - "examples": list of LLM Q/A dicts
    """
    rng = random.Random(random_seed)

    alignment, newick_str = simulate_alignment_with_pyvolve(
        newick_tree=newick_tree,
        seq_len=seq_len,
        seqfile=temp_seqfile,
    )

    # Validate alignment is not empty
    if not alignment:
        raise RuntimeError(
            f"simulate_alignment_with_pyvolve returned empty alignment.\n"
            f"  Tree: {newick_tree[:100]}...\n"
            f"  Sequence length: {seq_len}"
        )

    # Sample n_convergent_taxa that are mutually distant in the tree
    taxa_list = choose_distant_taxa(
        newick_str,
        n_taxa=n_convergent_taxa,
        min_distance_edges=min_taxa_distance,
        rng=rng,
    )

    alignment_with_homoplasy, convergent_metadata = inject_convergent_blocks_tree_aware(
        alignment,
        newick_str,
        n_blocks=n_convergent_sites,
        taxa_list=taxa_list,
        length_convergent_block=length_convergent_block,
        rng=rng,
    )

    # Validate the output alignment
    if not alignment_with_homoplasy:
        raise RuntimeError("inject_convergent_blocks_tree_aware returned empty alignment.")
    
    seq_len_check = len(next(iter(alignment_with_homoplasy.values())))


    examples = build_llm_examples(
        newick_str=newick_str,
        alignment=alignment_with_homoplasy,
        convergent_metadata=convergent_metadata,
        simplified_prompt=simplified_prompt,
    )

    result = {
        "tree": newick_str,
        "alignment": alignment_with_homoplasy,
        "convergent_taxa": taxa_list,
        "convergent_metadata": convergent_metadata,
        "examples": examples,
    }
    return result
