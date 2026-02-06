# Exploring Relational Reasoning Capabilities in LLMs with REL

This repository contains the code and data for the paper "Exploring Relational Reasoning Capabilities in Large Language Models". We include the code for building the datasets, running the experiments, and analyzing the results.

## REL-A
* RPMs for REL-A1, REL-A2, REL-A3, and REL-A4 were generated using the code available at `https://github.com/IBM/raven-large-language-models`. We use no confounders or noise.
* We will make code for the RPTs in REL-A5, REL-A6, and REL-A7 available upon publication of the final manuscript.

## REL-B
* All questions along with model outputs are in `bio_data/`. Due to the size of the data it is split up but once you pull you can run rebuild_csv.sh to rebuild the csv file.
* Code to build these datasets and run the LLMs on the benchmark is available in `bio_benchmark/`

## REL-C
* The three questions are available at `chem_data/dataset_[c1,c2,c3].jsonl`.
* Code to build the datasets and run the LLMs on the benchmark is available in `chem_benchmark/`.
* Paper results are available in `paper_results/chemistry`.

## Benchmark Format

All datasets are available in a unified JSONL format in the `REL/` directory, organized by domain:
- `REL/chemistry/` - Chemistry datasets (REL-C1, REL-C2, REL-C3)
- `REL/biology/` - Biology datasets (REL-B1)
- `REL/algebra/` - Algebra datasets (REL-A1, REL-A2, REL-A3, REL-A4)

### Shared Format

All datasets follow this common structure:

```json
{
  "id": "unique_identifier",
  "domain": "chemistry" | "biology" | "algebra",
  "task": "task_name",
  "question": "question text",
  "answer": {
    // Domain-specific answer fields (see below)
  },
  "metadata": {
    // Optional metadata
  }
}
```

**Required fields:**
- `id`: Unique identifier for the record
- `domain`: One of "chemistry", "biology", or "algebra"
- `task`: Task identifier (e.g., "REL-C1", "REL-B1", "REL-A1")
- `question`: The question text/prompt
- `answer`: Object containing domain-specific answer fields

### Chemistry Format

**Task types:**
- `REL-C1`: Isomer set yes/no questions
- `REL-C2`: Largest common motif identification
- `REL-C3`: Missing isomers completion

**Answer structure:**
- `label`: "Yes" or "No" (for REL-C1)
- `smiles`: Single SMILES string (for REL-C2)
- `missing_smiles`: List of SMILES strings (for REL-C3)
- `molecules`: List of input molecule SMILES (optional, present in all tasks)

**Example (REL-C1):**
```json
{
  "id": "q2_n5_00001",
  "domain": "chemistry",
  "task": "REL-C1",
  "question": "Is this list of molecules a set of *constitutional isomers*...",
  "answer": {
    "label": "Yes",
    "molecules": ["CC(C)C(Br)=CBr", "CC1(Br)CC1(C)Br", ...]
  },
  "metadata": {"formula": "C5H8Br2", "constructed_label": "Yes"}
}
```

**Example (REL-C2):**
```json
{
  "id": "q1a_n5_00001",
  "domain": "chemistry",
  "task": "REL-C2",
  "question": "Given the following list of SMILES, what is the largest...",
  "answer": {
    "smiles": "O=CC1CCC(=O)N1",
    "molecules": ["O=C1CCC(C(=O)Oc2ccc(O)cc2)N1", ...]
  },
  "metadata": {"mcs_num_atoms": 8, "mcs_num_bonds": 8, ...}
}
```

**Example (REL-C3):**
```json
{
  "id": "q3_given5_00301",
  "domain": "chemistry",
  "task": "REL-C3",
  "question": "Given the following list of constitutional isomers...",
  "answer": {
    "missing_smiles": ["BrC(Br)C1CCC1", "BrC(Br)CC1CC1", ...],
    "molecules": ["BrCC(Br)C1CC1", "CC1CC(Br)C1Br", ...]
  },
  "metadata": {"formula": "C5H8Br2", "universe_size": 88, ...}
}
```

### Biology Format

**Task:**
- `REL-B1`: Homoplasy detection

**Answer structure:**
- `label`: "yes" or "no"
- `taxa`: List of integers (taxa IDs involved in homoplasy, empty if label is "no")

**Validation rules:**
- If `label` is "yes", `taxa` must be a non-empty list
- If `label` is "no", `taxa` must be an empty list

**Example:**
```json
{
  "id": "biology_REL-B1_00001",
  "domain": "biology",
  "task": "REL-B1",
  "question": "Homoplasy refers to structured convergence...",
  "answer": {
    "label": "yes",
    "taxa": [15, 49, 18, 28, 20, 39, 29, 42, 16, 31]
  },
  "metadata": {}
}
```

### Algebra Format

**Task types:**
- `REL-A1`, `REL-A2`, `REL-A3`, `REL-A4`: Raven's Progressive Matrices

**Answer structure:**
- `target`: Integer index (0-7) of the correct answer in the choices

**Note:** The raw panel data (`panels` and `choices`) is stored in the `metadata` field for reference.

**Example:**
```json
{
  "id": "algebra_REL-A1_00001",
  "domain": "algebra",
  "task": "REL-A1",
  "question": "Complete the Raven's progressive matrix. Only return the missing panel index (1-8)!\n\nPanel 0:\n[639.43, 25.01, 275.03]\n...\n\nAnswer set:\nAnswer 1: [123.45, 67.89, ...]\n...",
  "answer": {
    "target": 2
  },
  "metadata": {
    "panels": [[[639.43, 25.01, 275.03], ...], ...],
    "choices": [[[123.45, 67.89, ...], ...], ...]
  }
}
```