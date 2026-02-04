# Algebra benchmark

Self-contained numerical RPM/RPT benchmark: no visual inputs, no confounders. This repository is **independent** — run everything from this directory; no parent repo required.

## Environment setup (rpmllm)

All code in this folder is written for **Python 3.10** and is intended to be run inside the **`rpmllm`** environment. You can create and use it with **mamba** or **micromamba**.

### 1. Create the environment

With **mamba** (or **conda**):

```bash
mamba create --name rpmllm python=3.10
mamba activate rpmllm
```

With **micromamba**:

```bash
micromamba create --name rpmllm python=3.10
micromamba activate rpmllm
```

### 2. Install dependencies

From the **repository root** (this directory):

```bash
pip install -r requirements.txt
```

This installs PyYAML (needed for reading/writing dataset config YAML). The rest uses only the Python standard library.

### 3. Optional: PyTorch and CUDA

If you plan to run model-based evaluation (e.g. Hugging Face or other GPU code) from this repo, install PyTorch in the same environment, for example:

```bash
# Example for CUDA 12.1 (adjust cuda version as needed)
mamba install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Dataset generation and the evaluation stub do **not** require PyTorch or a GPU.

### 4. Verify the setup

From the **repository root** with `rpmllm` activated:

```bash
mamba activate rpmllm   # or: micromamba activate rpmllm

python tasks.py
python generators.py
python run_eval.py example_datasets/REL-A1/config.yml --stub
```

If these run without errors, the environment is ready.

## Task mapping

| Task ID   | Ground rule              |
|-----------|--------------------------|
| REL-A1    | constant                 |
| REL-A2    | progression              |
| REL-A3    | distribute-three / permutation |
| REL-A4    | arithmetic               |
| REL-A5    | 4-spatiotemporal         |
| REL-A6    | 5-spatiotemporal         |
| REL-A7    | neighborhoodsum          |

See `tasks.py` for `TASK_TO_RULE`, `build_config()`, and helpers.

## Data format

One **sample** is a JSON object with:

| Field     | Type   | Description |
|-----------|--------|-------------|
| `panels`  | list   | Context panels: list of numerical matrices (2D) or tensors (3D). Each panel is a nested list of numbers (e.g. 3×3 = list of 3 rows, each row list of 3 numbers). Typically 8 panels for a 3×3 RPM. |
| `target`  | int    | Index of the correct answer in `choices` (0-based). |
| `choices` | list   | Candidate answer panels: list of 8 panels (same shape as one context panel). |

A **dataset** is a JSON file in one of two shapes:

- **Object:** `{ "0": sample0, "1": sample1, ... }` — keys are sample IDs.
- **Array:** `[ sample0, sample1, ... ]` — order is sample index.

Use `validate_sample(sample)` and `load_sample_from_dataset(data, index)` from `format.py` to validate and read by index. See `format.py` for the full schema and `EXAMPLE_SAMPLE`.

## Dataset config schema

The **dataset config** (e.g. `config.yml` written by `create_dataset`) uses a minimal YAML under a top-level `data` key.

**Required keys (under `data`):**

| Key         | Type   | Description |
|-------------|--------|-------------|
| `path`      | string | Directory containing the dataset file. |
| `config`    | string | Config string from `build_config(task, gridsize, maxval)`. |
| `gridsize`  | int    | Matrix/tensor size (e.g. 3, 9, 15, 30). |
| `nattr`     | int    | Number of attributes (e.g. 3 for matrix, 1 for irpt). |
| `nshow`     | int    | Number of context panels shown (e.g. 3). |
| `ntest`     | int    | Number of samples (test/eval size). |

**Optional keys:** `task`, `num_samples`, `maxval` (written by the CLI for convenience).

**Excluded (do not use):** `nconf`, `uncertainty`, `maxval_uncert`, `permutation`, `ood_set_attr`, `ood_set_rule`, `offset`, `angle`, `scaling`, etc.

Use `validate_config(data_section)` and `build_data_config(...)` from `config_schema.py` to validate and build the `data` dict.

## Generation

All seven tasks (REL-A1 … REL-A7) can be generated with `--generate`:

- **REL-A1 (constant)** — all 8 context panels are the same matrix; answer is that matrix; distractors are random.
- **REL-A2 (progression)** — 9 matrices form a progression (delta in {-2,-1,1,2}); context = first 8, answer = 9th.
- **REL-A3 (distribute-three)** — one matrix with cyclic row permutations (first row = n distinct values, each row = roll(prev, ±1)); all 8 panels and answer are that matrix.
- **REL-A4 (arithmetic)** — one matrix where each row has (a, b, a+b); all 8 panels and answer are that matrix.
- **REL-A5 … REL-A7** (4-spatiotemporal, 5-spatiotemporal, neighborhoodsum) use placeholder samples until rule-specific generators are added.

The REL-A1–REL-A4 generators are inspired by I-RAVEN-X–style generation logic and adapted to numerical-only panels.

Create a dataset for one task (run from the repository root):

```bash
python create_dataset.py --task REL-A1 --gridsize 3 --num_samples 125 --output_dir example_datasets/REL-A1 --generate --seed 42
```

Generate example datasets (125 samples per task) for all seven tasks:

```bash
python create_example_datasets.py
```

Without `--generate`, the CLI only writes the manifest and an empty `dataset.json`.

## Evaluation

- **rpm_numeric.py**: `sample_to_context()`, `sample_to_answer_choices()`, `build_query()` — turn a numerical sample into prompt text.
- **solver_pred.py**: `text2num()` (parse "Answer 3" → index 2), `majority_vote()`, `guard_answer()`.
- **loader.py**: `load_config()`, `load_dataset()`, `load_config_and_dataset()` — read config.yml and dataset.json.
- **run_eval.py**: `run_eval(config_path, model_fn=None, ...)` and CLI.

To verify your setup, run with the **stub model** (always predicts Answer 1) to test the pipeline:

```bash
python run_eval.py example_datasets/REL-A1/config.yml --stub
```

To use your own model, pass a callable `model_fn(prompt, query) -> response_string` to `run_eval(..., model_fn=your_fn)`.
