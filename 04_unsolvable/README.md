Inconsistent/Unsolvable Physics Dataset
=======================================

This module generates physics problems that contain internal inconsistencies (contradictory numerical values, mutually exclusive statements, or mismatched units). These are intended for training/evaluating LLMs to detect and explain inconsistencies rather than solve the problem numerically.

Quick start
-----------

Run the generator to create a small sample dataset:

```bash
python 04_unsolvable/generate_inconsistent_dataset.py \
  --count 50 \
  --seed 123 \
  --outdir 04_unsolvable/samples
```

Outputs (JSONL)
---------------

- `unsolvable.jsonl`: One record per problem with annotated inconsistencies and an explanation (`rationale`).
- `schema_unsolvable.jsonl`: Minimal schema hints.

Record schema
-------------

Each line is a JSON object with:

```json
{
  "category": "kinematics|newton|energy|momentum|circuits",
  "prompt": "str",
  "important_facts": ["str", "..."],
  "inconsistent_facts": ["str", "..."],
  "inconsistencies": [
    {
      "type": "contradictory_value|unit_mismatch|equation_conflict",
      "field": "e.g., mass|force|time|voltage",
      "value_a": "str",
      "value_b": "str",
      "units": "str",
      "description": "human-readable explanation"
    }
  ],
  "rationale": "why the problem is unsolvable as stated",
  "label": "inconsistent",
  "metadata": {"equation": "..."}
}
```

Intended training usage
-----------------------

- Detection/critique training: Given `prompt`, train models to identify `inconsistencies` and produce a `rationale` without attempting a numeric answer.
- Data validation tasks: Use as negative cases for problem generators or graders.

Design notes
------------

- Each problem is built from a solvable core template, then minimally perturbed to introduce 1–2 inconsistencies.
- Numeric formatting matches `03_extraneous_info_dataset` for consistency.

Solvability-Labeled Physics Dataset
===================================

This module produces simple physics problems labeled as solvable or unsolvable. Prompts are length-normalized to mitigate leakage from text length.

Quick start
-----------

```bash
python 04_unsolvable/generate_solvability_dataset.py \
  --count 200 \
  --seed 42 \
  --target_chars 280 \
  --jitter 0 \
  --outdir 04_unsolvable/samples
```

Outputs (JSONL)
---------------

- `solvability.jsonl`: One record per problem labeled with `solvable: true|false`.
- `schema_solvability.jsonl`: Minimal schema hints.

Record schema
-------------

Each line is a JSON object with:

```json
{
  "category": "kinematics|newton|energy|momentum|circuits",
  "prompt": "str (length-normalized)",
  "solvable": true,
  "metadata": {"type": "..."}
}
```

Notes on length normalization
-----------------------------

- `--target_chars` clamps or pads prompts to approximately the same length.
- Optional `--jitter` can add small ± variability to avoid identical lengths.

Intended training usage
-----------------------

- Binary classification (is the problem solvable?).
- As a filtering or routing signal before solution generation.


