Extraneous-Info Physics Dataset
================================

This module generates solvable physics problems that intentionally include extraneous information. It produces two JSONL files suitable for LLM fine-tuning with supervised learning and reinforcement learning from human feedback (RLHF)/preference optimization.

Quick start
-----------

Run the generator to create a small sample dataset:

```bash
uv run python 03_extraneous_info_dataset/generate_extraneous_dataset.py \
  --count 100 \
  --seed 42 \
  --outdir 03_extraneous_info_dataset/samples
```

Outputs (JSONL)
---------------

- `supervised.jsonl`: One record per problem with solution steps and final answer for standard supervised fine-tuning.
- `preference.jsonl`: Preference pairs (`chosen` vs `rejected`) to train reward models or do DPO/PPO-style preference optimization.
- `schema_*.jsonl`: Minimal schema hints.

Supervised record schema
------------------------

Each line is a JSON object with:

```json
{
  "category": "kinematics|newton|energy|momentum|circuits",
  "prompt": "str",
  "important_facts": ["str", "..."],
  "extraneous_facts": ["str", "..."],
  "solution_steps": ["str", "..."],
  "final_answer": "str",
  "metadata": {"equation": "..."}
}
```

Preference pair schema
----------------------

Each line is a JSON object with:

```json
{
  "category": "...",
  "prompt": "...",
  "chosen": {
    "reasoning": "high-quality reasoning using important facts",
    "answer": "correct numeric/units",
    "uses_extraneous": false,
    "correct": true
  },
  "rejected": {
    "reasoning": "mentions extraneous details / ignores physics",
    "answer": "perturbed or incorrect",
    "uses_extraneous": true,
    "correct": false
  },
  "metadata": {"equation": "..."}
}
```

Intended training usage
-----------------------

- Supervised fine-tuning: Concatenate `prompt` with an instruction and train to generate `solution_steps` + `final_answer`.
- Preference training (RLHF/DPO): Train a reward model or DPO directly with `chosen` preferred over `rejected` given the same `prompt`.

Design notes
------------

- Problems are algorithmically generated across classical mechanics and circuits with built-in extraneous details (colors, names, ambient conditions) that do not affect the solution.
- Numeric formatting uses compact significant figures or scientific notation when appropriate.
- All problems are intended to be solvable with standard undergraduate physics formulas indicated in `metadata`.


