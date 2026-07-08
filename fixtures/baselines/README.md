# External Baseline Contract

This directory holds **immutable stored baseline outputs** for external
systems (e.g. `glm_5_2.json`, `deepseek.json`, `claude.json`). It is empty by
design until the owner stores a real baseline.

Rules (enforced by `src/amali/benchmarks/comparison.py` and its tests):

- A missing baseline file makes every external comparison **NOT_PROVEN**.
  No code path may fabricate, estimate, or default a baseline score.
- A stored baseline must be an immutable capture of the external system's
  raw outputs on the *same* benchmark items, scored by the *same* local
  deterministic scorer. Storing it is an owner action, never automatic.
- Expected file shape: `{"baseline_name": str, "captured_at": iso8601,
  "items": [{"item_id": str, "output": str}, ...]}`.
- Nothing in this repository downloads baselines. Adding one requires the
  owner to obtain outputs legitimately and commit them explicitly.

Claim boundary: the only comparison the Base AI Gate proves is
*AMALI deterministic pipeline vs naive deterministic baseline* on local
items. Frontier-superiority claims are out of scope without this contract.
