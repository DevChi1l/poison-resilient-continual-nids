# Unified Team Workflow

## Ownership and branches

Developers A, B, and C now work as one owner. The team owns all implementation
and integration areas: `src/models/`, `src/training/`, `src/data/`,
`src/continual_learning/`, `src/novelty/`, `src/poisoning/`, `src/mitigation/`,
`src/evaluation/`, experiment entry points, configurations, and shared project
documentation. Work is not assigned back to a former lettered owner.

Use focused branches for focused changes. Preserve the existing historical
branches because they show where the model and data foundations originated.
The ongoing unified Review-2 work uses `feature/unified-novelty`, originally
based on the verified `feature/model-nids` foundation. The branch now also
contains clean continual/replay, static poisoning, and the prepared
replay-label consistency baseline. Do not assume `main` has these APIs.

## Shared-file owner

The unified team owns shared integration files, including:

- `run_pipeline.py`
- `README.md`
- `configs/`
- `requirements.txt` or `pyproject.toml`
- `PROJECT_STATUS.md`
- `ARCHITECTURE.md`
- `HANDOFF.md`

Shared-file changes must remain scoped, documented, and compatible with the
interfaces in `ARCHITECTURE.md`.

## Coordination rules

- Create a focused branch from the verified foundation required by the task;
  never work directly on `main`.
- Keep each task focused and do not refactor stable core modules incidentally.
- Before implementing, create a minimal smoke test or example that validates the owned interface.
- When blocked, implement against the documented interface with a temporary mock, and mark it clearly.
- Merge through a pull request after smoke-test evidence is recorded; do not
  merge feature work directly into `main` during an implementation task.

## Handoff format

Every session updates `PROJECT_STATUS.md` and `HANDOFF.md` with date/time, developer, branch, completed work, current work, changed files, tests, observed results, known problems, decisions, and next action.
