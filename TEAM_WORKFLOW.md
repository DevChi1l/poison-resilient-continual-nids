# Three-Developer Workflow

## Ownership and branches

| Developer | Branch | Owns | Deliverable for Review 2 |
| --- | --- | --- | --- |
| A - Model/NIDS | `feature/model-nids` | `src/models/`, `src/training/`, model entry points | Small Transformer, training, inference, checkpoint API |
| B - Data/Continual Learning | `feature/data-continual` | `src/data/`, `src/continual_learning/`, data metadata | CSV preprocessing, Task 1/2 stream, replay/incremental driver |
| C - Novelty/Poisoning/Evaluation | `feature/novelty-poisoning` | `src/novelty/`, `src/poisoning/`, `src/evaluation/`, `experiments/` | Novelty baseline, label-flip harness, metrics and saved results |

Developer B coordinates the integration after components expose the interfaces in `ARCHITECTURE.md`.

## Shared-file owner

Only Developer B changes shared integration files during the Review-2 build:

- `run_pipeline.py`
- `README.md`
- `configs/`
- `requirements.txt` or `pyproject.toml`
- `PROJECT_STATUS.md`
- `ARCHITECTURE.md`
- `HANDOFF.md`

Other developers propose changes in their pull request or issue rather than editing these files directly.

## Coordination rules

- Create a focused branch from updated `main`; never work directly on `main`.
- One owner changes one core module. Do not split a module among agents.
- Before implementing, create a minimal smoke test or example that validates the owned interface.
- When blocked, implement against the documented interface with a temporary mock, and mark it clearly.
- Merge through a pull request after smoke-test evidence is recorded.

## Handoff format

Every session updates `PROJECT_STATUS.md` and `HANDOFF.md` with date/time, developer, branch, completed work, current work, changed files, tests, observed results, known problems, decisions, and next action.
