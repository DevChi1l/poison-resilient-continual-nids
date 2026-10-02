# Agent Instructions

This repository is a Review-2 capstone prototype. Developers A, B, and C now
operate as one unified owner of the model, data, continual-learning, novelty,
poisoning, mitigation, evaluation, and integration code. The repository, not a
chat session, is the source of truth.

## Start-of-session protocol

Before changing code, read in this order:

1. `PROJECT_CONTEXT.md`
2. `TEAM_WORKFLOW.md`
3. `ARCHITECTURE.md`
4. `PROJECT_STATUS.md`
5. `HANDOFF.md`
6. `REVIEW2_CHECKLIST.md`

Then run `git status --short --branch`, inspect the recent log, and work only in the assigned ownership area.

## Non-negotiable rules

- Do not fabricate data, metrics, experiment results, or literature claims.
- Preserve unrelated work and document interface changes before changing an existing core module.
- Do not download large datasets or commit datasets, checkpoints, secrets, or generated bulk results without explicit coordination.
- Prefer a working, small, testable baseline over sophisticated unfinished methods.
- Use fixed seeds and record inputs, configuration, and generated outputs for every experiment.
- Before ending a work session, update `PROJECT_STATUS.md` and `HANDOFF.md`.

## Review-2 priority

The required goal is one end-to-end, reproducible prototype: data loading, preprocessing, a small Transformer classifier, prediction/metrics, confidence-threshold novelty baseline, a two-task continual-learning demonstration, and configurable label-flip poisoning. Advanced mitigation is optional.
