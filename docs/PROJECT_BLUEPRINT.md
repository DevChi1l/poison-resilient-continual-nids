# Project Blueprint

## Research goal

Evaluate whether a lightweight secure-replay controller can reduce the persistence of poisoned training evidence in a continual Transformer NIDS without preventing legitimate, emerging attack classes from being learned.

## Scope decisions

| Decision | Initial choice | Reason |
| --- | --- | --- |
| Data type | Flow-level tabular data | Feasible on student hardware and suitable for tabular Transformers. |
| Primary dataset | Combined local flow collection | The available Parquet file has compatible tabular flow features; its per-row source-dataset provenance is unavailable and must be stated as a limitation. |
| Learning setting | Class-incremental learning (CI) | Matches the new-attack learning research question. |
| Continual baseline | Class-balanced experience replay | Strong, understandable anti-forgetting baseline. |
| Core attack | Label flipping | Controlled, explainable first poisoning threat. |
| Initial defense | Label consistency + prototype distance | Lightweight and separately measurable. |
| Novelty handling | Quarantine rather than immediate discard | Avoids conflating a novel class with poisoned data. |

These are starting choices, not claims of optimality. Changes require a short decision record and an updated experiment plan.

## Architecture boundaries

```text
src/
  data/                Dataset loading, validation, splitting, preprocessing
  models/              Tabular Transformer and classifier interfaces
  continual_learning/  Replay buffer, task streams, update strategies
  novelty/             Thresholding, prototype support, quarantine logic
  poisoning/           Threat-model simulations and secure-admission signals
  evaluation/          Metrics, reports, figures, experiment summaries
  utils/               Seeding, configuration, logging, shared helpers
```

Keep attacks separate from defenses. A poisoning harness should generate an explicitly recorded corruption; it must never silently alter normal training data.

## Experiment ladder

| Stage | Deliverable | Exit criterion |
| --- | --- | --- |
| 0. Data audit | Reproducible preprocessing and class/task split | Schema, labels, counts, and split seed recorded. |
| 1. Static baseline | Transformer trained on one fixed split | Macro-F1 and per-class metrics saved. |
| 2. Continual baselines | Sequential fine-tuning and replay | Old/new-class performance and forgetting reported. |
| 3. Poisoning | Label-flip harness at bounded rates | Clean vs poisoned comparison reproducible. |
| 4. Defense baselines | Confidence and/or robust-distance filter | Poison admission and clean rejection measured. |
| 5. Proposed controller | Admission + quarantine design | Security, retention, novelty, and efficiency trade-off reported. |
| 6. Validation | Repeated seeds and secondary data setting | Mean/spread and limitations documented. |

## Required metrics

- Detection: macro-F1, per-class precision/recall/F1, confusion matrix.
- Continual learning: old-class retention, new-class performance, average forgetting.
- Security: poison admission rate, poison-detection TPR/FPR, attack success rate when a backdoor is implemented.
- Novelty: legitimate-new-class quarantine/rejection rate and eventual learning rate.
- Efficiency: replay size, admission latency, update time, model size.

## Threat model v1

The attacker may corrupt a bounded portion of candidate update/replay samples by changing labels or selected controllable flow features. The attacker cannot modify model weights, gradients, or the operating system. This is a semantic-data-poisoning model; file hashes alone are not a defense for it.

## Definition of done

A research change is complete only when it has:

1. Focused implementation and basic tests.
2. A documented configuration and fixed seed.
3. A runnable command or notebook entry point.
4. Recorded output location and metrics.
5. A brief interpretation that distinguishes observed results from hypotheses.
