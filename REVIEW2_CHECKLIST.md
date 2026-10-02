# Review-2 Checklist

## Working prototype requirements

- [ ] Documented, manageable flow-level CSV subset is available locally.
- [x] Preprocessing fits on training data only and emits the canonical data contract.
- [ ] Small Transformer can train and predict in a CPU smoke run.
- [ ] Classification metrics are generated and saved.
- [x] Confidence-threshold novelty baseline marks low-confidence samples as unknown (synthetic component verification only).
- [ ] Task 1 then Task 2 incremental experiment runs.
- [ ] Task 1 score is captured before and after Task 2; forgetting is calculated.
- [ ] Label flipping runs at configurable rates, including clean (`0%`) and poisoned conditions.
- [ ] Clean and poisoned outputs are saved separately.
- [ ] One command executes the configured demo pipeline.
- [ ] README documents setup, input data, and demo command.
- [ ] Architecture and component ownership can be explained by the team.

## Time-boxed plan

| Window | Owner | Deliverable |
| --- | --- | --- |
| Hour 0-1 | All | Read persistent context, create branches, freeze dependencies/data subset. |
| Hours 1-4 | A / B / C in parallel | Model / data-task stream / novelty-poisoning-evaluation smoke tests. |
| Hours 4-6 | B with A and C support | Integrate first complete run and fix interface mismatches. |
| Hours 6-7 | Owners | Finish the smallest missing mandatory component. |
| Hours 7-8 | C with B | Saved results, README, demo evidence. |
| Hours 8-9 | All | Stabilize, rerun smoke configuration, rehearse explanation. |

## Deferred to Review 3 unless the core pipeline is stable

- Timing backdoor attack
- Advanced replay defense and periodic audit
- Embedding-space novelty detector
- Secondary dataset and repeated-seed study
- Dashboard, deployment, and broad hyperparameter tuning
