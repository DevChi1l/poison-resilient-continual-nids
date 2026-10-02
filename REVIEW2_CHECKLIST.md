# Review-2 Checklist

## Working prototype requirements

- [ ] Documented, manageable flow-level CSV subset is available locally.
- [x] Preprocessing fits on training data only and emits the canonical data contract.
- [ ] Small Transformer can train and predict in a CPU smoke run.
- [x] Classification metrics are generated and saved in recorded Kaggle runs.
- [x] Confidence-threshold novelty baseline evaluated on held-out Task 2 classes; recall was low.
- [x] Task 1 then Task 2 clean incremental experiment ran on Kaggle.
- [x] Task 1 scores before/after Task 2 and signed forgetting were recorded.
- [x] Static and replay-label poisoning comparisons ran on bounded Kaggle subsets.
- [x] Clean/static-poisoning outputs were saved separately.
- [ ] One command executes the configured demo pipeline.
- [ ] README documents setup, input data, and demo command.
- [x] Architecture and unified A+B+C ownership are documented.

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
