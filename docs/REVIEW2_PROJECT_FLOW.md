# Review 2: End-to-End Project Flow

This document explains the intended Review-2 prototype from input data to saved experiment outputs. It describes the repository's documented design; it is not evidence that the pipeline already runs.

## Current repository state

At the time this guide was updated, the repository contains planning/interface documents and a local `data/cic-collection.parquet` file of about 979 MiB, but no dependency manifest, Python source implementation, executable pipeline, tests, or experiment results. The file is tracked in local commit `80018dc` (`Added dataset`), and Git reports local `main` one commit ahead of `origin/main`. This conflicts with the repository rule against committing raw datasets: do not push that commit. Preserve the local file while untracking it and amending the unpushed commit, following [Developer B's guide](DEVELOPER_B_GUIDE.md). The dataset's provenance, schema, and labels have not been verified, so do not assume it is the intended CIC-IDS2017 data. No experiment has been run. Check `PROJECT_STATUS.md` for future updates.

The repository defines three active technical roles: Developer A (model/training), Developer B (data/continual learning and integration), and Developer C (novelty/poisoning/evaluation). An earlier team prompt describes a fourth integration/QA person. Agree as a team whether that person is an additional reviewer/helper or whether the repository's three-role ownership table should be updated. Until then, follow the checked-in `TEAM_WORKFLOW.md` ownership and branch rules.

## Pipeline at a glance

```text
Documented flow CSV/Parquet subset
       |
       v
Audit columns, labels, rows, missing/non-finite values, duplicates
       |
       v
Deterministic split and train-only preprocessing fit
       |
       +---------------------------> Frozen validation/test partitions
       |
       v
Prepared numeric features + stable integer class IDs
       |
       +--> Static Transformer baseline --> class predictions/probabilities
       |                                      |
       |                                      +--> classification metrics
       |                                      +--> confidence novelty baseline
       |
       +--> Task 1 data --> train/evaluate Task 1
                             |
                             +--> evaluate held-out future classes as unknown
                             |
                             v
                       Task 2 data --> update model
                             |
                             +--> re-evaluate Task 1 and Task 2
                             +--> calculate forgetting/retention
       |
       +--> Clean training labels / seeded label-flip labels
                                      |
                                      v
                          Clean vs poisoned comparison
                                      |
                                      v
                  Config + seed + commit + metrics saved as JSON
```

## Stage-by-stage flow and ownership

### 1. Choose and document a manageable dataset input — Developer B

The project blueprint selects CIC-IDS2017 as the initial primary dataset. A local `data/cic-collection.parquet` has been added, but its source, schema, labels, and relationship to CIC-IDS2017 are unverified. Inspect and identify it before treating it as the chosen dataset. Record source/version, file name, labels, selection rule, row count, and checksum or other useful provenance in `data/metadata/`. Do not commit raw data or large processed files. See [Developer B's guide](DEVELOPER_B_GUIDE.md) for how to preserve this local file while removing it from the unpushed commit.

For Review 2, one reproducible subset is enough. Keep synthetic records strictly for code smoke checks; never report them as NIDS experiment evidence.

### 2. Audit, split, and preprocess — Developer B

Inspect the actual CSV schema and label values before writing assumptions into code. Define the target label and fixed feature order. Exclude identifiers or fields that would reveal a target or duplicate the split. Decide how to handle invalid rows, infinities, missing values, duplicate flows, and categorical columns, and record those rules.

Create seeded train/validation/test partitions before fitting data-dependent transforms. Fit scaling/encoding only on training data and apply that fitted transform to the other partitions. Save or reconstruct the transform and label mapping so a resumed run uses the same representation. Keep held-out test data out of both fitting and task updates.

Canonical interface from `ARCHITECTURE.md`:

```python
features: numpy.ndarray       # (n_samples, n_features), float32
labels: numpy.ndarray         # (n_samples,), integer class IDs
feature_names: list[str]      # fixed order matching features columns
class_names: dict[int, str]   # integer ID -> original readable label
```

The data interface must also expose or document split membership and enough metadata to reproduce the run. Stable class IDs must not silently change between Task 1 and Task 2.

### 3. Train the static Transformer baseline — Developer A

Developer A owns `src/models/` and `src/training/`. A small tabular Transformer accepts the prepared numeric features and integer labels, trains using configurable settings, and supports prediction and class probabilities. GPU is intended for medium/final experiments; a tiny CPU configuration is needed for local integration. The classifier interface and the `class_ids` order are defined in `ARCHITECTURE.md`.

Do not treat this classifier as the research contribution by itself. Save its configuration and checkpoint using the team's agreed artifact policy.

### 4. Produce known-class metrics and novelty baseline — Developer C

Developer C owns novelty and evaluation. Classification metrics should include macro-F1 and per-class precision/recall/F1; accuracy can be reported as a supplementary metric. Keep a confusion matrix when practical.

The Review-2 novelty baseline uses model probabilities only: a sample is marked `UNKNOWN` when its maximum class probability is below a configured threshold. Select/calibrate the threshold using validation data, then evaluate on a held-out set containing known and, where the task design permits, genuinely unseen classes. Report known/unknown behavior separately; a threshold baseline is not a sophisticated novelty claim.

### 5. Run the continual-learning sequence — Developer B with A's model API

Define at least two class-incremental tasks and publish their class membership and sample counts. Keep a fixed evaluation set for Task 1 classes and a separate evaluation set for Task 2 classes. Train on Task 1, measure its score, update on Task 2, and measure Task 1 again plus Task 2 performance. The change in Task 1 performance is the basic forgetting/retention result.

Before implementation, agree with Developer A how the classifier's output head handles newly introduced classes. The current classifier contract states prediction shapes and class IDs but does not yet specify an output-head expansion method. If replay is implemented, keep it optional and bounded; it must not delay a working sequential fine-tuning baseline.

### 6. Compare clean and poisoned updates — Developer C

Developer C owns the label-flip harness and poisoning evaluation. Preserve original labels, make the corruption rate and seed configurable, record changed row indices and achieved rate, and compare a clean (`0%`) condition with at least one nonzero label-flip condition. Do not poison validation or test labels. Keep the poisoned training run's outputs separate from the clean run.

Any mitigation is a separate optional experiment. State exactly what it filters and measure both poison rejection and legitimate-data rejection; do not call a generic filter a complete defense.

### 7. Integrate and save an auditable run — Developer B (integration lead in repository docs)

The eventual `run_pipeline.py` should take one config and call the components in a documented order. It should save a JSON result manifest under `results/` containing at least the experiment name, Git commit, dataset provenance, subset/split/task definitions, seeds, model and training settings, novelty threshold, poisoning settings, metrics, output paths, and limitations. Keep bulky artifacts out of Git.

There is no runnable command yet. The README gives this only as the intended future command, not an available command:

```bash
python run_pipeline.py --config configs/review2_smoke.yaml
```

Mark it runnable only after its code, config, dependencies, input instructions, and a successful observed smoke run are present.

## Review-2 completion checks

- A real, documented, manageable flow-level CSV subset is available to the team.
- Preprocessing yields the canonical arrays, stable feature order, class mapping, and train-only fitted transforms.
- Train/validation/test splits and Task 1/Task 2 class membership are reproducible.
- The Transformer trains and predicts on the tiny smoke configuration.
- Static classification, confidence novelty, sequential Task 1/Task 2, and clean/poisoned runs produce saved metrics.
- Task 1 score before and after Task 2 is reported, along with Task 2 performance and forgetting.
- A single documented command runs the integrated smoke pipeline and creates an auditable result manifest.
- README, architecture, status, and handoff reflect what was actually implemented and run.

## Explicitly later work

Advanced embedding-space novelty methods, sophisticated replay defenses, backdoor attacks, broad multi-dataset studies, repeated-seed statistical analysis, dashboards, deployment, and broad hyperparameter searches belong to Review 3 unless the Review-2 pipeline is already stable.

## Related repository documents

- [Project context](../PROJECT_CONTEXT.md)
- [Team workflow](../TEAM_WORKFLOW.md)
- [Architecture and interfaces](../ARCHITECTURE.md)
- [Project status](../PROJECT_STATUS.md)
- [Review-2 checklist](../REVIEW2_CHECKLIST.md)
- [Developer B step-by-step guide](DEVELOPER_B_GUIDE.md)
