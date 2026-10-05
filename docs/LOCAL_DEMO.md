# Local artifact demonstration

The Streamlit interface is a bounded local demonstration over existing private
artifacts. It does not train, mutate checkpoints, rewrite replay data, or claim
that playback metrics were produced by the UI.

## Set up

Use Python 3.13 for the simplest PyTorch compatibility and to match the
recorded Kaggle environment:

```bash
python3.13 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python -m pip install 'torch==2.11.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
```

Copy and edit `configs/demo_artifacts.example.json` only if your private
artifact locations differ. Paths are relative to its `base_dir`; absolute
paths are also accepted. Keep correct SHA-256 values so a mismatched
checkpoint, preprocessor, or result ZIP fails closed.

The sample exporter also requires the original Parquet and persisted full-test
row indices declared in the `dataset` configuration section. The returned
artifact bundle did not include the split arrays. On this machine they were
re-created outside Git with the original seed-42 split generator; the resulting
`splits.npy` SHA-256 exactly matched the run's recorded
`445182b087636e68089a5efd5d9645a1cc41109940d17ee778b81c935f0d2ca3`.
If the arrays are absent elsewhere, reproduce and verify them without training:

```bash
.venv-demo/bin/python -c "from src.data.large_data import prepare_full_partitions; prepare_full_partitions('data/cic-collection.parquet', '.local/nids-demo/split', seed=42, progress=print)"
sha256sum .local/nids-demo/split/splits.npy \
  .local/nids-demo/split/test_indices.npy
```

Expected hashes are `445182b0...2ca3` and `89f24093...9a8`. Run the
read-only preflight before opening the app:

```bash
.venv-demo/bin/python scripts/demo_preflight.py \
  --config configs/demo_artifacts.example.json \
  --database .local/nids-demo-rehearsal/quarantine.sqlite3
```

It checks configured hashes, both checkpoint/preprocessor contracts, replay
dimensions, named Task 2 evidence, held-out data provenance, and the SQLite
destination. A failure includes a concrete `Fix:` line.

Export a deterministic five-per-class held-out sample. This scans labels and
materializes only 40 selected rows; it never runs the model or selects rows by
prediction correctness:

```bash
.venv-demo/bin/python scripts/export_demo_sample.py \
  --config configs/demo_artifacts.example.json \
  --output-dir uploads/rehearsal \
  --rows-per-class 5 --seed 42
```

The upload is `uploads/rehearsal/heldout_flows.csv`: the static preprocessor's
54 raw features in exact order plus `ClassLabel`. Original Parquet row IDs are
separate in `uploads/rehearsal/heldout_flow_provenance.csv`. Both and their
manifest are ignored by Git.

Launch from the repository root:

```bash
.venv-demo/bin/streamlit run streamlit_app.py -- \
  --config configs/demo_artifacts.example.json \
  --database .local/nids-demo-rehearsal/quarantine.sqlite3
```

## Required private artifacts

The example configuration expects:

- static flow inference: clean eight-class `best.pt` plus the matching
  `preprocessing_static.json`;
- poisoning/quarantine: five-class Task 1 `best.pt`, its matching
  `preprocessing_continual.json`, `replay_buffer.npz`, and saved Task 1
  `novelty.json`;
- playback and gate calibration: the original `data/task2/results.zip` with
  SHA-256
  `0c73a44cca06cb92f7c21c98652808f739df67236d58d790f5cf4b93027d832e`.

The checkpoint, replay buffer, uploaded flow file, result ZIP, and generated
SQLite queue remain outside Git. The default queue is
`.local/nids-demo/quarantine.sqlite3`, an ignored path.
Use `--database` for a faculty run so rehearsal and earlier review decisions
are not mixed or overwritten.

## Flow input contract

Upload CSV or Parquet containing at most 2,000 rows. It must contain exactly
the configured 54 raw numeric feature columns in the recorded order. An
optional broad-label column may be selected for metrics and is excluded before
schema validation. Inputs with missing, extra, duplicate, reordered, or
non-numeric feature columns fail before inference.

The default is the clean eight-class static model and static preprocessing.
The Task 1 model always uses continual preprocessing. Its weak saved
max-confidence novelty cutoff is available only for the five-class teacher;
the application never applies it to the eight-class static model.

## Teacher and quarantine demonstration

The same deterministic demo can be run without the UI:

```bash
.venv-demo/bin/python scripts/demo_teacher_quarantine.py \
  --config configs/demo_artifacts.example.json
```

This performs CPU inference over the 500 supplied replay rows, changes 80 of
400 eligible old-attack labels to Benign with seed 42, applies the saved
Task-1-teacher label-consistency gate, and queues rejected rows. It prints an
experiment-only ground-truth audit. The persistent queue stores original ID,
supplied label, score, threshold, model identity, row payload, status, and
decision history—but never the simulator poison mask. Reviewer release is an
audited queue decision only and cannot trigger training.

## Saved comparison playback

The continual view reads only named JSON evidence from the ZIP, after checking
its hash and rejecting unsafe member paths, symlinks, encrypted members, or
oversized JSON. Bundled source and checkpoints are not imported or executed.
The compact evidence index is
`docs/evidence/task2_targeted_summary.json`; the original ZIP remains the
authoritative artifact and is not modified.

The exact timed narration and click sequence is in
`docs/FIVE_MINUTE_DEMO.md`. The completed local rehearsal is summarized in
`docs/evidence/demo_rehearsal_20261005.json`. Its 40-row metrics are bounded
sample behavior, not full-test performance.
