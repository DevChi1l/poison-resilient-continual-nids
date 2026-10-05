# Dataset Archive Preparation

`prepare_raw_data.py` safely prepares source archives for the team's data
pipeline. It extracts ZIP files only: it does not read Parquet schemas, change
labels, sample rows, normalize features, or create train/test splits.

## Colab command

After mounting Google Drive, run this command from the repository root:

```bash
python scripts/prepare_raw_data.py \
  --source-dir "/content/drive/MyDrive/datasets/cic-ids-source" \
  --raw-dir "/content/cic-ids/raw"
```

Replace `--source-dir` with the mounted directory that contains the original
ZIP files. The destination can be a fast Colab local disk directory, a Lightning
workspace directory, or a mounted persistent volume. Re-running the same command
skips files whose expected uncompressed size already exists.

Use this first when checking paths without writing files:

```bash
python scripts/prepare_raw_data.py \
  --source-dir "/content/drive/MyDrive/datasets/cic-ids-source" \
  --raw-dir "/content/cic-ids/raw" \
  --dry-run
```

## Safety guarantees

- Searches recursively and processes ZIP files in a stable sorted order.
- Never changes, deletes, or moves the original ZIP archives.
- Rejects unsafe archive paths and ZIP symlinks.
- Uses a `.part` file and atomic rename for each extracted file.
- Skips existing same-size files and fails instead of overwriting a conflicting file.
- Stops at raw extraction; loading and preprocessing remain separate team-owned stages.

## Expected output layout

Archive-relative paths are preserved under `--raw-dir`. The current local
archives would produce this layout:

```text
<raw-dir>/
  cic-collection.parquet
  Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet
  Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet
  DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet
  DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet
  DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet
  DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet
  Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet
  Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet
  Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet
  Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet
```

Important: these names identify the currently available local archives as
**CSE-CIC-IDS2018-style** artifacts. Do not label them CIC-IDS2021 in an
experiment report until the team verifies their provenance and schema.

## Local demo preparation

The artifact-backed Streamlit demonstration has two no-training helpers:

```bash
python scripts/demo_preflight.py \
  --config configs/demo_artifacts.example.json \
  --database .local/nids-demo-rehearsal/quarantine.sqlite3

python scripts/export_demo_sample.py \
  --config configs/demo_artifacts.example.json \
  --output-dir uploads/rehearsal --rows-per-class 5 --seed 42
```

Preflight checks configured hashes and contracts without changing artifacts or
creating the queue. Export reads only the original Parquet and saved held-out
row indices, never predictions. Generated flow/provenance files stay under the
ignored `uploads/` tree; split arrays and SQLite queues stay under `.local/`.

## Kaggle clean Task 2 retention study

`run_task2_clean_retention.py` is the GPU-only runner used by the corresponding
Kaggle notebook. It consumes the verified prepared-data folder and completed
Task 1 run, never retrains Task 1, and writes a separate resumable output:

```bash
python scripts/run_task2_clean_retention.py \
  --prepared-dir /kaggle/input/<prepared-dataset>/review2_large_prepare_<id> \
  --task1-run-dir /kaggle/input/<task1-dataset>/full_task1_<id>
```

Use `--resume-run-dir` with an entire prior output folder to continue from
durable epoch boundaries. The runner requires CUDA and deliberately refuses
non-Kaggle input/output paths.
