# Dataset Archive Preparation

`prepare_raw_data.py` safely prepares source archives for Developer B's data
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
- Stops at raw extraction; Developer B owns all loading and preprocessing.

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
experiment report until Developer B verifies their provenance and schema.
