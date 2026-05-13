# sliding_window/timesurface

Three scripts for time-based sliding-window time surface dataset creation and ResNet-18 training.

## Order of execution

```
1. extract_samples_timesurface.py  --window_ms <N>
2. merge_timesurface.py            --window_ms <N>
3. train_timesurface.py            --window_ms <N>
```

---

## extract_samples_timesurface.py

Extracts time-based sliding-window time surface samples from all recordings.

**Sliding window**
Duration = `--window_ms` ms, stride = `window_ms / 2` ms (50% overlap)

**Extraction range**
`t_initial` → `t_initial + 300ms`

**Filters**
Windows with fewer than 100 events are skipped.

**Resolution**
Events downsampled from 1280×720 → 640×360.

**Time surface encoding**
Uses Prophesee `MostRecentTimestampBuffer` to record the most recent event timestamp per pixel. Timestamps are normalised to [0, 1] within the window; pixels with no events remain 0. Output is a single-channel map.

**Output** (batch files saved to `SLIDING_BASE/<window_ms>ms/batches/`)

```
timesurface_data_batch_0.npy       float32  (≤500, 1, 360, 640)
timesurface_labels_batch_0.npy     int64    (≤500,)
timesurface_recids_batch_0.npy     int64    (≤500,)
...
```

Each batch holds up to 500 samples. Recording ID tracks which recording each sample came from (used for split).

---

## merge_timesurface.py

Reads batch files from `SLIDING_BASE/<window_ms>ms/batches/`, writes merged dataset to `SLIDING_BASE/<window_ms>ms/merged/` using memory-mapped files. Deletes each batch immediately after writing to keep peak disk usage at merged_so_far + one_batch.

**Output** (saved to `SLIDING_BASE/<window_ms>ms/merged/`)

```
timesurface_data.npy                float32  (N, 1, 360, 640)
timesurface_labels.npy              int64    (N,)
timesurface_recording_ids.npy       int64    (N,)
```

All batch files deleted after merge.

---

## train_timesurface.py

Trains a ResNet-18 on single-channel time surface event representations.

**Architecture**
ResNet-18 with first conv layer replaced: `Conv2d(1→64, k=7, stride=2, padding=3)`.

**Normalisation**
Per-sample: divide by max value (applied in dataset loader).

**Split strategy** (recording-level, prevents leakage from sliding windows)

1. Unique recording IDs split into test vs rest (stratified, seed 42, 20%)
2. Rest split into train vs val (stratified, seed 123, 10% of total)
3. Every sample assigned to the set its recording belongs to

**Training config**

```
Optimizer  : Adam, lr=0.001
Loss       : CrossEntropyLoss (inverse-frequency class weights)
LR schedule: ReduceLROnPlateau, factor=0.5, patience=3 (on val acc)
Batch size : 32
Max epochs : 50
Early stop : patience=10 epochs (no val-acc improvement)
Best model : saved on best validation accuracy
```

**Output** (saved to `SLIDING_BASE/<window_ms>ms/merged/`)

```
model_timesurface_<window_ms>ms_best.pth
metrics_<window_ms>ms.txt
```

Test recording IDs also saved to `SLIDING_BASE/test_recording_ids.npy` for use by `extract_test_samples.py`.

---

## .env variables required

```
RECORDINGS_DIR=   path to root folder containing gesture subfolders
DIR=              recording session subfolder (e.g. recording_session_1)
SLIDING_BASE=     base dir per representation; window-size subdirs created automatically
                  e.g. /Volumes/T7/thesis/sliding_window_time/timesurface
```
