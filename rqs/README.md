# rqs

Two scripts for RQ evaluation: test sample extraction and inference across all three research questions.

## Order of execution

```
1. extract_test_samples.py  --repr <repr> --window_ms <N>
2. evaluate.py              --repr <repr> --window_ms <N>
```

Run both scripts once per representation × window size combination.

---

## extract_test_samples.py

Extracts fixed-position test windows from held-out recordings for RQ1 and RQ2 evaluation.

**Prerequisites**

- `SLIDING_BASE/test_recording_ids.npy` must exist (saved by any `train_<repr>.py` run)
- Raw recordings must be accessible at `RECORDINGS_DIR/DIR`

**RQ1 samples**
One window per recording: `[t_initial, t_initial + window_ms]`

**RQ2 samples**
Six windows per recording, one per offset: `[t_initial + offset, t_initial + offset + window_ms]`
Offsets: 0, 20, 40, 60, 80, 100 ms

**Filter**
Windows with fewer than 100 events are skipped (matching training extraction).

**Supported representations**

| `--repr`      | Channels | Encoding                                     |
| ------------- | -------- | -------------------------------------------- |
| `histogram`   | 2        | ON/OFF event counts                          |
| `voxel`       | 5        | Signed polarity weights across temporal bins |
| `timesurface` | 1        | Normalised most-recent-timestamp per pixel   |

**Output** (saved to `SLIDING_BASE/<window_ms>ms/test_samples/`)

```
rq1_data.npy                float32  (N, C, 360, 640)
rq1_labels.npy              int64    (N,)
rq1_recording_ids.npy       int64    (N,)

rq2_data.npy                float32  (M, C, 360, 640)
rq2_labels.npy              int64    (M,)
rq2_offsets_ms.npy          int64    (M,)   offset for each sample
rq2_recording_ids.npy       int64    (M,)
```

---

## evaluate.py

Loads a trained model and evaluates it on the extracted test samples for RQ1, RQ2, and RQ3.

**RQ1** — accuracy of the `window_ms` model on its matched test window (one data point per model in the RQ1 curve)

**RQ2** — accuracy of the `window_ms` model across all six temporal offsets (0–100ms)

**RQ3** — accuracy at the 30ms baseline; only runs when `--window_ms 30`

**Output** (saved to `SLIDING_BASE/<window_ms>ms/results/`)

```
rq1_accuracy.npy            float64  (1,)
rq1_window_ms.npy           int64    (1,)
rq1_result.txt

rq2_accuracies.npy          float64  (6,)
rq2_offsets_ms.npy          int64    (6,)
rq2_results.txt

rq3_<repr>_acc_30ms.npy     float64  (1,)   only when window_ms == 30
rq3_<repr>_result.txt       only when window_ms == 30
```

---

## .env variables required

```
RECORDINGS_DIR=   path to root folder containing gesture subfolders
DIR=              recording session subfolder (e.g. recording_session_1)
SLIDING_BASE=     base dir for the representation being evaluated
                  e.g. /Volumes/T7/thesis/sliding_window_time/histogram
```
