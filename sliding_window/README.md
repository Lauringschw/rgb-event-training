# sliding_window

Time-based sliding-window extraction, merging, and ResNet-18 training for three event representations.

## Structure

```
sliding_window/
├── histogram/       2-channel ON/OFF event count histogram
├── voxel/           5-bin voxel grid with signed polarity weights
└── timesurface/     1-channel most-recent-timestamp surface
```

Each subdirectory is self-contained with identical pipeline structure:

```
1. extract_samples_<repr>.py          --window_ms <N>
2. merge_<repr>.py                    --window_ms <N>
3. train_<repr>.py                    --window_ms <N>
4. ../rqs/extract_test_samples.py     --repr <repr> --window_ms <N>
5. ../rqs/evaluate.py                 --repr <repr> --window_ms <N>
```

Steps 1–3 live in the representation subdirectory. Steps 4–5 live in `rqs/` and are shared across all representations. See the README in each subdirectory for representation-specific details, and `rqs/README.md` for evaluation details.

## Shared design

**Window**: duration = `--window_ms` ms, stride = `window_ms / 2` ms (50% overlap), extracted from `t_initial` → `t_initial + 300ms` per recording.

**Resolution**: 1280×720 → 640×360.

**Filter**: windows with fewer than 100 events are skipped.

**Split**: recording-level 70/10/20 train/val/test, stratified by gesture, seeds 42 (test) and 123 (val). Identical across all representations and window sizes.

**Model**: ResNet-18 with first conv layer adapted to the representation's channel count (1, 2, or 5).

**Output layout** (per representation, per window size):

```
SLIDING_BASE/<window_ms>ms/batches/    <- extraction output (deleted after merge)
SLIDING_BASE/<window_ms>ms/merged/     <- merged dataset + model weights + metrics
SLIDING_BASE/test_recording_ids.npy    <- held-out recording IDs for RQ evaluation
```

## .env variables required

```
RECORDINGS_DIR=   path to root folder containing gesture subfolders
DIR=              recording session subfolder (e.g. recording_session_1)
SLIDING_BASE=     set to the representation's base dir before running
                  e.g. /Volumes/T7/thesis/sliding_window_time/histogram
                       /Volumes/T7/thesis/sliding_window_time/voxel
                       /Volumes/T7/thesis/sliding_window_time/timesurface
```
