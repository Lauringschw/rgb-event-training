# rgb-event-training

Training and evaluation pipeline for low-latency hand gesture recognition from event camera data.

Consumes labeled recordings produced by [rgb-event-labeling](https://github.com/Lauringschw/rgb-event-labeling). Event data is converted into one of three representations and classified with ResNet-18 across three research questions on window length, temporal offset, and representation choice.

---

## Repository structure

```
rgb-event-training/
├── sliding_window/
│   ├── histogram/           2-channel ON/OFF event count histogram
│   ├── voxel/               5-bin voxel grid with signed polarity weights
│   └── timesurface/         1-channel most-recent-timestamp surface
├── rqs/                     RQ evaluation: test extraction + inference
└── .env                     paths config (not committed)
```

---

## Pipeline

Run once per representation × window size. Set `SLIDING_BASE` in `.env` to the representation's base dir before running.

```
1. sliding_window/<repr>/extract_samples_<repr>.py   --window_ms <N>
2. sliding_window/<repr>/merge_<repr>.py             --window_ms <N>
3. sliding_window/<repr>/train_<repr>.py             --window_ms <N>
4. rqs/extract_test_samples.py                       --repr <repr> --window_ms <N>
5. rqs/evaluate.py                                   --repr <repr> --window_ms <N>
```

---

## Research questions

| RQ  | Question                                             | Operationalisation                                                                       |
| --- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| RQ1 | How does accuracy vary across window lengths?        | Separate model per window size (10, 20, 30, 50, 70, 100ms); one accuracy value per model |
| RQ2 | How does temporal offset affect accuracy?            | Fixed window size, extracted at t+0 to t+100ms offsets in 20ms steps                     |
| RQ3 | Which representation achieves best accuracy at 30ms? | Histogram vs voxel vs timesurface at the 30ms baseline                                   |

---

## Hardware

| Device                           | Role                                          |
| -------------------------------- | --------------------------------------------- |
| Prophesee Gen4 HD DVS (1280×720) | Event camera — primary data source            |
| Basler RGB (~140 fps)            | Labeling oracle (see rgb-event-labeling)      |
| HP EliteBook 855 G7 (Ubuntu)     | Data extraction (Metavision SDK — Linux only) |
| MacBook Pro M2 Max (64GB, MPS)   | Training                                      |
| Samsung T7 2TB SSD               | Processed dataset storage                     |
| Seagate 2TB HDDD                 | Unprocessed dataset storage                   |

---

## .env variables required

```
RECORDINGS_DIR=   path to root folder containing gesture subfolders
DIR=              recording session subfolder (e.g. recording_session_1)
SLIDING_BASE=     base dir for the active representation
                  e.g. /Volumes/T7/thesis/sliding_window_time/histogram
                       /Volumes/T7/thesis/sliding_window_time/voxel
                       /Volumes/T7/thesis/sliding_window_time/timesurface
```

---

## Dependencies

- Python 3.x, PyTorch (MPS backend: `PYTORCH_ENABLE_MPS_FALLBACK=1`)
- Prophesee Metavision SDK (extraction only, Linux)
- `metavision_sdk_core` (time surface only)
- NumPy, scikit-learn, python-dotenv, torchvision
