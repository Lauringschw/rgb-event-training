import argparse
from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent.parent / '.env')

# == configs ===================================================================
SENSOR_HEIGHT = 360
SENSOR_WIDTH  = 640
ORIG_HEIGHT   = 720
ORIG_WIDTH    = 1280
N_BINS        = 5
EXTRACTION_RANGE_US        = 300_000
BATCH_SIZE                 = 500
MAX_RECORDINGS_PER_GESTURE = 320

# == paths =====================================================================
RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR            = os.getenv("DIR")
SLIDING_BASE   = Path(os.getenv("SLIDING_BASE"))

GESTURE_TO_LABEL = {'rock': 0, 'paper': 1, 'scissor': 2}


# == representation ============================================================

def events_to_voxel(events):
    voxel = np.zeros((N_BINS, SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.float32)
    if len(events) == 0:
        return voxel

    x = (events['x'].astype(np.int32) * SENSOR_WIDTH  // ORIG_WIDTH)
    y = (events['y'].astype(np.int32) * SENSOR_HEIGHT // ORIG_HEIGHT)
    valid = (x >= 0) & (x < SENSOR_WIDTH) & (y >= 0) & (y < SENSOR_HEIGHT)
    x, y = x[valid], y[valid]
    t = events['t'][valid].astype(np.float64)
    p = events['p'][valid]

    if len(t) == 0:
        return voxel

    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        bin_idx = np.zeros(len(t), dtype=np.int32)
    else:
        t_norm  = (t - t_min) / (t_max - t_min)
        bin_idx = np.clip((t_norm * N_BINS).astype(np.int32), 0, N_BINS - 1)

    weights = np.where(p == 1, 1.0, -1.0).astype(np.float32)
    np.add.at(voxel, (bin_idx, y, x), weights)
    return voxel


# == TIME-BASED sliding window =================================================

def extract_time_windows(events, t_start_us, t_end_us, window_us, stride_us):
    samples = []
    if len(events) == 0:
        return samples
    current_t = t_start_us
    while current_t + window_us <= t_end_us:
        window_end = current_t + window_us
        mask = (events['t'] >= current_t) & (events['t'] < window_end)
        window_events = events[mask]
        if len(window_events) < 100:
            print(f"      Warning: only {len(window_events)} events in [{current_t}, {window_end}), skipping")
            current_t += stride_us
            continue
        samples.append(events_to_voxel(window_events))
        current_t += stride_us
    return samples


# == per-recording processing ==================================================

def process_recording(folder: Path, window_us: int, stride_us: int):
    labels_file = folder / "labels.npy"
    raw_file    = folder / "prophesee_events.raw"
    if not labels_file.exists() or not raw_file.exists():
        print(f"  !! Missing files in {folder.name}")
        return None
    labels    = np.load(labels_file, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    t_start   = t_initial
    t_end     = t_initial + EXTRACTION_RANGE_US
    chunks = [ev for ev in EventsIterator(str(raw_file))]
    if not chunks:
        print(f"  !! No events in {folder.name}")
        return None
    all_events = np.concatenate(chunks)
    mask   = (all_events['t'] >= t_start) & (all_events['t'] < t_end)
    events = all_events[mask]
    if len(events) == 0:
        print(f"  !! No events in [{t_start}, {t_end}) for {folder.name}")
        return None
    samples = extract_time_windows(events, t_start, t_end, window_us, stride_us)
    duration_ms = (events['t'][-1] - events['t'][0]) / 1000.0
    print(f"  -> {len(samples)} samples from {len(events)} events ({duration_ms:.1f}ms)")
    return samples


# == batch helpers =============================================================

def save_batch(batch_samples, batch_labels, batch_rec_ids, batch_num, sliding_dir):
    np.save(sliding_dir / f"voxel_data_batch_{batch_num}.npy",
            np.array(batch_samples, dtype=np.float32))
    np.save(sliding_dir / f"voxel_labels_batch_{batch_num}.npy",
            np.array(batch_labels, dtype=np.int64))
    np.save(sliding_dir / f"voxel_recids_batch_{batch_num}.npy",
            np.array(batch_rec_ids, dtype=np.int64))
    print(f"  [batch {batch_num}] saved {len(batch_samples)} samples")


# == main ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_ms', type=int, required=True,
                        help='Window duration in ms (e.g. 10, 20, 30, ...)')
    args = parser.parse_args()

    window_us = args.window_ms * 1_000
    stride_us = window_us // 2   # 50% overlap

    SLIDING_DIR = SLIDING_BASE / f"{args.window_ms}ms" / "batches"
    SLIDING_DIR.mkdir(parents=True, exist_ok=True)

    base = RECORDINGS_DIR / DIR

    batch_samples, batch_labels, batch_rec_ids = [], [], []
    batch_num       = 0
    total_processed = 0
    total_failed    = 0
    total_samples   = 0
    recording_id    = 0

    print("=" * 60)
    print(f"VOXEL EXTRACTION ({args.window_ms}ms windows, 50% overlap)")
    print("=" * 60)
    print(f"Window duration : {args.window_ms} ms")
    print(f"Stride          : {args.window_ms // 2} ms")
    print(f"N_BINS          : {N_BINS}")
    print(f"Output dir      : {SLIDING_DIR}\n")

    for gesture in GESTURE_TO_LABEL:
        prefix          = gesture[0]
        label           = GESTURE_TO_LABEL[gesture]
        gesture_samples = 0
        gesture_ok      = 0

        for i in range(1, MAX_RECORDINGS_PER_GESTURE + 1):
            folder = base / gesture / f"{prefix}_{i}"
            if not folder.exists():
                break

            print(f"\n{gesture}/{prefix}_{i}  (rec_id={recording_id})")
            samples = process_recording(folder, window_us, stride_us)

            if samples:
                for s in samples:
                    batch_samples.append(s)
                    batch_labels.append(label)
                    batch_rec_ids.append(recording_id)
                    if len(batch_samples) >= BATCH_SIZE:
                        save_batch(batch_samples, batch_labels, batch_rec_ids, batch_num, SLIDING_DIR)
                        batch_samples, batch_labels, batch_rec_ids = [], [], []
                        batch_num += 1
                gesture_samples += len(samples)
                total_samples   += len(samples)
                gesture_ok      += 1
                total_processed += 1
            else:
                total_failed += 1

            recording_id += 1

        print(f"\n{gesture.upper()}: {gesture_ok} recordings, {gesture_samples} samples")

    if batch_samples:
        save_batch(batch_samples, batch_labels, batch_rec_ids, batch_num, SLIDING_DIR)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_processed} recordings -> {total_samples} samples")
    print(f"Failed: {total_failed} recordings")
    print(f"Batches saved to: {SLIDING_DIR}")
    print(f"Next step: python3 merge_voxel.py --window_ms {args.window_ms}")