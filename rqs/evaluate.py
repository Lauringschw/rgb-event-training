import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
from torchvision import models

load_dotenv(Path(__file__).parent.parent / '.env')


SLIDING_BASE = Path(os.getenv("SLIDING_BASE"))

RQ2_OFFSETS_MS  = [0, 20, 40, 60, 80, 100]
N_BINS          = 5
LABEL_TO_GESTURE = {0: 'rock', 1: 'paper', 2: 'scissor'}


# == models ====================================================================

class ResNet18Histogram(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNet18Voxel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(N_BINS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNet18TimeSurface(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


MODEL_CLS = {
    'histogram':   ResNet18Histogram,
    'voxel':       ResNet18Voxel,
    'timesurface': ResNet18TimeSurface,
}


def get_model_path(repr_name: str, window_ms: int) -> Path:
    if repr_name == 'histogram':
        return SLIDING_BASE / f"{window_ms}ms" / "merged" / f"model_histogram_{window_ms}ms_best.pth"
    else:
        return SLIDING_BASE / f"{window_ms}ms" / "merged" / f"model_{repr_name}_{window_ms}ms_best.pth"


# == normalisation =============================================================

def normalize(batch, repr_name):
    for j in range(len(batch)):
        max_val = np.abs(batch[j]).max() if repr_name == 'voxel' else batch[j].max()
        if max_val > 0:
            batch[j] /= max_val
    return batch


# == device ====================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# == inference =================================================================

def run_inference(model, data, device, repr_name, batch_size=32):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = normalize(data[i:i+batch_size].copy().astype(np.float32), repr_name)
            out   = model(torch.from_numpy(batch).to(device))
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def accuracy(preds, labels):
    return 100.0 * np.mean(preds == labels)


def per_class_acc(preds, labels):
    return {LABEL_TO_GESTURE[i]: 100.0 * np.mean(preds[labels == i] == i) for i in range(3)}


# == RQ1 =======================================================================

def evaluate_rq1(model, device, repr_name, window_ms, test_dir, results_dir):
    """
    RQ1 for this window_ms: single accuracy value at the trained window size.
    One model = one data point in the RQ1 curve.
    """
    print("\n" + "=" * 55)
    print(f"RQ1 — {window_ms}ms model accuracy")
    print("=" * 55)

    data   = np.load(test_dir / "rq1_data.npy")
    labels = np.load(test_dir / "rq1_labels.npy")

    preds   = run_inference(model, data, device, repr_name)
    overall = accuracy(preds, labels)
    pc      = per_class_acc(preds, labels)

    line = (f"window={window_ms}ms  overall={overall:.2f}%  "
            f"rock={pc['rock']:.2f}%  paper={pc['paper']:.2f}%  scissor={pc['scissor']:.2f}%")
    print(line)

    np.save(results_dir / "rq1_accuracy.npy",    np.array([overall]))
    np.save(results_dir / "rq1_window_ms.npy",   np.array([window_ms]))
    (results_dir / "rq1_result.txt").write_text(f"RQ1 — {repr_name} {window_ms}ms\n{line}\n")
    print(f"Saved -> {results_dir}/rq1_*")
    return overall


# == RQ2 =======================================================================

def evaluate_rq2(model, device, repr_name, window_ms, test_dir, results_dir):
    """
    RQ2: accuracy of this window_ms model across offsets 0, 20, 40, 60, 80, 100ms.
    Window = [t_initial + offset, t_initial + offset + window_ms].
    """
    print("\n" + "=" * 55)
    print(f"RQ2 — {window_ms}ms model, temporal offset effect")
    print("=" * 55)

    data    = np.load(test_dir / "rq2_data.npy")
    labels  = np.load(test_dir / "rq2_labels.npy")
    offsets = np.load(test_dir / "rq2_offsets_ms.npy")

    results = {}
    lines   = [f"RQ2 — {repr_name} {window_ms}ms window"]
    header  = f"{'Offset':>10}  {'Window':>20}  {'Overall':>8}  {'Rock':>8}  {'Paper':>8}  {'Scissor':>8}"
    print(header)
    print("-" * len(header))
    lines.append(header)

    for off in RQ2_OFFSETS_MS:
        mask    = offsets == off
        if not np.any(mask):
            continue
        preds   = run_inference(model, data[mask], device, repr_name)
        overall = accuracy(preds, labels[mask])
        pc      = per_class_acc(preds, labels[mask])
        results[off] = overall
        window_label = f"t+{off}-{off+window_ms}ms"
        line = (f"{off:>8}ms  {window_label:>20}  {overall:>7.2f}%"
                f"  {pc['rock']:>7.2f}%  {pc['paper']:>7.2f}%  {pc['scissor']:>7.2f}%")
        print(line)
        lines.append(line)

    offsets_out = [o for o in RQ2_OFFSETS_MS if o in results]
    np.save(results_dir / "rq2_accuracies.npy", np.array([results[o] for o in offsets_out]))
    np.save(results_dir / "rq2_offsets_ms.npy", np.array(offsets_out))
    (results_dir / "rq2_results.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved -> {results_dir}/rq2_*")
    return results


# == RQ3 =======================================================================

def evaluate_rq3(rq1_acc, repr_name, window_ms, results_dir):
    """RQ3: accuracy at 30ms baseline (only meaningful when window_ms == 30)."""
    if window_ms != 30:
        return
    print("\n" + "=" * 55)
    print(f"RQ3 — {repr_name} at 30ms baseline")
    print("=" * 55)
    print(f"Accuracy: {rq1_acc:.2f}%")
    np.save(results_dir / f"rq3_{repr_name}_acc_30ms.npy", np.array([rq1_acc]))
    (results_dir / f"rq3_{repr_name}_result.txt").write_text(
        f"RQ3 — {repr_name} at τ=0, Δt=30ms\n{rq1_acc:.4f}\n")
    print(f"Saved -> {results_dir}/rq3_{repr_name}_*")


# == main ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repr', required=True, choices=['histogram', 'voxel', 'timesurface'])
    parser.add_argument('--window_ms', type=int, required=True,
                        help='Window duration in ms — must match extracted test samples')
    args = parser.parse_args()

    TEST_DIR    = SLIDING_BASE / f"{args.window_ms}ms" / "test_samples"
    RESULTS_DIR = SLIDING_BASE / f"{args.window_ms}ms" / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"EVALUATION — {args.repr} | {args.window_ms}ms (RQ1 + RQ2 + RQ3)")
    print("=" * 55)

    device = get_device()
    print(f"\nDevice: {device}")

    model_path = get_model_path(args.repr, args.window_ms)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run train_histogram.py --window_ms {args.window_ms} first.")

    model = MODEL_CLS[args.repr]().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded: {model_path}\n")

    for fname in ["rq1_data.npy", "rq1_labels.npy", "rq2_data.npy", "rq2_labels.npy", "rq2_offsets_ms.npy"]:
        if not (TEST_DIR / fname).exists():
            raise FileNotFoundError(
                f"Missing: {TEST_DIR / fname}\n"
                f"Run: python3 extract_test_samples.py --repr {args.repr} --window_ms {args.window_ms}")

    rq1_acc = evaluate_rq1(model, device, args.repr, args.window_ms, TEST_DIR, RESULTS_DIR)
    evaluate_rq2(model, device, args.repr, args.window_ms, TEST_DIR, RESULTS_DIR)
    evaluate_rq3(rq1_acc, args.repr, args.window_ms, RESULTS_DIR)

    print("\n" + "=" * 55)
    print(f"COMPLETE — results: {RESULTS_DIR}")
    print("=" * 55)