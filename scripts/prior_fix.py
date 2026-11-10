"""Does correcting the model's city bias help, without training again?

    uv run python scripts/prior_fix.py results/baseline_seed0/model.onnx --workers 2

The confusion matrix of the starter recipe says the model puts 35.6% of the
validation photos in Tokyo, while each city is 20% of the photos, and that it
finds only 0.378 of the Busan photos. This script asks whether subtracting a
constant from each city's score, big enough to undo that over-prediction,
raises the balanced accuracy on the validation photos: the same measure the
leaderboard uses.

The constants come from the model's own predictions on the *training* photos,
so the validation photos stay untouched. Logits are cached, so the sweep over
the strength costs nothing after the first pass. A recipe could do the same
inside final_model, where the same ResNet18 comes back with its biases changed.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from rules import CITIES  # noqa: E402


def logits_for(onnx_path, rows, size, threads, batch=32):
    """The 5 scores of every row, in order."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def prepare(row):
        return rules.grader_input(rules.load_photo(row[0]), size)

    out = []
    with ThreadPoolExecutor(threads) as pool:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            x = np.stack(list(pool.map(prepare, chunk)))
            out.append(session.run(["logits"], {"image": x})[0])
    return np.concatenate(out, 0)


def balanced(logits, true):
    score, per_city, _ = rules.balanced_accuracy(true, logits.argmax(1).tolist())
    return score, per_city


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--data", default=str(ROOT / "data" / "starter"))
    ap.add_argument("--size", type=int, default=rules.TRACK_A_IMAGE_SIZE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train-every", type=int, default=15, help="use every Nth training photo")
    args = ap.parse_args()

    val_rows = rules.list_photos(Path(args.data) / "val")
    train_rows = rules.list_photos(Path(args.data) / "train")[::args.train_every]
    val_true = [CITIES.index(c) for _, c in val_rows]
    print(f"validation photos {len(val_rows)}, training photos for the bias {len(train_rows)}",
          flush=True)

    cache = ROOT / "results" / "prior_fix"
    cache.mkdir(parents=True, exist_ok=True)

    for onnx_path in args.onnx:
        name = Path(onnx_path).parent.name  # baseline_seed0
        f_val = cache / f"{name}_val_logits.npy"
        f_train = cache / f"{name}_train_logits.npy"
        if f_val.exists() and f_train.exists():
            val_logits = np.load(f_val)
            train_logits = np.load(f_train)
        else:
            val_logits = logits_for(onnx_path, val_rows, args.size, args.workers)
            np.save(f_val, val_logits)
            train_logits = logits_for(onnx_path, train_rows, args.size, args.workers)
            np.save(f_train, train_logits)

        base, per_city = balanced(val_logits, val_true)
        pred_share = np.bincount(val_logits.argmax(1), minlength=5) / len(val_logits)
        print(f"\n{name}: no correction, balanced accuracy {base:.4f}", flush=True)
        print("  " + " ".join(f"{c} {per_city[c]:.3f}" for c in CITIES))
        print("  predicted share " + " ".join(f"{c} {v:.3f}" for c, v in zip(CITIES, pred_share)))

        # The model's own bias, measured on training photos, never on validation.
        counts = np.bincount(train_logits.argmax(1), minlength=5).astype(np.float64) + 1.0
        prior = counts / counts.sum()
        offset = np.log(prior) - np.log(prior).mean()  # mean 0, only the ratios matter
        print("  bias from training photos " + " ".join(f"{c} {o:+.3f}" for c, o in zip(CITIES, offset)))

        print("  strength  balanced accuracy   per city", flush=True)
        best = (base, 0.0)
        for tau in np.arange(-1.5, 1.51, 0.25):
            score, pc = balanced(val_logits - tau * offset, val_true)
            star = ""
            if score > best[0]:
                best = (score, tau)
                star = "  <-"
            print(f"  {tau:+.2f}     {score:.4f}   "
                  + " ".join(f"{c} {pc[c]:.3f}" for c in CITIES) + star, flush=True)
        print(f"  best {best[0]:.4f} at strength {best[1]:+.2f} "
              f"({best[0] - base:+.4f} against no correction)")


if __name__ == "__main__":
    main()