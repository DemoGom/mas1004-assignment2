"""Score a Track A model on the validation photos, split by where each photo came from.

    uv run python scripts/val_by_source.py results/baseline_seed0/model.onnx
    uv run python scripts/val_by_source.py results/ema_seed0/model.onnx

Every starter photo's file name says where it came from: commons_<id>.jpg
(Wikimedia Commons), osv5m_<id>.jpg (a camera on a car or a bike) and
youtube_<video id>_<seconds>.jpg (a frame of a walking or driving video). The
leaderboard uses frames from channels that are not in the starter data and the
final ranking uses phone street photos, so a model that gains only on commons
gains only on the photos I can measure.

It reads the ONNX file only, with the same preparation as check_submission.py.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from rules import CITIES  # noqa: E402

SOURCES = ["commons", "osv5m", "youtube"]


def source_of(path):
    """The part of the file name before the first underscore."""
    return Path(path).name.split("_")[0]


def predict(onnx_path, rows, size, threads, batch=32):
    """Predicted city for every row, in order."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def prepare(row):
        return rules.grader_input(rules.load_photo(row[0]), size)

    preds = []
    with ThreadPoolExecutor(threads) as pool:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            x = np.stack(list(pool.map(prepare, chunk)))
            logits = session.run(["logits"], {"image": x})[0]
            preds += logits.argmax(1).tolist()
    return preds


def report(title, rows, preds):
    """Balanced accuracy of one group of rows, and the line for the table."""
    true = [CITIES.index(city) for _, city in rows]
    score, per_city, _ = rules.balanced_accuracy(true, preds)
    cities = " ".join(f"{c} {per_city[c]:.3f}" for c in CITIES)
    print(f"{title:<8} {len(rows):>5} photos  balanced accuracy {score:.4f}   {cities}")
    return score


def main():
    ap = __import__("argparse").ArgumentParser(description=__doc__,
                                              formatter_class=__import__("argparse").RawDescriptionHelpFormatter)
    ap.add_argument("onnx", nargs="+", help="one or more model.onnx files")
    ap.add_argument("--data", default=str(ROOT / "data" / "starter"))
    ap.add_argument("--size", type=int, default=rules.TRACK_A_IMAGE_SIZE)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    args = ap.parse_args()

    rows = rules.list_photos(Path(args.data) / "val")
    if not rows:
        raise SystemExit(f"no validation photos under {args.data}/val")
    groups = {"all": rows}
    for s in SOURCES:
        groups[s] = [r for r in rows if source_of(r[0]) == s]
    unknown = [r for r in rows if source_of(r[0]) not in SOURCES]
    if unknown:
        groups["other"] = unknown

    for onnx_path in args.onnx:
        print(f"\n{onnx_path}")
        preds = predict(onnx_path, rows, args.size, args.workers)
        by_row = dict(zip([r[0] for r in rows], preds))
        for name, group in groups.items():
            report(name, group, [by_row[r[0]] for r in group])


if __name__ == "__main__":
    main()