"""Write track_a/selection.csv: the starter training photos, minus the osv5m ones.

The osv5m photos are frames from cameras on cars and bikes. On the mistakes
sheets the model's confident mistakes were full of road-view frames, so the
student decided to leave those photos out. The reason, in the student's words:
"차의 로드뷰". Nothing else is left out; if track_a/dropped.csv exists, the
photos listed there are left out as well.

Run: uv run python scripts/make_selection.py
Then: uv run python src/curate.py count
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "starter" / "train"
OUT = ROOT / "track_a" / "selection.csv"
DROPPED = ROOT / "track_a" / "dropped.csv"


def main():
    dropped = set()
    if DROPPED.exists():
        with DROPPED.open(newline="", encoding="utf-8-sig") as f:
            dropped = {row["path"] for row in csv.DictReader(f)}

    photos = sorted(p.relative_to(TRAIN).as_posix() for p in TRAIN.glob("*/*.jpg"))
    road_view = [p for p in photos if Path(p).name.startswith("osv5m_")]
    keep = [p for p in photos if p not in set(road_view) and p not in dropped]

    OUT.write_text("path\n" + "\n".join(keep) + "\n", encoding="utf-8")
    print(f"{len(keep)} photos written to {OUT.relative_to(ROOT)}")
    print(f"  left out {len(road_view)} osv5m road-view photos and {len(dropped)} from dropped.csv")
    print(f"  of {len(photos)} photos in data/starter/train")


if __name__ == "__main__":
    main()