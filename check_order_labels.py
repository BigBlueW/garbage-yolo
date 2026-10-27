"""Validate grasp-order sidecar files and report annotation counts."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from grasp_order import load_relations, read_yolo_objects


def main() -> int:
    parser = argparse.ArgumentParser(description="Check grasp-order annotation integrity")
    parser.add_argument("dataset_root", nargs="?", default="robotic-garbage-obb")
    args = parser.parse_args()
    root = Path(args.dataset_root)
    totals = Counter()
    errors = 0

    for split in ("train", "val", "test"):
        for order_path in sorted((root / split / "grasp_order").glob("*.json")):
            label_path = root / split / "labels" / f"{order_path.stem}.txt"
            try:
                if not label_path.exists():
                    raise ValueError("missing YOLO label file")
                pairs = load_relations(order_path, len(read_yolo_objects(label_path)))
            except Exception as exc:
                print(f"ERROR {split}/{order_path.stem}: {exc}")
                errors += 1
                continue
            totals[f"{split}_images"] += 1
            totals[f"{split}_relations"] += len(pairs)

    print("\nGrasp-order annotation summary")
    for split in ("train", "val", "test"):
        print(f"  {split:<5}: {totals[f'{split}_images']} images, {totals[f'{split}_relations']} relations")
    print(f"  errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
