"""Validate grasp-order sidecar files and report annotation progress."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from grasp_order import load_document, read_yolo_objects, relation_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check grasp-order annotation integrity and progress")
    parser.add_argument("dataset_root", nargs="?", default="robotic-garbage-obb")
    parser.add_argument("--strict", action="store_true", help="Return an error until every image is reviewed")
    args = parser.parse_args()
    root = Path(args.dataset_root)
    totals = Counter()
    errors = 0

    for split in ("train", "val", "test"):
        images_dir = root / split / "images"
        if not images_dir.is_dir():
            continue
        images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in images:
            totals["images"] += 1
            totals[f"{split}_images"] += 1
            label_path = root / split / "labels" / f"{image_path.stem}.txt"
            try:
                objects = read_yolo_objects(label_path)
                document_path = relation_path(root, split, image_path.stem)
                document, issues = load_document(document_path, image_path.name, objects)
            except Exception as exc:
                print(f"ERROR {split}/{image_path.stem}: {exc}")
                errors += 1
                continue
            for issue in issues:
                print(f"{issue.severity.upper()} {split}/{image_path.stem}: {issue.message}")
                if issue.severity == "error":
                    errors += 1
            if document.get("reviewed", False) and not issues:
                totals["reviewed"] += 1
                totals[f"{split}_reviewed"] += 1
                totals["relations"] += len(document.get("relations", []))
            else:
                totals["unreviewed"] += 1

    print("\nGrasp-order annotation summary")
    print(f"  images    : {totals['images']}")
    print(f"  reviewed  : {totals['reviewed']}")
    print(f"  unreviewed: {totals['unreviewed']}")
    print(f"  relations : {totals['relations']}")
    for split in ("train", "val", "test"):
        print(f"  {split:<5}: {totals[f'{split}_reviewed']}/{totals[f'{split}_images']} reviewed")
    print(f"  errors    : {errors}")
    return 1 if errors or (args.strict and totals["unreviewed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
