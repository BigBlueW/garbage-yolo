"""Convert InstaOrder + COCO annotations into auxiliary pairwise order samples."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np

from external_order_data import relative_image_path, write_manifest


def annotation_points(annotation: dict) -> np.ndarray:
    polygons = annotation.get("segmentation")
    if isinstance(polygons, list):
        flat = [np.asarray(item, dtype=np.float32).reshape(-1, 2) for item in polygons if len(item) >= 6]
        if flat:
            return cv2.boxPoints(cv2.minAreaRect(np.vstack(flat)))
    x, y, width, height = map(float, annotation["bbox"])
    return np.asarray([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float32)


def parse_occlusion(order: str) -> tuple[int, int] | None:
    if "&" in order:
        return None  # mutual/cyclic occlusion is not a safe grasp-order label
    match = re.fullmatch(r"\s*(\d+)\s*<\s*(\d+)\s*", order)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def convert(instaorder_json: Path, coco_json: Path, images: Path, output: Path,
            negative_ratio: float, seed: int, max_images: int = 0) -> dict:
    insta = json.loads(instaorder_json.read_text(encoding="utf-8"))
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    image_by_id = {int(item["id"]): item for item in coco["images"]}
    annotation_by_id = {int(item["id"]): item for item in coco["annotations"]}
    positives, negatives = [], []
    ambiguous = 0
    records = insta["annotations"][:max_images or None]
    for record in records:
        image_id = int(record["image_id"])
        info = image_by_id.get(image_id)
        if info is None:
            continue
        image_path = images / info["file_name"]
        instance_ids = [int(value) for value in record["instance_ids"]]
        points = [annotation_points(annotation_by_id[value]) for value in instance_ids]
        # In official InstaOrder, i<j means j occludes i. The occluder j is
        # converted to "grasp before" i. Mutual occlusion is deliberately dropped.
        before: set[tuple[int, int]] = set()
        for item in record.get("occlusion", []):
            parsed = parse_occlusion(str(item.get("order", "")))
            if parsed is None:
                ambiguous += 1
                continue
            hidden, occluder = parsed
            before.add((occluder, hidden))
        for a in range(len(points)):
            for b in range(a + 1, len(points)):
                if (a, b) in before:
                    target = 0
                elif (b, a) in before:
                    target = 1
                else:
                    target = 2
                row = {"image": relative_image_path(image_path, output), "a_points": points[a].tolist(),
                       "b_points": points[b].tolist(), "target": target, "source": "InstaOrder",
                       "key": f"{image_id}/{a}-{b}"}
                (positives if target != 2 else negatives).append(row)
    rng = random.Random(seed)
    rng.shuffle(negatives)
    limit = int(round((len(positives) / 2) * max(negative_ratio, 0.0)))
    negatives = negatives[:limit] if positives else []
    rows = []
    for row in positives + negatives:
        rows.append(row)
        mirror = dict(row)
        mirror["a_points"], mirror["b_points"] = row["b_points"], row["a_points"]
        mirror["target"] = 1 - row["target"] if row["target"] in (0, 1) else 2
        mirror["key"] = row["key"] + "/mirror"
        rows.append(mirror)
    rng.shuffle(rows)
    write_manifest(output, rows)
    return {"images": len(records), "positive_pairs": len(positives), "negative_pairs": len(negatives),
            "rows": len(rows), "ambiguous_orders_dropped": ambiguous}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instaorder-json", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    stats = convert(Path(args.instaorder_json), Path(args.coco_json), Path(args.images), Path(args.output),
                    args.negative_ratio, args.seed, args.max_images)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
