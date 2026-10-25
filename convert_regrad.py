"""Convert REGRAD Relation Part V2 into the shared pairwise JSONL format."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np

from external_order_data import relative_image_path, write_manifest


def object_id(value) -> int | None:
    if isinstance(value, int):
        return value
    match = re.search(r"-(\d+)$", str(value))
    return int(match.group(1)) if match else None


def object_points(label: dict, segment: np.ndarray | None = None) -> np.ndarray | None:
    points = label.get("MinAreaRect")
    if points is not None:
        array = np.asarray(points, dtype=np.float32)
        if array.size == 8:
            return array.reshape(4, 2)
    obj_id = int(label.get("obj_id", -1))
    if segment is not None and obj_id > 0:
        ys, xs = np.where(segment == obj_id)
        if len(xs) >= 3:
            rect = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
            return cv2.boxPoints(rect)
    bbox = label.get("bbox")
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = map(float, bbox)
        return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return None


def transitive_before(ids: set[int], graph: dict) -> set[tuple[int, int]]:
    # REGRAD/VMRD: graph[parent] contains children. A child (upper object)
    # must be grasped before its parent (lower/supporting object).
    adjacency = {item: set() for item in ids}
    for parent_name, child_names in graph.items():
        parent = object_id(parent_name)
        if parent not in ids:
            continue
        for child_name in child_names:
            child = object_id(child_name)
            if child in ids and child != parent:
                adjacency[child].add(parent)
    closure: set[tuple[int, int]] = set()
    for source in ids:
        stack = list(adjacency[source])
        while stack:
            target = stack.pop()
            if (source, target) in closure:
                continue
            closure.add((source, target))
            stack.extend(adjacency.get(target, ()))
    return closure


def convert(source: Path, output: Path, negative_ratio: float, seed: int, max_scenes: int = 0) -> dict:
    rgb_dir = source / "RGBImages"
    segment_dir = source / "SegmentationImages"
    objects = json.loads((source / "objects.json").read_text(encoding="utf-8"))
    graphs = json.loads((source / "mrts.json").read_text(encoding="utf-8"))
    image_paths = sorted(rgb_dir.glob("*.jpg"))
    scene_ids = sorted({path.stem.rsplit("_", 1)[0] for path in image_paths})
    if max_scenes > 0:
        scene_ids = scene_ids[:max_scenes]
    allowed = set(scene_ids)
    positives, negatives = [], []
    skipped = 0
    for image_path in image_paths:
        scene_id, view_id = image_path.stem.rsplit("_", 1)
        if scene_id not in allowed:
            continue
        scene_labels = objects.get(scene_id, {})
        labels = scene_labels.get(view_id, scene_labels.get(str(int(view_id)), []))
        if not isinstance(labels, list):
            skipped += 1
            continue
        segment_path = segment_dir / f"{image_path.stem}.png"
        segment = cv2.imread(str(segment_path), cv2.IMREAD_UNCHANGED) if segment_path.exists() else None
        item_points = {}
        for label in labels:
            obj_id = int(label.get("obj_id", -1))
            points = object_points(label, segment)
            if obj_id > 0 and points is not None:
                item_points[obj_id] = points
        ids = set(item_points)
        before = transitive_before(ids, graphs.get(scene_id, {}))
        for offset, a_id in enumerate(sorted(ids)):
            for b_id in sorted(ids)[offset + 1:]:
                if (a_id, b_id) in before:
                    target = 0
                elif (b_id, a_id) in before:
                    target = 1
                else:
                    target = 2
                row = {
                    "image": relative_image_path(image_path, output),
                    "a_points": item_points[a_id].tolist(), "b_points": item_points[b_id].tolist(),
                    "target": target, "source": "REGRAD", "key": f"{scene_id}/{view_id}/{a_id}-{b_id}",
                }
                (positives if target != 2 else negatives).append(row)
    rng = random.Random(seed)
    rng.shuffle(negatives)
    negative_limit = int(round((len(positives) / 2) * max(negative_ratio, 0.0)))
    negatives = negatives[:negative_limit] if positives else []
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
    return {"images": len(image_paths), "scenes": len(allowed), "positive_pairs": len(positives),
            "negative_pairs": len(negatives), "rows": len(rows), "skipped_views": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="REGRAD V2 split folder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, default=0)
    args = parser.parse_args()
    stats = convert(Path(args.source), Path(args.output), args.negative_ratio, args.seed, args.max_scenes)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
