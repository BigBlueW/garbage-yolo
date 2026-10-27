"""Graph utilities for grasp-order annotations.

Annotations are written by ``label_tool.py`` (Tab order mode) as
``<split>/grasp_order/<stem>.json`` = ``{"relations": [[top, bottom], ...]}``.
``top`` rests on ``bottom``, so it must be grasped first (edge ``top -> bottom``).
Object IDs are the zero-based line numbers in the matching YOLO label file.
A missing file means the image has no stacked objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def relation_path(dataset_root: str | Path, split: str, stem: str) -> Path:
    return Path(dataset_root) / split / "grasp_order" / f"{stem}.json"


def read_yolo_objects(label_path: str | Path) -> list[dict]:
    """Read HBB/OBB labels without changing the source file."""
    path = Path(label_path)
    objects: list[dict] = []
    if not path.exists():
        return objects
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in (5, 9):
            raise ValueError(f"{path}:{line_number + 1}: expected 5 or 9 values, got {len(parts)}")
        try:
            class_id = int(float(parts[0]))
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number + 1}: invalid numeric label") from exc
        objects.append({
            "id": len(objects),
            "class_id": class_id,
            "format": "obb" if len(parts) == 9 else "hbb",
            "coordinates": [round(value, 8) for value in coordinates],
        })
    return objects


def compute_layers(object_count: int, pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Return parallel grasp layers using a deterministic topological sort."""
    adjacency = {index: set() for index in range(object_count)}
    indegree = [0] * object_count
    for source, target in pairs:
        if source < 0 or target < 0 or source >= object_count or target >= object_count:
            raise ValueError("Relation references an object outside the current label file")
        if source == target:
            raise ValueError("Self relations are not allowed")
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1

    remaining = set(range(object_count))
    layers: list[list[int]] = []
    while remaining:
        layer = sorted(index for index in remaining if indegree[index] == 0)
        if not layer:
            raise ValueError("Relations contain a cycle")
        layers.append(layer)
        for source in layer:
            remaining.remove(source)
            for target in adjacency[source]:
                indegree[target] -= 1
    return layers


def load_relations(path: str | Path, object_count: int) -> list[tuple[int, int]]:
    """Read ``[top, bottom]`` pairs; raise ValueError on bad IDs or cycles."""
    path = Path(path)
    if not path.exists():
        return []
    pairs = [(int(top), int(bottom)) for top, bottom in json.loads(path.read_text(encoding="utf-8"))["relations"]]
    compute_layers(object_count, pairs)
    return pairs
