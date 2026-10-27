"""Graph and rendering utilities for complete grasp sequences.

An edge (u, v) means object u must be grasped before object v.  The complete
sequence is deterministic, but objects in the same topological layer are
independent; their numeric order is only a tie-break recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from grasp_order import compute_layers, load_document, read_yolo_objects, relation_pairs


CLASS_NAMES = {0: "Plastic", 1: "Metal", 2: "Paper", 3: "General Waste"}
CLASS_COLORS = {
    0: (0, 230, 0),
    1: (220, 100, 20),
    2: (0, 150, 255),
    3: (230, 0, 230),
}


@dataclass(frozen=True)
class SceneObject:
    id: int
    class_id: int
    points: np.ndarray
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        point = np.mean(self.points, axis=0)
        return float(point[0]), float(point[1])

    @property
    def area(self) -> float:
        return abs(float(cv2.contourArea(self.points.astype(np.float32))))


@dataclass(frozen=True)
class RankedObject:
    rank: int
    layer: int
    object_id: int
    class_id: int
    score: float
    uncertainty_risk: float = 0.0


def object_points(obj: dict, width: int, height: int) -> np.ndarray:
    coords = np.asarray(obj["coordinates"], dtype=np.float32)
    if obj.get("format") == "hbb" or len(coords) == 4:
        cx, cy, box_w, box_h = coords
        x1, x2 = cx - box_w / 2, cx + box_w / 2
        y1, y2 = cy - box_h / 2, cy + box_h / 2
        coords = np.asarray([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.float32)
    points = coords.reshape(4, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    return points


def scene_from_yolo(objects: Sequence[dict], image_shape: Sequence[int]) -> list[SceneObject]:
    height, width = int(image_shape[0]), int(image_shape[1])
    return [
        SceneObject(
            id=int(obj["id"]),
            class_id=int(obj["class_id"]),
            points=object_points(obj, width, height),
        )
        for obj in objects
    ]


def _path_exists(adjacency: dict[int, set[int]], start: int, goal: int) -> bool:
    stack = [start]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node not in visited:
            visited.add(node)
            stack.extend(adjacency.get(node, ()))
    return False


def make_acyclic_edges(
    object_count: int,
    weighted_edges: Iterable[tuple[int, int, float]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, float]]]:
    """Keep strongest edges while rejecting invalid, duplicate, or cyclic ones."""
    adjacency = {index: set() for index in range(object_count)}
    accepted: list[tuple[int, int]] = []
    rejected: list[tuple[int, int, float]] = []
    for source, target, confidence in sorted(weighted_edges, key=lambda item: item[2], reverse=True):
        source, target = int(source), int(target)
        if not (0 <= source < object_count and 0 <= target < object_count) or source == target:
            rejected.append((source, target, float(confidence)))
            continue
        if target in adjacency[source]:
            continue
        if _path_exists(adjacency, target, source):
            rejected.append((source, target, float(confidence)))
            continue
        adjacency[source].add(target)
        accepted.append((source, target))
    return accepted, rejected


def safety_score(obj: SceneObject, image_shape: Sequence[int], others: Sequence[SceneObject]) -> float:
    """Tie-break score for independent objects; graph constraints always win."""
    height, width = float(image_shape[0]), float(image_shape[1])
    cx, cy = obj.center
    boundary = min(cx / width, (width - cx) / width, cy / height, (height - cy) / height)
    boundary = float(np.clip(boundary * 4.0, 0.0, 1.0))
    diagonal = max((width * width + height * height) ** 0.5, 1.0)
    distances = [
        ((cx - other.center[0]) ** 2 + (cy - other.center[1]) ** 2) ** 0.5 / diagonal
        for other in others
        if other.id != obj.id
    ]
    isolation = float(np.clip((min(distances) if distances else 1.0) * 4.0, 0.0, 1.0))
    return 0.55 * float(obj.confidence) + 0.25 * boundary + 0.20 * isolation


def uncertain_incoming_risks(
    decisions: Iterable[tuple[int, int, int | None, float]],
    confidence_floor: float = 0.45,
) -> dict[int, float]:
    """Accumulate soft blocking evidence without turning it into graph edges.

    Each item is ``(a_id, b_id, label, confidence)`` where label 0 means
    A-before-B and label 1 means B-before-A.  Disagreement (label=None) and
    very weak evidence are ignored.  The result only breaks ties inside the
    same topological layer, so accepted relations remain authoritative.
    """
    risks: dict[int, float] = {}
    for a_id, b_id, label, confidence in decisions:
        if label not in (0, 1) or confidence < confidence_floor:
            continue
        target = b_id if label == 0 else a_id
        risks[target] = risks.get(target, 0.0) + float(confidence)
    return risks


def complete_sequence(
    objects: Sequence[SceneObject],
    edges: Iterable[tuple[int, int]],
    image_shape: Sequence[int],
    uncertainty_risk: Mapping[int, float] | None = None,
) -> tuple[list[RankedObject], list[list[int]]]:
    """Flatten topological layers into a stable complete 1..N recommendation."""
    layers = compute_layers(len(objects), edges)
    by_id = {obj.id: obj for obj in objects}
    risk_by_id = uncertainty_risk or {}
    ranked: list[RankedObject] = []
    for layer_index, layer in enumerate(layers, start=1):
        ordered = sorted(
            layer,
            key=lambda object_id: (
                float(risk_by_id.get(object_id, 0.0)),
                -safety_score(by_id[object_id], image_shape, objects),
                object_id,
            ),
        )
        for object_id in ordered:
            obj = by_id[object_id]
            ranked.append(RankedObject(
                rank=len(ranked) + 1,
                layer=layer_index,
                object_id=object_id,
                class_id=obj.class_id,
                score=safety_score(obj, image_shape, objects),
                uncertainty_risk=float(risk_by_id.get(object_id, 0.0)),
            ))
    return ranked, layers


def load_annotated_scene(
    dataset_root: str | Path,
    split: str,
    image_name: str,
) -> tuple[np.ndarray, list[SceneObject], list[tuple[int, int]]]:
    root = Path(dataset_root)
    image_path = root / split / "images" / image_name
    label_path = root / split / "labels" / f"{Path(image_name).stem}.txt"
    order_path = root / split / "grasp_order" / f"{Path(image_name).stem}.json"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    objects = read_yolo_objects(label_path)
    document, issues = load_document(order_path, image_name, objects)
    errors = [issue.message for issue in issues if issue.severity == "error"]
    if errors:
        raise ValueError("; ".join(errors))
    if not document.get("reviewed"):
        raise ValueError(f"Order annotation is not reviewed: {order_path}")
    return image, scene_from_yolo(objects, image.shape), relation_pairs(document)


def render_sequence(
    image: np.ndarray,
    objects: Sequence[SceneObject],
    ranked: Sequence[RankedObject],
    edges: Iterable[tuple[int, int]] = (),
    uncertain_pairs: Iterable[tuple[int, int]] = (),
) -> np.ndarray:
    canvas = image.copy()
    by_id = {obj.id: obj for obj in objects}
    for source, target in edges:
        a = tuple(int(v) for v in by_id[source].center)
        b = tuple(int(v) for v in by_id[target].center)
        cv2.arrowedLine(canvas, a, b, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.08)
    for a_id, b_id in uncertain_pairs:
        a = np.asarray(by_id[a_id].center, dtype=np.float32)
        b = np.asarray(by_id[b_id].center, dtype=np.float32)
        distance = max(float(np.linalg.norm(b - a)), 1.0)
        direction = (b - a) / distance
        for start in np.arange(0.0, distance, 14.0):
            p1 = a + direction * start
            p2 = a + direction * min(start + 7.0, distance)
            cv2.line(canvas, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)),
                     (180, 180, 180), 2, cv2.LINE_AA)
        midpoint = tuple(np.rint((a + b) / 2).astype(int))
        cv2.putText(canvas, "?", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (220, 220, 220), 2, cv2.LINE_AA)
    for item in ranked:
        obj = by_id[item.object_id]
        color = CLASS_COLORS.get(obj.class_id, (255, 255, 255))
        polygon = np.rint(obj.points).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, color, 3, cv2.LINE_AA)
        cx, cy = (int(v) for v in obj.center)
        radius = 22
        cv2.circle(canvas, (cx, cy), radius, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)
        label = str(item.rank)
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas, label, (cx - size[0] // 2, cy + size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
