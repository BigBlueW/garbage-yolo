"""Schema and graph utilities for grasp-order annotations.

An edge ``source -> target`` means source must be grasped before target.
Object IDs are the zero-based line numbers in the matching YOLO label file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable


SCHEMA_VERSION = 1
RELATION_TYPE = "before"


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


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


def object_signature(objects: Iterable[dict]) -> str:
    canonical = json.dumps(list(objects), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def new_document(image_name: str, objects: list[dict]) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "image": image_name,
        "reviewed": False,
        "object_signature": object_signature(objects),
        "objects": objects,
        "relations": [],
    }


def relation_pairs(document: dict) -> list[tuple[int, int]]:
    pairs = []
    for edge in document.get("relations", []):
        pairs.append((int(edge["source"]), int(edge["target"])))
    return pairs


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


def validate_document(document: dict, current_objects: list[dict] | None = None) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(document, dict):
        return [ValidationIssue("error", "document", "Annotation must be a JSON object")]
    if document.get("version") != SCHEMA_VERSION:
        issues.append(ValidationIssue("error", "version", f"Expected schema version {SCHEMA_VERSION}"))
    if not isinstance(document.get("reviewed"), bool):
        issues.append(ValidationIssue("error", "reviewed", "reviewed must be true or false"))

    objects = document.get("objects")
    if not isinstance(objects, list):
        issues.append(ValidationIssue("error", "objects", "objects must be a list"))
        objects = []
    ids = [obj.get("id") for obj in objects if isinstance(obj, dict)]
    if ids != list(range(len(objects))):
        issues.append(ValidationIssue("error", "object_ids", "Object IDs must be contiguous YOLO line IDs"))

    if current_objects is not None:
        expected = object_signature(current_objects)
        if document.get("object_signature") != expected:
            issues.append(ValidationIssue(
                "error", "stale_objects",
                "The YOLO label changed after this grasp-order annotation; reset and review it again",
            ))

    relations = document.get("relations")
    if not isinstance(relations, list):
        issues.append(ValidationIssue("error", "relations", "relations must be a list"))
        return issues

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, edge in enumerate(relations):
        if not isinstance(edge, dict):
            issues.append(ValidationIssue("error", "edge", f"Relation {index} must be an object"))
            continue
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, int) or not isinstance(target, int):
            issues.append(ValidationIssue("error", "edge_ids", f"Relation {index} needs integer source/target"))
            continue
        if edge.get("type") != RELATION_TYPE:
            issues.append(ValidationIssue("error", "edge_type", f"Relation {index} must have type '{RELATION_TYPE}'"))
        pair = (source, target)
        if source == target:
            issues.append(ValidationIssue("error", "self_edge", f"Object {source} cannot precede itself"))
        if source < 0 or target < 0 or source >= len(objects) or target >= len(objects):
            issues.append(ValidationIssue("error", "edge_range", f"Relation {index} references a missing object"))
        if pair in seen:
            issues.append(ValidationIssue("error", "duplicate_edge", f"Duplicate relation {source}->{target}"))
        if (target, source) in seen:
            issues.append(ValidationIssue("error", "reverse_edge", f"Contradictory relations {source}<->{target}"))
        seen.add(pair)
        pairs.append(pair)

    if not any(issue.severity == "error" for issue in issues):
        try:
            compute_layers(len(objects), pairs)
        except ValueError as exc:
            issues.append(ValidationIssue("error", "cycle", str(exc)))
    return issues


def load_document(path: str | Path, image_name: str, current_objects: list[dict]) -> tuple[dict, list[ValidationIssue]]:
    path = Path(path)
    if not path.exists():
        document = new_document(image_name, current_objects)
        return document, []
    document = json.loads(path.read_text(encoding="utf-8"))
    return document, validate_document(document, current_objects)


def save_document(path: str | Path, document: dict, current_objects: list[dict]) -> None:
    path = Path(path)
    document = dict(document)
    document["version"] = SCHEMA_VERSION
    document["objects"] = current_objects
    document["object_signature"] = object_signature(current_objects)
    issues = [issue for issue in validate_document(document, current_objects) if issue.severity == "error"]
    if issues:
        raise ValueError("; ".join(issue.message for issue in issues))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
