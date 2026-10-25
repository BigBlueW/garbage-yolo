"""Build pairwise samples from reviewed grasp-order annotations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import torch
from torch.utils.data import Dataset

from grasp_order import read_yolo_objects, validate_document
from grasp_sequence import scene_from_yolo
from order_pair_model import pair_is_candidate, pair_tensor


@dataclass(frozen=True)
class PairSample:
    image_path: Path
    label_path: Path
    a_id: int
    b_id: int
    target: int


def transitive_relations(object_count: int, relations: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    adjacency = {index: set() for index in range(object_count)}
    for source, target in relations:
        adjacency[source].add(target)
    closure: set[tuple[int, int]] = set()
    for source in range(object_count):
        stack = list(adjacency[source])
        while stack:
            target = stack.pop()
            if (source, target) in closure:
                continue
            closure.add((source, target))
            stack.extend(adjacency[target])
    return closure


def build_pair_samples(
    dataset_root: str | Path,
    split: str,
    negative_ratio: float = 2.0,
    seed: int = 42,
) -> list[PairSample]:
    root = Path(dataset_root)
    order_dir = root / split / "grasp_order"
    rng = random.Random(seed)
    positives: list[PairSample] = []
    negatives: list[PairSample] = []
    for order_path in sorted(order_dir.glob("*.json")):
        document = json.loads(order_path.read_text(encoding="utf-8"))
        if not document.get("reviewed"):
            continue
        image_name = document.get("image") or f"{order_path.stem}.png"
        image_path = root / split / "images" / image_name
        label_path = root / split / "labels" / f"{order_path.stem}.txt"
        if not image_path.exists() or not label_path.exists():
            continue
        yolo_objects = read_yolo_objects(label_path)
        errors = [issue for issue in validate_document(document, yolo_objects) if issue.severity == "error"]
        if errors:
            raise ValueError(f"Invalid {order_path}: " + "; ".join(issue.message for issue in errors))
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        objects = scene_from_yolo(yolo_objects, image.shape)
        direct = [(int(edge["source"]), int(edge["target"])) for edge in document.get("relations", [])]
        before = transitive_relations(len(objects), direct)
        for a_id in range(len(objects)):
            for b_id in range(a_id + 1, len(objects)):
                if (a_id, b_id) in before:
                    positives.append(PairSample(image_path, label_path, a_id, b_id, 0))
                    positives.append(PairSample(image_path, label_path, b_id, a_id, 1))
                elif (b_id, a_id) in before:
                    positives.append(PairSample(image_path, label_path, a_id, b_id, 1))
                    positives.append(PairSample(image_path, label_path, b_id, a_id, 0))
                elif pair_is_candidate(objects[a_id], objects[b_id], image.shape):
                    negatives.append(PairSample(image_path, label_path, a_id, b_id, 2))
    rng.shuffle(negatives)
    # Negatives are also presented in both orientations.  This prevents the
    # arbitrary YOLO line order from becoming a shortcut for the classifier.
    negative_limit = int(round((len(positives) / 2) * max(negative_ratio, 0.0)))
    if positives:
        negatives = negatives[:negative_limit]
    mirrored_negatives = []
    for sample in negatives:
        mirrored_negatives.append(sample)
        mirrored_negatives.append(PairSample(
            sample.image_path, sample.label_path, sample.b_id, sample.a_id, 2,
        ))
    samples = positives + mirrored_negatives
    rng.shuffle(samples)
    return samples


class OrderPairDataset(Dataset):
    def __init__(self, samples: list[PairSample], image_size: int = 160, augment: bool = False,
                 preload: bool = True) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self._cached_path: Path | None = None
        self._cached_image = None
        self._cached_objects = None
        self._preloaded = None
        if preload:
            self._preloaded = [None] * len(samples)
            # Grouping by image makes every high-resolution source decode once.
            for index in sorted(range(len(samples)), key=lambda item: str(samples[item].image_path)):
                sample = samples[index]
                image, objects = self._load(sample)
                self._preloaded[index] = pair_tensor(
                    image, objects[sample.a_id], objects[sample.b_id], self.image_size,
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, sample: PairSample):
        if sample.image_path != self._cached_path:
            image = cv2.imread(str(sample.image_path))
            if image is None:
                raise FileNotFoundError(sample.image_path)
            yolo_objects = read_yolo_objects(sample.label_path)
            self._cached_path = sample.image_path
            self._cached_image = image
            self._cached_objects = scene_from_yolo(yolo_objects, image.shape)
        return self._cached_image, self._cached_objects

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self._preloaded is None:
            image, objects = self._load(sample)
            visual, geometry = pair_tensor(image, objects[sample.a_id], objects[sample.b_id], self.image_size)
        else:
            visual, geometry = self._preloaded[index]
        if self.augment:
            visual = visual.clone()
            if random.random() < 0.5:
                visual[:3] = torch.clamp(visual[:3] * random.uniform(0.85, 1.15), 0.0, 1.0)
            if random.random() < 0.25:
                visual[:3] = torch.clamp(visual[:3] + torch.randn_like(visual[:3]) * 0.015, 0.0, 1.0)
        return visual, geometry, torch.tensor(sample.target, dtype=torch.long)
