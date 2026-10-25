"""Shared JSONL pair format for external occlusion/order datasets."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from grasp_sequence import SceneObject
from order_pair_model import pair_tensor


@dataclass(frozen=True)
class ExternalPair:
    image_path: Path
    a_points: np.ndarray
    b_points: np.ndarray
    target: int
    source: str
    key: str


def write_manifest(path: str | Path, rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifest(path: str | Path, max_samples: int = 0, seed: int = 42) -> list[ExternalPair]:
    manifest = Path(path).resolve()
    samples: list[ExternalPair] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = (manifest.parent / image_path).resolve()
            target = int(row["target"])
            if target not in (0, 1, 2):
                raise ValueError(f"{manifest}:{line_number}: invalid target {target}")
            samples.append(ExternalPair(
                image_path=image_path,
                a_points=np.asarray(row["a_points"], dtype=np.float32).reshape(4, 2),
                b_points=np.asarray(row["b_points"], dtype=np.float32).reshape(4, 2),
                target=target,
                source=str(row.get("source", "external")),
                key=str(row.get("key", line_number)),
            ))
    if max_samples > 0 and len(samples) > max_samples:
        random.Random(seed).shuffle(samples)
        samples = samples[:max_samples]
    return samples


def relative_image_path(image_path: Path, manifest_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(manifest_path.resolve().parent).as_posix()
    except ValueError:
        import os
        return Path(os.path.relpath(image_path.resolve(), manifest_path.resolve().parent)).as_posix()


class ExternalOrderDataset(Dataset):
    def __init__(self, samples: list[ExternalPair], image_size: int = 160, augment: bool = False) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self._cached_path: Path | None = None
        self._cached_image: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if sample.image_path != self._cached_path:
            image = cv2.imread(str(sample.image_path))
            if image is None:
                raise FileNotFoundError(sample.image_path)
            self._cached_path, self._cached_image = sample.image_path, image
        assert self._cached_image is not None
        a = SceneObject(0, -1, sample.a_points)
        b = SceneObject(1, -1, sample.b_points)
        visual, geometry = pair_tensor(self._cached_image, a, b, self.image_size)
        if self.augment:
            visual = visual.clone()
            if random.random() < 0.5:
                visual[:3] = torch.clamp(visual[:3] * random.uniform(0.85, 1.15), 0.0, 1.0)
            if random.random() < 0.25:
                visual[:3] = torch.clamp(visual[:3] + torch.randn_like(visual[:3]) * 0.015, 0.0, 1.0)
        return visual, geometry, torch.tensor(sample.target, dtype=torch.long)


class ImageGroupedBatchSampler:
    """Shuffle image groups while keeping each image's pairs contiguous.

    REGRAD/InstaOrder contain many pairs per high-resolution image. Grouping
    avoids decoding the same image again for nearly every randomly shuffled
    sample while retaining epoch-level randomization.
    """

    def __init__(self, samples: list[ExternalPair], batch_size: int, shuffle: bool, seed: int = 42) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        groups: dict[Path, list[int]] = {}
        for index, sample in enumerate(samples):
            groups.setdefault(sample.image_path, []).append(index)
        self.groups = list(groups.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)
        batch: list[int] = []
        for group in groups:
            for index in group:
                batch.append(index)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        sample_count = sum(len(group) for group in self.groups)
        return (sample_count + self.batch_size - 1) // self.batch_size
