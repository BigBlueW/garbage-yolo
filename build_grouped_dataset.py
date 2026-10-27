"""Build a new leakage-controlled dataset without modifying the original."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Item:
    key: str
    split: str
    image: Path
    label: Path
    order: Path
    positive: int


class UnionFind:
    def __init__(self, keys: list[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def load_items(root: Path) -> dict[str, Item]:
    items = {}
    for split in ("train", "val", "test"):
        for image in sorted((root / split / "images").iterdir()):
            if not image.is_file():
                continue
            key = f"{split}/{image.name}"
            label = root / split / "labels" / f"{image.stem}.txt"
            order = root / split / "grasp_order" / f"{image.stem}.json"
            positive = 0
            if order.exists():
                positive = int(bool(json.loads(order.read_text(encoding="utf-8")).get("relations")))
            items[key] = Item(key, split, image, label, order, positive)
    return items


def assign_groups(groups: list[list[Item]], ratios: dict[str, float], seed: int) -> dict[str, list[Item]]:
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda group: (len(group), sum(item.positive for item in group)), reverse=True)
    total_images = sum(map(len, groups))
    total_positive = sum(item.positive for group in groups for item in group)
    targets = {split: (total_images * ratio, total_positive * ratio) for split, ratio in ratios.items()}
    assigned = {split: [] for split in ratios}
    counts = {split: [0, 0] for split in ratios}
    for group in groups:
        size, positives = len(group), sum(item.positive for item in group)
        best_split = max(ratios, key=lambda split: (
            (targets[split][0] - counts[split][0]) / max(targets[split][0], 1)
            + (targets[split][1] - counts[split][1]) / max(targets[split][1], 1)
        ))
        assigned[best_split].extend(group)
        counts[best_split][0] += size
        counts[best_split][1] += positives
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="robotic-garbage-obb")
    parser.add_argument("--audit", default="order_runs/scene_split_audit_grouping.json")
    parser.add_argument("--output", default="robotic-garbage-obb-grouped")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true", help="Copy files after writing the plan")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    items = load_items(source)
    union = UnionFind(list(items))
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    for candidate in audit["candidates"]:
        if candidate["left"] in items and candidate["right"] in items:
            union.union(candidate["left"], candidate["right"])
    grouped: dict[str, list[Item]] = {}
    for item in items.values():
        grouped.setdefault(union.find(item.key), []).append(item)
    groups = list(grouped.values())
    assigned = assign_groups(groups, {"train": .70, "val": .15, "test": .15}, args.seed)
    destination_for = {item.key: split for split, values in assigned.items() for item in values}
    leakage = []
    for group in groups:
        destinations = {destination_for[item.key] for item in group}
        if len(destinations) != 1:
            leakage.append([item.key for item in group])
    plan = {
        "source": str(source), "output": str(output), "similarity_threshold": audit["threshold"],
        "groups": len(groups), "largest_group": max(map(len, groups)), "leaking_groups": leakage,
        "splits": {split: {"images": len(values), "positive_images": sum(item.positive for item in values)}
                   for split, values in assigned.items()},
        "assignments": {split: [item.key for item in values] for split, values in assigned.items()},
    }
    plan_path = Path("order_runs/grouped_dataset_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in plan.items() if key != "assignments"}, indent=2))
    if not args.apply:
        print("dry_run=true; add --apply after reviewing the plan")
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    for split, values in assigned.items():
        for subdir in ("images", "labels", "grasp_order"):
            (output / split / subdir).mkdir(parents=True, exist_ok=True)
        for item in values:
            shutil.copy2(item.image, output / split / "images" / item.image.name)
            if item.label.exists():
                shutil.copy2(item.label, output / split / "labels" / item.label.name)
            if item.order.exists():
                shutil.copy2(item.order, output / split / "grasp_order" / item.order.name)
    (output / "dataset.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\n\n"
        "names:\n  0: plastic\n  1: metal\n  2: paper\n  3: general_waste\n", encoding="utf-8"
    )
    print(f"created={output}")


if __name__ == "__main__":
    main()
