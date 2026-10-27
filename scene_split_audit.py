"""Find visually similar cross-split images with ImageNet embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="robotic-garbage-obb")
    parser.add_argument("--output", default="order_runs/scene_split_audit.json")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(args.dataset)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    records = []
    for split in ("train", "val", "test"):
        for path in sorted((root / split / "images").iterdir()):
            if path.suffix.lower() in SUFFIXES:
                records.append({"split": split, "path": path})

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights).features.to(device).eval()
    transform = weights.transforms()
    features = []
    with torch.no_grad():
        for start in range(0, len(records), 32):
            images = []
            for record in records[start:start + 32]:
                bgr = cv2.imread(str(record["path"]))
                if bgr is None:
                    raise FileNotFoundError(record["path"])
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                images.append(transform(torch.from_numpy(rgb).permute(2, 0, 1)))
            batch = torch.stack(images).to(device)
            embedding = model(batch).mean(dim=(2, 3))
            embedding = torch.nn.functional.normalize(embedding, dim=1)
            features.append(embedding.cpu())
    matrix = torch.cat(features).numpy()
    similarity = matrix @ matrix.T
    candidates = []
    cross_split_count = 0
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            score = float(similarity[left, right])
            if score >= args.threshold:
                cross_split = records[left]["split"] != records[right]["split"]
                cross_split_count += int(cross_split)
                candidates.append({
                    "similarity": score,
                    "left": f"{records[left]['split']}/{records[left]['path'].name}",
                    "right": f"{records[right]['split']}/{records[right]['path'].name}",
                    "cross_split": cross_split,
                })
    candidates.sort(key=lambda item: item["similarity"], reverse=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"threshold": args.threshold, "images": len(records),
                                  "candidates": candidates}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "images": len(records),
                      "all_candidates": len(candidates), "cross_split_candidates": cross_split_count,
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
