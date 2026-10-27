"""Render the strongest scene-similarity candidates for visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="robotic-garbage-obb")
    parser.add_argument("--audit", default="order_runs/scene_split_audit.json")
    parser.add_argument("--output", default="order_runs/scene_split_pairs.jpg")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    root = Path(args.dataset)
    candidates = json.loads(Path(args.audit).read_text(encoding="utf-8"))["candidates"][:args.count]
    rows = []
    for item in candidates:
        tiles = []
        for value in (item["left"], item["right"]):
            split, name = value.split("/", 1)
            image = cv2.imread(str(root / split / "images" / name))
            if image is None:
                raise FileNotFoundError(value)
            scale = min(300 / image.shape[1], 190 / image.shape[0])
            resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            canvas = np.full((220, 320, 3), 245, np.uint8)
            y = 25 + (190 - resized.shape[0]) // 2
            x = (320 - resized.shape[1]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            cv2.putText(canvas, value, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(canvas)
        row = np.hstack(tiles)
        cv2.putText(row, f"similarity={item['similarity']:.4f}", (245, 216),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 180), 1, cv2.LINE_AA)
        rows.append(row)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))
    print(output)


if __name__ == "__main__":
    main()
