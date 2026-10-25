"""Split an external JSONL manifest by scene/image groups without leakage."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.input)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    # REGRAD key format: scene/view/object-pair. For other sources, fall
    # back to image path so every image remains in exactly one split.
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row.get("key", ""))
        group = key.split("/", 1)[0] if row.get("source") == "REGRAD" and "/" in key else str(row["image"])
        groups.setdefault(group, []).append(row)

    group_names = sorted(groups)
    random.Random(args.seed).shuffle(group_names)
    val_count = max(1, int(round(len(group_names) * args.val_ratio)))
    val_groups = set(group_names[:val_count])
    train_groups = set(group_names) - val_groups
    train_rows = [row for group in group_names if group in train_groups for row in groups[group]]
    val_rows = [row for group in group_names if group in val_groups for row in groups[group]]

    for output_name, output_rows in ((args.train_output, train_rows), (args.val_output, val_rows)):
        output = Path(output_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "groups": len(group_names), "train_groups": len(train_groups),
        "val_groups": val_count, "train_rows": len(train_rows), "val_rows": len(val_rows),
        "overlap": sorted(train_groups & val_groups),
    }, indent=2))


if __name__ == "__main__":
    main()
