"""Pretrain MobileNetV3 pairwise order model from external JSONL manifests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from external_order_data import ExternalOrderDataset, ImageGroupedBatchSampler, read_manifest
from order_pair_model import GEOMETRY_SIZE, PAIR_CLASSES, build_pair_model, swap_pair_batch, swap_pair_targets
from train_order_model import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", default="order_runs/order_model_regrad_pretrained.pt")
    parser.add_argument("--architecture", choices=("tiny_cnn", "mobilenet_v3_small"), default="mobilenet_v3_small")
    parser.add_argument("--init-checkpoint", default="",
                        help="Optional earlier external checkpoint for sequential pretraining")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--symmetry-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    train_samples = read_manifest(args.train_manifest, args.max_train_samples, args.seed)
    val_samples = read_manifest(args.val_manifest, args.max_val_samples, args.seed)
    if not train_samples or not val_samples:
        raise RuntimeError("Both external train and val manifests need samples")
    train_loader = DataLoader(ExternalOrderDataset(train_samples, args.image_size, True),
                              batch_sampler=ImageGroupedBatchSampler(train_samples, args.batch_size, True, args.seed),
                              num_workers=0)
    val_loader = DataLoader(ExternalOrderDataset(val_samples, args.image_size, False),
                            batch_sampler=ImageGroupedBatchSampler(val_samples, args.batch_size, False, args.seed),
                            num_workers=0)
    counts = torch.tensor([sum(item.target == i for item in train_samples) for i in range(3)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / torch.clamp(counts * 3, min=1)).to(device))
    model = build_pair_model(args.architecture, pretrained=not bool(args.init_checkpoint)).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        checkpoint_architecture = checkpoint.get("architecture", "tiny_cnn")
        if checkpoint_architecture != args.architecture:
            raise ValueError(
                f"Checkpoint architecture {checkpoint_architecture!r} does not match "
                f"--architecture {args.architecture!r}"
            )
        model.load_state_dict(checkpoint["model_state"])
        print(f"initialized_from={args.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    best_f1, stale = -1.0, 0
    print(f"device={device} train_pairs={len(train_samples)} val_pairs={len(val_samples)}")
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for visual, geometry, target in train_loader:
            visual, geometry, target = visual.to(device), geometry.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits_ab = model(visual, geometry)
            swapped_visual, swapped_geometry = swap_pair_batch(visual, geometry)
            logits_ba = model(swapped_visual, swapped_geometry)
            target_ba = swap_pair_targets(target)
            ba_as_ab = logits_ba[:, [1, 0, 2]]
            supervised = 0.5 * (criterion(logits_ab, target) + criterion(logits_ba, target_ba))
            consistency = torch.mean((torch.softmax(logits_ab, dim=1) - torch.softmax(ba_as_ab, dim=1)) ** 2)
            loss = supervised + args.symmetry_weight * consistency
            loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
        result = evaluate(model, val_loader, criterion, device)
        print(f"epoch={epoch:03d} train_loss={np.mean(losses):.4f} val_acc={result['accuracy']:.3f} "
              f"val_macro_f1={result['macro_f1']:.3f}")
        if result["macro_f1"] > best_f1 + 1e-4:
            best_f1, stale = result["macro_f1"], 0
            torch.save({"model_state": model.state_dict(), "geometry_size": GEOMETRY_SIZE,
                        "architecture": args.architecture, "pair_classes": PAIR_CLASSES,
                        "image_size": args.image_size, "best_val_metrics": result,
                        "pretraining_source": [args.train_manifest, args.val_manifest], "args": vars(args)}, output)
        else:
            stale += 1
            if stale >= args.patience:
                break
    output.with_suffix(".metrics.json").write_text(json.dumps({"best_val_macro_f1": best_f1}, indent=2), encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
