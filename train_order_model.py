"""Train the pairwise occlusion relation model from local annotations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from order_pair_data import OrderPairDataset, build_pair_samples
from order_pair_model import GEOMETRY_SIZE, PAIR_CLASSES, build_pair_model, swap_pair_batch, swap_pair_targets


def metrics(targets: list[int], predictions: list[int]) -> dict:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1
    per_class = []
    for index, name in enumerate(PAIR_CLASSES):
        tp = confusion[index, index]
        precision = tp / max(confusion[:, index].sum(), 1)
        recall = tp / max(confusion[index, :].sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append({"class": name, "precision": precision, "recall": recall, "f1": f1})
    return {
        "accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, targets, predictions = [], [], []
    for visual, geometry, target in loader:
        visual, geometry, target = visual.to(device), geometry.to(device), target.to(device)
        logits = model(visual, geometry)
        losses.append(float(criterion(logits, target).item()))
        targets.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    result = metrics(targets, predictions)
    result["loss"] = float(np.mean(losses)) if losses else 0.0
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="robotic-garbage-obb")
    parser.add_argument("--output", default="order_runs/order_model_mobilenet_best.pt")
    parser.add_argument("--architecture", choices=("tiny_cnn", "mobilenet_v3_small"),
                        default="mobilenet_v3_small")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not initialize MobileNetV3 with ImageNet weights")
    parser.add_argument("--init-checkpoint", default="",
                        help="Initialize from an external-pretraining checkpoint, but reset optimizer")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Sample all available classes evenly each epoch; use with a large negative ratio")
    parser.add_argument("--samples-per-epoch", type=int, default=0,
                        help="Balanced-sampler draws per epoch; 0 uses six times the smaller direction class")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12,
                        help="0 = no early stopping: cosine LR over all epochs and keep the last epoch")
    parser.add_argument("--symmetry-weight", type=float, default=0.2,
                        help="Penalty for disagreement between AB and swapped BA predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    train_samples = build_pair_samples(args.dataset, "train", args.negative_ratio, args.seed)
    val_samples = build_pair_samples(args.dataset, "val", args.negative_ratio, args.seed)
    if not train_samples or not val_samples:
        raise RuntimeError("Both train and val need reviewed positive grasp-order relations")
    print(f"device={device} train_pairs={len(train_samples)} val_pairs={len(val_samples)}")
    for split, samples in (("train", train_samples), ("val", val_samples)):
        counts = [sum(sample.target == index for sample in samples) for index in range(3)]
        print(f"{split}: " + ", ".join(f"{PAIR_CLASSES[i]}={counts[i]}" for i in range(3)))
    train_dataset = OrderPairDataset(train_samples, args.image_size, True)
    train_counts = [sum(sample.target == index for sample in train_samples) for index in range(3)]
    if args.balanced_sampler:
        sample_weights = [1.0 / max(train_counts[sample.target], 1) for sample in train_samples]
        positive_floor = max(min(train_counts[0], train_counts[1]), 1)
        samples_per_epoch = args.samples_per_epoch or 6 * positive_floor
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=samples_per_epoch, replacement=True, generator=generator,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        print(f"balanced_sampler samples_per_epoch={samples_per_epoch}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(OrderPairDataset(val_samples, args.image_size, False), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    counts = torch.tensor(train_counts, dtype=torch.float32)
    weights = counts.sum() / torch.clamp(counts * 3.0, min=1.0)
    criterion = nn.CrossEntropyLoss(weight=None if args.balanced_sampler else weights.to(device))
    model = build_pair_model(args.architecture, pretrained=not args.no_pretrained).to(device)
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
    backbone = getattr(model, "rgb_backbone", None)
    if backbone is not None:
        model.set_backbone_trainable(args.freeze_backbone_epochs <= 0)
        backbone_ids = {id(parameter) for parameter in backbone.parameters()}
        head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
        optimizer = torch.optim.AdamW([
            {"params": backbone.parameters(), "lr": args.lr * 0.1},
            {"params": head_parameters, "lr": args.lr},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_f1, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        if backbone is not None and epoch == args.freeze_backbone_epochs + 1:
            model.set_backbone_trainable(True)
            print(f"unfreeze_backbone epoch={epoch}")
        model.train()
        if backbone is not None and epoch <= args.freeze_backbone_epochs:
            backbone.eval()
        train_losses = []
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
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        result = evaluate(model, val_loader, criterion, device)
        if args.patience > 0:
            scheduler.step(result["macro_f1"])
        else:
            scheduler.step()
        print(f"epoch={epoch:03d} train_loss={np.mean(train_losses):.4f} "
              f"val_loss={result['loss']:.4f} val_acc={result['accuracy']:.3f} val_macro_f1={result['macro_f1']:.3f}")
        if args.patience <= 0 or result["macro_f1"] > best_f1 + 1e-4:
            best_f1, stale = result["macro_f1"], 0
            torch.save({
                "model_state": model.state_dict(), "geometry_size": GEOMETRY_SIZE,
                "architecture": args.architecture,
                "pair_classes": PAIR_CLASSES, "image_size": args.image_size,
                "best_val_metrics": result, "args": vars(args),
            }, output)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch}")
                break
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"best_val_macro_f1": best_f1}, indent=2), encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
