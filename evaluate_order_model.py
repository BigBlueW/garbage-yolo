"""Evaluate pair accuracy and closed-loop first-pick safety on val/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from grasp_order import load_relations, read_yolo_objects
from grasp_sequence import complete_sequence, make_acyclic_edges, scene_from_yolo, uncertain_incoming_risks
from order_pair_data import OrderPairDataset, build_pair_samples, transitive_relations
from order_pair_model import PAIR_CLASSES, load_pair_model, predict_bidirectional_decision
from train_order_model import metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="robotic-garbage-obb")
    parser.add_argument("--model", default="order_runs/order_model_best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--relation-conf", type=float, default=0.80)
    parser.add_argument("--uncertain-risk-floor", type=float, default=0.45)
    parser.add_argument("--no-uncertain-risk", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--details-json", help="Optional path for per-image diagnostics, including unsafe first picks")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model, checkpoint = load_pair_model(args.model, device)
    samples = build_pair_samples(args.dataset, args.split, args.negative_ratio, 42)
    loader = DataLoader(OrderPairDataset(samples, int(checkpoint.get("image_size", 160))), batch_size=32,
                        shuffle=False, num_workers=0)
    targets, predictions = [], []
    with torch.no_grad():
        for visual, geometry, target in loader:
            logits = model(visual.to(device), geometry.to(device))
            targets.extend(target.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    report = metrics(targets, predictions)

    safe, evaluated, rejected_cycles = 0, 0, 0
    edge_tp, edge_fp, edge_fn = 0, 0, 0
    compliant_orders, order_violations = 0, 0
    image_details = []
    root = Path(args.dataset)
    for order_path in sorted((root / args.split / "grasp_order").glob("*.json")):
        image_path = next((root / args.split / "images").glob(f"{order_path.stem}.*"), None)
        label_path = root / args.split / "labels" / f"{order_path.stem}.txt"
        image = cv2.imread(str(image_path)) if image_path else None
        if image is None or not label_path.exists():
            continue
        yolo_objects = read_yolo_objects(label_path)
        objects = scene_from_yolo(yolo_objects, image.shape)
        if not objects:
            continue
        gt_direct = load_relations(order_path, len(yolo_objects))
        weighted = []
        pair_decisions = []
        risk_inputs = []
        for a_id in range(len(objects)):
            for b_id in range(a_id + 1, len(objects)):
                decision = predict_bidirectional_decision(
                    model, image, objects[a_id], objects[b_id],
                    int(checkpoint.get("image_size", 160)), device, args.relation_conf,
                )
                pair_decisions.append({"a": a_id, "b": b_id, **decision.to_dict()})
                if decision.status == "uncertain":
                    risk_inputs.append((a_id, b_id, decision.label, decision.confidence))
                if decision.status == "relation" and decision.label is not None:
                    weighted.append(
                        (a_id, b_id, decision.confidence)
                        if decision.label == 0 else (b_id, a_id, decision.confidence)
                    )
        predicted, rejected = make_acyclic_edges(len(objects), weighted)
        rejected_cycles += len(rejected)
        uncertainty_risk = {} if args.no_uncertain_risk else uncertain_incoming_risks(
            risk_inputs, args.uncertain_risk_floor,
        )
        ranked, _ = complete_sequence(objects, predicted, image.shape, uncertainty_risk)
        predicted_first = ranked[0].object_id
        gt_closure = transitive_relations(len(objects), gt_direct)
        predicted_set = set(predicted)
        edge_tp += len(predicted_set & gt_closure)
        edge_fp += len(predicted_set - gt_closure)
        edge_fn += len(gt_closure - predicted_set)
        gt_blocked = {target for _, target in gt_closure}
        positions = {item.object_id: index for index, item in enumerate(ranked)}
        violations = sorted(
            edge for edge in gt_closure if positions[edge[0]] >= positions[edge[1]]
        )
        order_violations += len(violations)
        compliant_orders += int(not violations)
        is_safe = predicted_first not in gt_blocked
        safe += int(is_safe)
        evaluated += 1
        if args.details_json:
            image_details.append({
                "image": str(image_path),
                "safe": is_safe,
                "predicted_first": predicted_first,
                "ground_truth_direct": [list(edge) for edge in gt_direct],
                "ground_truth_closure": [list(edge) for edge in sorted(gt_closure)],
                "ground_truth_blocked": sorted(gt_blocked),
                "predicted_edges": [list(edge) for edge in predicted],
                "false_positive_edges": [list(edge) for edge in sorted(predicted_set - gt_closure)],
                "missed_edges": [list(edge) for edge in sorted(gt_closure - predicted_set)],
                "order_violations": [list(edge) for edge in violations],
                "ranked_object_ids": [item.object_id for item in ranked],
                "uncertainty_risk": {str(key): value for key, value in sorted(uncertainty_risk.items())},
                "pair_decisions": pair_decisions,
            })
    report["closed_loop_first_pick_safety"] = safe / max(evaluated, 1)
    report["images_evaluated"] = evaluated
    report["complete_order_compliance"] = compliant_orders / max(evaluated, 1)
    report["complete_order_violations"] = order_violations
    report["cycle_edges_rejected"] = rejected_cycles
    report["uncertain_risk_enabled"] = not args.no_uncertain_risk
    report["uncertain_risk_floor"] = args.uncertain_risk_floor
    edge_precision = edge_tp / max(edge_tp + edge_fp, 1)
    edge_recall = edge_tp / max(edge_tp + edge_fn, 1)
    report["accepted_edge_metrics"] = {
        "true_positive": edge_tp,
        "false_positive": edge_fp,
        "false_negative": edge_fn,
        "precision": edge_precision,
        "recall": edge_recall,
        "f1": 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-12),
    }
    if args.details_json:
        details_path = Path(args.details_json)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(json.dumps({"summary": report, "images": image_details}, ensure_ascii=False, indent=2), encoding="utf-8")
        report["details_json"] = str(details_path)
        report["unsafe_images"] = [item["image"] for item in image_details if not item["safe"]]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
