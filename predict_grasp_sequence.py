"""Output a complete 1..N grasp order from annotations or learned models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from grasp_sequence import (
    CLASS_NAMES, SceneObject, complete_sequence, load_annotated_scene,
    make_acyclic_edges, render_sequence, scene_from_yolo, uncertain_incoming_risks,
)
from grasp_order import read_yolo_objects
from order_pair_model import load_pair_model, predict_bidirectional_decision


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--annotated", metavar="IMAGE_NAME",
                      help="Use the human relation JSON (ground-truth UI check; does not test the model)")
    mode.add_argument("--labeled-image", metavar="IMAGE_NAME",
                      help="Use saved OBB labels but predict order with the learned model")
    mode.add_argument("--image", help="Use YOLO detector + trained pair model")
    parser.add_argument("--dataset", default="robotic-garbage-obb")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--detector", default="best.pt")
    parser.add_argument("--order-model", default="order_runs/order_model_best.pt")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--relation-conf", type=float, default=0.80)
    parser.add_argument("--uncertain-risk-floor", type=float, default=0.45,
                        help="Minimum agreed directional confidence used only to avoid a risky first pick")
    parser.add_argument("--no-uncertain-risk", action="store_true",
                        help="Disable soft risk ordering from low-confidence directional agreement")
    parser.add_argument("--show-uncertain", action="store_true",
                        help="Draw diagnostic gray pair lines; hidden by default because they are not relations")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="Preview path; defaults beside source image")
    parser.add_argument("--json", dest="json_path", help="Optional machine-readable output path")
    parser.add_argument("--next-only", action="store_true", help="Print only the next grasp for closed-loop use")
    return parser.parse_args()


def detect_scene(image_path: str, detector_path: str, confidence: float) -> tuple[object, list[SceneObject]]:
    from ultralytics import YOLO
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    result = YOLO(detector_path).predict(source=image, conf=confidence, verbose=False)[0]
    if result.obb is None:
        return image, []
    points = result.obb.xyxyxyxy.detach().cpu().numpy()
    classes = result.obb.cls.detach().cpu().numpy().astype(int)
    confidences = result.obb.conf.detach().cpu().numpy()
    objects = [
        SceneObject(index, int(classes[index]), points[index].astype("float32"), float(confidences[index]))
        for index in range(len(points))
    ]
    return image, objects


def load_labeled_objects(dataset_root: str, split: str, image_name: str):
    source_path = Path(dataset_root) / split / "images" / image_name
    label_path = Path(dataset_root) / split / "labels" / f"{Path(image_name).stem}.txt"
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Cannot read labels: {label_path}")
    return source_path, image, scene_from_yolo(read_yolo_objects(label_path), image.shape)


@torch.no_grad()
def predict_edges(image, objects, model_path: str, relation_conf: float, device_name: str):
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else
                          "cpu" if device_name == "auto" else device_name)
    model, checkpoint = load_pair_model(model_path, device)
    image_size = int(checkpoint.get("image_size", 160))
    candidates = []
    uncertain = []
    for a_id in range(len(objects)):
        for b_id in range(a_id + 1, len(objects)):
            a, b = objects[a_id], objects[b_id]
            decision = predict_bidirectional_decision(
                model, image, a, b, image_size, device, relation_conf,
            )
            if decision.status == "uncertain":
                uncertain.append({"a_id": a_id, "b_id": b_id, **decision.to_dict()})
                continue
            if decision.status != "relation" or decision.label is None:
                continue
            label, confidence = decision.label, decision.confidence
            source, target = (a_id, b_id) if label == 0 else (b_id, a_id)
            candidates.append((source, target, confidence))
    accepted, rejected = make_acyclic_edges(len(objects), candidates)
    return accepted, rejected, uncertain


def main():
    args = parse_args()
    if args.annotated:
        image, objects, edges = load_annotated_scene(args.dataset, args.split, args.annotated)
        source_path = Path(args.dataset) / args.split / "images" / args.annotated
        rejected = []
        uncertain = []
        order_source = "human_annotations"
    elif args.labeled_image:
        source_path, image, objects = load_labeled_objects(args.dataset, args.split, args.labeled_image)
        edges, rejected, uncertain = predict_edges(image, objects, args.order_model, args.relation_conf, args.device)
        order_source = "learned_model_with_saved_obb"
    else:
        source_path = Path(args.image)
        image, objects = detect_scene(str(source_path), args.detector, args.det_conf)
        edges, rejected, uncertain = predict_edges(image, objects, args.order_model, args.relation_conf, args.device)
        order_source = "detector_and_learned_model"
    if not objects:
        raise RuntimeError("No objects were found")
    risk_inputs = [
        (item["a_id"], item["b_id"], item.get("label"), item["confidence"])
        for item in uncertain
    ]
    uncertainty_risk = {} if args.no_uncertain_risk else uncertain_incoming_risks(
        risk_inputs, args.uncertain_risk_floor,
    )
    ranked, layers = complete_sequence(objects, edges, image.shape, uncertainty_risk)
    shown = ranked[:1] if args.next_only else ranked
    print("NEXT GRASP (closed-loop authoritative):")
    first = ranked[0]
    print(f"  #{first.object_id} {CLASS_NAMES.get(first.class_id, first.class_id)} "
          f"layer={first.layer} score={first.score:.3f} risk={first.uncertainty_risk:.3f}")
    if not args.next_only:
        print("COMPLETE CURRENT-FRAME PLAN:")
        for item in shown:
            independent = " (same-layer tie-break)" if len(layers[item.layer - 1]) > 1 else ""
            print(f"  {item.rank}. #{item.object_id} {CLASS_NAMES.get(item.class_id, item.class_id)} "
                  f"layer={item.layer} score={item.score:.3f} risk={item.uncertainty_risk:.3f}{independent}")
    # Never place generated previews inside dataset/images: that would silently
    # contaminate a later train/val/test scan.
    output = (Path(args.output) if args.output else
              Path("order_runs") / "predictions" / f"{source_path.stem}_grasp_sequence.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    uncertain_pairs = ([(item["a_id"], item["b_id"]) for item in uncertain]
                       if args.show_uncertain else [])
    if not cv2.imwrite(str(output), render_sequence(image, objects, ranked, edges, uncertain_pairs)):
        raise OSError(f"Cannot write {output}")
    payload = {
        "source": str(source_path),
        "order_source": order_source,
        "next_object_id": first.object_id,
        "sequence": [item.__dict__ for item in ranked],
        "layers": layers,
        "relations": [{"source": a, "target": b} for a, b in edges],
        "uncertain_pairs": uncertain,
        "uncertainty_risk": {str(key): value for key, value in sorted(uncertainty_risk.items())},
        "uncertain_risk_floor": args.uncertain_risk_floor,
        "rejected_cycle_edges": [{"source": a, "target": b, "confidence": c} for a, b, c in rejected],
        "note": "Re-run after every physical grasp and new photo; later ranks are a current-frame plan.",
    }
    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if uncertain:
        visibility = "shown as gray diagnostics" if args.show_uncertain else "hidden; details remain in JSON"
        print(f"uncertain_pairs={len(uncertain)} ({visibility})")
    print(f"preview={output}")


if __name__ == "__main__":
    main()
