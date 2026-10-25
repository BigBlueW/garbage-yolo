"""Camera-based closed-loop grasp planner: SPACE re-captures and re-plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from grasp_sequence import (
    CLASS_NAMES, SceneObject, complete_sequence, make_acyclic_edges,
    render_sequence, uncertain_incoming_risks,
)
from order_pair_model import load_pair_model, predict_bidirectional_decision


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--detector", default="best.pt")
    parser.add_argument("--order-model", default="order_runs/order_model_best.pt")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--relation-conf", type=float, default=0.80)
    parser.add_argument("--uncertain-risk-floor", type=float, default=0.45)
    parser.add_argument("--no-uncertain-risk", action="store_true")
    parser.add_argument("--show-uncertain", action="store_true",
                        help="Draw diagnostic gray pair lines; hidden by default")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="order_runs/live")
    return parser.parse_args()


def detected_objects(result) -> list[SceneObject]:
    if result.obb is None:
        return []
    points = result.obb.xyxyxyxy.detach().cpu().numpy()
    classes = result.obb.cls.detach().cpu().numpy().astype(int)
    confidence = result.obb.conf.detach().cpu().numpy()
    return [SceneObject(i, int(classes[i]), points[i].astype("float32"), float(confidence[i]))
            for i in range(len(points))]


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    detector = YOLO(args.detector)
    relation_model, checkpoint = load_pair_model(args.order_model, device)
    image_size = int(checkpoint.get("image_size", 160))
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("SPACE: capture/re-plan after each grasp | Q or ESC: quit")
    last_plan = None
    sequence_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            display = last_plan if last_plan is not None else frame
            cv2.putText(display, "SPACE: re-capture/re-plan | Q: quit", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Closed-loop grasp order", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != 32:
                continue
            result = detector.predict(source=frame, conf=args.det_conf, verbose=False)[0]
            objects = detected_objects(result)
            if not objects:
                print("No object detected; re-position the camera and press SPACE again")
                last_plan = frame.copy()
                continue
            weighted = []
            uncertain = []
            with torch.no_grad():
                for a_id in range(len(objects)):
                    for b_id in range(a_id + 1, len(objects)):
                        decision = predict_bidirectional_decision(
                            relation_model, frame, objects[a_id], objects[b_id],
                            image_size, device, args.relation_conf,
                        )
                        if decision.status == "uncertain":
                            uncertain.append({"a_id": a_id, "b_id": b_id, **decision.to_dict()})
                        elif decision.status == "relation" and decision.label is not None:
                            label, confidence = decision.label, decision.confidence
                            weighted.append((a_id, b_id, confidence) if label == 0 else
                                            (b_id, a_id, confidence))
            edges, rejected = make_acyclic_edges(len(objects), weighted)
            risk_inputs = [
                (item["a_id"], item["b_id"], item.get("label"), item["confidence"])
                for item in uncertain
            ]
            uncertainty_risk = {} if args.no_uncertain_risk else uncertain_incoming_risks(
                risk_inputs, args.uncertain_risk_floor,
            )
            ranked, layers = complete_sequence(objects, edges, frame.shape, uncertainty_risk)
            uncertain_pairs = ([(item["a_id"], item["b_id"]) for item in uncertain]
                               if args.show_uncertain else [])
            last_plan = render_sequence(frame, objects, ranked, edges, uncertain_pairs)
            next_item = ranked[0]
            sequence_number += 1
            payload = {
                "capture": sequence_number,
                "next_object_id": next_item.object_id,
                "next_class": CLASS_NAMES.get(next_item.class_id, str(next_item.class_id)),
                "sequence": [item.__dict__ for item in ranked],
                "layers": layers,
                "relations": [{"source": a, "target": b} for a, b in edges],
                "uncertain_pairs": uncertain,
                "uncertainty_risk": {str(key): value for key, value in sorted(uncertainty_risk.items())},
                "rejected_cycle_edges": len(rejected),
            }
            cv2.imwrite(str(output_dir / "latest.jpg"), last_plan)
            (output_dir / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"capture={sequence_number} NEXT=#{next_item.object_id} "
                  f"{payload['next_class']} complete_order=" +
                  "->".join(str(item.object_id) for item in ranked))
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
