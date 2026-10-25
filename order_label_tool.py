"""Interactive GUI for annotating grasp precedence on existing YOLO OBBs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from grasp_order import (
    RELATION_TYPE,
    compute_layers,
    load_document,
    new_document,
    read_yolo_objects,
    relation_pairs,
    relation_path,
    save_document,
)


CLASS_NAMES = ["Plastic", "Metal", "Paper", "GeneralWaste"]
CLASS_COLORS = [(0, 255, 0), (255, 230, 0), (0, 140, 255), (255, 0, 255)]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HEADER_HEIGHT = 94


class GraspOrderAnnotator:
    def __init__(self, dataset_root: Path, split: str = "train"):
        self.dataset_root = dataset_root
        self.splits = [name for name in ("train", "val", "test") if (dataset_root / name / "images").is_dir()]
        if not self.splits:
            raise FileNotFoundError(f"No train/val/test image folders under {dataset_root}")
        self.split_index = self.splits.index(split) if split in self.splits else 0
        self.image_index = 0
        self.image_paths: list[Path] = []
        self.raw_image: np.ndarray | None = None
        self.display_image: np.ndarray | None = None
        self.scale = 1.0
        self.objects: list[dict] = []
        self.polygons: list[np.ndarray] = []
        self.centers: list[tuple[int, int]] = []
        self.document: dict = {}
        self.document_path: Path | None = None
        self.issues = []
        self.selected_source: int | None = None
        self.mode = "ADD"
        self.history: list[dict] = []
        self.dirty = False
        self.message = ""
        self.last_click = None
        self.last_candidates: tuple[int, ...] = ()
        self.candidate_index = 0
        self.window_name = "Grasp Order Annotation Tool"
        self.refresh_images()
        self.load_current()

    @property
    def split(self) -> str:
        return self.splits[self.split_index]

    def refresh_images(self) -> None:
        folder = self.dataset_root / self.split / "images"
        self.image_paths = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not self.image_paths:
            raise FileNotFoundError(f"No images in {folder}")
        self.image_index = max(0, min(self.image_index, len(self.image_paths) - 1))

    def current_label_path(self) -> Path:
        return self.dataset_root / self.split / "labels" / f"{self.image_paths[self.image_index].stem}.txt"

    def object_polygon(self, item: dict, width: int, height: int) -> np.ndarray:
        coords = item["coordinates"]
        if item["format"] == "obb":
            points = [(coords[i] * width, coords[i + 1] * height) for i in range(0, 8, 2)]
        else:
            cx, cy, box_w, box_h = coords
            cx, cy, box_w, box_h = cx * width, cy * height, box_w * width, box_h * height
            points = [
                (cx - box_w / 2, cy - box_h / 2),
                (cx + box_w / 2, cy - box_h / 2),
                (cx + box_w / 2, cy + box_h / 2),
                (cx - box_w / 2, cy + box_h / 2),
            ]
        return np.asarray(points, dtype=np.float32)

    def load_current(self) -> None:
        path = self.image_paths[self.image_index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        height, width = image.shape[:2]
        max_width, max_height = 1280, 800
        self.scale = min(1.0, max_width / width, max_height / height)
        if self.scale < 1.0:
            image = cv2.resize(image, (round(width * self.scale), round(height * self.scale)), interpolation=cv2.INTER_AREA)
        self.raw_image = image
        disp_h, disp_w = image.shape[:2]
        self.objects = read_yolo_objects(self.current_label_path())
        self.polygons = [self.object_polygon(item, disp_w, disp_h) for item in self.objects]
        self.centers = [tuple(np.mean(poly, axis=0).astype(int)) for poly in self.polygons]
        self.document_path = relation_path(self.dataset_root, self.split, path.stem)
        self.document, self.issues = load_document(self.document_path, path.name, self.objects)
        self.selected_source = None
        self.history.clear()
        self.dirty = False
        self.message = ""
        self.last_click = None
        self.last_candidates = ()

    def editable(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def push_history(self) -> None:
        self.history.append(deepcopy(self.document))
        if len(self.history) > 30:
            self.history.pop(0)

    def save(self) -> None:
        if not self.dirty or not self.editable():
            return
        save_document(self.document_path, self.document, self.objects)
        self.dirty = False

    def choose_object(self, x: int, y: int) -> int | None:
        candidates = []
        for index, polygon in enumerate(self.polygons):
            if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
                center = self.centers[index]
                distance = (center[0] - x) ** 2 + (center[1] - y) ** 2
                candidates.append((distance, index))
        if not candidates:
            return None
        ordered = tuple(index for _, index in sorted(candidates))
        repeated = self.last_click is not None and abs(self.last_click[0] - x) <= 8 and abs(self.last_click[1] - y) <= 8
        if repeated and ordered == self.last_candidates:
            self.candidate_index = (self.candidate_index + 1) % len(ordered)
        else:
            self.candidate_index = 0
        self.last_click = (x, y)
        self.last_candidates = ordered
        return ordered[self.candidate_index]

    def apply_pair(self, target: int) -> None:
        if self.selected_source is None:
            self.selected_source = target
            self.message = f"Selected #{target} as FIRST grasp"
            return
        source = self.selected_source
        self.selected_source = None
        if source == target:
            self.message = "Source and target must be different"
            return
        pairs = relation_pairs(self.document)
        pair = (source, target)
        self.push_history()
        if self.mode == "ADD":
            if pair in pairs:
                self.message = f"#{source} -> #{target} already exists"
                self.history.pop()
                return
            if (target, source) in pairs:
                self.message = f"Remove reverse relation #{target} -> #{source} first"
                self.history.pop()
                return
            trial = pairs + [pair]
            try:
                compute_layers(len(self.objects), trial)
            except ValueError:
                self.message = "Rejected: this arrow would create a cycle"
                self.history.pop()
                return
            self.document.setdefault("relations", []).append({
                "source": source,
                "target": target,
                "type": RELATION_TYPE,
            })
            self.message = f"Added #{source} -> #{target}"
        else:
            before = len(self.document.get("relations", []))
            self.document["relations"] = [
                edge for edge in self.document.get("relations", [])
                if (int(edge["source"]), int(edge["target"])) != pair
            ]
            if len(self.document["relations"]) == before:
                self.message = f"No relation #{source} -> #{target}"
                self.history.pop()
                return
            self.message = f"Removed #{source} -> #{target}"
        self.dirty = True
        self.save()

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        image_y = y - HEADER_HEIGHT
        if image_y < 0 or not self.editable():
            return
        chosen = self.choose_object(x, image_y)
        if chosen is None:
            self.selected_source = None
            self.message = "Selection cleared"
            return
        self.apply_pair(chosen)

    def reset_stale(self) -> None:
        if self.editable():
            self.message = "Reset is only available for stale/invalid annotations"
            return
        self.document = new_document(self.image_paths[self.image_index].name, self.objects)
        self.issues = []
        self.history.clear()
        self.selected_source = None
        self.dirty = True
        self.save()
        self.message = "Stale relations reset; review this image again"

    def mark_reviewed(self) -> None:
        if not self.editable():
            return
        self.push_history()
        self.document["reviewed"] = True
        self.dirty = True
        self.save()
        self.message = "Reviewed and saved"

    def undo(self) -> None:
        if not self.history:
            self.message = "Nothing to undo"
            return
        self.document = self.history.pop()
        self.selected_source = None
        self.dirty = True
        self.save()
        self.message = "Undid last edit"

    def navigate(self, delta: int) -> None:
        self.save()
        self.image_index = max(0, min(len(self.image_paths) - 1, self.image_index + delta))
        self.load_current()

    def next_unreviewed(self) -> None:
        self.save()
        start = self.image_index
        for offset in range(1, len(self.image_paths) + 1):
            index = (start + offset) % len(self.image_paths)
            image_path = self.image_paths[index]
            label_path = self.dataset_root / self.split / "labels" / f"{image_path.stem}.txt"
            objects = read_yolo_objects(label_path)
            document_path = relation_path(self.dataset_root, self.split, image_path.stem)
            document, issues = load_document(document_path, image_path.name, objects)
            if issues or not document.get("reviewed", False):
                self.image_index = index
                self.load_current()
                return
        self.message = "All images in this split are reviewed"

    def switch_split(self) -> None:
        self.save()
        self.split_index = (self.split_index + 1) % len(self.splits)
        self.image_index = 0
        self.refresh_images()
        self.load_current()

    def draw(self) -> np.ndarray:
        assert self.raw_image is not None
        canvas = self.raw_image.copy()
        relations = relation_pairs(self.document)

        for source, target in relations:
            if source < len(self.centers) and target < len(self.centers):
                cv2.arrowedLine(canvas, self.centers[source], self.centers[target], (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.08)

        for index, (item, polygon, center) in enumerate(zip(self.objects, self.polygons, self.centers)):
            class_id = int(item["class_id"])
            color = CLASS_COLORS[class_id] if 0 <= class_id < len(CLASS_COLORS) else (200, 200, 200)
            points = polygon.astype(np.int32).reshape((-1, 1, 2))
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)
            thickness = 4 if index == self.selected_source else 2
            border = (255, 255, 255) if index == self.selected_source else color
            cv2.polylines(canvas, [points], True, border, thickness, cv2.LINE_AA)
            cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA)
            name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"Class{class_id}"
            text = f"#{index} {name}"
            tx, ty = int(polygon[0][0]), max(18, int(polygon[0][1]) - 7)
            cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        header = np.full((HEADER_HEIGHT, canvas.shape[1], 3), (32, 32, 32), dtype=np.uint8)
        path = self.image_paths[self.image_index]
        reviewed = bool(self.document.get("reviewed", False))
        state = "REVIEWED" if reviewed else "UNREVIEWED"
        state_color = (80, 230, 80) if reviewed else (0, 190, 255)
        cv2.putText(header, f"[{self.split.upper()}] {self.image_index + 1}/{len(self.image_paths)}  {path.name}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(header, f"{state} | {self.mode} | Left click: FIRST then LATER | R:mode M:review A/D:nav N:next-unreviewed Z:undo T:split Q:quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.43, state_color, 1, cv2.LINE_AA)

        if self.issues:
            detail = self.issues[0].message[:135]
            cv2.putText(header, f"ERROR: {detail} | K: reset stale annotation", (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 255), 1, cv2.LINE_AA)
        else:
            try:
                layers = compute_layers(len(self.objects), relations)
                layer_text = "  ".join(f"L{i + 1}:" + ",".join(f"#{obj}" for obj in layer) for i, layer in enumerate(layers))
            except ValueError as exc:
                layer_text = str(exc)
            info = self.message or layer_text
            cv2.putText(header, info[:150], (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
        return np.vstack([header, canvas])

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        print("\n=== 夾取順序標註工具 ===")
        print("左鍵依序點『先夾』物件、再點『後夾』物件，即建立箭頭。")
        print("R：切換新增/移除模式　M：確認本圖已完整審核　Z：復原")
        print("A/D/Space：前後圖片　N：下一張未審核　T：切換資料集　Q/Esc：離開")
        print("重疊框難以選取時，在同一位置連續點擊可輪換候選物件。")

        while True:
            cv2.imshow(self.window_name, self.draw())
            raw_key = cv2.waitKey(25)
            if raw_key == -1:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            key = raw_key & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self.save()
                break
            if key in (ord("r"), ord("R")):
                self.mode = "REMOVE" if self.mode == "ADD" else "ADD"
                self.selected_source = None
                self.message = f"Mode: {self.mode}"
            elif key in (ord("m"), ord("M")):
                self.mark_reviewed()
            elif key in (ord("z"), ord("Z")):
                self.undo()
            elif key in (ord("k"), ord("K")):
                self.reset_stale()
            elif key in (ord("a"), ord("A"), 81):
                self.navigate(-1)
            elif key in (ord("d"), ord("D"), 13, 32, 83):
                self.navigate(1)
            elif key in (ord("n"), ord("N")):
                self.next_unreviewed()
            elif key in (ord("t"), ord("T")):
                self.switch_split()
            elif key in (ord("s"), ord("S")):
                self.save()
                self.message = "Saved"
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate grasp precedence on existing YOLO OBB objects")
    parser.add_argument("--dataset-root", default=None, help="Dataset root; default is <project>/robotic-garbage-obb")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else project_root / "robotic-garbage-obb"
    GraspOrderAnnotator(dataset_root, args.split).run()


if __name__ == "__main__":
    main()
