import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from convert_instaorder import convert as convert_instaorder
from convert_regrad import convert as convert_regrad


class ExternalConverterTests(unittest.TestCase):
    def test_regrad_child_is_grasped_before_parent(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory) / "train"
            (root / "RGBImages").mkdir(parents=True)
            (root / "SegmentationImages").mkdir()
            cv2.imwrite(str(root / "RGBImages" / "00001_1.jpg"), np.zeros((40, 40, 3), np.uint8))
            labels = {"00001": {"1": [
                {"obj_id": 1, "MinAreaRect": [[2, 2], [20, 2], [20, 20], [2, 20]]},
                {"obj_id": 2, "MinAreaRect": [[10, 10], [28, 10], [28, 28], [10, 28]]},
            ]}}
            (root / "objects.json").write_text(json.dumps(labels), encoding="utf-8")
            (root / "mrts.json").write_text(json.dumps({"00001": {"box-1": ["cup-2"], "cup-2": []}}), encoding="utf-8")
            output = Path(directory) / "regrad.jsonl"
            stats = convert_regrad(root, output, negative_ratio=0, seed=1)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["positive_pairs"], 1)
            self.assertEqual(sorted(row["target"] for row in rows), [0, 1])
            original = next(row for row in rows if not row["key"].endswith("/mirror"))
            self.assertEqual(original["target"], 1)  # object 2 (B/child) before object 1 (A/parent)

    def test_instaorder_occluder_becomes_first(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            images = root / "train2017"; images.mkdir()
            cv2.imwrite(str(images / "one.jpg"), np.zeros((40, 40, 3), np.uint8))
            coco = {
                "images": [{"id": 7, "file_name": "one.jpg", "width": 40, "height": 40}],
                "annotations": [
                    {"id": 10, "image_id": 7, "bbox": [2, 2, 18, 18], "segmentation": []},
                    {"id": 11, "image_id": 7, "bbox": [10, 10, 18, 18], "segmentation": []},
                ],
            }
            insta = {"annotations": [{"image_id": 7, "instance_ids": [10, 11],
                                       "occlusion": [{"order": "0 < 1"}]}]}
            coco_path = root / "instances.json"; coco_path.write_text(json.dumps(coco), encoding="utf-8")
            insta_path = root / "insta.json"; insta_path.write_text(json.dumps(insta), encoding="utf-8")
            output = root / "manifest.jsonl"
            stats = convert_instaorder(insta_path, coco_path, images, output, 0, 1)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["positive_pairs"], 1)
            original = next(row for row in rows if not row["key"].endswith("/mirror"))
            self.assertEqual(original["target"], 1)  # B occludes A, so B first


if __name__ == "__main__":
    unittest.main()
