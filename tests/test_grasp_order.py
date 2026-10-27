import tempfile
import unittest
from pathlib import Path

from grasp_order import compute_layers, load_document, new_document, read_yolo_objects, save_document, validate_document


class GraspOrderTests(unittest.TestCase):
    def setUp(self):
        self.objects = [
            {"id": 0, "class_id": 0, "format": "obb", "coordinates": [0.1] * 8},
            {"id": 1, "class_id": 1, "format": "obb", "coordinates": [0.2] * 8},
            {"id": 2, "class_id": 2, "format": "obb", "coordinates": [0.3] * 8},
        ]

    def test_parallel_layers(self):
        self.assertEqual(compute_layers(3, [(0, 2), (1, 2)]), [[0, 1], [2]])

    def test_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            compute_layers(3, [(0, 1), (1, 2), (2, 0)])

    def test_validation_rejects_reverse_edges(self):
        document = new_document("sample.png", self.objects)
        document["relations"] = [
            {"source": 0, "target": 1, "type": "before"},
            {"source": 1, "target": 0, "type": "before"},
        ]
        codes = {issue.code for issue in validate_document(document, self.objects)}
        self.assertIn("reverse_edge", codes)

    def test_stale_obb_snapshot_is_detected(self):
        document = new_document("sample.png", self.objects)
        changed = [dict(item) for item in self.objects]
        changed[0] = dict(changed[0], class_id=3)
        codes = {issue.code for issue in validate_document(document, changed)}
        self.assertIn("stale_objects", codes)

    def test_round_trip_and_yolo_parse(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            label = root / "sample.txt"
            label.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n", encoding="utf-8")
            objects = read_yolo_objects(label)
            document = new_document("sample.png", objects)
            document["reviewed"] = True
            document["relations"] = []
            output = root / "sample.json"
            save_document(output, document, objects)
            loaded, issues = load_document(output, "sample.png", objects)
            self.assertFalse(issues)
            self.assertTrue(loaded["reviewed"])


if __name__ == "__main__":
    unittest.main()
