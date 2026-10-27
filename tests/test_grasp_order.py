import json
import tempfile
import unittest
from pathlib import Path

from grasp_order import compute_layers, load_relations, read_yolo_objects


class GraspOrderTests(unittest.TestCase):
    def test_parallel_layers(self):
        self.assertEqual(compute_layers(3, [(0, 2), (1, 2)]), [[0, 1], [2]])

    def test_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            compute_layers(3, [(0, 1), (1, 2), (2, 0)])

    def test_load_relations(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            label = root / "sample.txt"
            label.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n" * 3, encoding="utf-8")
            objects = read_yolo_objects(label)
            order = root / "sample.json"
            self.assertEqual(load_relations(order, len(objects)), [])
            order.write_text(json.dumps({"relations": [[0, 2], [1, 2]]}), encoding="utf-8")
            self.assertEqual(load_relations(order, len(objects)), [(0, 2), (1, 2)])
            order.write_text(json.dumps({"relations": [[0, 1], [1, 0]]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_relations(order, len(objects))
            order.write_text(json.dumps({"relations": [[0, 5]]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_relations(order, len(objects))


if __name__ == "__main__":
    unittest.main()
