import unittest
from unittest.mock import patch

import numpy as np
import torch

from grasp_sequence import SceneObject, complete_sequence, make_acyclic_edges, uncertain_incoming_risks
from label_overlay import FIRST_PICK_COLOR, grasp_order_color
from order_pair_data import transitive_relations
from order_pair_model import (
    RelationDecision, pair_candidate_kind, predict_grasp_plan, swap_pair_batch, swap_pair_targets,
)


def obj(index, x, y):
    points = np.asarray([[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]], dtype=np.float32)
    return SceneObject(index, index % 4, points, 0.9)


class GraspSequenceTests(unittest.TestCase):
    def test_transitive_relations(self):
        self.assertEqual(transitive_relations(3, [(0, 1), (1, 2)]), {(0, 1), (1, 2), (0, 2)})

    def test_cycle_repair_keeps_strong_edges(self):
        accepted, rejected = make_acyclic_edges(3, [(0, 1, .9), (1, 2, .8), (2, 0, .7)])
        self.assertEqual(accepted, [(0, 1), (1, 2)])
        self.assertEqual(len(rejected), 1)

    def test_complete_sequence_respects_dependencies(self):
        objects = [obj(0, 20, 20), obj(1, 50, 50), obj(2, 80, 80), obj(3, 90, 20)]
        ranked, layers = complete_sequence(objects, [(2, 1), (1, 0)], (100, 100, 3))
        positions = {item.object_id: item.rank for item in ranked}
        self.assertLess(positions[2], positions[1])
        self.assertLess(positions[1], positions[0])
        self.assertEqual(layers, [[2, 3], [1], [0]])

    def test_candidate_keeps_small_visible_gap_but_rejects_far_pair(self):
        a = obj(0, 20, 20)
        near = obj(1, 31, 20)  # one-pixel polygon gap
        far = obj(2, 60, 20)
        self.assertEqual(pair_candidate_kind(a, near, (100, 100, 3), near_gap_ratio=.01), "near")
        self.assertIsNone(pair_candidate_kind(a, far, (100, 100, 3), near_gap_ratio=.01))

    def test_pair_swap_is_an_involution(self):
        visual = torch.arange(2 * 5 * 3 * 3, dtype=torch.float32).reshape(2, 5, 3, 3)
        geometry = torch.arange(2 * 25, dtype=torch.float32).reshape(2, 25)
        swapped_visual, swapped_geometry = swap_pair_batch(visual, geometry)
        restored_visual, restored_geometry = swap_pair_batch(swapped_visual, swapped_geometry)
        self.assertTrue(torch.equal(visual, restored_visual))
        self.assertTrue(torch.equal(geometry, restored_geometry))
        targets = torch.tensor([0, 1, 2])
        self.assertTrue(torch.equal(swap_pair_targets(swap_pair_targets(targets)), targets))

    def test_uncertain_direction_demotes_only_the_likely_lower_object(self):
        objects = [obj(0, 20, 20), obj(1, 50, 50), obj(2, 80, 80)]
        risks = uncertain_incoming_risks([
            (0, 1, 0, 0.51),  # 0 may be above 1
            (0, 2, None, 0.90),  # disagreement is not directional evidence
            (1, 2, 0, 0.40),  # below the risk floor
        ])
        self.assertEqual(risks, {1: 0.51})
        ranked, _ = complete_sequence(objects, [], (100, 100, 3), risks)
        self.assertNotEqual(ranked[0].object_id, 1)
        self.assertEqual(ranked[-1].object_id, 1)

    def test_grasp_plan_returns_unique_first_pick_inside_layer_one(self):
        boxes = [
            (0, obj(0, 20, 20).points, 0.50),
            (1, obj(1, 50, 50).points, 0.99),
            (2, obj(2, 80, 80).points, 0.90),
        ]

        def relation_for_pair(_model, _image, a, b, _size, _device, _threshold):
            if (a.id, b.id) == (0, 2):
                return RelationDecision("relation", "intersect", 0, 0.95, "test")
            return RelationDecision("no_relation", "intersect", 2, 0.95, "test")

        with patch("order_pair_model.predict_bidirectional_decision", side_effect=relation_for_pair):
            layers, ranks = predict_grasp_plan(
                None, np.zeros((100, 100, 3), dtype=np.uint8), boxes, 160, torch.device("cpu"),
            )

        self.assertEqual(layers, [1, 1, 2])
        self.assertEqual(ranks, [2, 1, 3])

    def test_first_pick_is_red_and_other_layers_have_stable_distinct_colors(self):
        self.assertEqual(grasp_order_color(1, True), FIRST_PICK_COLOR)
        self.assertNotEqual(grasp_order_color(1), FIRST_PICK_COLOR)
        self.assertNotEqual(grasp_order_color(1), grasp_order_color(2))
        self.assertEqual(grasp_order_color(2), grasp_order_color(2))
        self.assertNotIn(grasp_order_color(7), {grasp_order_color(i) for i in range(1, 7)})


if __name__ == "__main__":
    unittest.main()
