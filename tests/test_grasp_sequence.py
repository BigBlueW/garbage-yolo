import unittest

import numpy as np
import torch

from grasp_sequence import SceneObject, complete_sequence, make_acyclic_edges, uncertain_incoming_risks
from order_pair_data import transitive_relations
from order_pair_model import pair_candidate_kind, swap_pair_batch, swap_pair_targets


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


if __name__ == "__main__":
    unittest.main()
