import unittest
import numpy as np
from evaluate_baseline import iou_dice


class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        m = np.array([[1, 0], [0, 1]])
        self.assertEqual(iou_dice(m, m), (1.0, 1.0))

    def test_disjoint(self):
        self.assertEqual(iou_dice([[1, 0]], [[0, 1]]), (0.0, 0.0))

    def test_partial(self):
        iou, dice = iou_dice([[1, 1], [0, 0]], [[1, 0], [1, 0]])
        self.assertAlmostEqual(iou, 1 / 3)
        self.assertAlmostEqual(dice, 1 / 2)

    def test_empty(self):
        self.assertEqual(iou_dice([[0, 0]], [[0, 0]]), (1.0, 1.0))

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            iou_dice([[1, 0]], [[1], [0]])


if __name__ == '__main__':
    unittest.main()
