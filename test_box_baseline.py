"""Small tests for the COCO-to-SAM2 box conversion."""
import unittest
import numpy as np
from run_box_baseline import coco_box_xyxy


class TestBoxConversion(unittest.TestCase):
    def test_normal_box(self):
        np.testing.assert_array_equal(coco_box_xyxy([10, 20, 30, 40], (100, 200, 3)), [10, 20, 40, 60])

    def test_clip_to_image(self):
        np.testing.assert_array_equal(coco_box_xyxy([-5, 10, 30, 100], (50, 60, 3)), [0, 10, 25, 50])

    def test_invalid_box(self):
        with self.assertRaises(ValueError):
            coco_box_xyxy([10, 20, 0, 40], (100, 200, 3))

    def test_outside_image(self):
        with self.assertRaises(ValueError):
            coco_box_xyxy([300, 20, 10, 40], (100, 200, 3))


if __name__ == "__main__":
    unittest.main()
