import unittest

import pandas as pd

from compare_prompts import bootstrap_ci, check_same_objects, compare


def make(iou, dice, prompt_ms, image_ids=(1, 1, 2), ann_ids=(10, 11, 12)):
    return pd.DataFrame({
        "image_id": list(image_ids), "annotation_id": list(ann_ids),
        "iou": iou, "dice": dice,
        "encoder_ms": [500.0, 500.0, 600.0], "prompt_ms": prompt_ms,
    })


class TestCompare(unittest.TestCase):
    def test_paired_difference(self):
        point = make([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [10.0, 10.0, 10.0])
        box = make([0.7, 0.5, 0.3], [0.8, 0.6, 0.4], [12.0, 12.0, 12.0])
        table, paired = compare(point, box)
        self.assertAlmostEqual(paired["mean_delta_iou"], 0.0)
        self.assertAlmostEqual(paired["box_better_share"], 1 / 3)
        self.assertAlmostEqual(paired["equal_share"], 1 / 3)
        self.assertAlmostEqual(paired["point_better_share"], 1 / 3)
        row = table.set_index("Metric").loc["Encoder latency, mean (ms)"]
        self.assertAlmostEqual(row["Point Prompt"], 550.0)   # once per image, not per object

    def test_different_objects_rejected(self):
        point = make([0.5] * 3, [0.6] * 3, [10.0] * 3)
        box = make([0.5] * 3, [0.6] * 3, [10.0] * 3, ann_ids=(10, 11, 99))
        with self.assertRaises(ValueError):
            check_same_objects(point, box)

    def test_bootstrap_is_reproducible(self):
        self.assertEqual(bootstrap_ci([0.1, 0.2, 0.3]), bootstrap_ci([0.1, 0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()