# Failure Analysis

Failure = IoU < 0.3 on the test split (6,941 objects). Point prompt: **1,443** failures (20.8%); box prompt: **189** (2.7%); both fail: 152. For 1,038 point failures the box reaches IoU ≥ 0.5; only 16 objects fail with the box but reach IoU ≥ 0.5 with the point.

Figure: `fig_failure_compact.png` (4 cases) / `fig_failure_cases.png` (6 cases). Green = ground truth, red = point (star), blue = box (yellow rectangle).

| Fig. label (4-case / 6-case) | Object (annotation id) | Area, px | Point IoU | Box IoU | Type |
|---|---|---:|---:|---:|---|
| a / a | person (475292) | 63,516 | 0.20 | 0.88 | point: part instead of whole |
| – / b | keyboard (1116028) | 43,753 | 0.01 | 0.88 | point: single key (mask 418 px) |
| b / c | carrot (1917299) | 20,106 | 0.58 | 0.25 | box: several objects merged |
| c / d | chair (373276) | 1,942 | 0.60 | 0.25 | box: occluded, fragmented GT |
| d / e | backpack (1169777) | 301 | 0.19 | 0.25 | both: person segmented instead |
| – / f | dining table (1616035) | 260,311 | 0.11 | 0.21 | both: annotation convention |

**Causes (hypotheses from the images and mask sizes, not tested experimentally):**
- *Point, part vs. whole:* 923 of 1,443 point failures have a mask smaller than half of the GT. One point does not say which level (arm / person) is meant. Large objects fail more often with the point (32.6% for area ≥ 96²) than small ones (18.5% for < 32²). Person is the top category (435).
- *Box, several objects in one box:* 142 of 189 box failures are over-segmentations (mask > 2× GT). Thin, rotated, occluded or crowded objects have loose boxes containing other objects. 130 of 189 (68.8%) are small objects; box failure rate 4.7% for small vs 2.2% for large.
- *Both, wrong object/level or annotation convention:* a small backpack gets replaced by the person; COCO dining-table masks seem to include dishes (58 point / 20 box failures in this category).
- *Score:* mean score of failures 0.743 (point) / 0.745 (box) vs 0.824 / 0.915 overall; corr(score, IoU) = 0.358 (point) / 0.746 (box). A low score warns about box failures; point failures often have a high score (0.87 in case a).

**Limitations:** COCO masks have label noise; box = oracle GT box; cases are hand-picked illustrations, not frequencies.
