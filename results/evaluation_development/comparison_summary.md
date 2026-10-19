## Comparison table

| Metric | Point Prompt | Box Prompt |
|---|---|---|
| Mean IoU | 0.5445 | 0.7497 |
| Median IoU | 0.6110 | 0.8032 |
| Mean Dice | 0.6470 | 0.8426 |
| Median Dice | 0.7585 | 0.8908 |
| Encoder latency (mean / median) | 615.73 ms / 605.88 ms | 569.21 ms / 565.88 ms |
| Prompt latency (mean / median) | 12.99 ms / 12.02 ms | 13.71 ms / 10.85 ms |

## Summary

On the same 345 objects from 50 development images, the box prompt improved segmentation quality over the one-point baseline: mean IoU rose from 0.5445 to 0.7497 (paired ΔIoU = +0.2051, 95% bootstrap CI [+0.1750, +0.2360]) and mean Dice from 0.6470 to 0.8426 (ΔDice = +0.1956). The box gave a higher IoU for 72% of the objects and a lower IoU for 28%, so the point prompt was still better on a noticeable share of objects, which the failure analysis should explain. Image-encoder latency does not depend on the prompt type; the observed 616 ms vs 569 ms most likely reflects run-to-run variation, because the two conditions were measured in separate runs. Prompt latency is comparable (median 12.02 ms for the point and 10.85 ms for the box); a 1 ms difference between single runs should not be read as a real speed difference. The mean box prompt latency (13.71 ms) is inflated by one slow call (691 ms, the first object of the first image), most likely a cold start, so the median is more representative. A box specifies the extent of the object, which probably removes the ambiguity of a single point (for example, a part of an object versus the whole object); this explanation has not been tested yet. Limitations: the box is the COCO ground-truth bounding box (an oracle prompt), not a detector output; the results are preliminary and come from 50 development images only, not from the test split; and both conditions were run once on a Mac (mps), so absolute latencies are indicative only.
