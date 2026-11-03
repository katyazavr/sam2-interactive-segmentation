# Comparison tables (6941 objects, 991 images, test split)

### Share of objects above an IoU threshold

| IoU threshold | Point (%) | Box (%) | Difference (pp) |
|---|---|---|---|
| >= 0.5 | 65.5 | 89.7 | 24.2 |
| >= 0.75 | 44.1 | 64.6 | 20.5 |
| >= 0.9 | 14.4 | 22.8 | 8.4 |

### Mean IoU by object size

| Size | Objects | Point mean IoU | Box mean IoU | Difference | Box better (%) |
|---|---|---|---|---|---|
| small | 2782 | 0.592 | 0.666 | 0.074 | 54.6 |
| medium | 2370 | 0.651 | 0.812 | 0.161 | 72.8 |
| large | 1789 | 0.548 | 0.844 | 0.296 | 84.1 |

### Categories with the largest and smallest box gain (>= 30 objects)

| Group | Category | Objects | Point mean IoU | Box mean IoU | Difference |
|---|---|---|---|---|---|
| largest gain | train | 42 | 0.399 | 0.905 | 0.506 |
| largest gain | bus | 52 | 0.475 | 0.889 | 0.414 |
| largest gain | toilet | 36 | 0.555 | 0.880 | 0.325 |
| largest gain | keyboard | 38 | 0.511 | 0.832 | 0.322 |
| largest gain | couch | 45 | 0.489 | 0.806 | 0.317 |
| smallest gain | spoon | 53 | 0.649 | 0.688 | 0.039 |
| smallest gain | umbrella | 69 | 0.736 | 0.772 | 0.036 |
| smallest gain | tennis racket | 41 | 0.768 | 0.798 | 0.030 |
| smallest gain | zebra | 49 | 0.815 | 0.832 | 0.017 |
| smallest gain | baseball bat | 38 | 0.766 | 0.748 | -0.019 |

### Size of the paired differences and almost wrong masks

| Statistic | Objects (%) |
|---|---|
| Box better by more than 0.1 | 38.8 |
| Point better by more than 0.1 | 5.2 |
| Box better by more than 0.3 | 26.1 |
| Point better by more than 0.3 | 0.6 |
| Difference at most 0.05 | 44.0 |
| Point IoU below 0.3 | 20.8 |
| Box IoU below 0.3 | 2.7 |

### Median predicted area / GT area

| Size | Point pred/GT area | Box pred/GT area |
|---|---|---|
| small | 1.089 | 1.308 |
| medium | 0.904 | 1.020 |
| large | 0.709 | 0.978 |
