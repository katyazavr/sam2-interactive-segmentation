
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
OUTPUT = RESULTS / "runtime_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

METHODS = {
    "Point Prompt": "point_center",
    "Box Prompt": "box_gt",
}

summary_rows = []
latency_data = {}

for label, condition in METHODS.items():
    df = pd.read_csv(RESULTS / f"{condition}_test.csv")

    with open(RESULTS / f"{condition}_test_run_info.json") as f:
        info = json.load(f)

    assert not df[["image_id", "annotation_id"]].duplicated().any()
    assert len(df) == info["objects"]
    assert df["image_id"].nunique() == info["images"]

    # Each image is encoded once.
    # Avoid counting the encoder time once per object.
    encoder = df.groupby("image_id")["encoder_ms"].first()
    prompt = df["prompt_ms"]

    latency_data[label] = {
        "encoder": encoder,
        "prompt": prompt,
    }

    summary_rows.append({
        "method": label,
        "images": len(encoder),
        "objects": len(prompt),
        "total_seconds": info["total_seconds"],
        "device": info["device"],
        "precision": info["precision"],
        "encoder_mean_ms": encoder.mean(),
        "encoder_median_ms": encoder.median(),
        "encoder_std_ms": encoder.std(),
        "prompt_mean_ms": prompt.mean(),
        "prompt_median_ms": prompt.median(),
        "prompt_std_ms": prompt.std(),
        "prompt_p95_ms": prompt.quantile(0.95),
        "prompt_p99_ms": prompt.quantile(0.99),
        "prompt_max_ms": prompt.max(),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT / "runtime_summary.csv", index=False)

print("\n=== RUNTIME ANALYSIS ===")
print(summary.round(3).to_string(index=False))

# Figure 1: Mean encoder latency
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(
    summary["method"],
    summary["encoder_mean_ms"],
)
ax.set_ylabel("Latency (ms/image)")
ax.set_title("Mean Image Encoder Latency")
ax.set_ylim(0, 750)
fig.tight_layout()
fig.savefig(OUTPUT / "encoder_latency.png", dpi=200)
plt.close(fig)

# Figure 2: Mean and median prompt latency
fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(summary))
width = 0.35

ax.bar(
    x - width / 2,
    summary["prompt_mean_ms"],
    width,
    label="Mean",
)
ax.bar(
    x + width / 2,
    summary["prompt_median_ms"],
    width,
    label="Median",
)

ax.set_xticks(x)
ax.set_xticklabels(summary["method"])
ax.set_ylabel("Latency (ms/object)")
ax.set_title("Prompt Prediction Latency")
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT / "prompt_latency.png", dpi=200)
plt.close(fig)

# Figure 3: Distribution of prompt latency
fig, ax = plt.subplots(figsize=(8, 4))

for label, values in latency_data.items():
    ax.hist(
        values["prompt"],
        bins=np.linspace(0, 50, 51),
        alpha=0.5,
        label=label,
    )

ax.set_xlabel("Prompt latency (ms/object)")
ax.set_ylabel("Number of predictions")
ax.set_title("Prompt Latency Distribution (0-50 ms)")
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT / "prompt_distribution.png", dpi=200)
plt.close(fig)

# Figure 4: Total runtime
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(
    summary["method"],
    summary["total_seconds"] / 60,
)
ax.set_ylabel("Total runtime (minutes)")
ax.set_title("Full Test Evaluation Runtime")

fig.tight_layout()
fig.savefig(OUTPUT / "total_runtime.png", dpi=200)
plt.close(fig)

print(f"\nSaved results to: {OUTPUT}")
