"""Comparison tables and figures for Point Prompt vs Box Prompt.

Run after evaluate_baseline.py was run for both conditions
(we need point_center_evaluation_objects.csv and box_gt_evaluation_objects.csv).

How to run (from the main project folder):
    python make_comparison.py --split test

Result in results/evaluation_<split>/comparison/:
    table_thresholds.csv     share of objects with IoU >= 0.5 / 0.75 / 0.9
    table_size.csv           mean IoU for small / medium / large objects
    table_category.csv       categories with the biggest and smallest gain of the box
    table_paired.csv         how big the wins are, share of almost wrong masks (IoU < 0.3)
    table_area_ratio.csv     predicted area / GT area (is the mask too small or too big)
    tables.md                all tables in one file (easy to copy)
    tables.tex               the same tables for Overleaf
    fig_iou_threshold.png/.pdf   share of objects with IoU >= t
    fig_iou_by_size.png/.pdf     mean IoU by object size
    fig_delta_iou.png/.pdf       histogram of IoU(box) - IoU(point) for each object
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POINT, BOX = "point_center", "box_gt"
# colors: blue = point, orange = box (checked for color blindness)
C_POINT, C_BOX = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SIZE_ORDER = ["small", "medium", "large"]


def load_pairs(eval_dir: Path) -> pd.DataFrame:
    """Read both conditions and put them in one table, one row per object."""
    p = pd.read_csv(eval_dir / f"{POINT}_evaluation_objects.csv")
    b = pd.read_csv(eval_dir / f"{BOX}_evaluation_objects.csv")
    # the comparison is fair only if both have the same objects
    if set(p.annotation_id) != set(b.annotation_id) or len(p) != len(b):
        raise ValueError("Point and box results have different objects")
    keep = ["annotation_id", "image_id", "category_name", "area", "iou", "dice", "pred_area"]
    m = p[keep].merge(b[["annotation_id", "iou", "dice", "pred_area"]], on="annotation_id",
                      suffixes=("_point", "_box"))
    m["d_iou"] = m.iou_box - m.iou_point
    # COCO size groups by mask area: small < 32x32, medium < 96x96, large >= 96x96
    m["size"] = pd.cut(m.area, [0, 32 ** 2, 96 ** 2, np.inf], right=False, labels=SIZE_ORDER)
    return m


# tables
def table_thresholds(m):
    rows = []
    for t in (0.5, 0.75, 0.9):
        p, b = (m.iou_point >= t).mean() * 100, (m.iou_box >= t).mean() * 100
        rows.append({"IoU threshold": f">= {t}", "Point (%)": p, "Box (%)": b, "Difference (pp)": b - p})
    return pd.DataFrame(rows)


def table_size(m):
    rows = []
    for s in SIZE_ORDER:
        g = m[m["size"] == s]
        rows.append({"Size": s, "Objects": len(g), "Point mean IoU": g.iou_point.mean(),
                     "Box mean IoU": g.iou_box.mean(), "Difference": g.d_iou.mean(),
                     "Box better (%)": (g.d_iou > 1e-9).mean() * 100})
    return pd.DataFrame(rows)


def table_paired(m):
    """How big the wins are, and how often the mask is almost fully wrong (IoU < 0.3)."""
    d = m.d_iou
    rows = [("Box better by more than 0.1", (d > 0.1).mean() * 100),
            ("Point better by more than 0.1", (d < -0.1).mean() * 100),
            ("Box better by more than 0.3", (d > 0.3).mean() * 100),
            ("Point better by more than 0.3", (d < -0.3).mean() * 100),
            ("Difference at most 0.05", (d.abs() <= 0.05).mean() * 100),
            ("Point IoU below 0.3", (m.iou_point < 0.3).mean() * 100),
            ("Box IoU below 0.3", (m.iou_box < 0.3).mean() * 100)]
    return pd.DataFrame(rows, columns=["Statistic", "Objects (%)"])


def table_area_ratio(m):
    """Median of predicted area / GT area. Below 1: mask too small, above 1: mask too big."""
    rows = []
    for s in SIZE_ORDER:
        g = m[m["size"] == s]
        rows.append({"Size": s, "Point pred/GT area": (g.pred_area_point / g.area).median(),
                     "Box pred/GT area": (g.pred_area_box / g.area).median()})
    return pd.DataFrame(rows)


def table_category(m, min_n, k):
    g = m.groupby("category_name").agg(Objects=("d_iou", "size"), point=("iou_point", "mean"),
                                       box=("iou_box", "mean"), diff=("d_iou", "mean"))
    g = g[g.Objects >= min_n].sort_values("diff", ascending=False)
    # if there are few categories, do not show the same category in both groups
    top, low = g.head(k).assign(Group="largest gain"), g.iloc[k:].tail(k).assign(Group="smallest gain")
    out = pd.concat([top, low]).reset_index()
    out = out.rename(columns={"category_name": "Category", "point": "Point mean IoU",
                              "box": "Box mean IoU", "diff": "Difference"})
    return out[["Group", "Category", "Objects", "Point mean IoU", "Box mean IoU", "Difference"]]


def to_markdown(df, title):
    cols = list(df.columns)
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v, c) for c, v in zip(cols, r)) + " |")
    return "\n".join(lines) + "\n"


def to_latex(df, caption):
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", r"\small",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(cols).replace("%", r"\%") + r" \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(fmt(v, c).replace(">=", r"$\geq$") for c, v in zip(cols, r)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def fmt(v, col=""):
    # percents with 1 decimal, IoU with 3 decimals
    if isinstance(v, (float, np.floating)):
        return f"{v:.1f}" if ("%" in col or "pp" in col) else f"{v:.3f}"
    return str(v)


# figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def fig_threshold(m, out):
    t = np.linspace(0, 1, 101)
    p = [(m.iou_point >= x).mean() * 100 for x in t]
    b = [(m.iou_box >= x).mean() * 100 for x in t]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    style(ax)
    ax.plot(t, b, color=C_BOX, linewidth=2, label="Box prompt")
    ax.plot(t, p, color=C_POINT, linewidth=2, label="Point prompt")
    ax.set_xlabel("IoU threshold t", fontsize=8, color=INK)
    ax.set_ylabel("Objects with IoU ≥ t (%)", fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 102)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    save(fig, out, "fig_iou_threshold")


def fig_size(m, out):
    ts = table_size(m)
    x = np.arange(len(ts))
    w = 0.36
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    style(ax)
    # small gap between the two bars of one group
    ax.bar(x - w / 2 - 0.01, ts["Point mean IoU"], w, color=C_POINT, label="Point prompt")
    ax.bar(x + w / 2 + 0.01, ts["Box mean IoU"], w, color=C_BOX, label="Box prompt")
    for i, r in ts.iterrows():
        ax.text(i + w / 2 + 0.01, r["Box mean IoU"] + 0.015, f"{r['Difference']:+.2f}",
                ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [f"{s}\n(n={n})" for s, n in zip(ts.Size, ts.Objects)])
    ax.set_ylabel("Mean IoU", fontsize=8, color=INK)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    save(fig, out, "fig_iou_by_size")


def fig_delta(m, out):
    bins = np.linspace(-1, 1, 41)
    counts, edges = np.histogram(m.d_iou, bins=bins)
    share = counts / len(m) * 100
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [C_BOX if c > 0 else C_POINT for c in centers]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    style(ax)
    ax.bar(centers, share, width=(edges[1] - edges[0]) * 0.9, color=colors)
    ax.axvline(0, color=INK2, linewidth=0.8)
    box_better = (m.d_iou > 1e-9).mean() * 100
    point_better = (m.d_iou < -1e-9).mean() * 100
    ax.text(0.98, 0.95, f"box better: {box_better:.1f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=INK)
    ax.text(0.02, 0.95, f"point better: {point_better:.1f}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color=INK)
    ax.set_xlabel("IoU(box) − IoU(point) for one object", fontsize=8, color=INK)
    ax.set_ylabel("Objects (%)", fontsize=8, color=INK)
    ax.set_xlim(-1, 1)
    save(fig, out, "fig_delta_iou")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["development", "validation", "test"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--min-category", type=int, default=30,
                    help="use only categories with at least this many objects")
    ap.add_argument("--top", type=int, default=5, help="how many categories to show on each side")
    args = ap.parse_args()

    eval_dir = Path(args.results_dir) / f"evaluation_{args.split}"
    out = eval_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    m = load_pairs(eval_dir)

    t_thr, t_size = table_thresholds(m), table_size(m)
    t_cat = table_category(m, args.min_category, args.top)
    t_thr.to_csv(out / "table_thresholds.csv", index=False)
    t_size.to_csv(out / "table_size.csv", index=False)
    t_cat.to_csv(out / "table_category.csv", index=False)
    t_pair, t_area = table_paired(m), table_area_ratio(m)
    t_pair.to_csv(out / "table_paired.csv", index=False)
    t_area.to_csv(out / "table_area_ratio.csv", index=False)

    n = f"{len(m)} objects, {m.image_id.nunique()} images, {args.split} split"
    md = [f"# Comparison tables ({n})", "",
          to_markdown(t_thr, "Share of objects above an IoU threshold"),
          to_markdown(t_size, "Mean IoU by object size"),
          to_markdown(t_cat, f"Categories with the largest and smallest box gain (>= {args.min_category} objects)"),
          to_markdown(t_pair, "Size of the paired differences and almost wrong masks"),
          to_markdown(t_area, "Median predicted area / GT area")]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    tex = [to_latex(t_thr, "Share of objects above an IoU threshold."),
           to_latex(t_size, "Mean IoU by object size (COCO area groups)."),
           to_latex(t_cat, "Categories with the largest and smallest box gain.")]
    (out / "tables.tex").write_text("\n".join(tex), encoding="utf-8")

    fig_threshold(m, out)
    fig_size(m, out)
    fig_delta(m, out)

    print(n)
    print("\n".join(md))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
