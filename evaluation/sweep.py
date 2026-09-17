"""Sweep the thin-wall min_thickness_mm threshold on the dev set and plot
precision/recall against it. Dev set only, per the project's honesty rule:
thresholds are tuned on dev, then the default config is run once on test.

Usage:
    python evaluation/sweep.py --data-dir data/dev --out docs/images/thin_wall_sweep.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from partcheck.checks.thin_wall import ThinWallCheck
from partcheck.loader import load_part

THRESHOLDS = [round(0.5 + 0.1 * i, 1) for i in range(16)]  # 0.5 .. 2.0


def sweep(data_dir: Path, units: str = "mm") -> list[dict]:
    with open(data_dir / "labels.csv") as f:
        rows = list(csv.DictReader(f))

    parts_dir = data_dir / "parts"
    meshes = {row["part_id"]: load_part(str(parts_dir / f"{row['part_id']}.stl"), units=units) for row in rows}
    truth = {row["part_id"]: int(row["thin_wall"]) for row in rows}

    results = []
    for threshold in THRESHOLDS:
        config = {"min_thickness_mm": threshold, "ray_offset_mm": 0.01}
        tp = fp = fn = tn = 0
        for part_id, mesh in meshes.items():
            predicted = len(ThinWallCheck().run(mesh, config)) > 0
            actual = bool(truth[part_id])
            if actual and predicted:
                tp += 1
            elif not actual and predicted:
                fp += 1
            elif actual and not predicted:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        results.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall})

    return results


def plot(results: list[dict], out_path: Path) -> None:
    thresholds = [r["threshold"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, precision, marker="o", label="Precision")
    ax.plot(thresholds, recall, marker="s", label="Recall")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="Default (1.0 mm)")
    ax.set_xlabel("min_thickness_mm threshold")
    ax.set_ylabel("Score")
    ax.set_title("Thin-wall check: precision/recall vs. threshold (dev set)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/images/thin_wall_sweep.png"))
    args = parser.parse_args()

    results = sweep(args.data_dir)
    print(f"{'threshold':>10s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} {'precision':>10s} {'recall':>10s}")
    for r in results:
        print(f"{r['threshold']:10.1f} {r['tp']:4d} {r['fp']:4d} {r['fn']:4d} {r['tn']:4d} {r['precision']:10.2f} {r['recall']:10.2f}")

    plot(results, args.out)


if __name__ == "__main__":
    main()
