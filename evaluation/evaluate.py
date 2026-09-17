"""Run the PartCheck pipeline over a generated dataset and report per-check metrics.

Usage:
    python evaluation/evaluate.py --data-dir data/dev --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from partcheck.checks import CHECKS
from partcheck.loader import load_part

LABEL_NAMES = ["thin_wall", "sharp_corner", "overhang"]


def load_labels(data_dir: Path) -> list[dict]:
    with open(data_dir / "labels.csv") as f:
        return list(csv.DictReader(f))


def run_pipeline(stl_path: Path, units: str, config: dict) -> tuple[dict[str, bool], float, int]:
    start = time.perf_counter()
    mesh = load_part(str(stl_path), units=units)
    predicted = {}
    for name, check_cls in CHECKS.items():
        findings = check_cls().run(mesh, config["checks"].get(name, {}))
        predicted[name] = len(findings) > 0
    elapsed = time.perf_counter() - start
    return predicted, elapsed, len(mesh.faces)


def evaluate(data_dir: Path, config: dict) -> tuple[list[dict], dict[str, dict]]:
    rows = load_labels(data_dir)
    units = config.get("units", "mm")
    parts_dir = data_dir / "parts"

    predictions: list[dict] = []
    for row in rows:
        stl_path = parts_dir / f"{row['part_id']}.stl"
        predicted, elapsed, face_count = run_pipeline(stl_path, units, config)
        record = {
            "part_id": row["part_id"],
            "family": row["family"],
            "runtime_s": round(elapsed, 4),
            "face_count": face_count,
        }
        for name in LABEL_NAMES:
            record[f"{name}_true"] = int(row[name])
            record[f"{name}_pred"] = int(predicted[name])
        predictions.append(record)

    metrics = {name: _confusion_metrics(predictions, name) for name in LABEL_NAMES}
    return predictions, metrics


def _confusion_metrics(predictions: list[dict], check: str) -> dict:
    tp = fp = fn = tn = 0
    for p in predictions:
        actual, pred = p[f"{check}_true"], p[f"{check}_pred"]
        if actual and pred:
            tp += 1
        elif not actual and pred:
            fp += 1
        elif actual and not pred:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def print_report(predictions: list[dict], metrics: dict[str, dict]) -> None:
    print(f"{'check':15s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} {'precision':>10s} {'recall':>10s} {'f1':>10s}")
    for name, m in metrics.items():
        print(
            f"{name:15s} {m['tp']:4d} {m['fp']:4d} {m['fn']:4d} {m['tn']:4d} "
            f"{m['precision']:10.2f} {m['recall']:10.2f} {m['f1']:10.2f}"
        )

    total_time = sum(p["runtime_s"] for p in predictions)
    avg_faces = sum(p["face_count"] for p in predictions) / len(predictions)
    print(f"\n{len(predictions)} parts, avg runtime {total_time / len(predictions):.3f}s/part, avg {avg_faces:.0f} faces")


def write_predictions_csv(predictions: list[dict], path: Path) -> None:
    fieldnames = list(predictions[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=None, help="Path to write predictions.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    predictions, metrics = evaluate(args.data_dir, config)
    print_report(predictions, metrics)

    out_path = args.out or (args.data_dir / "predictions.csv")
    write_predictions_csv(predictions, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
