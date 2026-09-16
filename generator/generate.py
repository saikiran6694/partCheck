"""Generate a labelled dataset of synthetic parts.

Usage:
    python generator/generate.py --seed 1 --n 30 --out data/dev
    python generator/generate.py --seed 2 --n 20 --out data/test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from families import FAMILIES, export_shape

LABEL_NAMES = ["thin_wall", "sharp_corner", "overhang"]


def generate_dataset(n: int, seed: int, out_dir: Path) -> None:
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    family_names = list(FAMILIES)
    rows = []

    for i in range(n):
        family = family_names[i % len(family_names)]
        part_id = f"{family}_{i:03d}"
        rng = random.Random(f"{seed}-{part_id}")

        shape, params, labels = FAMILIES[family](rng)

        stl_path = parts_dir / f"{part_id}.stl"
        export_shape(shape, str(stl_path))

        row = {
            "part_id": part_id,
            "family": family,
            "seed": seed,
            "params_json": json.dumps(params),
            **{name: int(labels[name]) for name in LABEL_NAMES},
        }
        rows.append(row)
        print(f"[{i + 1}/{n}] {part_id}  labels={labels}")

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["part_id", "family", "seed", "params_json", *LABEL_NAMES]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} parts and {labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    generate_dataset(n=args.n, seed=args.seed, out_dir=args.out)


if __name__ == "__main__":
    main()
