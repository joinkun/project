#!/usr/bin/env python
"""build_reduced_basis の低次元表現を実データで検証する(ステージ5.5 §C)。

k = 5, 8, 10, 13, 15, 20 それぞれについて、以下の再構成誤差
‖x - B B^T x‖ / ‖x‖ を計算する:

- 基底 B を全社(非金融)の x で学習し、全社で再構成誤差を測る
- 基底 B を class 0(非金融)だけの x で学習し、class 2 だけで
  再構成誤差を測る(class 0 の主成分で class 2 を表現できるか)

使い方:
    uv run python scripts/verify_reduced_basis.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.subspace import build_reduced_basis, reconstruction_error

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_KS = (5, 8, 10, 13, 15, 20)


def load_normalized_vectors(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> tuple[np.ndarray, list[int | None]]:
    label_by_doc: dict[str, int] = {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] != "":
                label_by_doc[row["docID"]] = int(row["label"])

    rows: list[list[float]] = []
    labels: list[int | None] = []
    with aggregated_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["parse_status"] != "ok" or row.get("is_financial") == "True":
                continue
            assets_str = row.get("total_assets", "")
            if assets_str in ("", "0"):
                continue
            assets = float(assets_str)
            values = [float(row[n]) if row[n] != "" else 0.0 for n in names]
            rows.append([v / assets for v in values])
            labels.append(label_by_doc.get(row["docID"]))
    return np.array(rows), labels


def _stats(values: np.ndarray) -> str:
    v = values.tolist()
    if not v:
        return "(データなし)"
    q1, _, q3 = statistics.quantiles(v, n=4)
    return f"中央値 {statistics.median(v):.4f}, 四分位 [{q1:.4f}, {q3:.4f}], 最大 {max(v):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)

    x, labels = load_normalized_vectors(args.aggregated_csv, args.labels_csv, names)
    x0 = x[[i for i, lab in enumerate(labels) if lab == 0]]
    x2 = x[[i for i, lab in enumerate(labels) if lab == 2]]
    print(f"全社: {len(x)}社, class0: {len(x0)}社, class2: {len(x2)}社\n")

    print("=== 基底を全社で学習し、全社で再構成誤差を測る ===")
    for k in _KS:
        b_all = build_reduced_basis(c, x, k)
        err = reconstruction_error(b_all, x)
        print(f"k={k:>2}: {_stats(err)}")

    print("\n=== 基底を class0 だけで学習した場合 ===")
    for k in _KS:
        b0 = build_reduced_basis(c, x0, k)
        err_on_0 = reconstruction_error(b0, x0)
        err_on_2 = reconstruction_error(b0, x2)
        print(f"k={k:>2}: class0で再構成 {_stats(err_on_0)}")
        print(f"        class2で再構成 {_stats(err_on_2)}")


if __name__ == "__main__":
    main()
