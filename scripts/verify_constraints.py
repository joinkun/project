#!/usr/bin/env python
"""制約行列 C・射影 P を実データで検証する(ステージ5 §B.5)。

`data/interim/values/aggregated_core.csv` (inspect_values.py の出力)から
状態ベクトル x を作り、非金融会社について以下を確認する:

- rank(C) が期待値と一致するか
- P が冪等(P@P≈P)・対称(P≈P.T)か
- ‖C^T x‖ / Assets がほぼ0か(会計恒等式をどれだけ満たしているか)
- P @ x ≈ x か(実データは既に制約部分空間にあるので射影で動かないはず)

include_subtotals の True/False 両方で検証する。

使い方:
    uv run python scripts/verify_constraints.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    build_projection,
    compute_rank,
    load_canonical_spec,
    state_vector_names,
)

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")


def load_state_vectors(
    aggregated_csv: Path, names: tuple[str, ...]
) -> tuple[list[str], np.ndarray]:
    """非金融・parse_status=ok の会社について、名前の順で並んだ x を返す。"""
    doc_ids: list[str] = []
    rows: list[list[float]] = []
    with aggregated_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["parse_status"] != "ok":
                continue
            if row.get("is_financial") == "True":
                continue
            try:
                values = [float(row[n]) if row[n] != "" else 0.0 for n in names]
            except (KeyError, ValueError):
                continue
            doc_ids.append(row["docID"])
            rows.append(values)
    return doc_ids, np.array(rows)


def verify(canonical_path: Path, aggregated_csv: Path, include_subtotals: bool) -> None:
    spec = load_canonical_spec(canonical_path)
    names = state_vector_names(spec, include_subtotals=include_subtotals)
    c = build_constraint_matrix(spec, include_subtotals=include_subtotals)
    p = build_projection(c)
    rank = compute_rank(c)
    dim = c.shape[0]

    print(f"--- include_subtotals={include_subtotals} ---")
    print(f"D={dim}, K={c.shape[1]}, rank(C)={rank}, 実効次元={dim - rank}")
    print(f"P idempotent (max|P@P - P|): {np.max(np.abs(p @ p - p)):.2e}")
    print(f"P symmetric (max|P - P.T|): {np.max(np.abs(p - p.T)):.2e}")

    doc_ids, x = load_state_vectors(aggregated_csv, names)
    if x.size == 0:
        print("実データが読み込めませんでした。aggregated_core.csv を確認してください。")
        return

    assets_idx = names.index("jppfs_cor_Assets") if include_subtotals else None
    if assets_idx is not None:
        assets = x[:, assets_idx]
    else:
        # include_subtotals=False では Assets が状態に無いので、資産側基礎科目の和を使う。
        assets_side_idx = [
            i for i, b in enumerate(spec.basis) if b.side == "assets"
        ]
        assets = x[:, assets_side_idx].sum(axis=1)

    residual_norm = np.linalg.norm(x @ c, axis=1)  # 各社の ‖C^T x‖ (x@C = (C^T x)^T の各成分)
    safe_assets = np.where(assets == 0, 1.0, assets)
    rel_residual = residual_norm / np.abs(safe_assets)

    px = x @ p.T  # 各行が x_i の射影(P対称なのでx@Pでも同じ)
    move_norm = np.linalg.norm(px - x, axis=1)
    rel_move = move_norm / np.abs(safe_assets)

    n = len(doc_ids)
    print(f"検証対象(非金融・parse_status=ok): {n}社")
    print(f"‖C^T x‖/Assets: {_stats(rel_residual)}")
    print(f"‖Px - x‖/Assets: {_stats(rel_move)}")
    n_nonzero_exact = int(np.sum(residual_norm > 1e-6))
    print(f"‖C^T x‖ が厳密に0でない社: {n_nonzero_exact}/{n}")
    print()


def _stats(values: np.ndarray) -> str:
    v = values.tolist()
    if not v:
        return "(データなし)"
    q1, _, q3 = statistics.quantiles(v, n=4)
    return f"中央値 {statistics.median(v):.2e}, 四分位 [{q1:.2e}, {q3:.2e}], 最大 {max(v):.2e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    args = parser.parse_args()

    verify(args.canonical, args.aggregated_csv, include_subtotals=False)
    verify(args.canonical, args.aggregated_csv, include_subtotals=True)


if __name__ == "__main__":
    main()
