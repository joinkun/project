#!/usr/bin/env python
"""PCAで観測された class 間の低次元性・部分空間の違いが、単なる標本サイズ効果
でないかを null 分布で検証する(ステージ5.5 §B)。

class 2(GC注記あり、68社)は class 0(非金融、約3949社)よりずっと小さい
標本なので、「class 2 が低次元に見える」「主角度が大きい」という
ステージ5 §D の観測が、単にサンプルサイズが小さいことの統計的な
アーティファクトである可能性がある。これを検証するため、class 0 から
class 2 と同じサイズの部分集合を無作為抽出して同じ統計量を計算し、
その分布(null分布)の中に実測値がどう位置するかを見る。

使い方:
    uv run python scripts/pca_null_test.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    orthonormal_basis,
    state_vector_names,
)

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")

_N_ITER = 200
_TOP_K = 5
_SEED = 0


def load_normalized_vectors(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> tuple[np.ndarray, list[int | None]]:
    """{非金融・parse_status=ok} の全社について、総資産で正規化した x と label を返す。"""
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


def pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """中心化PCA。(累積寄与率, 主成分方向(行が単位ベクトル、分散降順))を返す。"""
    mean = x.mean(axis=0)
    centered = x - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    n = x.shape[0]
    explained_variance = (s**2) / max(n - 1, 1)
    total = explained_variance.sum()
    ratio = explained_variance / total if total > 0 else explained_variance
    cumulative: np.ndarray = np.cumsum(ratio)
    return cumulative, vt


def n_components_for(cumulative: np.ndarray, threshold: float) -> int:
    idx = np.searchsorted(cumulative, threshold)
    return int(idx) + 1 if idx < len(cumulative) else len(cumulative)


def principal_angles_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = a @ b.T
    s = np.linalg.svd(m, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    result: np.ndarray = np.degrees(np.arccos(s))
    return result


def percentile_rank(value: float, distribution: np.ndarray) -> float:
    return float(np.mean(distribution <= value) * 100)


def null_dist_ncomponents(
    pool: np.ndarray, q: np.ndarray, size: int, n_iter: int, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    """pool から size 社を無作為抽出し、null(C^T) 内PCAで95%/99%必要主成分数を
    n_iter 回記録する。"""
    n95s: list[int] = []
    n99s: list[int] = []
    for _ in range(n_iter):
        idx = rng.choice(len(pool), size=size, replace=False)
        z = pool[idx] @ q
        cumulative, _vt = pca(z)
        n95s.append(n_components_for(cumulative, 0.95))
        n99s.append(n_components_for(cumulative, 0.99))
    return n95s, n99s


def _stats_line(dist: list[int] | np.ndarray) -> str:
    arr = np.array(dist)
    return (
        f"中央値={np.median(arr):.1f}, 5%tile={np.percentile(arr, 5):.1f}, "
        f"95%tile={np.percentile(arr, 95):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    q = orthonormal_basis(c)

    x, labels = load_normalized_vectors(args.aggregated_csv, args.labels_csv, names)
    x0 = x[[i for i, lab in enumerate(labels) if lab == 0]]
    x1 = x[[i for i, lab in enumerate(labels) if lab == 1]]
    x2 = x[[i for i, lab in enumerate(labels) if lab == 2]]
    print(f"class0: {len(x0)}社, class1: {len(x1)}社, class2: {len(x2)}社")

    rng = np.random.default_rng(_SEED)

    # --- B.1: 主成分数のnull分布 ---
    print(f"\n=== B.1 主成分数のnull分布({_N_ITER}回、null(C^T)内でPCA) ===")

    z2 = x2 @ q
    cum2, _vt2 = pca(z2)
    n95_2, n99_2 = n_components_for(cum2, 0.95), n_components_for(cum2, 0.99)
    n95_null, n99_null = null_dist_ncomponents(x0, q, len(x2), _N_ITER, rng)
    print(f"[class0から{len(x2)}社を抽出したnull分布(class2のサイズ)]")
    print(f"  95%必要主成分数: {_stats_line(n95_null)}")
    print(f"  99%必要主成分数: {_stats_line(n99_null)}")
    pct95_2 = percentile_rank(n95_2, np.array(n95_null))
    pct99_2 = percentile_rank(n99_2, np.array(n99_null))
    print(
        f"  class2実測: 95%={n95_2}(パーセンタイル{pct95_2:.1f}%), "
        f"99%={n99_2}(パーセンタイル{pct99_2:.1f}%)"
    )

    z1 = x1 @ q
    cum1, _vt1 = pca(z1)
    n95_1, n99_1 = n_components_for(cum1, 0.95), n_components_for(cum1, 0.99)
    n95_null_137, n99_null_137 = null_dist_ncomponents(x0, q, len(x1), _N_ITER, rng)
    print(f"\n[class0から{len(x1)}社を抽出したnull分布(class1のサイズ)]")
    print(f"  95%必要主成分数: {_stats_line(n95_null_137)}")
    print(f"  99%必要主成分数: {_stats_line(n99_null_137)}")
    print(
        f"  class1実測: 95%={n95_1}"
        f"(パーセンタイル{percentile_rank(n95_1, np.array(n95_null_137)):.1f}%), "
        f"99%={n99_1}(パーセンタイル{percentile_rank(n99_1, np.array(n99_null_137)):.1f}%)"
    )

    # --- B.2: 主角度のnull分布 ---
    print(f"\n=== B.2 主角度のnull分布({_N_ITER}回、class0内の2組{len(x2)}社ずつ) ===")
    angle_null = np.zeros((_N_ITER, _TOP_K))
    for i in range(_N_ITER):
        perm = rng.permutation(len(x0))
        idx_a = perm[: len(x2)]
        idx_b = perm[len(x2) : 2 * len(x2)]
        _cum_a, vt_a = pca(x0[idx_a])
        _cum_b, vt_b = pca(x0[idx_b])
        angle_null[i] = principal_angles_degrees(vt_a[:_TOP_K], vt_b[:_TOP_K])

    _cum0, vt0 = pca(x0)
    _cum2_full, vt2_full = pca(x2)
    actual_angles = principal_angles_degrees(vt0[:_TOP_K], vt2_full[:_TOP_K])

    for k in range(_TOP_K):
        dist = angle_null[:, k]
        pct = percentile_rank(actual_angles[k], dist)
        print(
            f"  第{k + 1}主角度: 実測(class0 vs class2)={actual_angles[k]:.2f}度, "
            f"null分布[中央値={np.median(dist):.2f}, 5%tile={np.percentile(dist, 5):.2f}, "
            f"95%tile={np.percentile(dist, 95):.2f}], "
            f"実測のパーセンタイル={pct:.1f}%"
        )


if __name__ == "__main__":
    main()
