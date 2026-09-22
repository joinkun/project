#!/usr/bin/env python
"""実データの分散が実効次元 D-1 のうち実質何次元に乗っているかを PCA で推定する
(ステージ5 §D)。

全社(非金融)の状態ベクトル x を各社の総資産で正規化し(会社規模の影響を
除くため)、以下の3種類のPCAを行う:

1. 生の x(D次元)に対する PCA。会計恒等式(C^T x = 0)により、理論上は
   ちょうど rank(C) 個の主成分の固有値が(数値誤差の範囲で)0になる
   はずで、これは §B で確認した rank(C) の数値的な裏付けにもなる。
2. 制約部分空間 null(C^T) への正規直交基底 Q(D, D-1)で座標変換した
   z = Q^T x (D-1次元)に対する PCA。x は既に C^T x = 0 を満たすので
   Q Q^T x = x であり、情報の損失なくこの (D-1)次元表現に変換できる。
   実効的にはこちらが本質(1.の「自明にゼロの1次元」を除いた表現)。
3. class 0 / class 1 / class 2 それぞれで別々に PCA し、上位主成分が
   張る部分空間どうしの主角度(principal angles)を求める。

出力:
    data/interim/figures/pca_cumulative_variance.png
    data/interim/figures/pca_null_space_cumulative_variance.png

使い方:
    uv run python scripts/estimate_intrinsic_dimension.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Hiragino Sans", *plt.rcParams["font.sans-serif"]]

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    orthonormal_basis,
    state_vector_names,
)

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_FIG_DIR = Path("data/interim/figures")
_TOP_K_FOR_ANGLES = 5


@dataclass(frozen=True)
class PcaResult:
    explained_variance_ratio: np.ndarray  # (min(n,D),)
    cumulative: np.ndarray
    components: np.ndarray  # (min(n,D), D) 主成分方向(行が単位ベクトル、分散降順)


def pca(x: np.ndarray) -> PcaResult:
    mean = x.mean(axis=0)
    centered = x - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    n = x.shape[0]
    explained_variance = (s**2) / max(n - 1, 1)
    total = explained_variance.sum()
    ratio = explained_variance / total if total > 0 else explained_variance
    cumulative = np.cumsum(ratio)
    return PcaResult(explained_variance_ratio=ratio, cumulative=cumulative, components=vt)


def n_components_for(cumulative: np.ndarray, threshold: float) -> int:
    idx = np.searchsorted(cumulative, threshold)
    return int(idx) + 1 if idx < len(cumulative) else len(cumulative)


def principal_angles_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: (k, D) 正規直交な行ベクトル集合(各部分空間の基底)。主角度(度)を返す。"""
    m = a @ b.T
    s = np.linalg.svd(m, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


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


def plot_cumulative(result: PcaResult, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    x_axis = np.arange(1, len(result.cumulative) + 1)
    ax.plot(x_axis, result.cumulative, marker="o", markersize=3)
    ax.axhline(0.95, color="tab:orange", linestyle="--", label="95%")
    ax.axhline(0.99, color="tab:red", linestyle="--", label="99%")
    n95 = n_components_for(result.cumulative, 0.95)
    n99 = n_components_for(result.cumulative, 0.99)
    ax.axvline(n95, color="tab:orange", linestyle=":", alpha=0.7)
    ax.axvline(n99, color="tab:red", linestyle=":", alpha=0.7)
    ax.set_xlabel("主成分数")
    ax.set_ylabel("累積寄与率")
    ax.set_title(f"{title}\n95%に必要な主成分数={n95}, 99%に必要な主成分数={n99}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    parser.add_argument("--fig-dir", type=Path, default=_DEFAULT_FIG_DIR)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    q = orthonormal_basis(c)  # (D, D-1)

    x, labels = load_normalized_vectors(args.aggregated_csv, args.labels_csv, names)
    print(f"対象社数(非金融・parse_status=ok): {x.shape[0]}社, D={x.shape[1]}")

    # 1. 生の x に対する PCA
    result_raw = pca(x)
    n95_raw = n_components_for(result_raw.cumulative, 0.95)
    n99_raw = n_components_for(result_raw.cumulative, 0.99)
    print("\n=== 1. 生の x(総資産で正規化)に対する PCA ===")
    print(f"95%に必要な主成分数: {n95_raw} / D={x.shape[1]}")
    print(f"99%に必要な主成分数: {n99_raw} / D={x.shape[1]}")
    print(f"下位5固有値の寄与率(理論上 rank(C)={c.shape[1]} 個がほぼ0のはず): ")
    print(f"  {result_raw.explained_variance_ratio[-5:]}")
    plot_cumulative(
        result_raw,
        "生の状態ベクトル(総資産正規化)の累積寄与率",
        args.fig_dir / "pca_cumulative_variance.png",
    )

    # 2. null(C^T) 座標系での PCA
    z = x @ q  # (n, D-1)
    result_null = pca(z)
    n95_null = n_components_for(result_null.cumulative, 0.95)
    n99_null = n_components_for(result_null.cumulative, 0.99)
    print("\n=== 2. 制約部分空間 null(C^T) 内(D-1次元)での PCA ===")
    print(f"95%に必要な主成分数: {n95_null} / (D-1)={z.shape[1]}")
    print(f"99%に必要な主成分数: {n99_null} / (D-1)={z.shape[1]}")
    plot_cumulative(
        result_null,
        "制約部分空間 null(C^T) 内での累積寄与率",
        args.fig_dir / "pca_null_space_cumulative_variance.png",
    )

    # 3. class別PCAと主角度
    print("\n=== 3. class別PCAと主成分部分空間の主角度 ===")
    by_class: dict[int, np.ndarray] = {}
    for label in (0, 1, 2):
        idx = [i for i, lab in enumerate(labels) if lab == label]
        if len(idx) < _TOP_K_FOR_ANGLES:
            print(f"class {label}: 社数不足({len(idx)}社)のためスキップ")
            continue
        x_label = x[idx]
        result = pca(x_label)
        n95 = n_components_for(result.cumulative, 0.95)
        print(f"class {label}: {len(idx)}社, 95%に必要な主成分数={n95}")
        by_class[label] = result.components[:_TOP_K_FOR_ANGLES]

    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        if a not in by_class or b not in by_class:
            continue
        angles = principal_angles_degrees(by_class[a], by_class[b])
        print(
            f"class {a} vs class {b} の主角度(上位{_TOP_K_FOR_ANGLES}主成分, 度): "
            f"{np.round(angles, 2)}"
        )

    print(f"\n図を出力しました: {args.fig_dir / 'pca_cumulative_variance.png'}")
    print(f"図を出力しました: {args.fig_dir / 'pca_null_space_cumulative_variance.png'}")


if __name__ == "__main__":
    main()
