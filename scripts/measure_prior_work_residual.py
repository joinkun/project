#!/usr/bin/env python
"""先行研究方式(資産側のみ操作)で貸借がどれだけ崩れるかを実測する(ステージ5 §C)。

これは Lashkevich & Zelenkov (2026) の手法の再実装ではない。彼らは資産側の
3項目(棚卸資産・その他流動資産・その他固定資産)のみを介入対象とし、
貸借一致への言及がない。**同じ設計方針(資産側だけを動かす)を取ると
貸借がどれだけ壊れるか**を、単純な反実仮想生成(class 0 の中央値へ向けて
線形補間)で最小限に実証する。

先行研究の3項目への対応(canonical 基礎科目名):
    棚卸資産       -> inventories
    その他流動資産  -> jppfs_cor_OtherCA
    その他固定資産  -> __OTHER__jppfs_cor_NoncurrentAssets
        (「その他固定資産」に直接対応する単独の基礎科目が canonical に
        無いため、NoncurrentAssets 配下の残余科目で代用する。最も近い
        概念だが完全な対応ではないことに注意。)

比較として、資産側の全基礎科目を動かすケースと、提案手法(射影 P を
適用)では残差が常に厳密に0になることも同じ図に示す。

使い方:
    uv run python scripts/measure_prior_work_residual.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

# 日本語グリフ(DejaVu Sansには無い)を描画するため、macOSに同梱されている
# Hiragino Sans をフォールバックの先頭に挿入する。
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", *plt.rcParams["font.sans-serif"]]

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    build_projection,
    load_canonical_spec,
    state_vector_names,
)

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_OUT_PNG = Path("data/interim/figures/prior_work_residual.png")

_ALPHAS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
_PRIOR_WORK_TARGETS: tuple[str, ...] = (
    "inventories",
    "jppfs_cor_OtherCA",
    "__OTHER__jppfs_cor_NoncurrentAssets",
)


def load_labels(labels_csv: Path) -> dict[str, int]:
    label_by_doc: dict[str, int] = {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "":
                continue
            label_by_doc[row["docID"]] = int(row["label"])
    return label_by_doc


def load_class_vectors(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> dict[int, np.ndarray]:
    """{label: x行列(n, D)} を、非金融・parse_status=ok の会社について返す。"""
    label_by_doc = load_labels(labels_csv)
    by_label: dict[int, list[list[float]]] = {0: [], 1: [], 2: []}
    with aggregated_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["parse_status"] != "ok" or row.get("is_financial") == "True":
                continue
            label = label_by_doc.get(row["docID"])
            if label is None:
                continue
            values = [float(row[n]) if row[n] != "" else 0.0 for n in names]
            by_label[label].append(values)
    return {label: np.array(rows) for label, rows in by_label.items()}


def counterfactual(
    x: np.ndarray, target_idx: list[int], median: np.ndarray, alpha: float
) -> np.ndarray:
    """target_idx の成分だけを、median へ向けて alpha だけ線形補間する。"""
    x2 = x.copy()
    x2[target_idx] = x[target_idx] + alpha * (median[target_idx] - x[target_idx])
    return x2


def relative_residuals(
    class2: np.ndarray,
    target_idx: list[int],
    median: np.ndarray,
    c: np.ndarray,
    assets_idx: list[int],
) -> dict[float, list[float]]:
    out: dict[float, list[float]] = {}
    for alpha in _ALPHAS:
        rel: list[float] = []
        for x in class2:
            x2 = counterfactual(x, target_idx, median, alpha)
            assets = float(np.sum(x2[assets_idx]))
            residual = float(np.abs((c.T @ x2)[0]))
            rel.append(residual / assets if assets else 0.0)
        out[alpha] = rel
    return out


def proposed_method_residuals(
    class2: np.ndarray,
    target_idx: list[int],
    median: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    assets_idx: list[int],
) -> dict[float, list[float]]:
    """反実仮想を作った後に射影 P を適用した場合の残差(常に厳密に0のはず)。"""
    out: dict[float, list[float]] = {}
    for alpha in _ALPHAS:
        rel: list[float] = []
        for x in class2:
            x2 = counterfactual(x, target_idx, median, alpha)
            x2p = p @ x2
            assets = float(np.sum(x2p[assets_idx]))
            residual = float(np.abs((c.T @ x2p)[0]))
            rel.append(residual / assets if assets else 0.0)
        out[alpha] = rel
    return out


def normalize_by_assets(x: np.ndarray, assets_idx: list[int]) -> np.ndarray:
    """各行を Σ(資産側科目)(=Assets相当)で割る。x̃ = x / Assets(§A)。"""
    assets = x[:, assets_idx].sum(axis=1, keepdims=True)
    assets = np.where(assets == 0, 1.0, assets)
    result: np.ndarray = x / assets
    return result


def normalized_residuals(
    class2_norm: np.ndarray, target_idx: list[int], median_norm: np.ndarray, c: np.ndarray
) -> dict[float, list[float]]:
    """正規化済み x̃ に対する ‖C^T x̃'‖(既に規模で割ってあるので、さらには割らない)。"""
    out: dict[float, list[float]] = {}
    for alpha in _ALPHAS:
        rel: list[float] = []
        for x in class2_norm:
            x2 = counterfactual(x, target_idx, median_norm, alpha)
            residual = float(np.abs((c.T @ x2)[0]))
            rel.append(residual)
        out[alpha] = rel
    return out


def normalized_proposed_method_residuals(
    class2_norm: np.ndarray,
    target_idx: list[int],
    median_norm: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
) -> dict[float, list[float]]:
    out: dict[float, list[float]] = {}
    for alpha in _ALPHAS:
        rel: list[float] = []
        for x in class2_norm:
            x2 = counterfactual(x, target_idx, median_norm, alpha)
            x2p = p @ x2
            residual = float(np.abs((c.T @ x2p)[0]))
            rel.append(residual)
        out[alpha] = rel
    return out


def _stats(values: list[float]) -> str:
    if not values:
        return "(データなし)"
    q1, _, q3 = statistics.quantiles(values, n=4)
    return (
        f"中央値 {statistics.median(values):.4e}, 四分位 [{q1:.4e}, {q3:.4e}], "
        f"最大 {max(values):.4e}"
    )


def _plot_panel(
    ax: Axes,
    results_3: dict[float, list[float]],
    results_all: dict[float, list[float]],
    results_proposed: dict[float, list[float]],
    ylabel: str,
    title: str,
) -> None:
    width = 0.6
    group_gap = 2.4
    positions_3 = [i * group_gap for i in range(len(_ALPHAS))]
    positions_all = [p + width for p in positions_3]
    positions_proposed = [p + 2 * width for p in positions_3]

    def _box(
        positions: list[float], data: dict[float, list[float]], color: str, label: str
    ) -> None:
        values = [data[a] for a in _ALPHAS]
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            boxprops={"edgecolor": color},
            whiskerprops={"color": color},
            capprops={"color": color},
            medianprops={"color": color},
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.6)
        ax.plot([], [], color=color, linewidth=8, alpha=0.6, label=label)

    _box(positions_3, results_3, "tab:orange", "先行研究方式(資産側3項目のみ)")
    _box(positions_all, results_all, "tab:red", "資産側全科目を動かす")
    _box(positions_proposed, results_proposed, "tab:blue", "提案手法(射影 P を適用)")

    ax.set_xticks(positions_all)
    ax.set_xticklabels([f"α={a}" for a in _ALPHAS])
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_xlabel("反実仮想の移動幅 α")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)


def plot(
    results_3: dict[float, list[float]],
    results_all: dict[float, list[float]],
    results_proposed: dict[float, list[float]],
    results_3_norm: dict[float, list[float]],
    results_all_norm: dict[float, list[float]],
    results_proposed_norm: dict[float, list[float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5.5))

    _plot_panel(
        ax_left,
        results_3,
        results_all,
        results_proposed,
        "相対残差 ‖C^T x'‖ / Assets",
        "左: 生の値(会社規模の影響を含む)",
    )
    _plot_panel(
        ax_right,
        results_3_norm,
        results_all_norm,
        results_proposed_norm,
        "残差 ‖C^T x̃'‖ (x̃ = x/Assets で正規化済み)",
        "右: 総資産で正規化(会社規模の影響を除去)",
    )
    fig.suptitle("資産側のみを動かす先行研究方式 vs 提案手法(射影)の貸借残差")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT_PNG)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    p = build_projection(c)
    assets_idx = [i for i, b in enumerate(spec.basis) if b.side == "assets"]

    vectors = load_class_vectors(args.aggregated_csv, args.labels_csv, names)
    class0, class2 = vectors[0], vectors[2]
    print(f"class0(非金融): {len(class0)}社, class2(GC注記・非金融): {len(class2)}社")
    if len(class0) == 0 or len(class2) == 0:
        print("class0 または class2 のデータが不足しています。")
        return

    median = np.median(class0, axis=0)

    target_idx_3 = [names.index(n) for n in _PRIOR_WORK_TARGETS]

    results_3 = relative_residuals(class2, target_idx_3, median, c, assets_idx)
    results_all = relative_residuals(class2, assets_idx, median, c, assets_idx)
    results_proposed = proposed_method_residuals(class2, target_idx_3, median, c, p, assets_idx)

    print("\n=== [生の値] 相対残差 ‖C^T x'‖ / Assets (α別) ===")
    for alpha in _ALPHAS:
        print(f"α={alpha}")
        print(f"  先行研究方式(3項目):     {_stats(results_3[alpha])}")
        print(f"  資産側全科目:             {_stats(results_all[alpha])}")
        print(f"  提案手法(射影):           {_stats(results_proposed[alpha])}")

    # §A(ステージ5.5): 総資産で正規化した版。既に正規化済みなので割らない。
    class0_norm = normalize_by_assets(class0, assets_idx)
    class2_norm = normalize_by_assets(class2, assets_idx)
    median_norm = np.median(class0_norm, axis=0)

    results_3_norm = normalized_residuals(class2_norm, target_idx_3, median_norm, c)
    results_all_norm = normalized_residuals(class2_norm, assets_idx, median_norm, c)
    results_proposed_norm = normalized_proposed_method_residuals(
        class2_norm, target_idx_3, median_norm, c, p
    )

    print("\n=== [正規化: x̃=x/Assets] 残差 ‖C^T x̃'‖ (α別) ===")
    for alpha in _ALPHAS:
        print(f"α={alpha}")
        print(f"  先行研究方式(3項目):     {_stats(results_3_norm[alpha])}")
        print(f"  資産側全科目:             {_stats(results_all_norm[alpha])}")
        print(f"  提案手法(射影):           {_stats(results_proposed_norm[alpha])}")

    plot(
        results_3,
        results_all,
        results_proposed,
        results_3_norm,
        results_all_norm,
        results_proposed_norm,
        args.out,
    )
    print(f"\n図を出力しました: {args.out}")


if __name__ == "__main__":
    main()
