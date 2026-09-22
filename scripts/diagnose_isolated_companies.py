#!/usr/bin/env python
"""輸送計画から脱落する孤立企業を診断する(ステージ6.5 §C)。

§C.1: コスト行列の分布、脱落企業の最近傍距離、filerNameと推測される理由。
§C.2: unbalanced OT(``ot.sinkhorn_unbalanced``)で脱落社数がどう変わるか。
§C.3: 対数変換(``sign(x) * log1p(|x|)``)でコスト行列の分布がどう変わるか。

どの対策を採用するかはこの段階では決めない(数値を見て判断する)。

使い方:
    uv run python scripts/diagnose_isolated_companies.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.subspace import build_reduced_basis
from fsgen.models.transport import cost_matrix, solve_ot_verbose, solve_unbalanced_ot

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_K = 13
_DEFAULT_EPSILON = 0.05
_TAUS = (0.1, 1.0, 10.0)
_PERCENTILES = (0, 10, 25, 50, 75, 90, 95, 99, 100)


def load_dataset(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    agg = pd.read_csv(aggregated_csv, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"] != "True")]
    labels_df = pd.read_csv(labels_csv, dtype={"docID": str})
    labels_df = labels_df[labels_df["label"] != ""]
    merged = agg.merge(labels_df[["docID", "filerName", "label"]], on="docID", how="inner")
    merged = merged[merged["total_assets"].notna() & (merged["total_assets"] != 0)]
    x = merged[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = merged["total_assets"].to_numpy(dtype=float)
    return merged, x, assets


def _percentile_summary(values: np.ndarray) -> str:
    ps = np.percentile(values, _PERCENTILES)
    return ", ".join(f"{p}%={v:.3g}" for p, v in zip(_PERCENTILES, ps, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)

    merged, x, assets = load_dataset(args.aggregated_csv, args.labels_csv, names)
    labels = merged["label"].astype(int).to_numpy()
    doc_ids = merged["docID"].to_numpy()
    filer_names = merged["filerName"].to_numpy()

    x_tilde = x / assets[:, None]
    b = build_reduced_basis(c, x_tilde, k=_K)
    z = x_tilde @ b
    idx0 = np.where(labels == 0)[0]
    idx2 = np.where(labels == 2)[0]
    z0, z2 = z[idx0], z[idx2]

    # --- C.1: コスト行列の分布 ---
    print("=== C.1 コスト行列の分布(正規化前、z空間の二乗ユークリッド距離) ===")
    cost = cost_matrix(z2, z0)
    print(f"全体: {_percentile_summary(cost.ravel())}")
    print(f"中央値: {np.median(cost):.4g}(正規化はこれで割る)")
    print(f"最大/中央値 比: {cost.max() / np.median(cost):.1f}")

    nn_dist = cost.min(axis=1)  # 各class2企業の最近傍コスト
    print(f"\n各社の最近傍コスト(未正規化)の分布: {_percentile_summary(nn_dist)}")

    result = solve_ot_verbose(z2, z0, _DEFAULT_EPSILON)
    zero_mask = result.plan.sum(axis=1) < 1e-12
    dropped_idx = np.where(zero_mask)[0]
    kept_idx = np.where(~zero_mask)[0]
    print(f"\nε={_DEFAULT_EPSILON} で脱落した社: {len(dropped_idx)}/{len(idx2)}")
    print(f"脱落社の最近傍コスト: {_percentile_summary(nn_dist[dropped_idx])}")
    print(f"非脱落社の最近傍コスト: {_percentile_summary(nn_dist[kept_idx])}")

    basis_sides = [bb.side for bb in spec.basis]
    net_assets_col_idx = [i for i, side in enumerate(basis_sides) if side == "net_assets"]

    print("\n脱落企業一覧(filerName, 最近傍コスト, 純資産/総資産比, 総資産):")
    for i in dropped_idx:
        row_x = x[idx2[i]]
        net_assets_ratio = row_x[net_assets_col_idx].sum() / assets[idx2[i]]
        print(
            f"  {doc_ids[idx2[i]]} {filer_names[idx2[i]]}: "
            f"最近傍コスト={nn_dist[i]:.4g}, 純資産/総資産={net_assets_ratio:.3f}, "
            f"総資産={assets[idx2[i]]:,.0f}円"
        )

    # --- C.2: unbalanced OT ---
    print("\n=== C.2 unbalanced OT(ot.sinkhorn_unbalanced)での脱落社数 ===")
    for tau in _TAUS:
        plan_u = solve_unbalanced_ot(z2, z0, _DEFAULT_EPSILON, tau)
        row_sums = plan_u.sum(axis=1)
        n_zero = int(np.sum(row_sums < 1e-12))
        print(
            f"tau={tau}: 質量ゼロの始点={n_zero}/{len(idx2)}, "
            f"行質量の分布: {_percentile_summary(row_sums)}"
        )

    # --- C.3: 対数変換版 ---
    print("\n=== C.3 対数変換版(sign(x)*log1p(|x|))でのコスト行列 ===")
    x_log = np.sign(x_tilde) * np.log1p(np.abs(x_tilde))
    # 対数変換後は「Cが定義する恒等式」が非線形になり build_reduced_basis は
    # そのままでは使えないが、コスト行列の分布への影響を見るだけなら
    # 通常のPCA(非中心化SVD)で十分。
    _u, _s, vt_log = np.linalg.svd(x_log, full_matrices=False)
    z_log = x_log @ vt_log[:_K].T
    z0_log, z2_log = z_log[idx0], z_log[idx2]
    cost_log = cost_matrix(z2_log, z0_log)
    print(f"対数変換後の全体分布: {_percentile_summary(cost_log.ravel())}")
    print(f"対数変換後 最大/中央値 比: {cost_log.max() / np.median(cost_log):.1f}")
    print(f"(参考)対数変換前 最大/中央値 比: {cost.max() / np.median(cost):.1f}")

    result_log = solve_ot_verbose(z2_log, z0_log, _DEFAULT_EPSILON)
    n_zero_log = int(np.sum(result_log.plan.sum(axis=1) < 1e-12))
    print(f"対数変換版でε={_DEFAULT_EPSILON}の脱落社数: {n_zero_log}/{len(idx2)}")


if __name__ == "__main__":
    main()
