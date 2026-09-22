#!/usr/bin/env python
"""ステージ8 §A: ステージ7報告の事前確認(A.1〜A.5)。

A.1 sign の集計を科目ごとに列挙し直す(前回報告の集計ミスの確認)。
A.2 自己株式の方向制約が実際に効いているかを検証する。
A.3 射影後の総資産変化の分布(68社)。
A.4 目標に到達できていない企業(射影後の移動量 / 未射影目標までの距離)。
A.5 重みに床を入れた版(delta=1e-3)と白色化版の比較。

使い方:
    uv run python scripts/verify_stage7_followup.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.projection_qp import (
    apply_projection_to_paths,
    default_weights,
    load_direction_constraints,
    project_onto_feasible,
)
from fsgen.accounting.subspace import build_reduced_basis
from fsgen.errors import SolverNotConvergedError
from fsgen.models.transport import paths_to_state_vectors, single_target_paths, solve_ot

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_K = 13
_EPSILON = 0.05
_TS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_dataset(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[int]]:
    agg = pd.read_csv(aggregated_csv, dtype={"docID": str})
    is_financial = agg["is_financial"].astype(str)
    agg = agg[(agg["parse_status"] == "ok") & (is_financial != "True")]
    labels_df = pd.read_csv(labels_csv, dtype={"docID": str})
    labels_df = labels_df[pd.to_numeric(labels_df["label"], errors="coerce").notna()]
    merged = agg.merge(labels_df[["docID", "filerName", "label"]], on="docID", how="inner")
    merged = merged[merged["total_assets"].notna() & (merged["total_assets"] != 0)]
    x = merged[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = merged["total_assets"].to_numpy(dtype=float)
    labels = merged["label"].astype(int).tolist()
    return merged["docID"].tolist(), merged["filerName"].tolist(), x, assets, labels


def _solve_unwhitened(
    y: np.ndarray,
    spec: object,
    x_start: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    index: dict[str, int],
) -> np.ndarray:
    """A.5比較専用: 白色化せず、生の x に重み付き目的関数を直接課して解く
    (project_onto_feasible の白色化を外した版。診断専用でライブラリには入れない)。
    """
    dim = len(w)
    scale = max(float(np.max(np.abs(y))), float(np.max(np.abs(x_start))), 1.0)
    y_s = y / scale
    x = cp.Variable(dim)
    objective = cp.Minimize(cp.sum_squares(cp.multiply(np.sqrt(w), x - y_s)))
    constraints = [c.T @ x == 0]
    for b in spec.basis:  # type: ignore[attr-defined]
        i = index[b.name]
        if b.sign == "nonneg":
            constraints.append(x[i] >= 0)
        elif b.sign == "nonpos":
            constraints.append(x[i] <= 0)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, max_iter=50000, eps_abs=1e-8, eps_rel=1e-8)
    if x.value is None or prob.status != "optimal":
        raise SolverNotConvergedError(f"QPが最適解でない: status={prob.status}")  # §A.1
    return np.asarray(x.value, dtype=float) * scale


def section_a1(spec: object) -> None:
    print("=== A.1 sign の全科目一覧(core.yaml の記載順) ===")
    counts: dict[str, int] = {"nonneg": 0, "nonpos": 0, "free": 0}
    n_residual = 0
    for b in spec.basis:  # type: ignore[attr-defined]
        is_residual = b.name.startswith("__OTHER__")
        n_residual += int(is_residual)
        marker = " [残余]" if is_residual else ""
        counts[b.sign] += 1
        print(f"  {b.name:<50} sign={b.sign:<7}{marker}  {b.sign_reason}")
    print(f"合計: {sum(counts.values())} 科目(内訳 nonneg={counts['nonneg']}, "
          f"nonpos={counts['nonpos']}, free={counts['free']}, うち残余={n_residual})")
    non_free_residual = [
        b.name
        for b in spec.basis  # type: ignore[attr-defined]
        if b.name.startswith("__OTHER__") and b.sign != "free"
    ]
    print(f"free でない残余科目: {non_free_residual}(空なら修正不要)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    direction_constraints = load_direction_constraints(args.canonical)

    section_a1(spec)

    doc_ids, filer_names, x, assets, labels = load_dataset(
        args.aggregated_csv, args.labels_csv, names
    )
    x_tilde = x / assets[:, None]
    b = build_reduced_basis(c, x_tilde, k=_K)
    z = x_tilde @ b

    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    z0, z2 = z[idx0], z[idx2]
    assets2 = assets[idx2]
    x2_true = x[idx2]
    doc_ids2 = [doc_ids[i] for i in idx2]
    filer2 = [filer_names[i] for i in idx2]

    plan = solve_ot(z2, z0, _EPSILON)
    z_linear, _targets = single_target_paths(z2, z0, plan, _TS)
    x_unprojected = paths_to_state_vectors(z_linear, z2, x2_true, b, assets2)

    x_primary, _infos = apply_projection_to_paths(
        x_unprojected, x2_true, assets2, spec, mode="every_step",
        enforce_inactive=False, weighted=True, direction_constraints=direction_constraints,
    )

    # --- A.2: 方向制約の効果検証 ---
    print("\n=== A.2 自己株式の方向制約の効果 ===")
    net_assets_idx = [i for i, bb in enumerate(spec.basis) if bb.side == "net_assets"]
    treasury_idx = names.index("jppfs_cor_TreasuryStock")
    net_assets_t0 = x2_true[:, net_assets_idx].sum(axis=1)
    debt_overhang_mask = net_assets_t0 < 0
    print(f"債務超過(純資産t=0が負)企業数: {int(debt_overhang_mask.sum())}/{len(doc_ids2)}")

    treasury_t0 = x_primary[:, 0, treasury_idx]
    treasury_t1 = x_primary[:, -1, treasury_idx]
    # QPソルバのeps_abs/eps_rel由来の数値誤差(他の符号違反判定と同じ基準、総資産の0.01%)を
    # 実質的な制約違反と混同しないよう、絶対1円ではなく総資産比の許容誤差を使う。
    tol = assets2 * 1e-4
    increased_magnitude = np.abs(treasury_t1) > np.abs(treasury_t0) + tol

    n_violate_overhang = int(np.sum(increased_magnitude & debt_overhang_mask))
    n_increase_non_overhang = int(np.sum(increased_magnitude & ~debt_overhang_mask))
    n_overhang = int(debt_overhang_mask.sum())
    n_non_overhang = int((~debt_overhang_mask).sum())
    print(
        f"債務超過企業のうち |自己株式| が増加した社数: {n_violate_overhang}/{n_overhang}"
        "(0でなければ制約が効いていない)"
    )
    print(
        f"非債務超過企業のうち |自己株式| が増加した社数: "
        f"{n_increase_non_overhang}/{n_non_overhang}(制約対象外なので増えてよい)"
    )
    if n_violate_overhang > 0:
        for i in np.where(increased_magnitude & debt_overhang_mask)[0][:10]:
            print(f"  {doc_ids2[i]} {filer2[i]}: treasury t0={treasury_t0[i]:,.0f} -> "
                  f"t1={treasury_t1[i]:,.0f}, net_assets(0)={net_assets_t0[i]:,.0f}")

    # --- A.3: 総資産変化の分布 ---
    print("\n=== A.3 射影後の総資産変化の分布(Assets(t=1)/Assets(t=0) - 1) ===")
    assets_idx = [i for i, bb in enumerate(spec.basis) if bb.side == "assets"]
    assets_t0 = x_primary[:, 0, assets_idx].sum(axis=1)
    assets_t1 = x_primary[:, -1, assets_idx].sum(axis=1)
    ratio = assets_t1 / assets_t0 - 1.0
    qs = np.percentile(ratio, [0, 25, 50, 75, 100])
    print(
        f"分布: min={qs[0]:.4f}, Q1={qs[1]:.4f}, 中央値={qs[2]:.4f}, "
        f"Q3={qs[3]:.4f}, max={qs[4]:.4f}"
    )

    # --- A.4: 目標に到達できない企業 ---
    print("\n=== A.4 目標に到達できない企業(‖x(1)-x(0)‖ / ‖x_target_unprojected-x(0)‖ < 0.5) ===")
    reach_ratio = np.zeros(len(doc_ids2))
    for i in range(len(doc_ids2)):
        moved = np.linalg.norm(x_primary[i, -1] - x_primary[i, 0])
        target_moved = np.linalg.norm(x_unprojected[i, -1] - x_unprojected[i, 0])
        reach_ratio[i] = moved / target_moved if target_moved > 0 else 1.0
    low = np.where(reach_ratio < 0.5)[0]
    print(f"該当社数: {len(low)}/{len(doc_ids2)}")
    order = low[np.argsort(reach_ratio[low])]
    for i in order:
        print(f"  {doc_ids2[i]} {filer2[i]}: 到達率={reach_ratio[i]:.3f}, "
              f"純資産(t=0)={net_assets_t0[i]:,.0f}")

    # --- A.5: 重みの床 ---
    print("\n=== A.5 重みに床を入れた版(delta=1e-3)との比較 ===")
    delta = 1e-3
    index = {n: i for i, n in enumerate(names)}
    max_ratio_floored = []
    max_ratio_current = []
    rel_diff_floored_vs_current = []
    rel_diff_whitened_vs_unwhitened = []
    for i in range(len(doc_ids2)):
        x_start = x2_true[i]
        a = float(assets2[i])
        w_current = default_weights(x_start, a)
        w_floored = 1.0 / (np.abs(x_start) + delta * a)
        max_ratio_current.append(w_current.max() / w_current.min())
        max_ratio_floored.append(w_floored.max() / w_floored.min())

        y = x_unprojected[i, -1]
        x_w_current, _ = project_onto_feasible(y, spec, x_start=x_start, weights=w_current)
        x_w_floored, _ = project_onto_feasible(y, spec, x_start=x_start, weights=w_floored)
        denom = max(float(np.linalg.norm(x_w_current)), 1.0)
        rel_diff_floored_vs_current.append(
            float(np.linalg.norm(x_w_floored - x_w_current)) / denom
        )

        # 床ありの重みで、白色化あり(project_onto_feasible)/なし(_solve_unwhitened)を比較。
        x_floored_unwhitened = _solve_unwhitened(y, spec, x_start, w_floored, c, index)
        denom2 = max(float(np.linalg.norm(x_w_floored)), 1.0)
        rel_diff_whitened_vs_unwhitened.append(
            float(np.linalg.norm(x_floored_unwhitened - x_w_floored)) / denom2
        )
    print(f"重み比の最大値: 現行(epsのみ)={max(max_ratio_current):,.0f}倍, "
          f"床あり(delta=1e-3)={max(max_ratio_floored):,.0f}倍")
    print(f"床あり版で 白色化あり/なし の解の相対差(‖差‖/‖白色化ありの解‖): "
          f"最大={max(rel_diff_whitened_vs_unwhitened):.4e}, "
          f"中央値={np.median(rel_diff_whitened_vs_unwhitened):.4e}"
          "(一致すれば、床ありでは白色化が無くても解けることになる)")
    print(f"現行版(epsのみ)と床あり版の解の相対差(‖差‖/‖現行解‖): "
          f"最大={max(rel_diff_floored_vs_current):.4e}, "
          f"中央値={np.median(rel_diff_floored_vs_current):.4e}")


if __name__ == "__main__":
    main()
