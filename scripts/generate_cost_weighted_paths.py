#!/usr/bin/env python
"""ステージ9: 科目ごとのコスト重みを輸送コスト・射影重みに反映し、
ステージ7/8の(重みなし)結果と比較する。

比較対象:
- baseline: ステージ7/8と同じ、低次元PCA空間の二乗ユークリッド距離コスト
- weighted: 元空間での科目別非対称コスト(costs.yaml、方式A=利益剰余金等を
  実質固定)
- weighted_low_dim: 元空間コストを低次元へ近似投影した版(§2.1後半、
  対応先の変化を比較する目的のみ)

使い方:
    uv run python scripts/generate_cost_weighted_paths.py
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
from fsgen.accounting.projection_qp import (
    apply_projection_to_paths,
    load_direction_constraints,
    project_onto_feasible,
)
from fsgen.accounting.subspace import build_reduced_basis
from fsgen.evaluation.classifiers import train_classifiers
from fsgen.models.cost import (
    combined_projection_weights,
    cost_vectors,
    load_cost_config,
    low_dim_symmetric_cost_matrix,
    weighted_asymmetric_cost_matrix,
)
from fsgen.models.transport import (
    cost_matrix as euclidean_cost_matrix,
)
from fsgen.models.transport import (
    effective_support_sizes,
    paths_to_state_vectors,
    single_target_paths,
    solve_ot,
    solve_ot_from_cost_verbose,
)
from fsgen.viz.balance_sheet import build_bs_lines, load_labels, render_markdown, render_text

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_COSTS = Path("conf/accounts/costs.yaml")
_DEFAULT_LABELS_JA = Path("conf/accounts/labels_ja.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_OUTPUT_DIR = Path("data/interim/generated_cost_weighted")
_K = 13
_EPSILON = 0.05
_TS = (0.0, 0.25, 0.5, 0.75, 1.0)
_SEED = 42
_VIOLATION_RELATIVE_TOL = 1e-4


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


def count_sign_violations(
    x_paths: np.ndarray, names: tuple[str, ...], expected: dict[str, str], assets: np.ndarray
) -> int:
    n, t_len, _d = x_paths.shape
    count = 0
    for i in range(n):
        tol = assets[i] * _VIOLATION_RELATIVE_TOL
        for t_i in range(t_len):
            for d, name in enumerate(names):
                sign = expected[name]
                if sign == "free":
                    continue
                v = x_paths[i, t_i, d]
                if (sign == "nonneg" and v < -tol) or (sign == "nonpos" and v > tol):
                    count += 1
    return count


def cost_matrix_stats(cost: np.ndarray, label: str) -> None:
    nonzero = cost[cost > 0]
    ratio = nonzero.max() / np.median(nonzero)
    log_ratio = np.log1p(nonzero).max() / np.median(np.log1p(nonzero))
    print(f"  [{label}] max/median比={ratio:.1f}倍, log1p変換後max/median比={log_ratio:.1f}倍")


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--costs", type=Path, default=_DEFAULT_COSTS)
    parser.add_argument("--labels-ja", type=Path, default=_DEFAULT_LABELS_JA)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    direction_constraints = load_direction_constraints(args.canonical)
    cost_config = load_cost_config(args.costs)
    # §1.3/§2.1の指示通り、輸送コスト・射影のいずれもnon_actionable科目には
    # freeze(w=1e6)を使う(仕様に明記された既定の方式A)。
    # "ignore"(w=0、対応先選びからnon_actionable科目を除外する代替案)も
    # 実装し、§3.2/§7で比較する(自明ではないトレードオフが見つかったため)。
    w_plus, w_minus = cost_vectors(cost_config, names, non_actionable_mode="freeze")
    w_plus_ot, w_minus_ot = w_plus, w_minus
    w_max_ot = np.maximum(w_plus_ot, w_minus_ot)
    w_plus_ot_ignore, w_minus_ot_ignore = cost_vectors(
        cost_config, names, non_actionable_mode="ignore"
    )
    re_idx = names.index("jppfs_cor_RetainedEarnings")

    doc_ids, filer_names, x, assets, labels = load_dataset(
        args.aggregated_csv, args.labels_csv, names
    )
    x_tilde = x / assets[:, None]
    b = build_reduced_basis(c, x_tilde, k=_K)
    z = x_tilde @ b

    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    z0, z2 = z[idx0], z[idx2]
    x0_tilde, x2_tilde = x_tilde[idx0], x_tilde[idx2]
    assets2 = assets[idx2]
    x2_true = x[idx2]
    doc_ids2 = [doc_ids[i] for i in idx2]
    filer2 = [filer_names[i] for i in idx2]
    print(f"class0: {len(idx0)}社, class2: {len(idx2)}社")

    # --- baseline(ステージ7/8): 低次元PCA空間の二乗ユークリッド距離 ---
    plan_base = solve_ot(z2, z0, _EPSILON)
    z_linear_base, targets_base = single_target_paths(z2, z0, plan_base, _TS)
    x_base_unprojected = paths_to_state_vectors(z_linear_base, z2, x2_true, b, assets2)
    x_base_proj, _infos_base = apply_projection_to_paths(
        x_base_unprojected, x2_true, assets2, spec, mode="every_step",
        weighted=True, direction_constraints=direction_constraints,
    )

    # --- weighted(ステージ9): 元空間での科目別非対称コスト ---
    print("\n=== §3.1 コスト行列の妥当性 ===")
    cost_base = euclidean_cost_matrix(z2, z0)
    cost_matrix_stats(cost_base, "baseline(旧、低次元PCA空間の二乗ユークリッド)")
    cost_elem = weighted_asymmetric_cost_matrix(x2_tilde, x0_tilde, w_plus_ot, w_minus_ot)
    cost_matrix_stats(cost_elem, "元空間コスト")
    ot_result = solve_ot_from_cost_verbose(cost_elem, n0=len(idx2), n1=len(idx0), epsilon=_EPSILON)
    print(f"  脱落した企業数(質量ほぼ0の行): {ot_result.n_zero_mass_rows}/{len(idx2)}")
    ess = effective_support_sizes(ot_result.plan)
    print(f"  実効サポートサイズ: 中央値={np.median(ess):.2f}, 最大={ess.max():.2f}")

    z_linear_w, targets_w = single_target_paths(z2, z0, ot_result.plan, _TS)
    x_w_unprojected = paths_to_state_vectors(z_linear_w, z2, x2_true, b, assets2)

    proj_weights_w = [
        combined_projection_weights(x2_true[i], float(assets2[i]), w_plus, w_minus)
        for i in range(len(idx2))
    ]
    x_w_proj = x_w_unprojected.copy()
    for i in range(len(idx2)):
        for t_i in range(len(_TS)):
            x_proj, _info = project_onto_feasible(
                x_w_unprojected[i, t_i], spec, x_start=x2_true[i],
                weights=proj_weights_w[i], direction_constraints=direction_constraints,
            )
            x_w_proj[i, t_i] = x_proj

    # --- 低次元近似版(§2.1後半、対応先の変化比較専用) ---
    cost_low_dim = low_dim_symmetric_cost_matrix(z2, z0, b, w_max_ot)
    cost_matrix_stats(cost_low_dim, "低次元近似コスト")
    plan_low_dim = solve_ot_from_cost_verbose(
        cost_low_dim, n0=len(idx2), n1=len(idx0), epsilon=_EPSILON
    ).plan
    targets_low_dim = np.argmax(plan_low_dim, axis=1)

    # --- §3.2 対応先の変化 ---
    print("\n=== §3.2 対応先の変化(baseline vs weighted[元空間]) ===")
    changed = int(np.sum(targets_base != targets_w))
    print(f"対応先が変わった社数: {changed}/{len(idx2)}")
    print(f"対応先に現れる異なる企業数: baseline={len(set(targets_base.tolist()))}, "
          f"weighted={len(set(targets_w.tolist()))}")
    x0_true = x[idx0]
    assets0 = assets[idx0]
    re_target_base = x0_true[targets_base, re_idx] / assets0[targets_base]
    re_target_w = x0_true[targets_w, re_idx] / assets0[targets_w]
    n_base_neg = int(np.sum(re_target_base < 0))
    n_w_neg = int(np.sum(re_target_w < 0))
    print(
        f"目標企業の利益剰余金が負の社数: baseline={n_base_neg}/{len(idx2)}, "
        f"weighted={n_w_neg}/{len(idx2)}"
    )

    print("\n=== §3.2続き: 元空間版 vs 低次元近似版の対応先の違い ===")
    changed_lowdim = int(np.sum(targets_w != targets_low_dim))
    print(f"対応先が変わった社数(元空間 vs 低次元近似): {changed_lowdim}/{len(idx2)}")

    # --- §7: non_actionableの扱い(freeze vs ignore)のトレードオフ ---
    print("\n=== §7 non_actionableの輸送コスト上の扱い: freeze(仕様既定) vs ignore(代替案) ===")
    cost_elem_ignore = weighted_asymmetric_cost_matrix(
        x2_tilde, x0_tilde, w_plus_ot_ignore, w_minus_ot_ignore
    )
    ot_ignore = solve_ot_from_cost_verbose(
        cost_elem_ignore, n0=len(idx2), n1=len(idx0), epsilon=_EPSILON
    )
    targets_ignore = np.argmax(ot_ignore.plan, axis=1)
    re_target_ignore = x0_true[targets_ignore, re_idx] / assets0[targets_ignore]
    n_ignore_neg = int(np.sum(re_target_ignore < 0))
    z_linear_ignore, _ = single_target_paths(z2, z0, ot_ignore.plan, _TS)
    x_ignore_unprojected = paths_to_state_vectors(z_linear_ignore, z2, x2_true, b, assets2)
    proj_weights_ignore = [
        combined_projection_weights(x2_true[i], float(assets2[i]), w_plus, w_minus)
        for i in range(len(idx2))
    ]
    x_ignore_proj = x_ignore_unprojected.copy()
    for i in range(len(idx2)):
        for t_i in range(len(_TS)):
            x_proj, _info = project_onto_feasible(
                x_ignore_unprojected[i, t_i], spec, x_start=x2_true[i],
                weights=proj_weights_ignore[i], direction_constraints=direction_constraints,
            )
            x_ignore_proj[i, t_i] = x_proj
    re_change_ignore = np.median(
        np.abs(x_ignore_proj[:, -1, re_idx] - x_ignore_proj[:, 0, re_idx]) / assets2
    )
    re_change_freeze = np.median(
        np.abs(x_w_proj[:, -1, re_idx] - x_w_proj[:, 0, re_idx]) / assets2
    )
    print(
        f"  freeze(w=1e6, 既定): 目標企業のREが負の社数={n_w_neg}/{len(idx2)}, "
        f"生成経路のRE変化(総資産比)中央値={re_change_freeze:.4e}"
    )
    print(
        f"  ignore(w=0, 代替案): 目標企業のREが負の社数={n_ignore_neg}/{len(idx2)}, "
        f"生成経路のRE変化(総資産比)中央値={re_change_ignore:.4e}"
    )
    print(
        "  => freezeは対応先選びをREが近い(同じくらい不健全な)相手に偏らせる副作用があるが、"
        "REの実際の変化量はignoreよりはるかに小さく抑えられる(最重要項目である"
        "REの非操作性をより確実に達成する)。"
    )

    # --- §3.3 生成結果(ステージ8指標の再計算) ---
    expected = {bb.name: bb.sign for bb in spec.basis}
    print("\n=== §3.3 生成結果(射影後) ===")
    for label, paths in (("baseline", x_base_proj), ("weighted", x_w_proj)):
        v = count_sign_violations(paths, names, expected, assets2)
        assets_idx = [i for i, bb in enumerate(spec.basis) if bb.side == "assets"]
        assets_t0 = paths[:, 0, assets_idx].sum(axis=1)
        assets_t1 = paths[:, -1, assets_idx].sum(axis=1)
        assets_change = np.median(np.abs(assets_t1 / assets_t0 - 1.0))
        prox = np.median(
            np.linalg.norm(paths[:, -1, :] - paths[:, 0, :], axis=1) / assets2
        )
        sparsity = np.median(
            np.sum(np.abs(paths[:, -1, :] - paths[:, 0, :]) / assets2[:, None] > 0.01, axis=1)
        )
        re_change = np.median(np.abs(paths[:, -1, re_idx] - paths[:, 0, re_idx]) / assets2)
        print(
            f"  {label:<10} 符号違反={v}件, 総資産変化率中央値={assets_change:.4e}, "
            f"proximity中央値={prox:.4f}, sparsity中央値={sparsity:.1f}, "
            f"利益剰余金変化(総資産比)中央値={re_change:.4e}"
        )

    # validity(3分類器、健全確率、高いほど良い)
    x_all_tilde = x / assets[:, None]
    is_class2 = np.array([lab == 2 for lab in labels])
    classifiers = train_classifiers(x_all_tilde, is_class2, seed=_SEED)
    print("  validity(健全確率、高いほど良い):")
    for label, paths in (("baseline", x_base_proj), ("weighted", x_w_proj)):
        endpoint_norm = paths[:, -1, :] / assets2[:, None]
        for clf_name, clf in classifiers.items():
            health = 1.0 - clf.predict_proba_class2(endpoint_norm)
            print(f"    {label:<10} {clf_name:<15} median={np.median(health):.4f}")

    # --- BS表(§3.4): 東邦亜鉛を含む5社、baseline/weighted両方 ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    labels_map = load_labels(args.labels_ja)

    valid_mask = ot_result.plan.sum(axis=1) >= 1e-12
    valid_indices = np.where(valid_mask)[0]
    net_assets_idx = [i for i, bb in enumerate(spec.basis) if bb.side == "net_assets"]
    net_assets_t0 = x_w_proj[:, 0, net_assets_idx].sum(axis=1)
    order = valid_indices[np.argsort(net_assets_t0[valid_indices])]
    debt_overhang = list(order[:2])
    rng = np.random.default_rng(_SEED)
    twqj_idx = doc_ids2.index("S100TWQJ") if "S100TWQJ" in doc_ids2 else None
    extra_required = [twqj_idx] if twqj_idx is not None else []
    must_include = list(dict.fromkeys([*debt_overhang, *extra_required]))
    remaining = [i for i in valid_indices if i not in must_include]
    n_extra = max(0, 5 - len(must_include))
    extra = list(rng.choice(remaining, size=min(n_extra, len(remaining)), replace=False))
    sample_indices = must_include + extra
    print(f"\nBS表サンプル: {[doc_ids2[i] for i in sample_indices]}")

    for i in sample_indices:
        for label, paths in (("baseline", x_base_proj), ("weighted", x_w_proj)):
            x_by_t = [paths[i, t_i, :] for t_i in range(len(_TS))]
            lines = build_bs_lines(spec, names, x_by_t, labels_map)
            text = render_text(lines, _TS, f"{filer2[i]} [{label}]", doc_ids2[i])
            md = render_markdown(lines, _TS, f"{filer2[i]} [{label}]", doc_ids2[i])
            print("\n" + text)
            (samples_dir / f"{doc_ids2[i]}_{label}.md").write_text(md, encoding="utf-8")

    print(f"\n出力先: {args.output_dir}")


if __name__ == "__main__":
    main()
