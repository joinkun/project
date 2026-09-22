#!/usr/bin/env python
"""射影つきの経路を生成し、ステージ7の比較(§5〜§6)を行う。

ステージ6.5の経路生成(OT + 線形補間、実測値を基準点とする移動量方式)に
射影(二次計画)を組み込む。以下を比較する:

- mode: every_step(各tで射影) / endpoint_only(終端のみ射影)
- enforce_inactive: x_i=0 の科目を0に固定するかどうか
- weighted: 重み付き射影 W=diag(1/(|x_start|+eps)) かどうか

使い方:
    uv run python scripts/generate_projected_paths.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.projection_qp import (
    ProjectionInfo,
    apply_projection_to_paths,
    default_weights,
    load_direction_constraints,
    project_onto_feasible,
)
from fsgen.accounting.subspace import build_reduced_basis
from fsgen.models.transport import paths_to_state_vectors, single_target_paths, solve_ot
from fsgen.viz.balance_sheet import build_bs_lines, load_labels, render_markdown, render_text

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_LABELS_JA = Path("conf/accounts/labels_ja.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_OUTPUT_DIR = Path("data/interim/generated_projected")
_K = 13
_EPSILON = 0.05
_TS = (0.0, 0.25, 0.5, 0.75, 1.0)
_SEED = 42


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


def expected_signs(spec: object) -> dict[str, str]:
    return {b.name: b.sign for b in spec.basis}  # type: ignore[attr-defined]


_VIOLATION_RELATIVE_TOL = 1e-4  # 総資産の0.01%。QPソルバのeps_abs/eps_rel由来の数値誤差を吸収する。


def count_sign_violations(
    x_paths: np.ndarray,
    names: tuple[str, ...],
    expected: dict[str, str],
    assets: np.ndarray,
) -> pd.DataFrame:
    """符号制約違反を数える。ソルバの数値誤差(実測: 最大でも総資産比1e-6台)を
    実質的な違反と混同しないよう、各社の総資産に対する相対許容誤差を使う
    (絶対0円厳密一致では、QPソルバの出力はほぼ全て「違反」と誤検出される)。
    """
    rows = []
    n, t_len, _d = x_paths.shape
    for i in range(n):
        tol = assets[i] * _VIOLATION_RELATIVE_TOL
        for t_i in range(t_len):
            for d, name in enumerate(names):
                sign = expected[name]
                if sign == "free":
                    continue
                v = x_paths[i, t_i, d]
                violated = (sign == "nonneg" and v < -tol) or (sign == "nonpos" and v > tol)
                if violated:
                    rows.append({"row_index": i, "t_index": t_i, "name": name, "value": v})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--labels-ja", type=Path, default=_DEFAULT_LABELS_JA)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    direction_constraints = load_direction_constraints(args.canonical)
    print(f"方向制約: {[(d.name, d.condition, d.rule) for d in direction_constraints]}")

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
    print(f"class0: {len(idx0)}社, class2: {len(idx2)}社")

    plan = solve_ot(z2, z0, _EPSILON)
    z_linear, _targets = single_target_paths(z2, z0, plan, _TS)
    x_unprojected = paths_to_state_vectors(z_linear, z2, x2_true, b, assets2)

    expected = expected_signs(spec)

    print("\n=== §4 未射影経路の符号違反(ステージ6.5の再確認) ===")
    v_unprojected = count_sign_violations(x_unprojected, names, expected, assets2)
    print(f"総違反件数: {len(v_unprojected)}")

    # --- §5.1: mode × enforce_inactive の比較(weighted=True固定) ---
    print("\n=== §5.1 mode × enforce_inactive の符号違反比較(射影後、weighted=True) ===")
    results: dict[tuple[str, bool], tuple[np.ndarray, list[ProjectionInfo]]] = {}
    for mode in ("every_step", "endpoint_only"):
        for enforce_inactive in (False, True):
            x_proj, infos = apply_projection_to_paths(
                x_unprojected,
                x2_true,
                assets2,
                spec,
                mode=mode,
                enforce_inactive=enforce_inactive,
                weighted=True,
                direction_constraints=direction_constraints,
            )
            results[(mode, enforce_inactive)] = (x_proj, infos)
            v = count_sign_violations(x_proj, names, expected, assets2)
            status_counts = Counter(info.status for info in infos)
            print(
                f"mode={mode:<13} enforce_inactive={enforce_inactive!s:<5}: "
                f"符号違反={len(v)}件, ソルバstatus={dict(status_counts)}, "
                f"求解時間平均={np.mean([i.solve_time_seconds for i in infos]) * 1000:.2f}ms"
            )

    # 主要な組み合わせ(採用): every_step, enforce_inactive=False, weighted=True
    x_primary, infos_primary = results[("every_step", False)]

    # --- §5.1 続き: infeasible/失敗の件数と原因 ---
    print("\n=== §8 ソルバがinfeasible/失敗した件数 ===")
    non_optimal = [info for info in infos_primary if info.status not in ("optimal",)]
    print(f"status != optimal: {len(non_optimal)}/{len(infos_primary)}")
    for info in non_optimal[:10]:
        print(f"  status={info.status}, max_violation={info.max_constraint_violation:.4e}")

    # 射影後、始点からほとんど動かなかった社(=実行可能な再建経路が乏しい)
    print("\n=== §7 射影後、終端がほとんど動かなかった社 ===")
    barely_moved = []
    for i in range(len(doc_ids2)):
        moved = np.linalg.norm(x_primary[i, -1] - x_unprojected[i, 0])
        target_moved = np.linalg.norm(x_unprojected[i, -1] - x_unprojected[i, 0])
        if target_moved > 0 and moved / target_moved < 0.05:
            barely_moved.append((doc_ids2[i], filer2[i], moved, target_moved))
    print(f"該当社: {len(barely_moved)}/{len(doc_ids2)}")
    for doc_id, filer, moved, target_moved in barely_moved[:10]:
        print(f"  {doc_id} {filer}: 移動量={moved:,.0f} (目標までの距離{target_moved:,.0f})")

    # --- 総資産の扱い(§7) ---
    print("\n=== §7 射影後の総資産合計の変化 ===")
    assets_idx = [i for i, bb in enumerate(spec.basis) if bb.side == "assets"]
    assets_before = x_unprojected[:, :, assets_idx].sum(axis=2)
    assets_after = x_primary[:, :, assets_idx].sum(axis=2)
    rel_change = np.abs(assets_after - assets_before) / assets2[:, None]
    print(
        f"射影前後の総資産相対変化: 中央値={np.median(rel_change):.4e}, "
        f"最大={np.max(rel_change):.4e}"
    )

    # --- §5.2: 経路の連続性 ---
    print("\n=== §5.2 経路の連続性(隣接t間の変化量 / 全体の変化量) ===")
    for label, x_paths in (("未射影", x_unprojected), ("射影後(every_step)", x_primary)):
        ratios = []
        for i in range(len(doc_ids2)):
            total = np.linalg.norm(x_paths[i, -1] - x_paths[i, 0])
            if total == 0:
                continue
            steps = [
                np.linalg.norm(x_paths[i, k + 1] - x_paths[i, k]) / total
                for k in range(len(_TS) - 1)
            ]
            ratios.append(steps)
        arr = np.array(ratios)
        print(f"{label}: 各区間の比率(社平均) = {arr.mean(axis=0)}")
        print(f"{label}: 区間比率の標準偏差(社間) = {arr.std(axis=0)}")

    # --- §6.5: 重み付きvs重みなし比較 ---
    print("\n=== §6.5(報告5) 重み付きvs重みなしの射影距離比較 ===")
    total_dist_weighted = []
    total_dist_unweighted = []
    per_account_move_w: dict[str, list[float]] = defaultdict(list)
    per_account_move_u: dict[str, list[float]] = defaultdict(list)
    for i in range(len(doc_ids2)):
        y = x_unprojected[i, -1]
        x_start = x2_true[i]
        w = default_weights(x_start, float(assets2[i]))
        x_w, info_w = project_onto_feasible(y, spec, x_start=x_start, weights=w)
        x_u, info_u = project_onto_feasible(y, spec, x_start=x_start, weights=None)
        total_dist_weighted.append(info_w.distance_before_after)
        total_dist_unweighted.append(info_u.distance_before_after)
        for d, name in enumerate(names):
            per_account_move_w[name].append(abs(x_w[d] - y[d]))
            per_account_move_u[name].append(abs(x_u[d] - y[d]))
    print(
        f"移動距離(円): 重み付き 中央値={np.median(total_dist_weighted):,.0f}, "
        f"重みなし 中央値={np.median(total_dist_unweighted):,.0f}"
    )
    print("科目別の移動量(中央値、円) 重み付き vs 重みなし(上位10科目、重み付きの移動量順):")
    per_account_summary = sorted(
        (
            (name, np.median(per_account_move_w[name]), np.median(per_account_move_u[name]))
            for name in names
        ),
        key=lambda t: -t[1],
    )
    for name, med_w, med_u in per_account_summary[:10]:
        print(f"  {name:<45} weighted={med_w:>15,.0f}  unweighted={med_u:>15,.0f}")

    # --- BS表サンプル(§6.7) ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    labels_map = load_labels(args.labels_ja)

    # 輸送計画の質量が数値的に0まで潰れた社(ステージ6.5 §Cで判明)は対応先が
    # 実質未定義で、経路が暴れて解釈しづらいBS表になるため候補から除外する。
    valid_mask = plan.sum(axis=1) >= 1e-12
    valid_indices = np.where(valid_mask)[0]
    print(f"\n輸送計画が有効な社: {len(valid_indices)}/{len(doc_ids2)}(サンプルはこの中から選ぶ)")

    net_assets_idx = [i for i, bb in enumerate(spec.basis) if bb.side == "net_assets"]
    net_assets_t0 = x_primary[:, 0, net_assets_idx].sum(axis=1)
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
    print(f"BS表サンプル: {[doc_ids2[i] for i in sample_indices]}")

    for i in sample_indices:
        x_by_t = [x_primary[i, t_i, :] for t_i in range(len(_TS))]
        lines = build_bs_lines(spec, names, x_by_t, labels_map)
        text = render_text(lines, _TS, filer2[i], doc_ids2[i])
        md = render_markdown(lines, _TS, filer2[i], doc_ids2[i])
        print("\n" + text)
        (samples_dir / f"{doc_ids2[i]}.md").write_text(md, encoding="utf-8")

    print(f"\n出力先: {args.output_dir}")


if __name__ == "__main__":
    main()
