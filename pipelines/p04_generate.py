"""ステージ6: 最小構成の経路生成(OT + 線形補間)。

Flow Matching・CAFM・ニューラルネットは使わない。目的は
(1) データから経路までの配管が通ることの確認、
(2) 生成された貸借対照表を人間が見て会計的に読めるかの判断、の2つ。

使い方:
    uv run python pipelines/p04_generate.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.subspace import build_reduced_basis
from fsgen.models.transport import (
    barycenter_paths,
    effective_support_sizes,
    paths_to_state_vectors,
    plan_entropy,
    single_target_paths,
    solve_ot_verbose,
)
from fsgen.viz.balance_sheet import build_bs_lines, load_labels, render_markdown, render_text

log = logging.getLogger(__name__)

SignExpectation = Literal["nonneg", "nonpos", "free"]


@dataclass(frozen=True)
class Dataset:
    doc_ids: list[str]
    filer_names: list[str]
    x: np.ndarray  # (n, D) 円
    assets: np.ndarray  # (n,)
    labels: list[int]


def load_dataset(aggregated_parquet: Path, labels_csv: Path, names: tuple[str, ...]) -> Dataset:
    agg = pd.read_parquet(aggregated_parquet)
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"] != True)]  # noqa: E712
    labels_df = pd.read_csv(labels_csv, dtype={"docID": str})
    labels_df = labels_df[labels_df["label"] != ""]
    merged = agg.merge(labels_df[["docID", "filerName", "label"]], on="docID", how="inner")
    merged = merged[merged["total_assets"].notna() & (merged["total_assets"] != 0)]

    x = merged[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = merged["total_assets"].to_numpy(dtype=float)
    return Dataset(
        doc_ids=merged["docID"].tolist(),
        filer_names=merged["filerName"].tolist(),
        x=x,
        assets=assets,
        labels=merged["label"].astype(int).tolist(),
    )


def expected_signs(x: np.ndarray, names: tuple[str, ...]) -> dict[str, SignExpectation]:
    """全社の実測値から、科目ごとの経験的な符号期待を機械的に求める
    (ステージ5.5 §Eと同じ規則: 負が1社もなければ nonneg、正が1社もなければ
    nonpos、両方あれば free)。"""
    result: dict[str, SignExpectation] = {}
    for i, name in enumerate(names):
        col = x[:, i]
        has_neg = bool(np.any(col < 0))
        has_pos = bool(np.any(col > 0))
        if not has_neg:
            result[name] = "nonneg"
        elif not has_pos:
            result[name] = "nonpos"
        else:
            result[name] = "free"
    return result


def count_sign_violations(
    x_paths: np.ndarray, names: tuple[str, ...], expected: dict[str, SignExpectation]
) -> pd.DataFrame:
    """x_paths (n, T, D) の各 (行, t, 科目) について符号違反を数える。

    返り値の row_index/t_index は x_paths の軸0/軸1のインデックス
    (呼び出し側で docID/t に対応付ける)。
    """
    rows = []
    n, t_len, _d = x_paths.shape
    for i in range(n):
        for t_i in range(t_len):
            for d, name in enumerate(names):
                exp = expected[name]
                if exp == "free":
                    continue
                v = x_paths[i, t_i, d]
                violated = (exp == "nonneg" and v < 0) or (exp == "nonpos" and v > 0)
                if violated:
                    rows.append(
                        {
                            "row_index": i,
                            "t_index": t_i,
                            "name": name,
                            "expected": exp,
                            "value": v,
                        }
                    )
    return pd.DataFrame(rows)


def net_assets_at_t0(x_linear: np.ndarray, spec_basis_sides: list[str]) -> np.ndarray:
    """各社の t=0 における純資産合計(純資産側基礎科目の和)。"""
    net_assets_idx = [i for i, side in enumerate(spec_basis_sides) if side == "net_assets"]
    result: np.ndarray = x_linear[:, 0, net_assets_idx].sum(axis=1)
    return result


def select_samples(
    net_assets_t0: np.ndarray, valid_mask: np.ndarray, n_samples: int, seed: int
) -> list[int]:
    """BS表出力用のサンプルを選ぶ。

    選定基準: 純資産合計(t=0)が最も小さい(債務超過が最も深刻な)2社を
    必ず含める。残りは、それ以外の会社から固定シードで無作為抽出する
    (再現性のため)。輸送計画の質量がほぼ0(数値アンダーフローで対応先が
    実質的に未定義)の会社は、意味のある経路になっていないため候補から
    除外する(§6報告の「質量ゼロの始点」参照)。
    """
    valid_indices = np.where(valid_mask)[0]
    order = valid_indices[np.argsort(net_assets_t0[valid_indices])]
    n_debt_overhang = min(2, len(order))
    must_include = list(order[:n_debt_overhang])
    remaining = [int(i) for i in order[n_debt_overhang:]]
    rng = np.random.default_rng(seed)
    n_extra = max(0, n_samples - len(must_include))
    extra = list(rng.choice(remaining, size=min(n_extra, len(remaining)), replace=False))
    return [int(i) for i in must_include] + [int(i) for i in extra]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    gen = cfg.generate
    spec = load_canonical_spec(Path(gen.canonical))
    names = state_vector_names(spec, include_subtotals=False)
    basis_sides = [b.side for b in spec.basis]
    c = build_constraint_matrix(spec, include_subtotals=False)

    dataset = load_dataset(Path(gen.aggregated_parquet), Path(gen.labels_csv), names)
    print(f"読み込み: {len(dataset.doc_ids)}社(非金融)")

    x_tilde = dataset.x / dataset.assets[:, None]

    b = build_reduced_basis(c, x_tilde, k=int(gen.k))
    print(f"低次元基底: D={b.shape[0]}, k={b.shape[1]}(全社で学習)")

    z = x_tilde @ b

    idx0 = [i for i, lab in enumerate(dataset.labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(dataset.labels) if lab == 2]
    z0, z2 = z[idx0], z[idx2]
    assets0 = dataset.assets[idx0]
    del assets0  # 終点側の総資産は使わない(始点企業の総資産だけをスケール復元に使う、§3.4)
    assets2 = dataset.assets[idx2]
    doc_ids0 = [dataset.doc_ids[i] for i in idx0]
    doc_ids2 = [dataset.doc_ids[i] for i in idx2]
    filer2 = [dataset.filer_names[i] for i in idx2]
    print(f"class0(健全、終点分布): {len(idx0)}社, class2(GC注記、始点分布): {len(idx2)}社")

    print("\n=== ε ごとの輸送計画の性質 ===")
    for eps in gen.epsilons:
        result = solve_ot_verbose(z2, z0, float(eps))
        ess = effective_support_sizes(result.plan)
        ess_valid = ess[np.isfinite(ess)]
        entropy = plan_entropy(result.plan)
        ess_summary = (
            f"中央値={np.median(ess_valid):.1f} [{ess_valid.min():.1f}, {ess_valid.max():.1f}]"
            if ess_valid.size
            else "(有効な行なし)"
        )
        print(
            f"ε={eps}: 収束={result.converged}"
            f"(周辺分布誤差={result.max_marginal_error:.2e}, "
            f"質量ゼロの始点={result.n_zero_mass_rows}/{len(idx2)}), "
            f"エントロピー={entropy:.3f}, "
            f"実効サポートサイズ(有効行のみ) {ess_summary} (終点数={len(idx0)})"
        )

    eps_default = float(gen.default_epsilon)
    default_result = solve_ot_verbose(z2, z0, eps_default)
    plan = default_result.plan
    print(f"\n以降は ε={eps_default} の輸送計画を使用。")
    valid_mask = plan.sum(axis=1) >= 1e-12
    zero_mass_doc_ids = [doc_ids2[i] for i in np.where(~valid_mask)[0]]
    if zero_mass_doc_ids:
        print(
            f"注意: {len(zero_mass_doc_ids)}/{len(idx2)}社は輸送計画の質量が"
            f"数値的に0まで潰れ、対応先が実質未定義(argmaxが恣意的)。"
            f"BS表サンプルの選定からは除外する: {zero_mass_doc_ids}"
        )

    ts = tuple(float(t) for t in gen.ts)
    z_linear, target_indices = single_target_paths(z2, z0, plan, ts)
    z_bary = barycenter_paths(z2, z0, plan, ts)

    x2_true = dataset.x[idx2]  # 実測値(円)。t=0 の基準点として使う(§B)。
    x_linear = paths_to_state_vectors(z_linear, z2, x2_true, b, assets2)
    x_bary = paths_to_state_vectors(z_bary, z2, x2_true, b, assets2)

    # --- 検証: 恒等式 ---
    residual_linear = np.einsum("ntd,dk->nt", x_linear, c)
    residual_bary = np.einsum("ntd,dk->nt", x_bary, c)
    print("\n=== 恒等式の検証 ===")
    max_rel_linear = np.max(np.abs(residual_linear) / assets2[:, None])
    max_rel_bary = np.max(np.abs(residual_bary) / assets2[:, None])
    print(f"線形補間: max|C^Tx|/Assets = {max_rel_linear:.2e}")
    print(f"バリセンター: max|C^Tx|/Assets = {max_rel_bary:.2e}")

    # --- 検証: 符号違反 ---
    expected = expected_signs(dataset.x, names)
    violations_linear = count_sign_violations(x_linear, names, expected)
    violations_bary = count_sign_violations(x_bary, names, expected)
    print("\n=== 符号違反の集計(線形補間) ===")
    if violations_linear.empty:
        print("違反なし")
    else:
        print(f"総違反件数: {len(violations_linear)}")
        print(violations_linear.groupby(["name", "t_index"]).size().unstack(fill_value=0))

    print("\n=== 符号違反の集計(バリセンター) ===")
    if violations_bary.empty:
        print("違反なし")
    else:
        print(f"総違反件数: {len(violations_bary)}")
        print(violations_bary.groupby(["name", "t_index"]).size().unstack(fill_value=0))

    # --- 到達点の分散比較 ---
    spread_linear = np.var(x_linear[:, -1, :] / assets2[:, None], axis=0).sum()
    spread_bary = np.var(x_bary[:, -1, :] / assets2[:, None], axis=0).sum()
    print(
        f"\n到達点(t=1)の分散(正規化空間): "
        f"線形補間={spread_linear:.4e}, バリセンター={spread_bary:.4e}"
    )

    # --- 出力 ---
    output_dir = Path(gen.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _paths_to_df(
        x_paths: np.ndarray, kind: str, target_ids: list[str | None]
    ) -> pd.DataFrame:
        rows = []
        for i, doc_id in enumerate(doc_ids2):
            for t_i, t in enumerate(ts):
                row: dict[str, object] = {
                    "docID": doc_id,
                    "filerName": filer2[i],
                    "kind": kind,
                    "target_docID": target_ids[i],
                    "t": t,
                }
                row.update({name: x_paths[i, t_i, d] for d, name in enumerate(names)})
                rows.append(row)
        return pd.DataFrame(rows)

    target_doc_ids: list[str | None] = [doc_ids0[j] for j in target_indices]
    df_linear = _paths_to_df(x_linear, "linear", target_doc_ids)
    df_bary = _paths_to_df(x_bary, "barycenter", [None] * len(doc_ids2))
    df_linear.to_parquet(output_dir / "paths_linear.parquet", index=False)
    df_bary.to_parquet(output_dir / "paths_barycenter.parquet", index=False)

    validation_rows = []
    for i, doc_id in enumerate(doc_ids2):
        for t_i, t in enumerate(ts):
            n_viol_linear = len(
                violations_linear[
                    (violations_linear["row_index"] == i) & (violations_linear["t_index"] == t_i)
                ]
            )
            n_viol_bary = len(
                violations_bary[
                    (violations_bary["row_index"] == i) & (violations_bary["t_index"] == t_i)
                ]
            )
            validation_rows.append(
                {
                    "docID": doc_id,
                    "t": t,
                    "residual_linear": residual_linear[i, t_i],
                    "residual_linear_rel": abs(residual_linear[i, t_i]) / assets2[i],
                    "residual_barycenter": residual_bary[i, t_i],
                    "residual_barycenter_rel": abs(residual_bary[i, t_i]) / assets2[i],
                    "n_sign_violations_linear": n_viol_linear,
                    "n_sign_violations_barycenter": n_viol_bary,
                    "transport_target_valid": bool(valid_mask[i]),
                }
            )
    pd.DataFrame(validation_rows).to_csv(output_dir / "validation.csv", index=False)

    # --- BS表サンプル ---
    labels_map = load_labels(Path(gen.labels_ja))
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    net_assets_t0 = net_assets_at_t0(x_linear, basis_sides)
    sample_indices = select_samples(
        net_assets_t0, valid_mask, n_samples=int(gen.n_samples), seed=int(gen.seed)
    )
    print(
        f"\nBS表サンプル({len(sample_indices)}社、うち債務超過上位2社を含む固定選定):"
        f" {[doc_ids2[i] for i in sample_indices]}"
    )

    for i in sample_indices:
        x_by_t = [x_linear[i, t_i, :] for t_i in range(len(ts))]
        lines = build_bs_lines(spec, names, x_by_t, labels_map)
        text = render_text(lines, ts, filer2[i], doc_ids2[i])
        md = render_markdown(lines, ts, filer2[i], doc_ids2[i])
        print("\n" + text)
        (samples_dir / f"{doc_ids2[i]}.md").write_text(md, encoding="utf-8")

    print(f"\n出力先: {output_dir}")


if __name__ == "__main__":
    main()
