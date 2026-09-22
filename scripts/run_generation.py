#!/usr/bin/env python
"""ステージ11-A: 経路生成専用スクリプト(torch/cvxpy/cvxpylayersのみ)。

macOSでtorchとlightgbmが同一プロセス内に共存すると、互いに異なる
libomp.dylib(torchが同梱するものとHomebrew版、lightgbmはHomebrew版を
要求する)が競合してセグフォルトする問題が確認された(§A.1)。
``KMP_DUPLICATE_LIB_OK=TRUE``は2つ目のランタイムを無視して処理を続行
させるだけで結果が静かに壊れる可能性があるため使わない。

代わりに、輸送計画の事前計算(precompute_transport.py)・経路生成
(本スクリプト、torch/cvxpy/cvxpylayersのみ)・指標計算(run_metrics.py、
lightgbm/sklearnのみ)の3プロセスに分離する。当初は2分割の想定だったが、
POT(``ot``パッケージ)が``ot.da``経由でsklearnを無条件にimportし、
sklearnがtorchとは別のlibomp.dylibを同梱しているため、torchを使う
プロセス内で``ot``をimportするだけでlibomp.dylibが2つロードされる
ことをvmmapで確認した(§A.3)。そのため本スクリプトは``ot``を一切
importせず、輸送計画はprecompute_transport.pyが計算したnpzを読み込む。

生成結果はparquetに保存し、run_metrics.pyが読み込んでvalidity/識別AUCを
計算する(状態ベクトルは常に総資産で正規化済み=tilde空間で保存するので、
run_metrics.py側でAssetsによる再正規化をしないこと——ステージ10.6で
この二重正規化バグを一度踏んでいる)。

前提: 先に ``uv run python scripts/precompute_transport.py`` を実行し、
``data/interim/generation_11a/fold_plans.npz`` を作っておくこと。

使い方:
    uv run python scripts/run_generation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import (
    CanonicalSpec,
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.projection_qp import (
    DirectionConstraintSpec,
    load_direction_constraints,
    project_onto_feasible,
)
from fsgen.accounting.subspace import build_reduced_basis
from fsgen.evaluation.baselines import (
    ConvexQpBaseline,
    build_prior_work_baseline,
    ot_target_for_new_point,
)
from fsgen.models.cost import combined_projection_weights, load_cost_config
from fsgen.models.differentiable_projection import build_projection_layer
from fsgen.models.flow_matching import VelocityField, euler_sample_batch, train_cafm, train_fm

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_COSTS = Path("conf/accounts/costs.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_OUTPUT_DIR = Path("data/interim/generation_11a")

_K = 13
_TS = (0.0, 1.0)  # 本スクリプトの用途(A.5/B.2/B.3)では始点・終点のみで十分
_N_FOLDS = 5
_SEEDS = (0, 1, 2)  # ステージ10/10.5/10.6と同一
_EPOCHS = 50
_LR = 1e-3
_HIDDEN = 128
_N_LAYERS = 3


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


def sample_and_convert(
    net: VelocityField, x2_tilde: np.ndarray, pos: np.ndarray, b: np.ndarray
) -> np.ndarray:
    z_start = x2_tilde[pos] @ b
    z_path = euler_sample_batch(net, z_start, _TS)
    displacement = z_path - z_start[:, None, :]
    x_tilde_path: np.ndarray = x2_tilde[pos][:, None, :] + np.einsum(
        "dk,ntk->ntd", b, displacement
    )
    return x_tilde_path


def project_path(
    spec: CanonicalSpec,
    x_tilde_path: np.ndarray,
    x_start_tilde: np.ndarray,
    weights: np.ndarray,
    dcs: tuple[DirectionConstraintSpec, ...],
    non_actionable: tuple[str, ...],
) -> np.ndarray:
    out = x_tilde_path.copy()
    for t_i in range(x_tilde_path.shape[0]):
        x_proj, _info = project_onto_feasible(
            x_tilde_path[t_i], spec, x_start=x_start_tilde, weights=weights,
            direction_constraints=dcs, frozen_accounts=non_actionable,
        )
        out[t_i] = x_proj
    return out


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(_DEFAULT_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(_DEFAULT_CANONICAL)
    cost_config = load_cost_config(_DEFAULT_COSTS)
    non_actionable = tuple(cost_config.non_actionable)
    dim = len(names)

    out_dir = _DEFAULT_OUTPUT_DIR
    plans_path = out_dir / "fold_plans.npz"
    if not plans_path.exists():
        raise FileNotFoundError(
            f"{plans_path} が見つかりません。先に "
            "`uv run python scripts/precompute_transport.py` を実行してください。"
        )
    fold_plans = np.load(plans_path)

    doc_ids, filer_names, x, assets, labels = load_dataset(
        _DEFAULT_AGGREGATED_CSV, _DEFAULT_LABELS_CSV, names
    )
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    x0_tilde = x_tilde[idx0]
    x2_tilde = x_tilde[idx2]
    doc_ids2 = [doc_ids[i] for i in idx2]
    filer2 = [filer_names[i] for i in idx2]
    n2 = len(idx2)
    print(f"class0: {len(idx0)}社, class2: {n2}社")

    # --- 生成本体: W1条件でPCFM/CAFM/FM(ステージ10.5と同一構成) ---
    w_plus_proj = np.ones(dim)
    w_minus_proj = np.ones(dim)
    proj_delta = 1e-3

    rows: list[dict[str, float | int | str]] = []

    def _row(
        seed: int, fold_i: int, doc_id: str, filer: str, method: str,
        x0_t: np.ndarray, x1_t: np.ndarray, assets_start: float,
    ) -> dict[str, float | int | str]:
        r: dict[str, float | int | str] = {
            "seed": seed, "fold": fold_i, "docID": doc_id, "filerName": filer,
            "method": method, "assets_start": assets_start,
        }
        for i, name in enumerate(names):
            r[f"x0_{name}"] = float(x0_t[i])
            r[f"x1_{name}"] = float(x1_t[i])
        return r

    print(f"\n=== 経路生成: 5-fold × seed{_SEEDS}(W1条件、FM/PCFM/CAFM) ===")
    for seed in _SEEDS:
        for fold_i in range(_N_FOLDS):
            train_pos = fold_plans[f"train_pos_{fold_i}"]
            test_pos = fold_plans[f"test_pos_{fold_i}"]
            plan_train = fold_plans[f"plan_train_{fold_i}"]
            plan_test = fold_plans[f"plan_test_{fold_i}"]

            pool_idx = idx0 + [idx2[p] for p in train_pos]
            x_pool_tilde = x_tilde[pool_idx]
            b = build_reduced_basis(c, x_pool_tilde, k=_K)
            z0 = x0_tilde @ b
            z_train = x2_tilde[train_pos] @ b
            z_test = x2_tilde[test_pos] @ b

            net_fm = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
            train_fm(
                z_train, z0, plan_train, net_fm, epochs=_EPOCHS, lr=_LR, seed=seed,
                z_val=z_test, plan_val=plan_test,
            )
            x_tilde_path_fm = sample_and_convert(net_fm, x2_tilde, test_pos, b)

            layers_train = []
            for p in train_pos:
                x_start_p = x2_tilde[p]
                weights_p = combined_projection_weights(
                    x_start_p, 1.0, w_plus_proj, w_minus_proj, proj_delta
                )
                layers_train.append(
                    build_projection_layer(spec, x_start_p, weights_p, dcs, non_actionable)
                )
            net_cafm = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
            x_start_train = x2_tilde[train_pos]
            result_cafm = train_cafm(
                z_train, z0, plan_train, net_cafm, layers_train, x_start_train, b,
                epochs=_EPOCHS, lr=_LR, seed=seed, z_val=z_test, plan_val=plan_test,
            )
            if result_cafm.solver_failures:
                print(f"    [注] CAFM射影求解失敗: {result_cafm.solver_failures}件(スキップ済み)")
            x_tilde_path_cafm = sample_and_convert(net_cafm, x2_tilde, test_pos, b)

            targets_train = np.argmax(plan_train, axis=1)

            for i_local, p in enumerate(test_pos):
                doc_id = doc_ids2[p]
                filer = filer2[p]
                x_start_tilde = x2_tilde[p]
                assets_start = float(assets[idx2[p]])
                weights = combined_projection_weights(
                    x_start_tilde, 1.0, w_plus_proj, w_minus_proj, proj_delta
                )

                raw_fm = x_tilde_path_fm[i_local]
                pcfm = project_path(spec, raw_fm, x_start_tilde, weights, dcs, non_actionable)
                cafm_raw = x_tilde_path_cafm[i_local]
                cafm = project_path(spec, cafm_raw, x_start_tilde, weights, dcs, non_actionable)

                rows.append(
                    _row(seed, fold_i, doc_id, filer, "FM", raw_fm[0], raw_fm[-1], assets_start)
                )
                rows.append(
                    _row(seed, fold_i, doc_id, filer, "PCFM", pcfm[0], pcfm[-1], assets_start)
                )
                rows.append(
                    _row(seed, fold_i, doc_id, filer, "CAFM", cafm[0], cafm[-1], assets_start)
                )

                ot = ot_target_for_new_point(
                    x2_tilde[p], 1.0, b, z0, z_train, targets_train
                )
                ts_arr = np.asarray(_TS)
                disp_z = ts_arr[:, None] * (ot.z_target - ot.z_start)[None, :]
                disp_x = np.einsum("dk,tk->td", b, disp_z)
                path_s9 = x_start_tilde[None, :] + disp_x
                path_s9_proj = project_path(
                    spec, path_s9, x_start_tilde, weights, dcs, non_actionable
                )
                rows.append(
                    _row(
                        seed, fold_i, doc_id, filer, "stage9_proposed",
                        path_s9_proj[0], path_s9_proj[-1], assets_start,
                    )
                )

            print(
                f"  seed={seed} fold={fold_i + 1}/{_N_FOLDS} 完了(test={len(test_pos)}社)"
            )

    # --- prior_work / convex_qp(class0中央値のみ、seed非依存) ---
    print("\n=== prior_work / convex_qp(参考ベースライン) ===")
    median0_tilde = np.median(x0_tilde, axis=0)
    prior_work = build_prior_work_baseline(median0_tilde, names)
    convex_qp = ConvexQpBaseline(
        median_tilde=median0_tilde, spec=spec, direction_constraints=dcs, weighted=True
    )
    for p in range(n2):
        doc_id = doc_ids2[p]
        filer = filer2[p]
        x_start_tilde = x2_tilde[p]
        assets_start = float(assets[idx2[p]])
        for method_name, baseline in (("prior_work", prior_work), ("convex_qp", convex_qp)):
            path_tilde = baseline.generate(x_start_tilde, 1.0, _TS)
            rows.append(
                _row(
                    0, -1, doc_id, filer, method_name,
                    path_tilde[0], path_tilde[-1], assets_start,
                )
            )

    gen_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_df.to_parquet(out_dir / "generated_endpoints.parquet", index=False)
    print(f"\n{len(gen_df)}行を書き出しました: {out_dir / 'generated_endpoints.parquet'}")
    print(f"出力先: {out_dir}")


if __name__ == "__main__":
    main()
