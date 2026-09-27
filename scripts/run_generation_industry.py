#!/usr/bin/env python
"""ステージ12: 業種考慮の輸送設計ごとにPCFM(every_step)で経路を生成する(torch専用プロセス)。

ステージ11-Aのプロセス分離を守る: 本スクリプトはlightgbm/sklearn/ot(POT)を
一切importしない。輸送計画はprecompute_industry_transport.pyが計算した
npzを読む。指標(validity・識別AUC等)はrun_metrics_industry.pyが計算する。

生成手法: PCFM(every_step)。重み条件: W3_eq_only(非操作科目の固定は
frozen_accounts等式制約に一任し、重みは``non_actionable_mode="keep"``の
コスト重み、ステージ10.6 §3.2)。FM学習は各(設計, seed, fold)で1回、
乱数シードはステージ10〜11-Aと同一(seed 0,1,2、fold分割seed=0、50epoch)。

使い方:
    uv run python scripts/run_generation_industry.py
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
from fsgen.models.cost import combined_projection_weights, cost_vectors, load_cost_config
from fsgen.models.flow_matching import (
    VelocityField,
    euler_sample_batch,
    seeded_velocity_field,
    train_fm,
)

_CANONICAL = Path("conf/accounts/core.yaml")
_COSTS = Path("conf/accounts/costs.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_IN_DIR = Path("data/interim/industry_12")

_DESIGNS = (
    "current_asis", "current", "soft_l0.5", "soft_l1", "soft_l2", "soft_l5", "soft_l10",
    "hard33", "hard_coarse",
)
_K = 13
_TS = (0.0, 1.0)
_N_FOLDS = 5
_SEEDS = (0, 1, 2)
_EPOCHS = 50
_LR = 1e-3
_HIDDEN = 128
_N_LAYERS = 3
_PROJ_DELTA = 1e-3


def load_dataset(
    names: tuple[str, ...],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[int]]:
    agg = pd.read_csv(_AGGREGATED_CSV, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS_CSV, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "filerName", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = m["total_assets"].to_numpy(dtype=float)
    return (
        m["docID"].tolist(), m["filerName"].tolist(), x, assets, m["label"].astype(int).tolist(),
    )


def sample_and_convert(
    net: VelocityField, x2_tilde: np.ndarray, pos: np.ndarray, b: np.ndarray
) -> np.ndarray:
    z_start = x2_tilde[pos] @ b
    z_path = euler_sample_batch(net, z_start, _TS)
    displacement = z_path - z_start[:, None, :]
    out: np.ndarray = x2_tilde[pos][:, None, :] + np.einsum("dk,ntk->ntd", b, displacement)
    return out


def project_path(
    spec: CanonicalSpec,
    path: np.ndarray,
    x_start: np.ndarray,
    weights: np.ndarray,
    dcs: tuple[DirectionConstraintSpec, ...],
    frozen: tuple[str, ...],
) -> np.ndarray:
    out = path.copy()
    for t_i in range(path.shape[0]):
        out[t_i], _ = project_onto_feasible(
            path[t_i], spec, x_start=x_start, weights=weights,
            direction_constraints=dcs, frozen_accounts=frozen,
        )
    return out


def main() -> None:
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(_CANONICAL)
    cost_config = load_cost_config(_COSTS)
    frozen = tuple(cost_config.non_actionable)
    # W3_eq_only: non_actionable_mode="keep"(freezeの1e6上書きをしない)
    w_plus, w_minus = cost_vectors(cost_config, names, non_actionable_mode="keep")

    doc_ids, filer_names, x, assets, labels = load_dataset(names)
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    x0_tilde, x2_tilde = x_tilde[idx0], x_tilde[idx2]
    doc2 = [doc_ids[i] for i in idx2]
    name2 = [filer_names[i] for i in idx2]
    folds = np.load(_IN_DIR / "folds.npz")
    print(f"class0: {len(idx0)}社, class2: {len(idx2)}社")

    rows: list[dict[str, float | int | str]] = []
    for design in _DESIGNS:
        plans = np.load(_IN_DIR / f"plans_{design}.npz")
        for seed in _SEEDS:
            for k in range(_N_FOLDS):
                train_pos, test_pos = folds[f"train_pos_{k}"], folds[f"test_pos_{k}"]
                pool_idx = idx0 + [idx2[p] for p in train_pos]
                b = build_reduced_basis(c, x_tilde[pool_idx], k=_K)
                z0 = x0_tilde @ b
                z_train = x2_tilde[train_pos] @ b
                z_test = x2_tilde[test_pos] @ b
                net = seeded_velocity_field(
                    seed * 1000 + k,
                    dim=_K,
                    hidden=_HIDDEN,
                    n_layers=_N_LAYERS,
                )
                train_fm(
                    z_train, z0, plans[f"plan_train_{k}"], net, epochs=_EPOCHS, lr=_LR,
                    seed=seed, z_val=z_test, plan_val=plans[f"plan_test_{k}"],
                )
                raw = sample_and_convert(net, x2_tilde, test_pos, b)
                for i_local, p in enumerate(test_pos):
                    x_start = x2_tilde[p]
                    weights = combined_projection_weights(
                        x_start, 1.0, w_plus, w_minus, _PROJ_DELTA
                    )
                    path = project_path(spec, raw[i_local], x_start, weights, dcs, frozen)
                    row: dict[str, float | int | str] = {
                        "design": design, "seed": seed, "fold": k, "docID": doc2[p],
                        "filerName": name2[p], "assets_start": float(assets[idx2[p]]),
                    }
                    for j, name in enumerate(names):
                        row[f"x0_{name}"] = float(path[0, j])
                        row[f"x1_{name}"] = float(path[-1, j])
                    rows.append(row)
            print(f"  {design:<13} seed={seed} 完了", flush=True)

    out = _IN_DIR / "generated_endpoints.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\n{len(rows)}行を書き出しました: {out}")


if __name__ == "__main__":
    main()
