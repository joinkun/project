#!/usr/bin/env python
"""ステージ16: FMを1回学習し、重み条件×追加制約の全設定で PCFM(毎ステップ射影)を生成する。

torch/cvxpyのプロセス(lightgbm/sklearn/POTをimportしない)。1プロセス=1(計画, seed)。

ステージ15では W3 を「W1条件のFM生出力を終点のみ再射影」した近似で評価していた。
ここではFMを1回だけ学習し(生出力は全設定で共通=FMは重み・制約に依存しない)、各設定の重みと
追加制約でパス(t=0,1)の**各点を射影**する(project_path)。パスはt∈{0,1}の2点なので、
毎ステップ射影=始点(既に実行可能)+終点の射影である。

設定名: ``{W1|W3}_{制約}``。制約は S0(なし) / S5(小計非負+純資産非悪化+残余下限)に
資本金の下限 α(a05/a10)と事業継続の下限 β(b00/b03/b05)を加えたもの。
  例: W1_S5_a05_b00 = W1・α=0.5・β=0 / W3_S5_a10_b03 = W3・α=1.0・β=0.3

出力: {out_dir}/{tag}__seed{seed}.parquet
  行=(config, method, docID)。method ∈ {FM(射影なし生出力), PCFM}。x0_*/x1_*はtilde空間。
使い方:
    uv run python scripts/run_generation_stage16.py --tag nearest_na0 \\
        --plans data/interim/stage16/plans_nearest_na0.npz --seed 0
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import (  # noqa: E402
    ConstraintLayers,
    PlausibilityConfig,
    apply_layers,
    build_linear_constraints,
    load_constraint_layers,
)
from fsgen.accounting.projection_qp import (  # noqa: E402
    load_direction_constraints,
    project_onto_feasible,
)
from fsgen.accounting.subspace import build_reduced_basis  # noqa: E402
from fsgen.errors import InfeasibleProjectionError  # noqa: E402
from fsgen.models.cost import (  # noqa: E402
    combined_projection_weights,
    cost_vectors,
    load_cost_config,
)
from fsgen.models.flow_matching import (  # noqa: E402
    euler_sample_batch,
    seeded_velocity_field,
    train_fm,
)
from fsgen.models.transport import check_plan_health  # noqa: E402

_CANONICAL = Path("conf/accounts/core.yaml")
_COSTS = Path("conf/accounts/costs.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_FOLDS = Path("data/interim/industry_12/folds.npz")
_K = 13
_TS = (0.0, 1.0)
_N_FOLDS = 5
_EPOCHS = 50
_LR = 1e-3
_HIDDEN = 128
_N_LAYERS = 3
_DELTA = 1e-3


def make_configs(
    layers: ConstraintLayers | None = None,
    re_cap: float | None = None,
) -> dict[str, tuple[str, PlausibilityConfig]]:
    """設定名 -> (重み条件, 追加制約)。layers.layer2=False なら全設定が第1層のみ(=S0)になる。"""
    layers = layers or ConstraintLayers()
    out: dict[str, tuple[str, PlausibilityConfig]] = {
        "W1_S0": ("W1", PlausibilityConfig(label="S0")),
    }
    for cond in ("W1", "W3"):
        for alpha, an in ((0.5, "a05"), (1.0, "a10")):
            for beta, bn in ((0.0, "b00"), (0.3, "b03"), (0.5, "b05")):
                name = f"{cond}_S5_{an}_{bn}"
                cfg = PlausibilityConfig(
                    True, True, alpha, None, True, retention_beta=beta or None,
                    contra_net_nonneg=True, label=name,
                )
                out[name] = (cond, apply_layers(cfg, layers))
    if re_cap is not None:  # ステージ27: 利益剰余金の増加の上限(R3 と併用)
        base = out["W1_S5_a05_b03"][1]
        name = f"W1_S5_a05_b03_re{re_cap:g}"
        capped = dataclasses.replace(base, re_growth_cap=re_cap, label=name)
        out[name] = ("W1", apply_layers(capped, layers))
    return out


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


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--costs", type=Path, default=_COSTS, help="コスト重み(ステージ24)")
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--configs", default="", help="設定名のカンマ区切り(省略=全設定)")
    ap.add_argument("--freeze", choices=("all", "no_re"), default="all")
    ap.add_argument("--epochs", type=int, default=_EPOCHS)
    ap.add_argument("--n-folds", type=int, default=_N_FOLDS)
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim/stage16"))
    ap.add_argument("--layers-from", type=Path, default=None,
                    help="constraint_layersを読むcore.yaml(省略時は全層有効)")
    ap.add_argument("--re-cap", type=float, default=None,
                    help="利益剰余金の増加の上限 γ(総資産比、ステージ27)")
    ap.add_argument("--k", type=int, default=_K,
                    help="低次元基底の次元(既定13。ステージ17 F: 感度確認用に14を1回試す)")
    args = ap.parse_args()
    layers = load_constraint_layers(args.layers_from) if args.layers_from else ConstraintLayers()
    torch.set_num_threads(1)

    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(_CANONICAL)
    cost_config = load_cost_config(args.costs)
    frozen = tuple(cost_config.non_actionable)
    if args.freeze == "no_re":
        frozen = tuple(n for n in frozen if n != "jppfs_cor_RetainedEarnings")
    dim = len(names)
    w1 = (np.ones(dim), np.ones(dim))
    w3 = cost_vectors(cost_config, names, non_actionable_mode="keep")
    weights_by_cond = {"W1": w1, "W3": w3}
    configs = make_configs(layers, args.re_cap)
    if args.configs:
        configs = {k: configs[k] for k in args.configs.split(",")}

    doc_ids, filer_names, x, assets, labels = load_dataset(names)
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    x0_tilde, x2_tilde = x_tilde[idx0], x_tilde[idx2]
    doc2 = [doc_ids[i] for i in idx2]
    name2 = [filer_names[i] for i in idx2]
    folds = np.load(_FOLDS)
    plans = np.load(args.plans)
    tag = f"{args.tag}__seed{args.seed}"
    print(f"[{tag}] configs={len(configs)} frozen={frozen}", flush=True)

    rows: list[dict[str, float | int | str]] = []

    def add(config: str, method: str, k: int, p: int, path: np.ndarray, status: str,
            active: str) -> None:
        row: dict[str, float | int | str] = {
            "tag": args.tag, "config": config, "seed": args.seed, "fold": k, "method": method,
            "docID": doc2[p], "filerName": name2[p], "assets_start": float(assets[idx2[p]]),
            "status": status, "active": active,
        }
        for j, name in enumerate(names):
            row[f"x0_{name}"] = float(path[0, j])
            row[f"x1_{name}"] = float(path[-1, j])
        rows.append(row)

    for k in range(args.n_folds):
        t0 = time.perf_counter()
        train_pos, test_pos = folds[f"train_pos_{k}"], folds[f"test_pos_{k}"]
        plan_train, plan_test = plans[f"plan_train_{k}"], plans[f"plan_test_{k}"]
        for plan in (plan_train, plan_test):
            n0, n1 = plan.shape
            check_plan_health(plan, np.full(n0, 1.0 / n0), None, check_marginals=False)
        pool_idx = idx0 + [idx2[p] for p in train_pos]
        b = build_reduced_basis(c, x_tilde[pool_idx], k=args.k)
        z0 = x0_tilde @ b
        z_train, z_test = x2_tilde[train_pos] @ b, x2_tilde[test_pos] @ b
        # ステージ26: 種を設定してからネットワークを作る(seeded_velocity_field)。以前は
        # train_fm の中で初めて種を設定していたため、fold 0 の初期値だけが実行ごとに変わっていた。
        net = seeded_velocity_field(
            args.seed * 1000 + k,
            dim=args.k,
            hidden=_HIDDEN,
            n_layers=_N_LAYERS,
        )
        train_fm(z_train, z0, plan_train, net, epochs=args.epochs, lr=_LR, seed=args.seed,
                 z_val=z_test, plan_val=plan_test)
        z_start = x2_tilde[test_pos] @ b
        z_path = euler_sample_batch(net, z_start, _TS)
        raw = x2_tilde[test_pos][:, None, :] + np.einsum(
            "dk,ntk->ntd", b, z_path - z_start[:, None, :]
        )
        for i_local, p in enumerate(test_pos):
            x_start = x2_tilde[p]
            add("raw", "FM", k, int(p), raw[i_local], "ok", "")
            for cname, (cond, cfg) in configs.items():
                wp, wm = weights_by_cond[cond]
                weights = combined_projection_weights(x_start, 1.0, wp, wm, _DELTA)
                extra = build_linear_constraints(spec, names, x_start, cfg)
                out = raw[i_local].copy()
                status, active = "ok", ""
                try:
                    for t_i in range(out.shape[0]):  # 毎ステップ射影(t=0,1)
                        out[t_i], info = project_onto_feasible(
                            raw[i_local][t_i], spec, x_start=x_start, weights=weights,
                            direction_constraints=dcs, frozen_accounts=frozen,
                            extra_constraints=extra,
                        )
                    active = ";".join(info.active_constraints)
                except InfeasibleProjectionError:
                    out = np.repeat(x_start[None, :], out.shape[0], axis=0)  # 動かせない
                    status = "infeasible"
                add(cname, "PCFM", k, int(p), out, status, active)
        print(f"  [{tag}] fold {k + 1}/{args.n_folds} 完了({time.perf_counter() - t0:.0f}s)",
              flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{tag}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[{tag}] {len(rows)}行を書き出しました: {out_path}", flush=True)


if __name__ == "__main__":
    main()
