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
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    orthonormal_basis,
    state_vector_names,
)
from fsgen.accounting.plausibility import (  # noqa: E402
    INACTIVE_ACCOUNTS,
    ConstraintLayers,
    PlausibilityConfig,
    apply_layers,
    build_linear_constraints,
    decompose_re_change,
    inactive_accounts,
    load_constraint_layers,
    re_decomposition_branches,
)
from fsgen.accounting.projection_qp import (  # noqa: E402
    load_direction_constraints,
    project_onto_feasible,
    project_onto_union,
)
from fsgen.accounting.subspace import build_reduced_basis  # noqa: E402
from fsgen.datasets.exclusion import (  # noqa: E402
    exclusion_masks,
    load_excluded_docs,
    subset_folds,
)
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




def reduced_basis(c: np.ndarray, x: np.ndarray, k: int, kind: str) -> np.ndarray:
    """低次元基底(ステージ44 A)。pca: null(C^T) 内の PCA 上位 k(現行)、null: null(C^T) の
    正規直交基底 U そのもの(C の完全 SVD の、rank を超える左特異ベクトル。PCA を使わない)。"""
    if kind == "pca":
        return build_reduced_basis(c, x, k=k)
    if kind == "null":
        u: np.ndarray = orthonormal_basis(c)
        return u
    raise ValueError(kind)

def inactive_targets(arg: str, names: tuple[str, ...]) -> tuple[str, ...]:
    """``--inactive`` の値 → 不活性制約の対象科目(ステージ41)。

    none: なし、selected: INACTIVE_ACCOUNTS(実データで選んだ科目)、all: 全基礎科目、
    それ以外: カンマ区切りの科目名。
    """
    if arg == "none":
        return ()
    if arg == "selected":
        return INACTIVE_ACCOUNTS
    if arg == "all":
        return names
    accs = tuple(a for a in arg.split(",") if a)
    unknown = sorted(set(accs) - set(names))
    if unknown:
        raise ValueError(f"基礎科目にない科目: {unknown}")
    return accs

def make_configs(
    layers: ConstraintLayers | None = None,
    re_cap: float | None = None,
    re_decomp: tuple[float, ...] = (),
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
    if re_cap is not None:  # ステージ44 B・C: 最終設定から制約を 1 つずつ外す・変える版
        final = out[f"W1_S5_a05_b03_re{re_cap:g}"][1]
        variants = {
            "s44_no_alpha": {"capital_min_ratio": None},
            "s44_no_beta": {"retention_beta": None},
            "s44_no_cap": {"re_growth_cap": None},
            "s44_no_all": {"capital_min_ratio": None, "retention_beta": None,
                           "re_growth_cap": None},
            "s44_alpha0": {"capital_min_ratio": 0.0},
            "s44_alpha03": {"capital_min_ratio": 0.3},
            "s44_alpha10": {"capital_min_ratio": 1.0},
            "s44_beta05": {"retention_beta": 0.5},
            "s44_beta10": {"retention_beta": 1.0},
            "s44_tscap": {"treasury_cap": True},
        }
        for name, changes in variants.items():
            v = dataclasses.replace(final, label=name, **changes)  # type: ignore[arg-type]
            out[name] = ("W1", apply_layers(v, layers))
    for gamma in re_decomp:  # ステージ29 R4: 欠損填補の振替と利益部分(上限 γ)の分離
        base = out["W1_S5_a05_b03"][1]
        name = f"W1_S5_a05_b03_r4g{gamma:g}"
        r4 = dataclasses.replace(base, re_decomposition_gamma=gamma, label=name)
        out[name] = ("W1", apply_layers(r4, layers))
    return out


def load_dataset(
    names: tuple[str, ...],
    excluded: frozenset[str] = frozenset(),
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[int], np.ndarray]:
    """最後の要素は、除外前の class 2 の並びで残す始点(ステージ28 の除外、fold の絞り込み用)。"""
    agg = pd.read_csv(_AGGREGATED_CSV, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS_CSV, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "filerName", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    row_keep, src_keep = exclusion_masks(
        m["docID"].tolist(), m["label"].astype(int).tolist(), excluded
    )
    m = m[row_keep]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = m["total_assets"].to_numpy(dtype=float)
    return (
        m["docID"].tolist(), m["filerName"].tolist(), x, assets, m["label"].astype(int).tolist(),
        src_keep,
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
    ap.add_argument("--exclude", type=Path, default=None,
                    help="除外する書類の CSV(列 docID、ステージ28)")
    ap.add_argument("--layers-from", type=Path, default=None,
                    help="constraint_layersを読むcore.yaml(省略時は全層有効)")
    ap.add_argument("--re-cap", type=float, default=None,
                    help="利益剰余金の増加の上限 γ(総資産比、ステージ27)")
    ap.add_argument("--re-decomp", default="",
                    help="R4 の利益部分の上限 γ のカンマ区切り(ステージ29。空なら作らない)")
    ap.add_argument("--basis", choices=("pca", "null"), default="pca",
                    help="pca = null(C^T) 内の PCA 上位 k(現行)、null = null(C^T) の"
                         "正規直交基底そのもの(PCA なし、31 次元。ステージ44)")
    ap.add_argument("--loss-log", action="store_true",
                    help="fold ごとの損失を JSON に書く(ステージ44)")
    ap.add_argument("--no-direction", action="store_true",
                    help="方向制約を外す(自己株式の上限への統合の確認、ステージ44)")
    ap.add_argument("--inactive", default="none",
                    help="不活性制約: none / selected / all / カンマ区切りの科目(ステージ41)")
    ap.add_argument("--allow-dropped-sources", action="store_true",
                    help="不均衡 OT で質量 0 になった訓練始点を許し、学習から除く(ステージ39)")
    ap.add_argument("--k", type=int, default=_K,
                    help="低次元基底の次元(既定13。ステージ17 F: 感度確認用に14を1回試す)")
    args = ap.parse_args()
    layers = load_constraint_layers(args.layers_from) if args.layers_from else ConstraintLayers()
    torch.set_num_threads(1)

    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    # ステージ44 C.6: --no-direction で方向制約(債務超過企業の自社株買い禁止)を外す(統合の確認用)
    dcs = () if args.no_direction else load_direction_constraints(_CANONICAL)
    cost_config = load_cost_config(args.costs)
    frozen = tuple(cost_config.non_actionable)
    if args.freeze == "no_re":
        frozen = tuple(n for n in frozen if n != "jppfs_cor_RetainedEarnings")
    dim = len(names)
    targets = inactive_targets(args.inactive, names)
    w1 = (np.ones(dim), np.ones(dim))
    w3 = cost_vectors(cost_config, names, non_actionable_mode="keep")
    weights_by_cond = {"W1": w1, "W3": w3}
    re_decomp = tuple(float(g) for g in args.re_decomp.split(",") if g)
    configs = make_configs(layers, args.re_cap, re_decomp)
    if args.configs:
        configs = {k: configs[k] for k in args.configs.split(",")}

    doc_ids, filer_names, x, assets, labels, src_keep = load_dataset(
        names, load_excluded_docs(args.exclude)
    )
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    x0_tilde, x2_tilde = x_tilde[idx0], x_tilde[idx2]
    doc2 = [doc_ids[i] for i in idx2]
    name2 = [filer_names[i] for i in idx2]
    folds = subset_folds(np.load(_FOLDS), src_keep)  # 除外なしなら元の分割のまま
    plans = np.load(args.plans)
    tag = f"{args.tag}__seed{args.seed}"
    print(f"[{tag}] configs={len(configs)} frozen={frozen}", flush=True)

    rows: list[dict[str, float | int | str]] = []
    loss_log: dict[str, dict[str, object]] = {}

    def add(config: str, method: str, k: int, p: int, path: np.ndarray, status: str,
            active: str, re_branch: int = -1, n_inactive: int = 0) -> None:
        # ステージ29: RE の変化の分解(振替・利益部分、総資産(0)比)。出力の状態ベクトルには含めない
        transfer, profit = decompose_re_change(path[0], path[-1], names)
        row: dict[str, float | int | str] = {
            "tag": args.tag, "config": config, "seed": args.seed, "fold": k, "method": method,
            "docID": doc2[p], "filerName": name2[p], "assets_start": float(assets[idx2[p]]),
            "status": status, "active": active, "re_branch": re_branch,
            "re_transfer": float(transfer), "re_profit": float(profit), "n_inactive": n_inactive,
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
            check_plan_health(plan, np.full(n0, 1.0 / n0), None, check_marginals=False,
                              allow_zero_mass_rows=args.allow_dropped_sources)
        # ステージ39: 不均衡 OT で質量 0 になった訓練始点は FM の学習から除く(ステージ13 と同じ)。
        # テストの始点は質量によらず全社を生成する
        alive = plan_train.sum(axis=1) > 1e-12
        if not alive.all():
            print(f"    [{tag}] fold {k}: 質量 0 の訓練始点 {int((~alive).sum())} 社を学習から除く",
                  flush=True)
        pool_idx = idx0 + [idx2[p] for p in train_pos]
        b = reduced_basis(c, x_tilde[pool_idx], args.k, args.basis)
        z0 = x0_tilde @ b
        z_train, z_test = x2_tilde[train_pos] @ b, x2_tilde[test_pos] @ b
        # ステージ26: 種を設定してからネットワークを作る(seeded_velocity_field)。以前は
        # train_fm の中で初めて種を設定していたため、fold 0 の初期値だけが実行ごとに変わっていた。
        net = seeded_velocity_field(
            args.seed * 1000 + k,
            dim=b.shape[1],
            hidden=_HIDDEN,
            n_layers=_N_LAYERS,
        )
        tr = train_fm(z_train[alive], z0, plan_train[alive], net, epochs=args.epochs, lr=_LR,
                      seed=args.seed, z_val=z_test, plan_val=plan_test)
        loss_log[str(k)] = {"train": tr.train_losses, "val": tr.val_losses,
                            "dim": int(b.shape[1])}
        z_start = x2_tilde[test_pos] @ b
        z_path = euler_sample_batch(net, z_start, _TS)
        raw = x2_tilde[test_pos][:, None, :] + np.einsum(
            "dk,ntk->ntd", b, z_path - z_start[:, None, :]
        )
        for i_local, p in enumerate(test_pos):
            x_start = x2_tilde[p]
            add("raw", "FM", k, int(p), raw[i_local], "ok", "")
            # ステージ41: 始点で 0 の対象科目を経路上でも始点の値に固定する(不活性制約)
            inactive = inactive_accounts(x_start, names, targets)
            frozen_i = frozen + tuple(a for a in inactive if a not in frozen)
            for cname, (cond, cfg) in configs.items():
                wp, wm = weights_by_cond[cond]
                weights = combined_projection_weights(x_start, 1.0, wp, wm, _DELTA)
                extra = build_linear_constraints(spec, names, x_start, cfg)
                out = raw[i_local].copy()
                status, active, branch = "ok", "", -1
                gamma = cfg.re_decomposition_gamma
                branches = (
                    re_decomposition_branches(spec, names, x_start, gamma)
                    if gamma is not None else None
                )
                try:
                    for t_i in range(out.shape[0]):  # 毎ステップ射影(t=0,1)
                        if branches is None:
                            out[t_i], info = project_onto_feasible(
                                raw[i_local][t_i], spec, x_start=x_start, weights=weights,
                                direction_constraints=dcs, frozen_accounts=frozen_i,
                                extra_constraints=extra,
                            )
                        else:  # R4: 2 つの分岐の和集合への射影(近い方の分岐を採る)
                            out[t_i], info, branch = project_onto_union(
                                raw[i_local][t_i], spec, x_start, branches, weights=weights,
                                extra_constraints=extra, direction_constraints=dcs,
                                frozen_accounts=frozen_i,
                            )
                    active = ";".join(info.active_constraints)
                except InfeasibleProjectionError:
                    out = np.repeat(x_start[None, :], out.shape[0], axis=0)  # 動かせない
                    status = "infeasible"
                add(cname, "PCFM", k, int(p), out, status, active, branch, len(inactive))
        print(f"  [{tag}] fold {k + 1}/{args.n_folds} 完了({time.perf_counter() - t0:.0f}s)",
              flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{tag}.parquet"
    if args.loss_log:  # ステージ44 A: fold ごとの訓練・検証損失(epoch ごと)
        (args.out_dir / f"{tag}__losses.json").write_text(json.dumps(loss_log), encoding="utf-8")
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[{tag}] {len(rows)}行を書き出しました: {out_path}", flush=True)


if __name__ == "__main__":
    main()
