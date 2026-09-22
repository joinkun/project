#!/usr/bin/env python
"""ステージ13: 収束版の輸送計画で FM / PCFM / CAFM を学習・生成する(torch専用プロセス)。

プロセス分離(ステージ11-A)を守る: lightgbm/sklearn/ot(POT)をimportしない。
輸送計画はprecompute_stage13_transport.py(POT側)かprecompute_industry_transport.py
が書いたnpzを読み、読み込み時に健全性チェック(§A.2)を行う。

1プロセス = 1(計画, 重み条件, seed)。5-foldを順に回す。並列化は呼び出し側
(シェルで複数プロセスを同時に起動)で行う。

CAFMは学習前に、微分可能射影層の解析勾配を数値勾配と照合する(§A.1、
ステージ10.5の勾配破綻の再発防止)。

出力: data/interim/stage13/{tag}__{condition}__seed{seed}.parquet
  行=(method, docID)。method ∈ {FM(射影なしの生出力), PCFM, CAFM_scratch, CAFM_warm}
  x0_*/x1_* は総資産で正規化済み(tilde空間)。

使い方:
    uv run python scripts/run_generation_stage13.py --tag balanced \\
        --plans data/interim/industry_12/plans_current.npz --condition W1_uniform --seed 0
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fsgen.accounting.constraints import (  # noqa: E402
    CanonicalSpec,
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.projection_qp import (  # noqa: E402
    DirectionConstraintSpec,
    load_direction_constraints,
    project_onto_feasible,
)
from fsgen.accounting.subspace import build_reduced_basis  # noqa: E402
from fsgen.models.cost import (  # noqa: E402
    combined_projection_weights,
    cost_vectors,
    load_cost_config,
)
from fsgen.models.differentiable_projection import (  # noqa: E402
    build_projection_layer,
    check_layer_gradient,
)
from fsgen.models.flow_matching import (  # noqa: E402
    VelocityField,
    euler_sample_batch,
    train_cafm,
    train_fm,
)
from fsgen.models.transport import check_plan_health  # noqa: E402

_CANONICAL = Path("conf/accounts/core.yaml")
_COSTS = Path("conf/accounts/costs.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_FOLDS = Path("data/interim/industry_12/folds.npz")
_OUT_DIR = Path("data/interim/stage13")

_K = 13
_TS = (0.0, 1.0)
_N_FOLDS = 5
_EPOCHS = 50
_LR = 1e-3
_HIDDEN = 128
_N_LAYERS = 3
_ALL_METHODS = ("FM", "PCFM", "CAFM_scratch", "CAFM_warm")


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


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--condition", choices=("W1_uniform", "W3_eq_only"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--methods", default=",".join(_ALL_METHODS))
    ap.add_argument("--freeze", choices=("all", "no_re"), default="all",
                    help="all=RE含め非操作科目を固定(R1) / no_re=REの固定を外す(R3)")
    ap.add_argument("--check-marginals", type=int, default=1,
                    help="1=周辺分布(一様)の誤差を検証(バランスOT) / 0=行質量のみ検証")
    ap.add_argument("--allow-dropped-sources", type=int, default=0,
                    help="1=不均衡OTで質量0になった始点を許し、FM学習から除外する(件数を表示)")
    ap.add_argument("--epochs", type=int, default=_EPOCHS)
    ap.add_argument("--n-folds", type=int, default=_N_FOLDS, help="スモークテスト用(既定=全5fold)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    methods = tuple(args.methods.split(","))

    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(_CANONICAL)
    cost_config = load_cost_config(_COSTS)
    frozen = tuple(cost_config.non_actionable)
    if args.freeze == "no_re":
        frozen = tuple(n for n in frozen if n != "jppfs_cor_RetainedEarnings")
    dim = len(names)
    if args.condition == "W1_uniform":
        w_plus, w_minus = np.ones(dim), np.ones(dim)
    else:
        w_plus, w_minus = cost_vectors(cost_config, names, non_actionable_mode="keep")
    delta = 1e-3

    doc_ids, filer_names, x, assets, labels = load_dataset(names)
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    x0_tilde, x2_tilde = x_tilde[idx0], x_tilde[idx2]
    doc2 = [doc_ids[i] for i in idx2]
    name2 = [filer_names[i] for i in idx2]
    folds = np.load(_FOLDS)
    plans = np.load(args.plans)
    rng_check = np.random.default_rng(args.seed)
    tag = f"{args.tag}__{args.condition}__seed{args.seed}"
    print(f"[{tag}] methods={methods} frozen={frozen}", flush=True)

    rows: list[dict[str, float | int | str]] = []

    def add(method: str, k: int, p: int, path: np.ndarray) -> None:
        row: dict[str, float | int | str] = {
            "tag": args.tag, "condition": args.condition, "seed": args.seed, "fold": k,
            "method": method, "docID": doc2[p], "filerName": name2[p],
            "assets_start": float(assets[idx2[p]]),
        }
        for j, name in enumerate(names):
            row[f"x0_{name}"] = float(path[0, j])
            row[f"x1_{name}"] = float(path[-1, j])
        rows.append(row)

    for k in range(args.n_folds):
        t0 = time.perf_counter()
        train_pos, test_pos = folds[f"train_pos_{k}"], folds[f"test_pos_{k}"]
        plan_train, plan_test = plans[f"plan_train_{k}"], plans[f"plan_test_{k}"]
        for plan in (plan_train, plan_test):  # §A.2: 健全性チェックを通った計画だけを使う
            n0, n1 = plan.shape
            a = np.full(n0, 1.0 / n0)
            b_marg = np.full(n1, 1.0 / n1)
            check_plan_health(
                plan, a, b_marg if args.check_marginals else None,
                check_marginals=bool(args.check_marginals),
                allow_zero_mass_rows=bool(args.allow_dropped_sources),
            )
        alive_train = plan_train.sum(axis=1) > 1e-12
        if not alive_train.all():
            n_drop = int((~alive_train).sum())
            print(f"    fold{k}: 質量0の訓練始点{n_drop}社をFM学習から除外", flush=True)
        pool_idx = idx0 + [idx2[p] for p in train_pos]
        b = build_reduced_basis(c, x_tilde[pool_idx], k=_K)
        z0 = x0_tilde @ b
        z_train = (x2_tilde[train_pos] @ b)[alive_train]
        z_test = x2_tilde[test_pos] @ b
        plan_train = plan_train[alive_train]
        train_pos_fit = train_pos[alive_train]  # 学習に使う始点(質量0の始点を除く)

        net_fm = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
        train_fm(
            z_train, z0, plan_train, net_fm, epochs=args.epochs, lr=_LR, seed=args.seed,
            z_val=z_test, plan_val=plan_test,
        )
        paths: dict[str, np.ndarray] = {"FM": sample_and_convert(net_fm, x2_tilde, test_pos, b)}

        if any(m.startswith("CAFM") for m in methods):
            layers = []
            for p in train_pos_fit:
                w_p = combined_projection_weights(x2_tilde[p], 1.0, w_plus, w_minus, delta)
                layers.append(build_projection_layer(spec, x2_tilde[p], w_p, dcs, frozen))
            for li in range(2):  # §A.1: 学習前に解析勾配を数値勾配と照合(破綻していれば例外)
                err = check_layer_gradient(layers[li], x2_tilde[train_pos_fit[li]], rng_check)
                print(f"    勾配チェック(fold{k}, 社{li}): 相対誤差中央値={err:.2e}", flush=True)
            if "CAFM_scratch" in methods:
                net_s = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
                res_s = train_cafm(
                    z_train, z0, plan_train, net_s, layers, x2_tilde[train_pos_fit], b,
                    epochs=args.epochs, lr=_LR, seed=args.seed, z_val=z_test, plan_val=plan_test,
                )
                paths["CAFM_scratch"] = sample_and_convert(net_s, x2_tilde, test_pos, b)
                print(f"    CAFM_scratch 射影失敗={res_s.solver_failures}", flush=True)
            if "CAFM_warm" in methods:
                net_w = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
                net_w.load_state_dict(net_fm.state_dict())
                res_w = train_cafm(
                    z_train, z0, plan_train, net_w, layers, x2_tilde[train_pos_fit], b,
                    epochs=args.epochs, lr=_LR, seed=args.seed, z_val=z_test, plan_val=plan_test,
                )
                paths["CAFM_warm"] = sample_and_convert(net_w, x2_tilde, test_pos, b)
                print(f"    CAFM_warm 射影失敗={res_w.solver_failures}", flush=True)

        for i_local, p in enumerate(test_pos):
            x_start = x2_tilde[p]
            weights = combined_projection_weights(x_start, 1.0, w_plus, w_minus, delta)
            add("FM", k, int(p), paths["FM"][i_local])
            for method, raw_paths in paths.items():
                name_out = "PCFM" if method == "FM" else method
                if name_out not in methods:
                    continue
                add(name_out, k, int(p),
                    project_path(spec, raw_paths[i_local], x_start, weights, dcs, frozen))
        elapsed = time.perf_counter() - t0
        print(f"  [{tag}] fold {k + 1}/{args.n_folds} 完了({elapsed:.0f}s)", flush=True)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / (f"{tag}.parquet" if args.n_folds == _N_FOLDS else f"SMOKE_{tag}.parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[{tag}] {len(rows)}行を書き出しました: {out}", flush=True)


if __name__ == "__main__":
    main()
