#!/usr/bin/env python
"""ステージ38: 遷移を分けた生成と重みの作用点(torch/cvxpy のプロセス)。

遷移は T1: class 2 → class 1、T2: class 1 → class 0。

作用点の組み合わせ(重みは ``costs_persistence.yaml``、増減で対称):

| setting | 輸送 | 射影 | 速度場 |
|---|---|---|---|
| 1 | 等重み | 一様 | 一様 |
| 2 | 重み | 一様 | 一様 |
| 3 | 重み | 重み | 一様 |
| 4 | 重み | 重み | 重み |
| 5 | 等重み | 重み | 重み |

- 射影の重み: 一様 = 最終設定の W1(1/(|x_start| + δ))、
  重み = c / (|x_start| + δ)(W3 と同じ形)
- 速度場の重み: 損失を e^T M e、M = B^T diag(c / mean(c)) B(元空間で重み付けした誤差と同じ)
- 制約は最終設定(第 2 層、α = 0.5、β = 0.3、R3 + 上限 γ = 0.757)。
  上限と第 2 層は各遷移の始点に対して
- ``--mode folds``: 始点の 5-fold(T1 は ``folds.npz``、T2 は ``folds_class1.npz``)
- ``--mode chain``: T2 の速度場を class 1 の全社で学習し、``--start-from``(T1 の出力)の終点に当てる

数値条件の記録(``diag_*.json``): 射影の重み比の最大、ステージ10.6 の最適性の検定
(f_W(x_W) ≤ f_W(x_U) かつ f_U(x_U) ≤ f_U(x_W))、速度場の損失の勾配の数値検証(中心差分)、
射影が実行不可能になった件数。

    uv run python scripts/run_generation_stage38.py --transition T1 --setting 1 --seed 0
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from run_generation_stage16 import (  # noqa: E402
    _CANONICAL,
    _DELTA,
    _EPOCHS,
    _HIDDEN,
    _K,
    _LR,
    _N_LAYERS,
    _TS,
    load_dataset,
    make_configs,
)

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import (  # noqa: E402
    ConstraintLayers,
    build_linear_constraints,
    decompose_re_change,
)
from fsgen.accounting.projection_qp import (  # noqa: E402
    load_direction_constraints,
    project_onto_feasible,
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
    VelocityField,
    euler_sample_batch,
    seeded_velocity_field,
    train_fm,
)
from fsgen.models.transport import check_plan_health  # noqa: E402

_GAMMA = 0.7565192502545388
_CONFIG = "W1_S5_a05_b03_re0.756519"
SETTINGS = {  # setting: (輸送の重み, 射影に重み, 速度場に重み)
    1: ("equal", False, False),
    2: ("persistence", False, False),
    3: ("persistence", True, False),
    4: ("persistence", True, True),
    5: ("equal", True, True),
}
TRANSITIONS = {"T1": (2, 1, Path("data/interim/industry_12/folds.npz")),
               "T2": (1, 0, Path("data/interim/stage38/folds_class1.npz"))}
_WEIGHT_COSTS = Path("conf/accounts/costs_persistence.yaml")
_OPT_RTOL = 1e-6


def grad_check(net: VelocityField, z: np.ndarray, u: np.ndarray, metric: np.ndarray,
               n_params: int = 20, eps: float = 1e-6) -> float:
    """重み付き損失 mean(e^T M e) の解析勾配と中心差分の相対誤差の最大(先頭の n_params 個)。"""
    zt = torch.tensor(z, dtype=torch.float64)
    ut = torch.tensor(u, dtype=torch.float64)
    tt = torch.full((len(z),), 0.5, dtype=torch.float64)
    mt = torch.tensor(metric, dtype=torch.float64)

    def loss() -> torch.Tensor:
        e = net(zt, tt) - ut
        return torch.mean(torch.einsum("ni,ij,nj->n", e, mt, e))

    params = [p for p in net.parameters()]
    net.zero_grad()
    loss().backward()  # type: ignore[no-untyped-call]
    worst = 0.0
    count = 0
    for p in params:
        assert p.grad is not None
        flat, gflat = p.data.view(-1), p.grad.view(-1)
        for i in range(min(flat.numel(), n_params - count)):
            orig = float(flat[i])
            with torch.no_grad():
                flat[i] = orig + eps
                lp = float(loss())
                flat[i] = orig - eps
                lm = float(loss())
                flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(gflat[i])
            worst = max(worst, abs(num - ana) / max(abs(num), abs(ana), 1e-8))
            count += 1
        if count >= n_params:
            break
    return worst


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transition", choices=tuple(TRANSITIONS), required=True)
    ap.add_argument("--setting", type=int, choices=tuple(SETTINGS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=("folds", "chain"), default="folds")
    ap.add_argument("--start-from", type=Path, default=None, help="chain: T1 の出力の parquet")
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage38/excluded.csv"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    src_label, tgt_label, folds_path = TRANSITIONS[args.transition]
    transport, proj_w, vel_w = SETTINGS[args.setting]
    plans = np.load(Path(f"data/interim/stage38/plans_{args.transition}_{transport}")
                    / "plans_nearest_na0.npz")

    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(_CANONICAL)
    frozen = tuple(n for n in load_cost_config(Path("conf/accounts/costs_equal.yaml"))
                   .non_actionable if n != "jppfs_cor_RetainedEarnings")  # R3
    dim = len(names)
    _cond, cfg = make_configs(ConstraintLayers(), _GAMMA)[_CONFIG]
    cw_plus, cw_minus = cost_vectors(load_cost_config(_WEIGHT_COSTS), names,
                                     non_actionable_mode="keep")
    cvec = np.maximum(cw_plus, cw_minus)
    ones = np.ones(dim)

    doc_all, filer_all, x_all, a_all, lab_all, _k = load_dataset(names)
    excluded = load_excluded_docs(args.exclude)
    row_keep, src_keep = exclusion_masks(doc_all, lab_all, excluded, src_label)
    doc = [d for d, k in zip(doc_all, row_keep, strict=True) if k]
    filer = [f for f, k in zip(filer_all, row_keep, strict=True) if k]
    x, assets = x_all[row_keep], a_all[row_keep]
    labels = [lab for lab, k in zip(lab_all, row_keep, strict=True) if k]
    x_tilde = x / assets[:, None]
    idx_t = [i for i, lab in enumerate(labels) if lab == tgt_label]
    idx_s = [i for i, lab in enumerate(labels) if lab == src_label]
    xt, xs = x_tilde[idx_t], x_tilde[idx_s]
    folds = subset_folds(np.load(folds_path), src_keep)

    diag: dict[str, object] = {"transition": args.transition, "setting": args.setting,
                               "seed": args.seed, "mode": args.mode, "n_sources": len(idx_s),
                               "n_targets": len(idx_t), "weight_ratio_cost": float(
                                   cvec.max() / cvec.min())}
    ratios: list[float] = []
    opt_viol = 0
    opt_checked = 0
    opt_worst = 0.0
    n_infeasible = 0
    grad_errs: list[float] = []
    rows: list[dict[str, float | int | str]] = []

    def train(train_s: np.ndarray, plan: np.ndarray, b: np.ndarray, seed_off: int,
              z_val: np.ndarray | None, plan_val: np.ndarray | None) -> VelocityField:
        check_plan_health(plan, np.full(len(plan), 1.0 / len(plan)), None,
                          check_marginals=False)
        net = seeded_velocity_field(args.seed * 1000 + seed_off, dim=_K, hidden=_HIDDEN,
                                    n_layers=_N_LAYERS)
        metric = None
        if vel_w:
            w = cvec / cvec.mean()
            metric = b.T @ np.diag(w) @ b
        train_fm(train_s @ b, xt @ b, plan, net, epochs=_EPOCHS, lr=_LR, seed=args.seed,
                 z_val=z_val, plan_val=plan_val, loss_metric=metric)
        if metric is not None and not grad_errs:
            rng = np.random.default_rng(args.seed)
            zz = (train_s @ b)[:8]
            uu = rng.normal(size=zz.shape)
            grad_errs.append(grad_check(net, zz, uu, metric))
        return net

    def project(raw: np.ndarray, x_start: np.ndarray) -> tuple[np.ndarray, str, str]:
        nonlocal opt_viol, opt_checked, opt_worst, n_infeasible
        w_u = combined_projection_weights(x_start, 1.0, ones, ones, _DELTA)
        w_c = combined_projection_weights(x_start, 1.0, cvec, cvec, _DELTA)
        w_use = w_c if proj_w else w_u
        ratios.append(float(w_use.max() / w_use.min()))
        extra = build_linear_constraints(spec, names, x_start, cfg,
                                         relax_residual_start=args.mode == "chain")
        out = raw.copy()
        active = ""
        try:
            for t_i in range(out.shape[0]):
                out[t_i], info = project_onto_feasible(
                    raw[t_i], spec, x_start=x_start, weights=w_use, direction_constraints=dcs,
                    frozen_accounts=frozen, extra_constraints=extra)
                active = ";".join(info.active_constraints)
            if proj_w:  # ステージ10.6 の最適性の検定(t = 1)
                y = raw[-1]
                x_u, _ = project_onto_feasible(
                    y, spec, x_start=x_start, weights=w_u, direction_constraints=dcs,
                    frozen_accounts=frozen, extra_constraints=extra)
                x_w = out[-1]
                fw_w, fw_u = np.sum(w_c * (x_w - y) ** 2), np.sum(w_c * (x_u - y) ** 2)
                fu_u, fu_w = np.sum(w_u * (x_u - y) ** 2), np.sum(w_u * (x_w - y) ** 2)
                opt_checked += 1
                ex1 = (fw_w - fw_u) / max(fw_u, 1e-300)
                ex2 = (fu_u - fu_w) / max(fu_w, 1e-300)
                opt_worst = max(opt_worst, ex1, ex2)
                if ex1 > _OPT_RTOL or ex2 > _OPT_RTOL:
                    opt_viol += 1
            return out, "ok", active
        except InfeasibleProjectionError:
            n_infeasible += 1
            return np.repeat(x_start[None, :], raw.shape[0], axis=0), "infeasible", ""

    def add(k: int, i: int, path: np.ndarray, status: str, active: str, method: str,
            x_orig: np.ndarray | None = None, extra: dict[str, float | int | str] | None = None
            ) -> None:
        x0 = path[0] if x_orig is None else x_orig
        transfer, profit = decompose_re_change(x0, path[-1], names)
        row: dict[str, float | int | str] = {
            "tag": f"{args.transition}_s{args.setting}", "config": f"s{args.setting}",
            "seed": args.seed, "fold": k, "method": method, "docID": doc[i],
            "filerName": filer[i], "assets_start": float(assets[i]), "status": status,
            "active": active, "re_transfer": float(transfer), "re_profit": float(profit),
        }
        for j, n in enumerate(names):
            row[f"x0_{n}"] = float(x0[j])
            row[f"x1_{n}"] = float(path[-1, j])
            if x_orig is not None:
                row[f"xm_{n}"] = float(path[0, j])
        if extra:
            row.update(extra)
        rows.append(row)

    if args.mode == "folds":
        for k in range(5):
            tr, te = folds[f"train_pos_{k}"], folds[f"test_pos_{k}"]
            if len(te) == 0:
                continue
            b = build_reduced_basis(c, x_tilde[idx_t + [idx_s[p] for p in tr]], k=_K)
            net = train(xs[tr], plans[f"plan_train_{k}"], b, k, xs[te] @ b,
                        plans[f"plan_test_{k}"])
            z0 = xs[te] @ b
            zp = euler_sample_batch(net, z0, _TS)
            raw = xs[te][:, None, :] + np.einsum("dk,ntk->ntd", b, zp - z0[:, None, :])
            for j, p in enumerate(te):
                out, st, act = project(raw[j], xs[p])
                add(k, idx_s[p], raw[j], "ok", "", "FM")
                add(k, idx_s[p], out, st, act, "PCFM")
    else:  # chain: T2 を class 1 の全社で学習し、T1 の終点に当てる
        if args.start_from is None or args.transition != "T2":
            raise ValueError("chain は --transition T2 と --start-from(T1 の出力)で使う")
        t1 = pd.read_parquet(args.start_from)
        t1 = t1[t1["method"] == "PCFM"].reset_index(drop=True)
        b = build_reduced_basis(c, x_tilde[idx_t + idx_s], k=_K)
        net = train(xs, plans["plan_full"], b, 5, None, None)
        mids = t1[[f"x1_{n}" for n in names]].to_numpy()
        origs = t1[[f"x0_{n}" for n in names]].to_numpy()
        z0 = mids @ b
        zp = euler_sample_batch(net, z0, _TS)
        raw = mids[:, None, :] + np.einsum("dk,ntk->ntd", b, zp - z0[:, None, :])
        doc_index = {d: i for i, d in enumerate(doc)}
        for j in range(len(t1)):
            i = doc_index[str(t1.at[j, "docID"])]
            out, st, act = project(raw[j], mids[j])
            add(int(t1.at[j, "fold"]), i, out, st, act, "PCFM", x_orig=origs[j],
                extra={"t1_status": str(t1.at[j, "status"])})

    diag.update({
        "proj_weight_ratio_max": float(max(ratios)) if ratios else None,
        "proj_weight_ratio_median": float(np.median(ratios)) if ratios else None,
        "optimality_checked": opt_checked, "optimality_violations": opt_viol,
        "optimality_worst_rel_excess": opt_worst, "infeasible": n_infeasible,
        "grad_check_max_rel_err": grad_errs[0] if grad_errs else None,
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out_dir / f"nearest_na0__seed{args.seed}.parquet",
                                  index=False)
    (args.out_dir / f"diag_seed{args.seed}.json").write_text(json.dumps(diag, indent=1),
                                                              encoding="utf-8")
    print(json.dumps(diag, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
