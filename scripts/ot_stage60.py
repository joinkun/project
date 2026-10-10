#!/usr/bin/env python
"""ステージ60 §0・A・B・C: 輸送計画の台と周辺分布(POT のプロセス。torch・sklearn を import しない)。

規則は ``results/stage60/criteria.md``(輸送・生成の前に記録)。コストは非対称 L1(中央値の尺度、
ステージ59)、C̃ = C / median(C)。すべて対数領域で解く。未収束などは例外になり、その設定は表に
「失敗」と理由を書いて QP に進めない(握りつぶさない)。

出力: ``results/stage60/plans_{c2,c1}.csv``(診断値)、``results/stage60/section0.md``、
``data/interim/stage60/targets_{c2,c1}.npz``(各設定のバリセントリック・最大質量の目標点)。

    uv run python scripts/ot_stage60.py --cls c2
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import ot
import pandas as pd
from density_stage51 import load_tilde
from prep_stage59 import OUTLIERS, deltas, l1_cost, l2_cost
from sampling_stage60 import stratified
from scales_stage59 import DEFAULT, SCALES, scale_table

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.errors import SolverNotConvergedError
from fsgen.models.cost import load_cost_config
from fsgen.models.industry import load_industry_config
from fsgen.models.transport import (
    effective_support_sizes,
    solve_ot_normalized,
    solve_semi_unbalanced_normalized,
)

OUT = Path("results/stage60")
DATA = Path("data/interim/stage60")
EPS = (0.005, 0.02, 0.1)
TAU1 = (0.001, 0.01, 0.1, 1.0, 10.0)
N1S = (110, 220, 550, 1100)
SEEDS = (0, 1, 2, 3, 4)
BETAS = (0.1, 0.5, 1.0, 5.0)
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
MAX_ITER = 100000


def weights(codes: list[str], names: tuple[str, ...], nact: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    t = scale_table(deltas(codes, names), nact, SCALES[DEFAULT])
    cp = t[t["dir"] == "+"].sort_values("j")["c"].to_numpy()
    cm = t[t["dir"] == "−"].sort_values("j")["c"].to_numpy()
    return cp, cm


def diagnose(plan: np.ndarray, xt: np.ndarray, cmat: np.ndarray, paid_idx: tuple[int, int]
             ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    rows = plan.sum(axis=1)
    if (rows <= 0).any():
        raise SolverNotConvergedError("質量 0 の行がある(バリセントリック射影は定義できない)")
    p = plan / rows[:, None]
    bary = p @ xt
    arg = xt[plan.argmax(axis=1)]
    recv = plan.sum(axis=0)
    sup = effective_support_sizes(plan)
    nn_dist = np.sqrt(((bary[:, None, :] - xt[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    paid = bary[:, paid_idx[0]] + bary[:, paid_idx[1]]
    am = plan.argmax(axis=1)
    d = {"mass": float(plan.sum()), "row_min": float(rows.min()),
         "support_med": float(np.median(sup)), "support_p90": float(np.percentile(sup, 90)),
         "support_min": float(sup.min()),
         "n_receive": int((recv > 0.01 / len(rows)).sum()),  # 1 始点の質量の 1% 超
         "top5_share": float(np.sort(recv)[-5:].sum() / recv.sum()),
         "n_argmax": int(len(set(am))), "argmax_max_count": int(np.bincount(am).max()),
         "var_bary": float(bary.var(axis=0).mean()), "var_argmax": float(arg.var(axis=0).mean()),
         "identity_resid": float(np.abs(bary @ cmat).max()),
         "paid_bary_med": float(np.median(paid)),
         "paid_argmax_med": float(np.median(arg[:, paid_idx[0]] + arg[:, paid_idx[1]])),
         "nn_dist_med": float(np.median(nn_dist))}
    return d, bary, arg


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    cmat = build_constraint_matrix(spec, include_subtotals=False)
    na = set(load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable)
    nact = np.array([n in na for n in names])
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    grp = "class 1 → class 0" if args.cls == "c2" else "class 2 → 回復"  # D.4 の母集団の分離
    cp, cm = weights(pairs[pairs["group"] == grp]["edinetCode"].tolist(), names, nact)
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    if args.cls == "c2":
        is_src = (lab == 2) & ~np.isin(docs_a, OUTLIERS)
    else:
        c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
        is_src = np.isin(docs_a, c1.loc[c1["reason"].fillna("") == "", "docID"].to_numpy())
    xs, xt = x[is_src], x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    ind_t = np.array(load_industry_config(Path("conf/accounts/industry.yaml")).labels(
        [code_of.get(d, "") for d in docs_a[lab == 0]]))
    c_raw = l1_cost(xs, xt, cp, cm)
    med = float(np.median(c_raw))
    ct = c_raw / med
    n0, n1 = ct.shape
    a = np.full(n0, 1.0 / n0)
    paid_idx = (names.index(_CS), names.index(_CSUR))
    print(f"[{args.cls}] n0 = {n0}、n1 = {n1}、n1/n0 = {n1 / n0:.1f}、median(C) = {med:.2f}",
          flush=True)

    rows: list[dict[str, object]] = []
    targets: dict[str, np.ndarray] = {"source_docs": docs_a[is_src],
                                      "target_docs": docs_a[lab == 0]}

    def record(name: str, family: str, params: dict[str, float],
               solve: Callable[[], tuple[np.ndarray, dict[str, float]]],
               cols: np.ndarray | None = None) -> None:
        t0 = time.perf_counter()
        row: dict[str, object] = {"setting": name, "family": family, **params}
        try:
            sub_plan, info = solve()
            plan = np.zeros((n0, n1))
            if cols is None:
                plan = sub_plan
            else:
                plan[:, cols] = sub_plan
            d, bary, arg = diagnose(plan, xt, cmat, paid_idx)
            row |= {"status": "ok", **info, **d}
            targets[f"{name}__bary"], targets[f"{name}__argmax"] = bary, arg
        except SolverNotConvergedError as e:
            row |= {"status": "失敗", "reason": str(e)}
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(f"  {name}: {row['status']} {row.get('support_med', '')} ({row['seconds']}s)",
              flush=True)

    def nn_plan(c: np.ndarray) -> Callable[[], tuple[np.ndarray, dict[str, float]]]:
        def f() -> tuple[np.ndarray, dict[str, float]]:
            p = np.zeros_like(c)
            p[np.arange(n0), c.argmin(axis=1)] = a
            return p, {"converged": 1.0, "row_err": 0.0, "col_err": float("nan")}
        return f

    def balanced(c: np.ndarray, b: np.ndarray, eps: float
                 ) -> Callable[[], tuple[np.ndarray, dict[str, float]]]:
        def f() -> tuple[np.ndarray, dict[str, float]]:
            r = solve_ot_normalized(c, a, b, eps, max_iter=MAX_ITER)
            col = float(np.abs(r.plan.sum(axis=0) - b).max())
            return r.plan, {"converged": float(r.converged), "n_iter": r.n_checkpoints,
                            "row_err": float(np.abs(r.plan.sum(axis=1) - a).max()),
                            "col_err": col}
        return f

    def semi(eps: float, tau: float) -> Callable[[], tuple[np.ndarray, dict[str, float]]]:
        def f() -> tuple[np.ndarray, dict[str, float]]:
            b = np.full(n1, 1.0 / n1)
            r = solve_semi_unbalanced_normalized(ct, a, b, eps, tau, max_iter=MAX_ITER * 10)
            return r.plan, {"converged": 1.0, "n_iter": r.n_checkpoints,
                            "row_err": r.max_marginal_error,
                            "col_err": float(np.abs(r.plan.sum(axis=0) - b).max())}
        return f

    record("nn", "0", {}, nn_plan(ct))
    record("nn_l2", "0'", {}, nn_plan(l2_cost(xs, xt, (~nact).astype(float))))
    # §0: 厳密解(ε → 0)
    t0 = time.perf_counter()
    emd, log = ot.emd(a, np.full(n1, 1.0 / n1), ct, numItermax=10_000_000, log=True)
    if log["warning"] is not None:
        raise SolverNotConvergedError(f"ot.emd: {log['warning']}")
    d_emd, b_emd, a_emd = diagnose(emd, xt, cmat, paid_idx)
    rows.append({"setting": "bal_emd", "family": "1", "status": "ok", "converged": 1.0,
                 "row_err": float(np.abs(emd.sum(axis=1) - a).max()),
                 "col_err": float(np.abs(emd.sum(axis=0) - 1.0 / n1).max()), **d_emd,
                 "seconds": round(time.perf_counter() - t0, 1)})
    targets["bal_emd__bary"], targets["bal_emd__argmax"] = b_emd, a_emd
    unif = np.full(n1, 1.0 / n1)
    for eps in EPS:
        record(f"bal_e{eps:g}", "1", {"eps": eps}, balanced(ct, unif, eps))
    for eps in EPS:
        for tau in TAU1:
            record(f"semi_t{tau:g}_e{eps:g}", "A", {"eps": eps, "tau1": tau}, semi(eps, tau))
    for n in N1S:
        for seed in SEEDS:
            cols = stratified(ind_t, n, seed)
            for eps in EPS:
                record(f"sub{n}_s{seed}_e{eps:g}", "B", {"eps": eps, "n1p": n, "seed": seed},
                       balanced(ct[:, cols], np.full(len(cols), 1.0 / len(cols)), eps), cols)
        k = n // n0 if args.cls == "c2" else max(1, round(n / n0))
        cols = np.unique(np.argsort(ct, axis=1)[:, :k])
        for eps in EPS:
            record(f"knn{k}_e{eps:g}", "B3", {"eps": eps, "n1p": len(cols), "k": k},
                   balanced(ct[:, cols], np.full(len(cols), 1.0 / len(cols)), eps), cols)
    mins = ct.min(axis=0)
    for beta in BETAS:
        w = np.exp(-beta * (mins - mins.min()))
        w /= w.sum()
        for eps in EPS:
            record(f"w{beta:g}_e{eps:g}", "C", {"eps": eps, "beta": beta}, balanced(ct, w, eps))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"plans_{args.cls}.csv", index=False)
    np.savez_compressed(DATA / f"targets_{args.cls}.npz", **targets)  # type: ignore[arg-type]
    meta = {"n0": n0, "n1": n1, "median_cost": med, "c_plus": cp.tolist(), "c_minus": cm.tolist(),
            "weights_group": grp}
    (DATA / f"meta_{args.cls}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
