#!/usr/bin/env python
"""ステージ64 A: 輸送問題として解く(POT のプロセス。torch・sklearn・cvxpy を import しない)。

規則は ``results/stage64/criteria.md``(実行の前に記録)。コストは非対称 L1(ステージ60 の c±)、
C̃ = C / median(C)。バランス OT(対数領域の Sinkhorn と厳密解)、不均衡 OT(τ0 = ∞、自前の対数領域)、
片側制約、最近傍を解き、健全性・最近傍との一致を記録する。未収束は表に「失敗」と書いて、目標を作らない。

出力: ``results/stage64/plans_{cls}.csv``・``mismatch_{cls}.csv``、
``data/interim/stage64/targets_{cls}.npz``(argmax の添字と重心)。

    uv run python scripts/ot_stage64.py --cls c2
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
from prep_stage59 import OUTLIERS, l1_cost

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.errors import SolverNotConvergedError
from fsgen.models.transport import (
    effective_support_sizes,
    solve_ot_normalized,
    solve_semi_unbalanced_normalized,
)

OUT = Path("results/stage64")
DATA = Path("data/interim/stage64")
EPS = (0.001, 0.005, 0.02, 0.1)
TAU1 = (0.001, 0.01, 0.1, 1.0, 10.0)
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    meta_path = Path(f"data/interim/stage60/meta_{args.cls}.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cp, cm = np.array(meta["c_plus"]), np.array(meta["c_minus"])
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    if args.cls == "c2":
        is_src = (lab == 2) & ~np.isin(docs_a, OUTLIERS)
    else:
        c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
        is_src = np.isin(docs_a, c1.loc[c1["reason"].fillna("") == "", "docID"].to_numpy())
    xs, xt = x[is_src], x[lab == 0]
    ct = l1_cost(xs, xt, cp, cm)
    ct = ct / np.median(ct)
    n0, n1 = ct.shape
    a, b = np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1)
    nn = ct.argmin(axis=1)
    paid_t = xt[:, names.index(_CS)] + xt[:, names.index(_CSUR)]
    src_docs, tgt_docs = docs_a[is_src], docs_a[lab == 0]
    print(f"[{args.cls}] n0 = {n0}、n1 = {n1}、n1/n0 = {n1 / n0:.1f}", flush=True)
    rows: list[dict[str, object]] = []
    store: dict[str, np.ndarray] = {"source_docs": src_docs, "target_docs": tgt_docs, "nn": nn}
    mism: list[dict[str, object]] = []

    def record(name: str, method: str, eps: float | None, tau: float | None,
               solve: Callable[[], tuple[np.ndarray, dict[str, float]]]) -> None:
        t0 = time.perf_counter()
        row: dict[str, object] = {"setting": name, "method": method, "eps": eps, "tau1": tau}
        try:
            plan, info = solve()
            rows_mass = plan.sum(axis=1)
            n_zero = int((rows_mass < 1e-12).sum())
            sup = effective_support_sizes(plan) if n_zero == 0 else np.full(n0, np.nan)
            am = plan.argmax(axis=1)
            match = am == nn
            row |= {"status": "ok", **info, "n_zero_rows": n_zero,
                    "min_row_mass": float(rows_mass.min()), "mass": float(plan.sum()),
                    "support_med": float(np.median(sup)),
                    "support_p90": float(np.percentile(sup, 90)),
                    "support_min": float(sup.min()), "support_max": float(sup.max()),
                    "nn_match": int(match.sum()), "n_argmax": int(len(set(am))),
                    "eligible": bool(n_zero == 0 and plan.sum() >= 0.5)}
            if n_zero == 0:
                store[f"{name}__argmax"] = am
                store[f"{name}__bary"] = (plan / rows_mass[:, None]) @ xt
            for i in np.where(~match)[0]:
                mism.append({"setting": name, "source": src_docs[i], "nn_target": tgt_docs[nn[i]],
                             "argmax_target": tgt_docs[am[i]],
                             "cost_diff": float(ct[i, am[i]] - ct[i, nn[i]]),
                             "paid_nn": float(paid_t[nn[i]]), "paid_argmax": float(paid_t[am[i]])})
        except SolverNotConvergedError as e:
            row |= {"status": "失敗", "reason": str(e), "eligible": False}
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(f"  {name}: {row['status']} 一致 {row.get('nn_match', '—')} / {n0} "
              f"サポート {row.get('support_med', '—')} ({row['seconds']}s)", flush=True)

    def balanced(eps: float) -> Callable[[], tuple[np.ndarray, dict[str, float]]]:
        def f() -> tuple[np.ndarray, dict[str, float]]:
            r = solve_ot_normalized(ct, a, b, eps, max_iter=200000)
            return r.plan, {"converged": float(r.converged), "n_iter": r.n_checkpoints,
                            "marg_err": r.max_marginal_error}
        return f

    def semi(eps: float, tau: float) -> Callable[[], tuple[np.ndarray, dict[str, float]]]:
        def f() -> tuple[np.ndarray, dict[str, float]]:
            r = solve_semi_unbalanced_normalized(ct, a, b, eps, tau, max_iter=2_000_000)
            return r.plan, {"converged": 1.0, "n_iter": r.n_checkpoints,
                            "marg_err": r.max_marginal_error}
        return f

    def onesided(eps: float) -> Callable[[], tuple[np.ndarray, dict[str, float]]]:
        def f() -> tuple[np.ndarray, dict[str, float]]:
            lg = -ct / eps
            w = np.exp(lg - lg.max(axis=1, keepdims=True))
            plan = w / w.sum(axis=1, keepdims=True) * a[:, None]
            if not np.isfinite(plan).all():
                raise SolverNotConvergedError("片側制約の計画が有限でない")
            return plan, {"converged": 1.0, "n_iter": 0,
                          "marg_err": float(np.abs(plan.sum(axis=1) - a).max())}
        return f

    def nearest() -> tuple[np.ndarray, dict[str, float]]:
        p = np.zeros_like(ct)
        p[np.arange(n0), nn] = a
        return p, {"converged": 1.0, "n_iter": 0, "marg_err": 0.0}

    def exact() -> tuple[np.ndarray, dict[str, float]]:
        p, log = ot.emd(a, b, ct, numItermax=10_000_000, log=True)
        if log["warning"] is not None:
            raise SolverNotConvergedError(f"ot.emd: {log['warning']}")
        return p, {"converged": 1.0, "n_iter": 0,
                   "marg_err": float(max(np.abs(p.sum(axis=1) - a).max(),
                                         np.abs(p.sum(axis=0) - b).max()))}

    record("nn", "最近傍", None, None, nearest)
    record("bal_emd", "バランス(厳密解)", 0.0, None, exact)
    for eps in EPS:
        record(f"bal_e{eps:g}", "バランス", eps, None, balanced(eps))
    for eps in EPS:
        for tau in TAU1:
            record(f"semi_t{tau:g}_e{eps:g}", "不均衡", eps, tau, semi(eps, tau))
    for eps in EPS:
        record(f"one_e{eps:g}", "片側制約", eps, None, onesided(eps))
    pd.DataFrame(rows).to_csv(OUT / f"plans_{args.cls}.csv", index=False)
    pd.DataFrame(mism).to_csv(OUT / f"mismatch_{args.cls}.csv", index=False)
    np.savez_compressed(DATA / f"targets_{args.cls}.npz", **store)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
