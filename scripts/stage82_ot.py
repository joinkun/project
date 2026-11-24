#!/usr/bin/env python
"""ステージ82: 企業ごとの単価 ψ_i による輸送計画(POT・numpy のプロセス)。

規則は ``results/stage82/criteria.md`` §2・§3。コスト C_ij = ψ_i(x_j − x_i)(利益剰余金・評価差額金は
0)、C̃ = C / median(C[~mask])、同一企業の組は除く。輸送の方式は ``stage82_common.OT_MODES``。

出力: ``data/interim/stage82/plan_{cls}_{cfg}_{コスト}_{方式}.npz``(``idx``・``pt``・``top10_idx``・
``top10_p``)、``results/stage82/ot/{cls}_{cfg}_{コスト}_{方式}.json``(診断)。

    uv run python scripts/stage82_ot.py --cls c1 --cfg f0 --pairs R1:prop,KDE0:prop
"""

from __future__ import annotations

import os
import sys

# POT の torch バックエンドと sklearn の読み込みを止める(precompute_transport.py と同じ)
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stage71_ot import candidates  # noqa: E402
from stage82_common import (  # noqa: E402
    COSTS,
    DATA,
    EPS,
    GAMMA,
    GAMMA_COL,
    GAMMA_COL_FALLBACK,
    OUT,
    TAU1,
    ZERO_OT,
)
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.errors import SolverNotConvergedError  # noqa: E402
from fsgen.models.sparse_transport import solve_sparse_semi_unbalanced  # noqa: E402
from fsgen.models.transport import solve_semi_unbalanced_normalized  # noqa: E402
from fsgen.models.unit_costs import inactive_mask, individual_theta, pairwise_cost  # noqa: E402


def theta_rows(cost: str, cfg: str, xs: np.ndarray, names: tuple[str, ...],
               tj: dict[str, object]) -> np.ndarray:
    """始点ごとの単価 (n0, 2, D)。利益剰余金・評価差額金は 0(輸送のコスト)。"""
    sp = COSTS[cost]
    if sp.pool == "alldist":
        if cfg != "full":
            raise ValueError("苦境全体のプールは full だけ")
        th = np.array(tj["survival"]["alldist"]["theta"])  # type: ignore[index]
    else:
        th = np.array(tj["configs"][cfg][sp.mode]["theta"])  # type: ignore[index]
    sigma = np.array(tj["sigma"], dtype=float)
    zero = np.array([n in ZERO_OT for n in names])
    rows = np.stack([individual_theta(th, x0, sigma, indiv_scale=sp.indiv,
                                      inactive=inactive_mask(x0, names) if sp.penalty else None)
                     for x0 in xs])
    rows[:, :, zero] = 0.0
    out: np.ndarray = rows
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--cfg", choices=("f0", "f1", "f2", "f3", "f4", "full"), required=True)
    ap.add_argument("--pairs", required=True, help="コスト:方式 のカンマ区切り")
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    tj = json.loads((DATA / "theta.json").read_text(encoding="utf-8"))
    x, lab, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    xt = x[lab == 0]
    tgt_codes = np.array([code_of.get(str(d), "") for d in docs_a[lab == 0]])
    m = sources(args.cls, lab, docs_a)
    xs, src = x[m], [str(d) for d in docs_a[m]]
    sc = np.array([code_of.get(d, "") for d in src])
    mask = (sc[:, None] == tgt_codes[None, :]) & (sc[:, None] != "")
    n0, n1 = len(xs), len(xt)
    kappa = n1 / n0
    tau_q = TAU1 * np.log(kappa) / (kappa * (kappa - 1.0))
    (OUT / "ot").mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}
    for pair in args.pairs.split(","):
        cost, ot_mode = pair.split(":")
        t0 = time.perf_counter()
        if cost not in cache:
            c = pairwise_cost(xs, xt, theta_rows(cost, args.cfg, xs, names, tj))
            ct = c / np.median(c[~mask])
            ct[mask] = 10.0 * ct.max()
            cache[cost] = ct
        ct = cache[cost]
        info: dict[str, object] = {"cls": args.cls, "cfg": args.cfg, "cost": cost, "ot": ot_mode}
        if ot_mode == "pure_lp":
            p = np.zeros_like(ct)
            p[np.arange(n0), np.where(mask, np.inf, ct).argmin(axis=1)] = 1.0
        elif ot_mode == "sinkhorn":
            r = solve_semi_unbalanced_normalized(ct, np.full(n0, 1 / n0), np.full(n1, 1 / n1),
                                                 EPS, TAU1, max_iter=2_000_000)
            p = r.plan / r.plan.sum(axis=1, keepdims=True)
        else:
            gamma, tau = {"prop": (GAMMA, tau_q), "top5": (GAMMA, tau_q),
                          "top8": (GAMMA, tau_q), "row": (GAMMA, 0.0),
                          "col": (GAMMA_COL, tau_q)}[ot_mode]
            allowed = ~mask
            if ot_mode in ("top5", "top8"):
                k = int(ot_mode[3:])
                order = np.argsort(np.where(mask, np.inf, ct), axis=1)[:, :k]
                allowed = np.zeros_like(mask)
                np.put_along_axis(allowed, order, True, axis=1)
            try:
                res = solve_sparse_semi_unbalanced(ct, gamma, tau, allowed=allowed)
            except SolverNotConvergedError:
                if ot_mode != "col":
                    raise
                gamma = GAMMA_COL_FALLBACK  # criteria §3: γ' = 10⁻⁵ で解き直す
                res = solve_sparse_semi_unbalanced(ct, gamma, tau, allowed=allowed)
            p = res.p
            info |= {"gamma": gamma, "tau": tau, "iters": res.n_iter, "residual": res.residual}
        top_idx = np.argsort(-p, axis=1)[:, :10]
        top_p = np.take_along_axis(p, top_idx, axis=1)
        if ot_mode == "sinkhorn":
            idx, pt, _k = candidates(p)
        else:
            kmax = int((p > 0).sum(axis=1).max())
            idx = np.argsort(-p, axis=1)[:, :kmax]
            pt = np.take_along_axis(p, idx, axis=1)
        if not np.allclose(pt.sum(axis=1), 1.0):
            raise ValueError("候補の重みの和が 1 でない")
        if (pt[np.take_along_axis(mask, idx, axis=1)] > 1e-12).any():
            raise ValueError("同一企業の組に質量がある")
        nz = (pt > 0).sum(axis=1)
        load = np.zeros(n1)
        np.add.at(load, idx.ravel(), pt.ravel())
        info |= {"nz_mean": float(nz.mean()), "nz_med": float(np.median(nz)),
                 "nz_max": int(nz.max()), "ess_med": float(np.median(1.0 / (pt ** 2).sum(axis=1))),
                 "max_load": float(kappa * load.max()), "unique_targets": int((load > 0).sum()),
                 "seconds": round(time.perf_counter() - t0, 1)}
        np.savez(DATA / f"plan_{args.cls}_{args.cfg}_{cost}_{ot_mode}.npz",
                 source_docs=np.array(src), idx=idx, pt=pt, top10_idx=top_idx, top10_p=top_p)
        (OUT / "ot" / f"{args.cls}_{args.cfg}_{cost}_{ot_mode}.json").write_text(
            json.dumps(info, ensure_ascii=False), encoding="utf-8")
        print(f"[{args.cls} {args.cfg} {cost} {ot_mode}] |S| 平均 {info['nz_mean']:.2f}、"
              f"ESS {info['ess_med']:.2f}、最大負荷 {info['max_load']:.1f}"
              f"({info['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
