#!/usr/bin/env python
"""ステージ83 パートC: 再建企業の out-of-fold の生成(cvxpy のプロセス)。

規則は ``results/stage83/criteria_stage83.md`` §3。``stage83_backtest_ot.py`` の計画の候補について、
KDE0p(主)・KDE0(base)と同じ QP(t123、F1)を解く。

- λ = λ̃ × s。s は cfg(fold f、c2rec は full)の class 1 の F1 の定義(D1 と同じ)。
- λ̃: 57 社の群 f では、主 = fold f 以外の 4 fold の λ̃* の中央値、参考 = λ̃*_f。c2rec では、
  単一の生成の λ̃(5 fold の中央値)。λ̃* はパートA の D1(``results/stage83/partA/summary.json``)。

出力: ``data/interim/stage83/bt_gen_{群}_{コスト}.parquet``(企業ごと・λ̃ の種類ごとの x0・y)。

    uv run python scripts/stage83_backtest_qp.py --group f0 --cost KDE0p
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qp_stage60 import SIGMA_FLOOR
from qp_stage66 import sources
from stage72_qp import liability_cap
from stage82_common import COSTS
from stage82_qp import tier_config
from stage83_common import D82, DATA, FOLDS, NAMES, OUT
from stage83_common import compositions as comps
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    INACTIVE_ACCOUNTS,
    build_linear_constraints,
    inactive_accounts,
    subtotal_members,
)
from fsgen.accounting.projection_qp import load_direction_constraints
from fsgen.accounting.recourse_qp import RecourseQP, RecourseWeights
from fsgen.errors import InfeasibleProjectionError
from fsgen.models.unit_costs import inactive_mask, individual_theta

_CANONICAL = Path("conf/accounts/core.yaml")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
_RE = "jppfs_cor_RetainedEarnings"
FROZEN = ("valuation_difference_securities",)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=(*FOLDS, "c2rec"), required=True)
    ap.add_argument("--cost", choices=("KDE0p", "KDE0"), required=True)
    args = ap.parse_args()
    cfg_name = "full" if args.group == "c2rec" else args.group
    sp = COSTS[args.cost]
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cfg = tier_config("t123")
    dcs = load_direction_constraints(_CANONICAL)
    na_coef = members["jppfs_cor_NetAssets"]
    tj = json.loads((DATA / "theta.json").read_text(encoding="utf-8"))
    sigma = np.maximum(np.array(tj["sigma"], dtype=float), SIGMA_FLOOR)
    w = 1.0 / sigma ** 2 / np.array([10.0 if n in _PAID else 1.0 for n in names]) ** 2
    th = np.array(tj["configs"][cfg_name][sp.mode]["theta"], dtype=float)
    re_idx = names.index(_RE)

    def theta_i(x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = individual_theta(th, x0, sigma, indiv_scale=sp.indiv,
                             inactive=inactive_mask(x0, names) if sp.penalty else None)
        t[:, re_idx] = 0.0
        return t[0], t[1]
    # λ̃
    sm = json.loads((OUT / "partA" / "summary.json").read_text(encoding="utf-8"))
    stars = sm["res"][NAMES[args.cost]]["star"]
    if any(stars[f] is None for f in FOLDS):
        raise ValueError("較正できない fold がある")
    if args.group == "c2rec":
        lams = {"main": float(np.median([stars[f] for f in FOLDS]))}
    else:
        lams = {"main": float(np.median([stars[f] for f in FOLDS if f != args.group])),
                "own": float(stars[args.group])}
    # s(D1 と同じ: cfg の class 1 の始点と計画)
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    xt = x[lab == 0]
    pdir = DATA if args.cost == "KDE0p" else D82
    z = np.load(pdir / f"plan_c1_{cfg_name}_{args.cost}_prop.npz", allow_pickle=True)
    m1 = sources("c1", x, lab, docs_a)
    if [str(v) for v in z["source_docs"]] != [str(d) for d in docs_a[m1]]:
        raise ValueError("始点の並びが一致しない")
    x1s, idx1, pt1 = x[m1], np.asarray(z["idx"]), np.asarray(z["pt"])
    ratios = []
    for i in range(len(x1s)):
        cp_i, cm_i = theta_i(x1s[i])
        d = xt[idx1[i]] - x1s[i]
        num = float((pt1[i] * 0.5 * (w * d ** 2).sum(axis=1)).sum())
        den = float((pt1[i] * (np.where(d > 0, cp_i, cm_i) * np.abs(d)).sum(axis=1)).sum())
        if den > 0:
            ratios.append(num / den)
    s = float(np.median(ratios))
    bt = np.load(DATA / f"bt_plan_{args.group}_{args.cost}.npz", allow_pickle=True)
    codes = [str(c) for c in bt["codes"]]
    x0s, _x1, _ta, ok = comps(codes, names)
    if ok != codes:
        raise ValueError("企業の並びが一致しない")
    idx, pt = np.asarray(bt["idx"]), np.asarray(bt["pt"])
    rows = []
    n_inf = 0
    for kind, lt in lams.items():
        for i, code in enumerate(codes):
            x0 = x0s[i]
            inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
            frozen_i = FROZEN + tuple(a for a in inactive if a not in FROZEN)
            extra = build_linear_constraints(spec, names, x0, cfg, relax_residual_start=True)
            extra = extra + liability_cap(members, x0)
            cp_i, cm_i = theta_i(x0)
            status = "ok"
            try:
                qp = RecourseQP(spec, x0, RecourseWeights(w, na_coef, 0.0, cp_i, cm_i, 0.0),
                                lt * s, extra_constraints=extra, direction_constraints=dcs,
                                frozen_accounts=frozen_i)
                k = pt[i] > 0
                ys = [qp.solve(xt[j]).y for j in idx[i][k]]
                y = (pt[i][k][:, None] * np.array(ys)).sum(axis=0)
            except InfeasibleProjectionError:
                y, status = x0.copy(), "infeasible"
                n_inf += 1
            row: dict[str, object] = {"group": args.group, "cost": args.cost, "lam_kind": kind,
                                      "lam_tilde": lt, "s": s, "code": code, "status": status}
            for j_, n in enumerate(names):
                row[f"x0_{n}"] = float(x0[j_])
                row[f"y_{n}"] = float(y[j_])
            rows.append(row)
    pd.DataFrame(rows).to_parquet(DATA / f"bt_gen_{args.group}_{args.cost}.parquet", index=False)
    print(f"[{args.group} {args.cost}] λ̃ = {lams}、s = {s:.4g}、不可能 {n_inf}", flush=True)


if __name__ == "__main__":
    main()
