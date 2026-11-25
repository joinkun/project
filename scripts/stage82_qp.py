#!/usr/bin/env python
"""ステージ82: 企業ごとの単価 ψ_i と制約の階層による F1 の生成(cvxpy のプロセス)。

規則は ``results/stage82/criteria.md`` §2・§4。第 2 段階は、候補 y_j ごとに次の強凸 QP を解き、
輸送の重み p̃_ij で平均する(F1)。

    min_{y ∈ 𝓕⁺_tier} ½ Σ w (y − y_j)² + λ [θ_i⁺ · u + θ_i⁻ · v]、y − x_i = u − v、u, v ≥ 0

(``RecourseQP``。利益剰余金の移動コストは 0、評価差額金は凍結。)θ_i は輸送と同じ ψ_i の単価。

- 制約の階層 ``--tier``: t1(会計の定義)、t12(+ 法規制)、t123(+ 実務、従来の 𝓕⁺)。
- ``--qp unified``: A_inact(i) を y_a = max(始点の値, 0)(実質 0)に固定する(2 本の不等式)。
  criteria §8: 「y_a = 0」は、始点が 10⁻⁶ 未満の正の値の科目で営業資産の保持の制約と矛盾した。
- λ = λ̃ × s。s は class 1 の F1 の定義(候補の重みつきの Σ ½w d² / Σ ψ_i(d) の中央値)。

出力: ``data/interim/stage82/{群}/gen_{cfg}/{cls}/{タグ}__l{λ̃}``、``results/stage82/qp/``。

    uv run python scripts/stage82_qp.py --cls c1 --cfg f0 --group e1 --cost R1 --shard 0/2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qp_stage60 import RE_CAP, S46_ALPHA, SIGMA_FLOOR, s47_final
from qp_stage66 import BETA_CASH, LAMS, sources
from stage72_qp import liability_cap
from stage82_common import COSTS, DATA, OUT, Setting
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    INACTIVE_ACCOUNTS,
    LinearConstraint,
    PlausibilityConfig,
    build_linear_constraints,
    decompose_re_change,
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


def tier_config(tier: str) -> PlausibilityConfig:
    """制約の階層(criteria §4)。t123 は従来の s47_final + 現金の下限。"""
    t1 = PlausibilityConfig(subtotal_nonneg=True, residual_nonneg=True, contra_net_nonneg=True,
                            label="tier1")
    if tier == "t1":
        return t1
    if tier == "t12":
        return dataclasses.replace(t1, treasury_cap=True, capital_min_ratio=S46_ALPHA,
                                   capital_stock_only=True, label="tier12")
    cfg = dataclasses.replace(s47_final(), cash_floor=BETA_CASH)
    if cfg.re_growth_cap != RE_CAP or not cfg.treasury_cap:
        raise ValueError("s47_final の想定と違う")
    return cfg


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--cfg", choices=("f0", "f1", "f2", "f3", "f4", "full"), required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--cost", choices=tuple(COSTS), required=True)
    ap.add_argument("--ot", default="prop")
    ap.add_argument("--qp", choices=("base", "unified"), default="base")
    ap.add_argument("--tier", choices=("t1", "t12", "t123"), default="t123")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--data", type=Path, default=DATA,
                    help="theta.json・計画・生成の場所(ステージ83 は data/interim/stage83)")
    ap.add_argument("--out", type=Path, default=OUT, help="記録の出力先")
    args = ap.parse_args()
    st = Setting(args.group, args.cost, args.ot, args.qp, args.tier)
    sp = COSTS[args.cost]
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cfg = tier_config(args.tier)
    dcs = load_direction_constraints(_CANONICAL) if args.tier == "t123" else ()
    na_coef = members["jppfs_cor_NetAssets"]
    tj = json.loads((args.data / "theta.json").read_text(encoding="utf-8"))
    sigma = np.maximum(np.array(tj["sigma"], dtype=float), SIGMA_FLOOR)
    w = 1.0 / sigma ** 2 / np.array([10.0 if n in _PAID else 1.0 for n in names]) ** 2
    th = np.array(tj["survival"]["alldist"]["theta"] if sp.pool == "alldist"
                  else tj["configs"][args.cfg][sp.mode]["theta"], dtype=float)
    re_idx = names.index(_RE)

    def theta_i(x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = individual_theta(th, x0, sigma, indiv_scale=sp.indiv,
                             inactive=inactive_mask(x0, names) if sp.penalty else None)
        t[:, re_idx] = 0.0
        return t[0], t[1]

    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]

    def plan(cls: str, src: list[str]) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(args.data / f"plan_{cls}_{args.cfg}_{args.cost}_{args.ot}.npz",
                    allow_pickle=True)
        if [str(v) for v in z["source_docs"]] != src:
            raise ValueError("始点の並びが一致しない")
        return np.asarray(z["idx"]), np.asarray(z["pt"])

    m1 = sources("c1", x, lab, docs_a)
    x1 = x[m1]
    idx1, pt1 = plan("c1", [str(d) for d in docs_a[m1]])
    ratios = []
    for i in range(len(x1)):
        cp_i, cm_i = theta_i(x1[i])
        d = xt[idx1[i]] - x1[i]
        num = float((pt1[i] * 0.5 * (w * d ** 2).sum(axis=1)).sum())
        den = float((pt1[i] * (np.where(d > 0, cp_i, cm_i) * np.abs(d)).sum(axis=1)).sum())
        if den > 0:
            ratios.append(num / den)
    s = float(np.median(ratios))
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(dd) for dd in docs_a[mask]]
    idx, pt = plan(args.cls, src)
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    lams = list(LAMS)[i_sh::n_sh]
    print(f"[{args.cls} {args.cfg} {st.tag}] s = {s:.4g}、λ̃ = {lams}", flush=True)
    summary = []
    for lt in lams:
        lam = lt * s
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        n_infeas, n_fb = 0, 0
        for i, doc in enumerate(src):
            x0 = xs[i]
            inactive = (inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
                        if args.tier == "t123" else ())
            frozen_i = FROZEN + tuple(a for a in inactive if a not in FROZEN)
            extra = build_linear_constraints(spec, names, x0, cfg)
            if args.qp == "unified":  # criteria §2・§8: A_inact(i) を max(始点, 0) に固定
                for j_ in np.where(inactive_mask(x0, names))[0]:
                    e = np.zeros(len(names))
                    e[j_] = 1.0
                    v = max(float(x0[j_]), 0.0)
                    extra = extra + (LinearConstraint(f"fix:{names[j_]}", e, v),
                                     LinearConstraint(f"fix-:{names[j_]}", -e, -v))
            if args.tier == "t123":
                extra = extra + liability_cap(members, x0)
            cp_i, cm_i = theta_i(x0)
            status = "ok"
            try:
                qp = RecourseQP(spec, x0, RecourseWeights(w, na_coef, 0.0, cp_i, cm_i, 0.0), lam,
                                extra_constraints=extra, direction_constraints=dcs,
                                frozen_accounts=frozen_i)
                k = pt[i] > 0
                ys = [qp.solve(xt[j]).y for j in idx[i][k]]
                y = (pt[i][k][:, None] * np.array(ys)).sum(axis=0)
                n_fb += qp.n_fallback
            except InfeasibleProjectionError:
                y, status = x0.copy(), "infeasible"
                n_infeas += 1
            tr, pr = decompose_re_change(x0, y, names)
            row: dict[str, object] = {
                "tag": "nearest_na0", "config": "s47_final", "fold": 0, "method": "PCFM",
                "docID": doc, "filerName": str(filer.get(doc, "")),
                "assets_start": float(agg.loc[doc, "total_assets"]), "status": status,
                "active": "", "re_branch": -1, "re_transfer": float(tr), "re_profit": float(pr),
                "n_inactive": len(inactive)}
            for j_, n in enumerate(names):
                row[f"x0_{n}"] = float(x0[j_])
                row[f"x1_{n}"] = float(y[j_])
            rows.append(row)
        out_dir = args.data / args.group / f"gen_{args.cfg}" / args.cls / f"{st.tag}__l{lt:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        for seed in (0, 1, 2):
            df.assign(seed=seed).to_parquet(out_dir / f"nearest_na0__seed{seed}.parquet",
                                            index=False)
        summary.append({"cls": args.cls, "cfg": args.cfg, "setting": st.tag, "lam_tilde": lt,
                        "lam": lam, "s": s, "n_infeasible": n_infeas, "n_fallback": n_fb,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  λ̃ = {lt:g}: 不可能 {n_infeas}、代替ソルバ {n_fb}"
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    (args.out / "qp").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(
        args.out / "qp" / f"{args.group}_{args.cls}_{args.cfg}_{st.tag}_{i_sh}of{n_sh}.csv",
        index=False)


if __name__ == "__main__":
    main()
