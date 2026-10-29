#!/usr/bin/env python
"""ステージ74: リークのないプール・相対コストでの F1 事後アンサンブル(cvxpy のプロセス)。

規則は ``results/stage74/criteria.md``。第 1 項は全 32 科目・m = 10、制約は s47_final + 現金の下限 +
負債の上限(1 − 10⁻⁶ の余裕)。凍結は評価差額金と不活性科目だけ。移動コストは始点ごとの c̃_i^±
(``stage74_costs.firm_costs``)。λ = λ̃ × s、s は設定の class 1 のプールとコストで、
class 1 の始点で計算。

    uv run python scripts/stage74_qp.py --cls c1 --setting A1 --shard 0/2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qp_stage60 import SIGMA_FLOOR, s47_final
from qp_stage66 import BETA_CASH, LAMS, sources
from stage72_qp import liability_cap
from stage74_costs import SETTINGS, entry, firm_costs, pool_of
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    INACTIVE_ACCOUNTS,
    build_linear_constraints,
    decompose_re_change,
    inactive_accounts,
    subtotal_members,
)
from fsgen.accounting.projection_qp import load_direction_constraints
from fsgen.accounting.recourse_qp import RecourseQP, RecourseWeights
from fsgen.errors import InfeasibleProjectionError

DATA = Path("data/interim/stage74")
OUT = Path("results/stage74")
_CANONICAL = Path("conf/accounts/core.yaml")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
FROZEN = ("valuation_difference_securities",)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    st = args.setting
    kind = SETTINGS[st][0]
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cfg = dataclasses.replace(s47_final(), cash_floor=BETA_CASH)
    dcs = load_direction_constraints(_CANONICAL)
    na_coef = members["jppfs_cor_NetAssets"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    w_full = 1.0 / np.maximum(sd["sigma_1"].to_numpy(), SIGMA_FLOOR) ** 2
    m = np.array([10.0 if n in _PAID else 1.0 for n in names])
    w = w_full / m ** 2
    wj = json.loads((DATA / "weights.json").read_text(encoding="utf-8"))
    floor = np.array(wj["floor"], dtype=float)
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    code_of = agg["edinetCode"].astype(str)
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]

    def plan(cls: str, src: list[str]) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(DATA / f"plan_{cls}.npz", allow_pickle=True)
        if [str(v) for v in z["source_docs"]] != src:
            raise ValueError("始点の並びが一致しない")
        key = f"{kind}{pool_of(st, cls)}"
        return np.asarray(z[f"{key}__idx"]), np.asarray(z[f"{key}__pt"])

    # λ の尺度 s(class 1、この設定の class 1 のプールとコスト)
    m1 = sources("c1", x, lab, docs_a)
    src1 = [str(d) for d in docs_a[m1]]
    idx1, pt1 = plan("c1", src1)
    num, den = np.zeros(len(src1)), np.zeros(len(src1))
    for i, doc in enumerate(src1):
        x0 = x[m1][i]
        cp, cm = firm_costs(kind, x0, entry(wj, pool_of(st, "c1"), code_of.get(doc, "")), floor,
                            names)
        d = xt[idx1[i]] - x0
        num[i] = (pt1[i] * 0.5 * (w * d ** 2).sum(axis=1)).sum()
        den[i] = (pt1[i] * (np.where(d > 0, cp, cm) * np.abs(d)).sum(axis=1)).sum()
    s = float(np.median(num[den > 0] / den[den > 0]))
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(dd) for dd in docs_a[mask]]
    idx, pt = plan(args.cls, src)
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    lams = list(LAMS)[i_sh::n_sh]
    print(f"[{args.cls} {st}] {kind}{pool_of(st, args.cls)}、s = {s:.4g}、λ̃ = {lams}", flush=True)
    summary = []
    for lt in lams:
        lam = lt * s
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        n_infeas, n_fb = 0, 0
        for i, doc in enumerate(src):
            x0 = xs[i]
            cp, cm = firm_costs(kind, x0, entry(wj, pool_of(st, args.cls), code_of.get(doc, "")),
                                floor, names)
            inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
            frozen_i = FROZEN + tuple(a for a in inactive if a not in FROZEN)
            extra = build_linear_constraints(spec, names, x0, cfg) + liability_cap(members, x0)
            status = "ok"
            try:
                qp = RecourseQP(spec, x0, RecourseWeights(w, na_coef, 0.0, cp, cm, 0.0), lam,
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
        out_dir = DATA / "gen" / args.cls / f"{st}__l{lt:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        for seed in (0, 1, 2):
            df.assign(seed=seed).to_parquet(out_dir / f"nearest_na0__seed{seed}.parquet",
                                            index=False)
        summary.append({"cls": args.cls, "setting": st, "lam_tilde": lt, "lam": lam, "s": s,
                        "n_infeasible": n_infeas, "n_fallback": n_fb,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  λ̃ = {lt:g}: 不可能 {n_infeas}、代替ソルバ {n_fb}"
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    (OUT / "qp").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "qp" / f"{args.cls}_{st}_{i_sh}of{n_sh}.csv", index=False)


if __name__ == "__main__":
    main()
