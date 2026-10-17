#!/usr/bin/env python
"""ステージ67 C: 利益剰余金に移動コストを与えた QP による生成(cvxpy のプロセス)。

規則は ``results/stage67/criteria.md``(生成の前に記録)。設定はすべて ``RecourseQP``。
対応先はステージ64 の不均衡 OT(τ1 = 0.01・ε = 0.001)の argmax、制約は s47_final + 現金の下限:

- ``s1r``: 第 1 項は全 32 科目・払込資本 m = 10、c± は中央値(利益剰余金 0)、η = 0
  (設定 1 の再現の確認)
- ``s2``: s1r + 利益剰余金の移動コスト
- ``s3``: s2 の第 1 項を階層分離(A_BS 20 科目 + NA、m なし)に
- ``s4``: s2 + η = 1.0
- ``s5``: s3 + η = 1.0

λ = λ̃ × s(s は設定ごとに class 1 の対応先で計算)。

    uv run python scripts/qp_stage67.py --cls c1 --setting s2 --shard 0/2
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
from qp_stage66 import BETA_CASH, LAMS, ROUND_RESIDUALS, lam_scale, sources, targets
from tilde_stage51 import load_tilde
from variance_stage58 import pooled_within

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
from fsgen.models.cost import load_cost_config

DATA = Path("data/interim/stage67")
OUT = Path("results/stage67")
_CANONICAL = Path("conf/accounts/core.yaml")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
# 設定名 → (第 1 項, c±, η)
SETTINGS = {"s1r": ("full", "base", 0.0), "s2": ("full", "re", 0.0), "s3": ("hier", "re", 0.0),
            "s4": ("full", "re", 1.0), "s5": ("hier", "re", 1.0)}


def cpm(cls: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    path = (Path(f"data/interim/stage60/meta_{cls}.json") if kind == "base"
            else DATA / f"weights_{cls}.json")
    m = json.loads(path.read_text(encoding="utf-8"))
    return np.array(m["c_plus"]), np.array(m["c_minus"])


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cfg = dataclasses.replace(s47_final(), cash_floor=BETA_CASH)
    dcs = load_direction_constraints(_CANONICAL)
    na = load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable
    frozen = tuple(n for n in na if n != "jppfs_cor_RetainedEarnings")
    na_coef = members["jppfs_cor_NetAssets"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    w_full = 1.0 / np.maximum(sd["sigma_1"].to_numpy(), SIGMA_FLOOR) ** 2
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    first, ckind, eta = SETTINGS[args.setting]
    if first == "full":
        m = np.array([10.0 if n in _PAID else 1.0 for n in names])
        w_bs, w_na = w_full / m ** 2, 0.0
    else:
        in_bs = np.array([(c == 0) and n not in ROUND_RESIDUALS
                          for n, c in zip(names, na_coef, strict=True)])
        w_bs = np.where(in_bs, w_full, 0.0)
        w_na = 1.0 / max(float(pooled_within((x @ na_coef)[:, None], lab)[0]), SIGMA_FLOOR) ** 2
    xt = x[lab == 0]
    m1 = sources("c1", x, lab, docs_a)
    cp1, cm1 = cpm("c1", ckind)
    s = lam_scale(x[m1], targets("c1", [str(d) for d in docs_a[m1]], xt, "ot64"),
                  RecourseWeights(w_bs, na_coef, w_na, cp1, cm1, eta))
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(d) for d in docs_a[mask]]
    cp, cm = cpm(args.cls, ckind)
    rw = RecourseWeights(w_bs, na_coef, w_na, cp, cm, eta)
    tau = targets(args.cls, src, xt, "ot64")
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    lams = list(LAMS)[i_sh::n_sh]
    print(f"[{args.cls} {args.setting}] s = {s:.4g}、λ̃ = {lams}", flush=True)
    summary = []
    for lt in lams:
        lam = lt * s
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        n_infeas, n_fb = 0, 0
        for i, doc in enumerate(src):
            x0 = xs[i]
            inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
            frozen_i = frozen + tuple(a for a in inactive if a not in frozen)
            extra = build_linear_constraints(spec, names, x0, cfg)
            qp = RecourseQP(spec, x0, rw, lam, extra_constraints=extra, direction_constraints=dcs,
                            frozen_accounts=frozen_i)
            status = "ok"
            try:
                y = qp.solve(tau[i]).y
            except InfeasibleProjectionError:
                y, status = x0.copy(), "infeasible"
                n_infeas += 1
            n_fb += qp.n_fallback
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
        out_dir = DATA / "gen" / args.cls / f"{args.setting}__l{lt:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        for seed in (0, 1, 2):
            df.assign(seed=seed).to_parquet(out_dir / f"nearest_na0__seed{seed}.parquet",
                                            index=False)
        summary.append({"cls": args.cls, "setting": args.setting, "lam_tilde": lt, "lam": lam,
                        "s": s, "n_infeasible": n_infeas, "n_fallback": n_fb,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  λ̃ = {lt:g}: 不可能 {n_infeas}、代替ソルバ {n_fb}"
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    (OUT / "qp").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "qp" / f"{args.cls}_{args.setting}_{i_sh}of{n_sh}.csv",
                                 index=False)


if __name__ == "__main__":
    main()
