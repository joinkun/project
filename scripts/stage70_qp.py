#!/usr/bin/env python
"""ステージ70 B: 新しいコストと目標(argmax・M1・M2)による生成(cvxpy のプロセス)。

規則は ``results/stage70/criteria.md`` §3。QP は ``RecourseQP``(第 1 項は全 32 科目、
η = 0)、制約は s47_final + 現金の下限。新設計: c± は 72 社・ε0 なし、凍結は評価差額金と
不活性科目だけ(非支配株主持分は c = 1/s で動かす)、利益剰余金は設定 6 だけ c = 1/s。
目標は ``stage70_ot.py`` の main 版。

    uv run python scripts/stage70_qp.py --cls c1 --setting s3a_m10 --shard 0/2
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
from qp_stage66 import BETA_CASH, LAMS, lam_scale, sources
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

DATA = Path("data/interim/stage70")
OUT = Path("results/stage70")
_CANONICAL = Path("conf/accounts/core.yaml")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
_RE = "jppfs_cor_RetainedEarnings"
FROZEN = ("valuation_difference_securities",)
# 設定の基本形 → (目標のキー, 利益剰余金にコスト)。それぞれ m = 10 と m = 1 で解く
BASE = {"s2": ("t0.01__argmax", False), "s3a": ("t0.05__m1", False), "s3b": ("t0.1__m1", False),
        "s4": ("bal__m1", False), "s5a": ("t0.05__m2", False), "s5b": ("t0.1__m2", False),
        "s6a": ("t0.05__m1", True), "s6b": ("t0.1__m1", True)}
SETTINGS = {f"{k}_m{m:g}": (key, re, float(m)) for k, (key, re) in BASE.items() for m in (10, 1)}


def move_cost(w: dict[str, list[float]], names: tuple[str, ...], re: bool
              ) -> tuple[np.ndarray, np.ndarray]:
    cp, cm = np.array(w["c_plus"], dtype=float), np.array(w["c_minus"], dtype=float)
    if not re:
        cp[names.index(_RE)] = cm[names.index(_RE)] = 0.0
    return cp, cm


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
    na_coef = members["jppfs_cor_NetAssets"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    w_full = 1.0 / np.maximum(sd["sigma_1"].to_numpy(), SIGMA_FLOOR) ** 2
    key, re, m_paid = SETTINGS[args.setting]
    m = np.array([m_paid if n in _PAID else 1.0 for n in names])
    w_bs = w_full / m ** 2
    wj = json.loads((DATA / "weights.json").read_text(encoding="utf-8"))
    cp, cm = move_cost(wj["pools"]["72"], names, re)
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    code_of = agg["edinetCode"].astype(str)

    def targets(cls: str, src: list[str]) -> np.ndarray:
        z = np.load(DATA / f"targets_{cls}_main.npz", allow_pickle=True)
        if [str(v) for v in z["source_docs"]] != src:
            raise ValueError("目標の始点の並びが一致しない")
        out: np.ndarray = np.asarray(z[key], dtype=float)
        return out

    m1 = sources("c1", x, lab, docs_a)
    s = lam_scale(x[m1], targets("c1", [str(d) for d in docs_a[m1]]),
                  RecourseWeights(w_bs, na_coef, 0.0, cp, cm, 0.0))
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(d) for d in docs_a[mask]]
    tau = targets(args.cls, src)
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    lams = list(LAMS)[i_sh::n_sh]
    print(f"[{args.cls} {args.setting}] 目標 {key}、s = {s:.4g}、λ̃ = {lams}", flush=True)
    summary = []
    for lt in lams:
        lam = lt * s
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        n_infeas, n_fb, n_loo = 0, 0, 0
        for i, doc in enumerate(src):
            x0 = xs[i]
            lk = f"72:{code_of.get(doc, '')}"
            if lk in wj["loo"]:  # 自分自身を除いたコスト
                cpi, cmi = move_cost(wj["loo"][lk], names, re)
                n_loo += 1
            else:
                cpi, cmi = cp, cm
            rw = RecourseWeights(w_bs, na_coef, 0.0, cpi, cmi, 0.0)
            inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
            frozen_i = FROZEN + tuple(a for a in inactive if a not in FROZEN)
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
        summary.append({"cls": args.cls, "setting": args.setting, "target": key,
                        "lam_tilde": lt, "lam": lam, "s": s, "n_infeasible": n_infeas,
                        "n_fallback": n_fb, "n_loo": n_loo,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  λ̃ = {lt:g}: 不可能 {n_infeas}、代替ソルバ {n_fb}、自分を除いたコスト {n_loo}"
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    (OUT / "qp").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "qp" / f"{args.cls}_{args.setting}_{i_sh}of{n_sh}.csv",
                                 index=False)


if __name__ == "__main__":
    main()
