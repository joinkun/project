#!/usr/bin/env python
"""ステージ68 B: 遷移ベースの目標(T0・T1・T2)による生成(cvxpy のプロセス)。

規則は ``results/stage68/criteria.md``(生成の前に記録)。目標は ``targets_stage68.py`` の
``data/interim/stage68/targets_{cls}.npz``。設定はすべて ``RecourseQP``(第 1 項は全 32 科目、
η = 0)、制約は s47_final + 現金の下限:

- ``t0m10``: T0、払込資本 m = 10、c± はステージ60(設定 1)
- ``t0m1``: T0、m = 1(設定 2)
- ``t1m1``: T1、m = 1(設定 3)
- ``t2m1``: T2、m = 1(設定 4)
- ``t2m1re``: T2、m = 1、c± はステージ67(利益剰余金にコスト、設定 5)

λ = λ̃ × s(s は設定ごとに class 1 の始点と目標で計算)。

    uv run python scripts/qp_stage68.py --cls c1 --setting t2m1 --shard 0/2
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
from fsgen.models.cost import load_cost_config

DATA = Path("data/interim/stage68")
OUT = Path("results/stage68")
_CANONICAL = Path("conf/accounts/core.yaml")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
# 設定名 → (目標, 払込資本の m, c±)
SETTINGS = {"t0m10": ("t0", 10.0, "base"), "t0m1": ("t0", 1.0, "base"),
            "t1m1": ("t1", 1.0, "base"), "t2m1": ("t2", 1.0, "base"),
            "t2m1re": ("t2", 1.0, "re")}


def cpm(cls: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    path = (Path(f"data/interim/stage60/meta_{cls}.json") if kind == "base"
            else Path(f"data/interim/stage67/weights_{cls}.json"))
    m = json.loads(path.read_text(encoding="utf-8"))
    return np.array(m["c_plus"]), np.array(m["c_minus"])


def targets(cls: str, src: list[str], key: str) -> np.ndarray:
    z = np.load(DATA / f"targets_{cls}.npz", allow_pickle=True)
    if [str(v) for v in z["source_docs"]] != src:
        raise ValueError("目標の始点の並びが一致しない")
    out: np.ndarray = np.asarray(z[key], dtype=float)
    return out


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
    key, m_paid, ckind = SETTINGS[args.setting]
    m = np.array([m_paid if n in _PAID else 1.0 for n in names])
    w_bs = w_full / m ** 2
    m1 = sources("c1", x, lab, docs_a)
    cp1, cm1 = cpm("c1", ckind)
    s = lam_scale(x[m1], targets("c1", [str(d) for d in docs_a[m1]], key),
                  RecourseWeights(w_bs, na_coef, 0.0, cp1, cm1, 0.0))
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(d) for d in docs_a[mask]]
    cp, cm = cpm(args.cls, ckind)
    rw = RecourseWeights(w_bs, na_coef, 0.0, cp, cm, 0.0)
    tau = targets(args.cls, src, key)
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
