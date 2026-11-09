#!/usr/bin/env python
"""ステージ80: γ' = 0.1 のスパース OT による SP-F1 と、A1・argmax の生成(cvxpy のプロセス)。

規則は ``results/stage80/criteria.md`` §2・§3。その他はステージ75 の F1 と同じ(𝓕⁺、m = 10、新設計、
利益剰余金の移動コスト 0)。

- ``{abs,R1,A1}_sp``: ``data/interim/stage80/plan_*`` の非ゼロの候補すべて(重み = p_ij)。
- ``A1_argmax``: ステージ79 の A1 の Sinkhorn の 1 位。尺度 s は F1 の定義(Sinkhorn の p̃)。

λ = λ̃ × s(s は class 1)。``--cfg full`` は 57 社の全体の A1・SP-F1(単一の生成、§3)。

出力: ``data/interim/stage80/gen_{cfg}/{cls}/{設定}__l{λ̃}``、``results/stage80/qp/``。

    uv run python scripts/stage80_qp.py --cls c1 --cfg f0 --setting A1_sp --shard 0/2
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

DATA = Path("data/interim/stage80")
OUT = Path("results/stage80")
_CANONICAL = Path("conf/accounts/core.yaml")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
_RE = "jppfs_cor_RetainedEarnings"
FROZEN = ("valuation_difference_securities",)
# 設定 → (コスト, 計画のファイルの場所, 計画のキー, argmax か)
SETTINGS = {"abs_sp": ("abs", DATA, "abssp", False), "R1_sp": ("R1", DATA, "R1sp", False),
            "A1_sp": ("A1", DATA, "A1sp", False),
            "A1_argmax": ("A1", Path("data/interim/stage79"), "A1", True)}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--cfg", choices=("f0", "f1", "f2", "f3", "f4", "full"), required=True)
    ap.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    kind, pdir, pkey, argmax = SETTINGS[args.setting]
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cfg = dataclasses.replace(s47_final(), cash_floor=BETA_CASH)
    dcs = load_direction_constraints(_CANONICAL)
    na_coef = members["jppfs_cor_NetAssets"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    sigma = np.maximum(sd["sigma_1"].to_numpy(float), SIGMA_FLOOR)
    w = 1.0 / sigma ** 2 / np.array([10.0 if n in _PAID else 1.0 for n in names]) ** 2
    wsrc = "stage75" if kind == "abs" else "stage79"
    e = json.loads(Path(f"data/interim/{wsrc}/weights.json").read_text(
        encoding="utf-8"))["configs"][args.cfg][kind]
    cp, cm = np.array(e["plus"], dtype=float), np.array(e["minus"], dtype=float)
    cp[names.index(_RE)] = cm[names.index(_RE)] = 0.0
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]

    def plan(cls: str, src: list[str]) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(pdir / f"plan_{cls}_{args.cfg}.npz", allow_pickle=True)
        if [str(v) for v in z["source_docs"]] != src:
            raise ValueError("始点の並びが一致しない")
        return np.asarray(z[f"{pkey}__idx"]), np.asarray(z[f"{pkey}__pt"])

    m1 = sources("c1", x, lab, docs_a)
    x1 = x[m1]
    idx1, pt1 = plan("c1", [str(d) for d in docs_a[m1]])  # s は F1 の重みで計算する
    d1 = xt[idx1] - x1[:, None, :]
    num = (pt1 * 0.5 * (w * d1 ** 2).sum(axis=2)).sum(axis=1)
    den = (pt1 * (np.where(d1 > 0, cp, cm) * np.abs(d1)).sum(axis=2)).sum(axis=1)
    s = float(np.median(num[den > 0] / den[den > 0]))
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(dd) for dd in docs_a[mask]]
    idx, pt = plan(args.cls, src)
    if argmax:
        pt = np.zeros_like(pt)
        pt[:, 0] = 1.0
    rw = RecourseWeights(w, na_coef, 0.0, cp, cm, 0.0)
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    lams = list(LAMS)[i_sh::n_sh]
    print(f"[{args.cls} {args.cfg} {args.setting}] s = {s:.4g}、λ̃ = {lams}", flush=True)
    summary = []
    for lt in lams:
        lam = lt * s
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        n_infeas, n_fb = 0, 0
        for i, doc in enumerate(src):
            x0 = xs[i]
            inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
            frozen_i = FROZEN + tuple(a for a in inactive if a not in FROZEN)
            extra = build_linear_constraints(spec, names, x0, cfg) + liability_cap(members, x0)
            status = "ok"
            try:
                qp = RecourseQP(spec, x0, rw, lam, extra_constraints=extra,
                                direction_constraints=dcs, frozen_accounts=frozen_i)
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
        out_dir = DATA / f"gen_{args.cfg}" / args.cls / f"{args.setting}__l{lt:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        for seed in (0, 1, 2):
            df.assign(seed=seed).to_parquet(out_dir / f"nearest_na0__seed{seed}.parquet",
                                            index=False)
        summary.append({"cls": args.cls, "cfg": args.cfg, "setting": args.setting,
                        "lam_tilde": lt, "lam": lam, "s": s, "n_infeasible": n_infeas,
                        "n_fallback": n_fb, "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  λ̃ = {lt:g}: 不可能 {n_infeas}、代替ソルバ {n_fb}"
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    (OUT / "qp").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(
        OUT / "qp" / f"{args.cls}_{args.cfg}_{args.setting}_{i_sh}of{n_sh}.csv", index=False)


if __name__ == "__main__":
    main()
