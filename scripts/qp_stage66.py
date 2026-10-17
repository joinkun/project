#!/usr/bin/env python
"""ステージ66 B: 階層分離・頻度加重・Elastic-Net の QP による生成(cvxpy のプロセス)。

規則は ``results/stage66/criteria.md``(実装・生成の前に記録)。設定:

- ``s2``: 階層分離、中央値の c±(ステージ59〜65)、η = 0、対応先はステージ64 の不均衡 OT の argmax
- ``s3``: 階層分離、頻度加重の c±、η = 0、対応先は頻度加重のコストの不均衡 OT の argmax
- ``s4``: s3 + η = 1.0(提案)。``s4e05``・``s4e2`` は η = 0.5・2.0 の感度
- ``s5``: s4 + 1 段階統合(下界で枝刈りし、QP の目的関数が最小の class 0 企業を対応先にする、K = 50)

λ = λ̃ × s、s は設定ごとに class 1 の対応先で計算する(s5 は s4 の値)。出力は既存の生成と同じ形式で、
QP は決定的なので seed 0/1/2 に同じ結果を書く。

    uv run python scripts/qp_stage66.py --cls c1 --setting s4 --shard 0/5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from prep_stage59 import OUTLIERS
from qp_stage60 import SIGMA_FLOOR, s47_final
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
from fsgen.accounting.recourse_qp import (
    RecourseQP,
    RecourseResult,
    RecourseWeights,
    lower_bound,
)
from fsgen.errors import InfeasibleProjectionError
from fsgen.models.cost import load_cost_config

DATA = Path("data/interim/stage66")
OUT = Path("results/stage66")
_CANONICAL = Path("conf/accounts/core.yaml")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
BETA_CASH = 0.11732
K = 50
ROUND_RESIDUALS = ("__OTHER__jppfs_cor_Assets", "__OTHER__jppfs_cor_NoncurrentAssets",
                   "__OTHER__jppfs_cor_Liabilities")
SETTINGS = {"s2": ("median", 0.0, "ot64"), "s3": ("freq", 0.0, "ot66"),
            "s4": ("freq", 1.0, "ot66"), "s4e05": ("freq", 0.5, "ot66"),
            "s4e2": ("freq", 2.0, "ot66"), "s5": ("freq", 1.0, "integrate")}


def sources(cls: str, x: np.ndarray, lab: np.ndarray, docs: np.ndarray) -> np.ndarray:
    if cls == "c2":
        out: np.ndarray = (lab == 2) & ~np.isin(docs, OUTLIERS)
        return out
    c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
    out = np.isin(docs, c1.loc[c1["reason"].fillna("") == "", "docID"].to_numpy())
    return out


def weights(cls: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "median":
        m = json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(encoding="utf-8"))
    else:
        m = json.loads((DATA / f"weights_{cls}.json").read_text(encoding="utf-8"))
    return np.array(m["c_plus"]), np.array(m["c_minus"])


def targets(cls: str, src: list[str], xt: np.ndarray, kind: str) -> np.ndarray:
    if kind == "ot64":
        z = np.load(f"data/interim/stage64/targets_{cls}.npz", allow_pickle=True)
        key = "semi_t0.01_e0.001__argmax"
    else:
        z = np.load(DATA / f"targets_{cls}.npz", allow_pickle=True)
        key = "semi"
    if [str(v) for v in z["source_docs"]] != src:
        raise ValueError("目標の始点の並びが一致しない")
    out: np.ndarray = xt[z[key].astype(int)]
    return out


def lam_scale(xs: np.ndarray, tau: np.ndarray, rw: RecourseWeights) -> float:
    """s = median_i [D(x_i, τ_i) / Ψ(τ_i − x_i)](criteria §1)。"""
    first = 0.5 * (rw.w_bs * (xs - tau) ** 2).sum(axis=1) + 0.5 * rw.w_na * (
        (xs - tau) @ rw.na_coef) ** 2
    d = tau - xs
    up = np.maximum(d, 0) * rw.c_plus
    dn = np.maximum(-d, 0) * rw.c_minus
    move = (up + dn).sum(axis=1) + 0.5 * rw.eta * (up ** 2 + dn ** 2).sum(axis=1)
    ok = move > 0
    return float(np.median(first[ok] / move[ok]))


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    ap.add_argument("--lams", default=",".join(f"{v:g}" for v in LAMS))
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
    in_bs = np.array([(c == 0) and n not in ROUND_RESIDUALS for n, c in zip(names, na_coef,
                                                                             strict=True)])
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    w_bs = np.where(in_bs, 1.0 / np.maximum(sd["sigma_1"].to_numpy(), SIGMA_FLOOR) ** 2, 0.0)
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    sig_na = float(pooled_within((x @ na_coef)[:, None], lab)[0])
    w_na = 1.0 / max(sig_na, SIGMA_FLOOR) ** 2
    wkind, eta, tkind = SETTINGS[args.setting]
    xt = x[lab == 0]
    # λ のスケールは class 1 の対応先で(s5 は s4 と同じ対応先)
    m1 = sources("c1", x, lab, docs_a)
    cp1, cm1 = weights("c1", wkind)
    tk1 = "ot66" if tkind == "integrate" else tkind
    rw1 = RecourseWeights(w_bs, na_coef, w_na, cp1, cm1, eta)
    s = lam_scale(x[m1], targets("c1", [str(d) for d in docs_a[m1]], xt, tk1), rw1)
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(d) for d in docs_a[mask]]
    cp, cm = weights(args.cls, wkind)
    rw = RecourseWeights(w_bs, na_coef, w_na, cp, cm, eta)
    tau_ot = targets(args.cls, src, xt, tk1)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]
    lams = [float(v) for v in args.lams.split(",")]
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    lams = lams[i_sh::n_sh]
    print(f"[{args.cls} {args.setting}] |A_BS| = {int(in_bs.sum())}、w_NA = {w_na:.2f}"
          f"(σ = {sig_na:.4f})、s = {s:.4g}、λ̃ = {lams}", flush=True)
    summary = []
    for lt in lams:
        lam = lt * s
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        n_extra, n_unpruned, n_infeas, n_fb = 0, 0, 0, 0
        for i, doc in enumerate(src):
            x0 = xs[i]
            inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
            frozen_i = frozen + tuple(a for a in inactive if a not in frozen)
            extra = build_linear_constraints(spec, names, x0, cfg)
            qp = RecourseQP(spec, x0, rw, lam, extra_constraints=extra, direction_constraints=dcs,
                            frozen_accounts=frozen_i)
            status, tgt = "ok", ""
            try:
                if tkind == "integrate":
                    lb = lower_bound(x0, xt, rw, lam)
                    order = np.argsort(lb)
                    best: tuple[float, int, RecourseResult | None] = (np.inf, -1, None)
                    solved = set()
                    for j in order[:K]:
                        r = qp.solve(xt[j])
                        solved.add(int(j))
                        if r.objective < best[0]:
                            best = (r.objective, int(j), r)
                    rest = [int(j) for j in order[K:] if lb[j] < best[0]]
                    n_unpruned += len(rest)
                    while rest:
                        j = rest.pop(0)
                        if lb[j] >= best[0] or j in solved:
                            continue
                        r = qp.solve(xt[j])
                        solved.add(j)
                        n_extra += 1
                        if r.objective < best[0]:
                            best = (r.objective, j, r)
                    assert best[2] is not None
                    res, tgt = best[2], str(docs_a[lab == 0][best[1]])
                else:
                    res = qp.solve(tau_ot[i])
                y = res.y
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
                "n_inactive": len(inactive), "target_doc": tgt}
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
                        "s": s, "n_infeasible": n_infeas, "n_unpruned_beyond_K": n_unpruned,
                        "n_extra_qp": n_extra, "n_fallback": n_fb,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  λ̃ = {lt:g}: 不可能 {n_infeas}、K 以降で枝刈りされない候補 {n_unpruned}、追加"
              f"の QP "
              f"{n_extra}({time.perf_counter() - t0:.0f}s)", flush=True)
    (OUT / "qp").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "qp" / f"{args.cls}_{args.setting}_{i_sh}of{n_sh}.csv",
                                 index=False)


if __name__ == "__main__":
    main()
