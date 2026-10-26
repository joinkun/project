#!/usr/bin/env python
"""ステージ72: 安全マージンつき 𝓕⁺ での F1(事後アンサンブル)と比較の設定(cvxpy のプロセス)。

規則は ``results/stage72/criteria.md``。新設計(ステージ70・71)、制約は s47_final + 現金の下限 +
負債の上限 L(y) ≤ (1 − 10⁻⁶) max(2 L(x0), 0.01 A(x0))。

- ``f1`` / ``f1m1``: τ1 = 0.05 の S(i) の事後アンサンブル(m = 10 / 1)
- ``a2`` / ``a2t05``: argmax(τ1 = 0.01 / 0.05)
- ``bal_l2``: バランス OT の事前 L2 重心(計画の全体)
- ``f1bal``: バランス OT の S(i) の事後アンサンブル
- ``l2nn``: 等重み L2 の最近傍 + 移動コスト λ Σ (y − x)²(利益剰余金・評価差額金を除く)
- ``pre05`` / ``pre05m1``: τ1 = 0.05 の S(i) で先に混ぜた目標(L2 重心 / 重み付き中央値)

    uv run python scripts/stage72_qp.py --cls c1 --setting f1 --shard 0/2
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
from stage70_qp import move_cost
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    INACTIVE_ACCOUNTS,
    LinearConstraint,
    build_linear_constraints,
    decompose_re_change,
    inactive_accounts,
    subtotal_members,
)
from fsgen.accounting.projection_qp import load_direction_constraints, project_onto_feasible
from fsgen.accounting.recourse_qp import RecourseQP, RecourseWeights
from fsgen.errors import InfeasibleProjectionError

DATA = Path("data/interim/stage72")
OUT = Path("results/stage72")
_CANONICAL = Path("conf/accounts/core.yaml")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
FROZEN = ("valuation_difference_securities",)
NO_L2 = ("jppfs_cor_RetainedEarnings", "valuation_difference_securities")
MARGIN = 1.0 - 1e-6
# 設定 → (種類, 払込資本の m)。種類: ens = 事後アンサンブル、tau = 目標 1 点、l2 = 等重み L2
SETTINGS = {"f1": ("ens", 10.0), "f1m1": ("ens", 1.0), "a2": ("tau", 10.0),
            "a2t05": ("tau", 10.0), "bal_l2": ("tau", 10.0), "f1bal": ("ens", 10.0),
            "l2nn": ("l2", 10.0), "pre05": ("tau", 10.0), "pre05m1": ("tau", 10.0),
            # 追補(addendum_criteria.md §3): F1 の τ1 の感度
            "f1t001": ("ens", 10.0), "f1t01": ("ens", 10.0)}


def liability_cap(members: dict[str, np.ndarray], x0: np.ndarray) -> tuple[LinearConstraint, ...]:
    """負債の上限 L(y) ≤ (1 − 10⁻⁶) max(2 L(x0), 0.01 A(x0))(L(x0) ≤ 0 なら課さない)。"""
    la = members["jppfs_cor_Liabilities"]
    l0 = float(la @ x0)
    if l0 <= 0:
        return ()
    cap = MARGIN * max(2.0 * l0, 0.01 * float(members["jppfs_cor_Assets"] @ x0))
    return (LinearConstraint("Liabilities<=(1-1e-6)*max(2*start,0.01*Assets)", -la, -cap),)


def load(st: str, cls: str, src: list[str]) -> dict[str, np.ndarray]:
    """設定ごとの目標(tau・l2)または候補(ens)。"""
    if st in ("a2", "a2t05", "bal_l2"):
        z = np.load(f"data/interim/stage70/targets_{cls}_main.npz", allow_pickle=True)
        key = {"a2": "t0.01__argmax", "a2t05": "t0.05__argmax", "bal_l2": "bal__l2"}[st]
    elif st in ("f1", "f1m1"):
        z = np.load(f"data/interim/stage71/plan_{cls}.npz", allow_pickle=True)
        key = "t0.05"
    elif st in ("f1t001", "f1t01"):
        z = np.load(DATA / f"tau_{cls}.npz", allow_pickle=True)
        key = {"f1t001": "t0.01", "f1t01": "t0.1"}[st]
    else:
        z = np.load(DATA / f"plan_{cls}.npz", allow_pickle=True)
        key = {"f1bal": "bal", "l2nn": "l2nn", "pre05": "pre05_l2", "pre05m1": "pre05_m1"}[st]
    if [str(v) for v in z["source_docs"]] != src:
        raise ValueError("始点の並びが一致しない")
    if SETTINGS[st][0] == "ens":
        return {"idx": np.asarray(z[f"{key}__idx"]), "pt": np.asarray(z[f"{key}__pt"])}
    return {"tau": np.asarray(z[key], dtype=float)}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--gen-root", type=Path, default=DATA / "gen", help="生成の出力先")
    args = ap.parse_args()
    st = args.setting
    kind, m_paid = SETTINGS[st]
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cfg = dataclasses.replace(s47_final(), cash_floor=BETA_CASH)
    dcs = load_direction_constraints(_CANONICAL)
    na_coef = members["jppfs_cor_NetAssets"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    w_full = 1.0 / np.maximum(sd["sigma_1"].to_numpy(), SIGMA_FLOOR) ** 2
    m = np.array([m_paid if n in _PAID else 1.0 for n in names])
    w = w_full / m ** 2
    act = np.array([n not in NO_L2 for n in names], dtype=float)
    wj = json.loads(Path("data/interim/stage70/weights.json").read_text(encoding="utf-8"))
    cp0, cm0 = move_cost(wj["pools"]["72"], names, False)
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    code_of = agg["edinetCode"].astype(str)
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]

    def psi(d: np.ndarray) -> np.ndarray:
        out: np.ndarray = (np.where(d > 0, cp0, cm0) * np.abs(d)).sum(axis=-1)
        return out

    m1 = sources("c1", x, lab, docs_a)
    x1 = x[m1]
    t1 = load(st, "c1", [str(d) for d in docs_a[m1]])
    if kind == "ens":
        d = xt[t1["idx"]] - x1[:, None, :]
        num = (t1["pt"] * 0.5 * (w * d ** 2).sum(axis=2)).sum(axis=1)
        den = (t1["pt"] * psi(d)).sum(axis=1)
        s = float(np.median(num[den > 0] / den[den > 0]))
    elif kind == "l2":
        d = t1["tau"] - x1
        num, den = 0.5 * (w * d ** 2).sum(axis=1), (act * d ** 2).sum(axis=1)
        s = float(np.median(num[den > 0] / den[den > 0]))
    else:
        s = lam_scale(x1, t1["tau"], RecourseWeights(w, na_coef, 0.0, cp0, cm0, 0.0))
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(dd) for dd in docs_a[mask]]
    tg = load(st, args.cls, src)
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    lams = list(LAMS)[i_sh::n_sh]
    print(f"[{args.cls} {st}] s = {s:.4g}、λ̃ = {lams}", flush=True)
    summary = []
    for lt in lams:
        lam = lt * s
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        n_infeas, n_fb = 0, 0
        for i, doc in enumerate(src):
            x0 = xs[i]
            lk = f"72:{code_of.get(doc, '')}"
            cpi, cmi = move_cost(wj["loo"][lk], names, False) if lk in wj["loo"] else (cp0, cm0)
            inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
            frozen_i = FROZEN + tuple(a for a in inactive if a not in FROZEN)
            extra = build_linear_constraints(spec, names, x0, cfg) + liability_cap(members, x0)
            status = "ok"
            try:
                if kind == "l2":
                    wl = w + 2.0 * lam * act
                    tau = (w * tg["tau"][i] + 2.0 * lam * act * x0) / wl
                    # 移動コストが 0 の RecourseQP は cvxpy が扱えないので、ステージ61 と同じく
                    # project_onto_feasible で解く(収束と制約違反を厳密に検査する)
                    y, _info = project_onto_feasible(tau, spec, x_start=x0, weights=wl,
                                                     direction_constraints=dcs,
                                                     frozen_accounts=frozen_i,
                                                     extra_constraints=extra)
                else:
                    rw = RecourseWeights(w, na_coef, 0.0, cpi, cmi, 0.0)
                    qp = RecourseQP(spec, x0, rw, lam, extra_constraints=extra,
                                    direction_constraints=dcs, frozen_accounts=frozen_i)
                    if kind == "ens":
                        k = tg["pt"][i] > 0
                        ys = [qp.solve(xt[j]).y for j in tg["idx"][i][k]]
                        y = (tg["pt"][i][k][:, None] * np.array(ys)).sum(axis=0)
                    else:
                        y = qp.solve(tg["tau"][i]).y
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
        out_dir = args.gen_root / args.cls / f"{st}__l{lt:g}"
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
