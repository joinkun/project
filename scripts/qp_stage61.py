#!/usr/bin/env python
"""ステージ61 A: λ のグリッド × 距離 3 通りの QP 生成。

cvxpy のプロセスで、torch・ot・sklearn は import しない。規則は ``results/stage61/criteria.md``。
目標は各距離での最近傍(実在企業)。距離は対応先の選択と移動コストの両方に使う:

- ``l2``: 対応先 = 等重み L2²(非操作 0)、移動コスト λ Σ_act (y − x)²。第 1 項と合わせて
  Σ (w + λ·act)(y − ỹ)²、ỹ = (w τ + λ·act x) / (w + λ·act) の射影として解く。
- ``l1s``: 対称 L1(方向を分けない |Δ| の中央値の尺度)、移動コスト λ Σ c |Δ|。
- ``l1a``: 非対称 L1(c±、ステージ59・60)、移動コスト λ Σ (c⁺u + c⁻v)。

λ = λ̃ × s_d(s_d は最近傍の目標点で 2 項の大きさをそろえる係数)。出力は既存の生成と同じ形式
(QP は決定的なので seed 0/1/2 に同じ結果)。

    uv run python scripts/qp_stage61.py --cls c2 --shard 0/8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from density_stage51 import load_tilde
from prep_stage59 import OUTLIERS, deltas, l1_cost, l2_cost
from qp_stage60 import SIGMA_FLOOR, s47_final
from scales_stage59 import DEFAULT, SCALES, scale_table

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    INACTIVE_ACCOUNTS,
    build_linear_constraints,
    decompose_re_change,
    inactive_accounts,
)
from fsgen.accounting.projection_qp import load_direction_constraints, project_onto_feasible
from fsgen.errors import InfeasibleProjectionError, SolverNotConvergedError
from fsgen.models.cost import load_cost_config

DATA = Path("data/interim/stage61")
OUT = Path("results/stage61")
_CANONICAL = Path("conf/accounts/core.yaml")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DISTS = ("l2", "l1s", "l1a")


def sym_weights(d: np.ndarray, nact: np.ndarray) -> np.ndarray:
    """方向を分けない |Δ|(Δ ≠ 0)の中央値の尺度の c(scales_stage59 と同じ規則)。"""
    t = scale_table(np.where(d != 0, np.abs(d), 0.0), nact, SCALES[DEFAULT])
    out: np.ndarray = t[t["dir"] == "+"].sort_values("j")["c"].to_numpy()
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    cfg = s47_final()
    dcs = load_direction_constraints(_CANONICAL)
    na = load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable
    nact = np.array([n in set(na) for n in names])
    act = (~nact).astype(float)
    frozen = tuple(n for n in na if n != "jppfs_cor_RetainedEarnings")
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    w = 1.0 / np.maximum(sd["sigma_1"].to_numpy(), SIGMA_FLOOR) ** 2
    meta_path = Path(f"data/interim/stage60/meta_{args.cls}.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cp, cm = np.array(meta["c_plus"]), np.array(meta["c_minus"])
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    cs = sym_weights(deltas(pairs[pairs["group"] == meta["weights_group"]]["edinetCode"].tolist(),
                            names), nact)
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    if args.cls == "c2":
        is_src = (lab == 2) & ~np.isin(docs_a, OUTLIERS)
    else:
        c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
        is_src = np.isin(docs_a, c1.loc[c1["reason"].fillna("") == "", "docID"].to_numpy())
    xs, xt = x[is_src], x[lab == 0]
    src = docs_a[is_src].tolist()
    tau = {"l2": xt[l2_cost(xs, xt, act).argmin(axis=1)],
           "l1s": xt[l1_cost(xs, xt, cs, cs).argmin(axis=1)],
           "l1a": xt[l1_cost(xs, xt, cp, cm).argmin(axis=1)]}

    def move(d: str, delta: np.ndarray) -> np.ndarray:
        if d == "l2":
            out: np.ndarray = (act * delta ** 2).sum(axis=-1)
        elif d == "l1s":
            out = (cs * np.abs(delta)).sum(axis=-1)
        else:
            out = (np.where(delta > 0, cp, cm) * np.abs(delta)).sum(axis=-1)
        return out

    scale = {}
    for d in DISTS:
        first = (w * (tau[d] - xs) ** 2).sum(axis=1)
        m = move(d, tau[d] - xs)
        scale[d] = float(np.median(first[m > 0] / m[m > 0]))
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    lab_df = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    filer = lab_df.drop_duplicates("docID").set_index("docID")["filerName"]
    jobs = [(d, lt) for d in DISTS for lt in LAMS]
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    jobs = jobs[i_sh::n_sh]
    print(f"[{args.cls}] {len(src)} 社、s = {scale}、{len(jobs)} 組", flush=True)
    summary: list[dict[str, object]] = []
    for d, lt in jobs:
        lam = lt * scale[d]
        name = f"{d}_l{lt:g}"
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        stat: dict[str, int] = {}
        first_t, move_t = [], []
        n_inf = 0
        try:
            for i, doc in enumerate(src):
                x0, t_i = xs[i], tau[d][i]
                inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
                frozen_i = frozen + tuple(a for a in inactive if a not in frozen)
                extra = build_linear_constraints(spec, names, x0, cfg)
                status, active = "ok", ""
                try:
                    if d == "l2":
                        wt = w + lam * act
                        y, info = project_onto_feasible(
                            (w * t_i + lam * act * x0) / wt, spec, x_start=x0, weights=wt,
                            direction_constraints=dcs, frozen_accounts=frozen_i,
                            extra_constraints=extra)
                    else:
                        c_up, c_dn = (cs, cs) if d == "l1s" else (cp, cm)
                        y, info = project_onto_feasible(
                            t_i, spec, x_start=x0, weights=w, direction_constraints=dcs,
                            frozen_accounts=frozen_i, extra_constraints=extra,
                            l1_move=(c_up, c_dn, lam))
                    stat[info.status] = stat.get(info.status, 0) + 1
                    active = ";".join(info.active_constraints)
                except InfeasibleProjectionError:
                    y, status = x0.copy(), "infeasible"
                    n_inf += 1
                first_t.append(float((w * (y - t_i) ** 2).sum()))
                move_t.append(float(move(d, y - x0)))
                tr, pr = decompose_re_change(x0, y, names)
                row: dict[str, object] = {
                    "tag": "nearest_na0", "config": "s47_final", "fold": 0, "method": "PCFM",
                    "docID": doc, "filerName": str(filer.get(doc, "")),
                    "assets_start": float(agg.loc[doc, "total_assets"]), "status": status,
                    "active": active, "re_branch": -1, "re_transfer": float(tr),
                    "re_profit": float(pr), "n_inactive": len(inactive)}
                for j, n in enumerate(names):
                    row[f"x0_{n}"] = float(x0[j])
                    row[f"x1_{n}"] = float(y[j])
                rows.append(row)
        except SolverNotConvergedError as e:
            summary.append({"setting": name, "dist": d, "lam_tilde": lt, "status": "失敗",
                            "reason": str(e)})
            print(f"  {name}: 失敗 {e}", flush=True)
            continue
        out_dir = DATA / "gen" / args.cls / name
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        for seed in (0, 1, 2):
            df.assign(seed=seed).to_parquet(out_dir / f"nearest_na0__seed{seed}.parquet",
                                            index=False)
        summary.append({"setting": name, "dist": d, "lam_tilde": lt, "lam": lam, "status": "ok",
                        "solver_status": json.dumps(stat), "n_infeasible": n_inf,
                        "first_term_med": float(np.median(first_t)),
                        "move_cost_med": float(np.median(move_t)),
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  {name}: {stat} 不可能 {n_inf} ({time.perf_counter() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / f"qp_{args.cls}_{i_sh}of{n_sh}.csv", index=False)
    targets = {d: np.array(tau[d]) for d in DISTS}
    np.savez_compressed(DATA / f"targets_{args.cls}.npz", source_docs=np.array(src),
                        **targets)  # type: ignore[arg-type]
    (DATA / f"scale_{args.cls}.json").write_text(json.dumps(scale), encoding="utf-8")


if __name__ == "__main__":
    main()
