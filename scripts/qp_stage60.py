#!/usr/bin/env python
"""ステージ60 D.3: 目標点 τ からの凸 QP による生成。

cvxpy のプロセスで、torch・ot・sklearn は import しない。

規則は ``results/stage60/criteria.md`` §3。各始点 x_i について、

    min_y Σ_a w_a (y_a − τ_ia)² + λ Σ_a (c_a⁺ u_a + c_a⁻ v_a)、y − x_i = u − v、u, v ≥ 0

を、s47_final と同じ制約(第 2 層・方向制約・凍結・不活性制約)の下で解く(``project_onto_feasible``、
白色化)。w_a = 1 / max(σ_a^pool, 0.001)²(案 1)。λ = λ̃ × s、s は設定 0(最近傍)の目標点で
第 1 項と第 2 項の大きさをそろえる係数。出力は既存の生成と同じ形式で、QP は決定的なので
seed 0/1/2 に同じ結果を書く(``metrics_stage39.py`` が 3 seed を要求するため)。

    uv run python scripts/qp_stage60.py --cls c2 --lams 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    INACTIVE_ACCOUNTS,
    ConstraintLayers,
    PlausibilityConfig,
    apply_layers,
    build_linear_constraints,
    decompose_re_change,
    inactive_accounts,
)
from fsgen.accounting.projection_qp import load_direction_constraints, project_onto_feasible
from fsgen.errors import InfeasibleProjectionError, SolverNotConvergedError
from fsgen.models.cost import load_cost_config

DATA = Path("data/interim/stage60")
OUT = Path("results/stage60")
_CANONICAL = Path("conf/accounts/core.yaml")
RE_CAP = 0.7565192502545388  # γ(ステージ27・47)
S46_ALPHA = 0.0019850022055580064  # α(ステージ46・47)
SIGMA_FLOOR = 0.001


def s47_final() -> PlausibilityConfig:
    """``run_generation_stage16.make_configs`` の s47_final と同じ設定。

    torch を読み込まないため複製した。
    """
    base = PlausibilityConfig(True, True, 0.5, None, True, retention_beta=0.3,
                              contra_net_nonneg=True, label="W1_S5_a05_b03")
    cfg = dataclasses.replace(base, re_growth_cap=RE_CAP, capital_min_ratio=S46_ALPHA,
                              capital_stock_only=True, treasury_cap=True, label="s47_final")
    return apply_layers(cfg, ConstraintLayers())


def lam_scale(xs: np.ndarray, tau: np.ndarray, w: np.ndarray, cp: np.ndarray,
              cm: np.ndarray) -> float:
    """s = median_i [Σ w (x − τ)² / Σ c± |τ − x|](criteria §3)。"""
    d = tau - xs
    first = (w * d ** 2).sum(axis=1)
    second = (np.where(d > 0, cp, cm) * np.abs(d)).sum(axis=1)
    ok = second > 0
    return float(np.median(first[ok] / second[ok]))


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--lams", default="1", help="λ̃ のカンマ区切り")
    ap.add_argument("--settings", default="", help="対象の設定(カンマ区切り、既定はすべて)")
    ap.add_argument("--shard", default="0/1", help="i/n: 設定を n 分割した i 番目だけ解く")
    ap.add_argument("--check-solver", action="store_true",
                    help="CLARABEL で解き直して既定のソルバとの差を記録する(ステージ10.6)")
    args = ap.parse_args()
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    cfg = s47_final()
    dcs = load_direction_constraints(_CANONICAL)
    na = load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable
    frozen = tuple(n for n in na if n != "jppfs_cor_RetainedEarnings")  # R3(--freeze no_re)
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    w = 1.0 / np.maximum(sd["sigma_1"].to_numpy(), SIGMA_FLOOR) ** 2
    meta = json.loads((DATA / f"meta_{args.cls}.json").read_text(encoding="utf-8"))
    cp, cm = np.array(meta["c_plus"]), np.array(meta["c_minus"])
    costed = (cp > 0) & (cm > 0)  # c = 0(非操作)の科目は u・v の分け方が一意でないので数えない
    tg = np.load(DATA / f"targets_{args.cls}.npz", allow_pickle=True)
    src = [str(d) for d in tg["source_docs"]]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    filer = lab.drop_duplicates("docID").set_index("docID")["filerName"]
    xs = (agg.loc[src, list(names)].fillna(0.0).to_numpy(dtype=float)
          / agg.loc[src, "total_assets"].to_numpy(dtype=float)[:, None])
    s = lam_scale(xs, tg["nn__bary"], w, cp, cm)
    keys = sorted(k for k in tg.files if k.endswith(("__bary", "__argmax")))
    if args.settings:
        want = set(args.settings.split(","))
        keys = [k for k in keys if k.rsplit("__", 1)[0] in want]
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    keys = keys[i_sh::n_sh]
    lams = [float(v) for v in args.lams.split(",")]
    print(f"[{args.cls}] {len(src)} 社、{len(keys)} 目標 × λ̃ {lams}、s = {s:.4g}、"
          f"重み比 1/σ²: {w.max() / w.min():.3g}", flush=True)
    summary: list[dict[str, object]] = []
    for key in keys:
        tau_all = tg[key]
        for lt in lams:
            lam = lt * s
            name = key if lt == 1.0 else f"{key}__l{lt:g}"
            t0 = time.perf_counter()
            rows: list[dict[str, object]] = []
            stat: dict[str, int] = {}
            n_both, n_both6, both_max, n_infeas, max_dev = 0, 0, 0.0, 0, 0.0
            first_t, move_t = [], []
            try:
                for i, doc in enumerate(src):
                    x0 = xs[i]
                    inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
                    frozen_i = frozen + tuple(a for a in inactive if a not in frozen)
                    extra = build_linear_constraints(spec, names, x0, cfg)
                    status, active = "ok", ""
                    try:
                        y, info = project_onto_feasible(
                            tau_all[i], spec, x_start=x0, weights=w, direction_constraints=dcs,
                            frozen_accounts=frozen_i, extra_constraints=extra,
                            l1_move=(cp, cm, lam))
                        stat[info.status] = stat.get(info.status, 0) + 1
                        active = ";".join(info.active_constraints)
                        assert info.move_up is not None and info.move_down is not None
                        both = np.minimum(info.move_up, info.move_down)[costed]  # c > 0 の科目
                        n_both += int((both > 1e-9).sum())
                        n_both6 += int((both > 1e-6).sum())
                        both_max = max(both_max, float(both.max()))
                        if args.check_solver:
                            y2, _ = project_onto_feasible(
                                tau_all[i], spec, x_start=x0, weights=w, direction_constraints=dcs,
                                frozen_accounts=frozen_i, extra_constraints=extra,
                                l1_move=(cp, cm, lam), solver="CLARABEL")
                            max_dev = max(max_dev, float(np.abs(y - y2).max()))
                    except InfeasibleProjectionError:
                        y, status = x0.copy(), "infeasible"
                        n_infeas += 1
                    d = y - x0
                    first_t.append(float((w * (y - tau_all[i]) ** 2).sum()))
                    move_t.append(float((np.where(d > 0, cp, cm) * np.abs(d)).sum()))
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
                summary.append({"target": name, "status": "失敗", "reason": str(e)})
                print(f"  {name}: 失敗 {e}", flush=True)
                continue
            out_dir = DATA / "gen" / args.cls / name
            out_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows)
            for seed in (0, 1, 2):
                df.assign(seed=seed).to_parquet(out_dir / f"nearest_na0__seed{seed}.parquet",
                                                index=False)
            summary.append({"target": name, "status": "ok", "lam_tilde": lt, "lam": lam,
                            "solver_status": json.dumps(stat), "n_infeasible": n_infeas,
                            "n_both_positive": n_both, "n_both_positive_1e6": n_both6,
                            "both_positive_max": both_max, "clarabel_max_dev": max_dev,
                            "first_term_med": float(np.median(first_t)),
                            "move_cost_med": float(np.median(move_t)),
                            "seconds": round(time.perf_counter() - t0, 1)})
            print(f"  {name}: {stat} 両方正 {n_both} 不可能 {n_infeas} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(
        OUT / f"qp_{args.cls}_{args.lams.replace(',', '-')}_{i_sh}of{n_sh}.csv", index=False)
    (DATA / f"lam_scale_{args.cls}.json").write_text(json.dumps({"s": s}), encoding="utf-8")


if __name__ == "__main__":
    main()
