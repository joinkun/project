#!/usr/bin/env python
"""ステージ14 §B/C: 保存済みのFM生出力を、追加制約の段階(S0〜S4)ごとに再射影する。

FMの学習は射影の制約に依存しない(PCFMはサンプリング時のみ射影)ので、ステージ13の
FM生出力(method=="FM": x0=始点、x1=射影前の終点)をそのまま使い、射影だけをやり直す。
cvxpy/numpy/pandasのみ(torch/lightgbm/sklearn/POTを使わない)。

段階(§B.3):
  S0  現行(基礎科目の符号のみ)
  S1  + 小計・総合計の非負
  S2  + 純資産の非悪化
  S3  + 資本金・資本剰余金の下限 α(α=0.5。S3_a0 / S3_a1 は α=0 / 1.0)
  S4  + 総資産の変化幅 [0.5, 2.0](S1〜S3と併用)。S4b は S0+(4)のみ(併用可否の検討)
  S5r 残余科目の非負のみ / S5 = S3+残余科目の非負 / S6 = S5+総資産の幅
      (残余科目=__OTHER__*のうち実データで負が実質0の7科目。§Bの4制約だけでは、
       固定負債その他が負になって小計の非負をすり抜ける抜け穴が残るため追加)
--freeze: all=REを含む非操作科目を等式で固定(R1/R2) / no_re=REの固定を外す(R3)

出力: data/interim/stage14/proj_{tag}_{freeze}.parquet
  行=(stage, seed, docID)。status(ok/infeasible)、active(拘束制約)、x0_*/x1_*(tilde)

使い方:
    uv run python scripts/run_projection_stage14.py --tag nearest_na0 --freeze all
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import PlausibilityConfig, build_linear_constraints
from fsgen.accounting.projection_qp import load_direction_constraints, project_onto_feasible
from fsgen.errors import InfeasibleProjectionError
from fsgen.models.cost import combined_projection_weights, cost_vectors, load_cost_config

_STAGES: dict[str, PlausibilityConfig] = {
    "S0": PlausibilityConfig(label="S0"),
    "S1": PlausibilityConfig(subtotal_nonneg=True, label="S1"),
    "S2": PlausibilityConfig(subtotal_nonneg=True, net_assets_non_deterioration=True, label="S2"),
    "S3_a0": PlausibilityConfig(True, True, 0.0, label="S3_a0"),
    "S3": PlausibilityConfig(True, True, 0.5, label="S3"),
    "S3_a1": PlausibilityConfig(True, True, 1.0, label="S3_a1"),
    "S4": PlausibilityConfig(True, True, 0.5, (0.5, 2.0), label="S4"),
    "S4b": PlausibilityConfig(assets_ratio_range=(0.5, 2.0), label="S4b"),
    # ステージ14で見つけた抜け穴(残余科目の負)を塞ぐ段階
    "S5r": PlausibilityConfig(residual_nonneg=True, label="S5r"),  # 残余科目の非負のみ
    # ステージ16: 残余科目の下限は総資産比-1e-3(丸め許容)。貸倒引当金の追加で緩和は不要
    "S5": PlausibilityConfig(True, True, 0.5, None, True, label="S5"),  # S3 + 残余科目の下限0
    "S5_a1": PlausibilityConfig(True, True, 1.0, None, True, label="S5_a1"),  # α=1.0(減資禁止)
    "S6": PlausibilityConfig(True, True, 0.5, (0.5, 2.0), True, label="S6"),  # S5 + 総資産の幅
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="ステージ13の設計タグ(例: nearest_na0)")
    ap.add_argument("--freeze", choices=("all", "no_re"), default="all")
    ap.add_argument("--stages", default=",".join(_STAGES))
    ap.add_argument("--condition", default="W1_uniform", help="射影の重み(W1_uniform/W3_eq_only)")
    ap.add_argument("--fm-condition", default="W1_uniform", help="FM生出力を読む条件")
    ap.add_argument("--out-dir", default="data/interim/stage14")
    args = ap.parse_args()

    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    dcs = load_direction_constraints(Path("conf/accounts/core.yaml"))
    cc = load_cost_config(Path("conf/accounts/costs.yaml"))
    frozen = tuple(cc.non_actionable)
    if args.freeze == "no_re":
        frozen = tuple(n for n in frozen if n != "jppfs_cor_RetainedEarnings")
    dim = len(names)
    if args.condition == "W1_uniform":
        w_plus = w_minus = np.ones(dim)
    elif args.condition == "W3_eq_only":  # 生成スクリプトと同じ(costs.yamlの値をそのまま=keep)
        w_plus, w_minus = cost_vectors(cc, names, non_actionable_mode="keep")
    else:
        raise SystemExit(f"未対応の条件: {args.condition}")

    stem = f"{args.tag}__{args.fm_condition}"
    files = sorted(glob.glob(f"data/interim/stage13/{stem}__seed*.parquet"))
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fm = fm[fm["method"] == "FM"].sort_values(["seed", "docID"]).reset_index(drop=True)
    x0_cols = [f"x0_{n}" for n in names]
    x1_cols = [f"x1_{n}" for n in names]
    print(f"{args.tag}/{args.condition}: FM生出力 {len(fm)}行、固定={frozen}")

    rows: list[dict[str, object]] = []
    for stage in args.stages.split(","):
        cfg = _STAGES[stage]
        t0 = time.perf_counter()
        n_inf = 0
        for _, r in fm.iterrows():
            x_start = r[x0_cols].to_numpy(dtype=float)
            y = r[x1_cols].to_numpy(dtype=float)
            weights = combined_projection_weights(x_start, 1.0, w_plus, w_minus, 1e-3)
            extra = build_linear_constraints(spec, names, x_start, cfg)
            row: dict[str, object] = {
                "stage": stage, "seed": int(r["seed"]), "fold": int(r["fold"]),
                "docID": r["docID"], "filerName": r["filerName"],
                "assets_start": float(r["assets_start"]),
            }
            try:
                x_proj, info = project_onto_feasible(
                    y, spec, x_start=x_start, weights=weights, direction_constraints=dcs,
                    frozen_accounts=frozen, extra_constraints=extra,
                )
                row["status"] = "ok"
                row["active"] = ";".join(info.active_constraints)
            except InfeasibleProjectionError:
                x_proj = x_start.copy()  # 動かせない(実行不可能)。件数を数える
                row["status"] = "infeasible"
                row["active"] = ""
                n_inf += 1
            for j, n in enumerate(names):
                row[f"x0_{n}"] = float(x_start[j])
                row[f"x1_{n}"] = float(x_proj[j])
            rows.append(row)
        print(f"  {stage:<6} 完了 infeasible={n_inf}/{len(fm)}({time.perf_counter() - t0:.0f}s)",
              flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.condition == "W1_uniform" else f"_{args.condition}"
    out = out_dir / f"proj_{args.tag}_{args.freeze}{suffix}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"出力: {out}")


if __name__ == "__main__":
    main()
