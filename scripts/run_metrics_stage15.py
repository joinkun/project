#!/usr/bin/env python
"""ステージ15 §A〜D: 設定(最近傍の距離×射影の重み)×制約段階ごとの指標。

lightgbm/sklearn/scipyのプロセス(torch/cvxpy/POTを使わない)。
run_projection_stage14.py(--out-dir data/interim/stage15)のparquetを読む。

区分(§C.4): 必須=11項目のチェック違反0 / 主指標=コスト重み付きproximity /
副指標=等重みproximity・sparsity・総資産変化・validity / 参考=識別AUC(始点自体を併記)。
科目別の動き(§C.5): |Δx_a|/Assets の中央値を account_moves_{out}.csv に保存。

使い方:
    uv run python scripts/run_metrics_stage15.py --out stage15 \\
        --files nearest_na0_all:重みあり+W1,nearestu_na0_all:等重み+W1,...
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import (  # noqa: E402
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402
from fsgen.models.cost import cost_vectors, load_cost_config  # noqa: E402

_DIR = Path("data/interim/stage15")
_STUCK_MOVE = 0.02  # 始点からの移動(tilde空間のL2)がこれ未満なら「動けない」
_SPARSITY_THR = 0.01  # row_metricsと同じ(総資産比の変化がこれを超えた科目数)
_AUC_REPEATS = 20
_CAPITAL_ACTIVE = "jppfs_cor_Capital"  # 資本金・資本剰余金の制約名の接頭辞


def load_real(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    x = x / m["total_assets"].to_numpy(dtype=float)[:, None]
    return x, m["label"].astype(int).to_numpy() == 2


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", required=True, help="proj_<X>.parquetのX:ラベル をカンマ区切り")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-auc", action="store_true")
    args = ap.parse_args()

    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    m = subtotal_members(spec, names)
    cc = load_cost_config(Path("conf/accounts/costs.yaml"))
    w_plus, w_minus = cost_vectors(cc, names, non_actionable_mode="keep")
    x_all, is2 = load_real(names)
    clfs = train_classifiers(x_all, is2, seed=42)
    real0 = x_all[~is2]
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]
    j_re = names.index("jppfs_cor_RetainedEarnings")
    signs = [b.sign for b in spec.basis]

    rows: list[dict[str, object]] = []
    moves: list[pd.DataFrame] = []
    base_auc: dict[str, float] = {}
    for spec_s in args.files.split(","):
        fname, label = spec_s.split(":")
        df = pd.read_parquet(_DIR / f"proj_{fname}.parquet")
        for stage, sub in df.groupby("stage", sort=False):
            sub = sub.sort_values(["seed", "docID"]).reset_index(drop=True)
            x0, x1 = sub[x0c].to_numpy(), sub[x1c].to_numpy()
            flags = evaluate_checks(x0, x1, spec, names)
            extra = evaluate_extra_checks(x1, x0, spec, names)
            s0 = (sub["seed"] == 0).to_numpy()
            delta = x1 - x0
            move = np.linalg.norm(delta, axis=1)
            viol = np.zeros(len(sub), dtype=int)
            for j, sg in enumerate(signs):
                if sg == "nonneg":
                    viol += (x1[:, j] < -1e-4).astype(int)
                elif sg == "nonpos":
                    viol += (x1[:, j] > 1e-4).astype(int)
            a0, a1 = x0 @ m["jppfs_cor_Assets"], x1 @ m["jppfs_cor_Assets"]
            na0, na1 = x0 @ m["jppfs_cor_NetAssets"], x1 @ m["jppfs_cor_NetAssets"]
            row: dict[str, object] = {"案": label, "段階": stage}
            for j, nm in enumerate(CHECK_NAMES):
                row[nm] = int(flags[s0, j].sum())
            for j, nm in enumerate(EXTRA_CHECK_NAMES):
                row[nm] = int(extra[s0, j].sum())
            anyv = flags.any(axis=1) | extra.any(axis=1)
            row["何らか違反(1〜11,seed0)"] = int(anyv[s0].sum())
            row["何らか違反(1〜11,3seed行)"] = int(anyv.sum())
            row["infeasible"] = int((sub["status"] == "infeasible").sum())
            row["動けない"] = int((move < _STUCK_MOVE).sum())
            row["prox_cost_w"] = float(
                np.median(np.sum(np.where(delta >= 0, w_plus, w_minus) * np.abs(delta), axis=1))
            )
            row["proximity_l2"] = float(np.median(move))
            row["sparsity"] = float(np.median(np.sum(np.abs(delta) > _SPARSITY_THR, axis=1)))
            row["総資産変化"] = float(np.median((a1 - a0) / a0))
            row["純資産変化/資産"] = float(np.median((na1 - na0) / a0))
            row["RE変化/資産(中央値)"] = float(np.median(delta[:, j_re]))
            row["validity_lgbm"] = float(np.median(1 - clfs["lightgbm"].predict_proba_class2(x1)))
            row["恒等式残差"] = float(np.max(np.abs(x1 @ c)))
            row["符号違反(基礎)"] = int(viol.sum())
            act = Counter(a for s in sub["active"] for a in (s.split(";") if s else []))
            cap = sum(1 for s in sub["active"] if any(
                k.startswith((f"{_CAPITAL_ACTIVE}Stock>=", f"{_CAPITAL_ACTIVE}Surplus>="))
                for k in (s.split(";") if s else [])))
            row["資本金拘束の行(204中)"] = cap
            cap_firms = {d for d, s in zip(sub["docID"], sub["active"], strict=True) if any(
                k.startswith((f"{_CAPITAL_ACTIVE}Stock>=", f"{_CAPITAL_ACTIVE}Surplus>="))
                for k in (s.split(";") if s else []))}
            row["資本金拘束の社(68中)"] = len(cap_firms)
            row["拘束制約Top"] = "; ".join(f"{k}:{v}" for k, v in act.most_common(5))
            if not args.no_auc:
                g0 = x1[s0]
                res = identification_auc_repeated(g0, real0, _AUC_REPEATS)
                row["AUC_lgbm"] = float(res["lightgbm"].mean())
                row["AUC_logreg"] = float(res["logreg"].mean())
                if "start" not in base_auc:
                    rb = identification_auc_repeated(x0[s0], real0, _AUC_REPEATS)
                    base_auc = {"lgbm": float(rb["lightgbm"].mean()),
                                "logreg": float(rb["logreg"].mean()), "start": 1.0}
            rows.append(row)
            mv = pd.DataFrame({
                "案": label, "段階": stage, "account": names,
                "median_abs_move": np.median(np.abs(delta), axis=0),
                "mean_abs_move": np.mean(np.abs(delta), axis=0),
                "cost_plus": w_plus, "cost_minus": w_minus,
            })
            moves.append(mv)
    out = pd.DataFrame(rows)
    out.to_csv(_DIR / f"summary_{args.out}.csv", index=False)
    pd.concat(moves).to_csv(_DIR / f"account_moves_{args.out}.csv", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.float_format", "{:.3g}".format)
    chk = [*CHECK_NAMES, *EXTRA_CHECK_NAMES, "何らか違反(1〜11,seed0)", "何らか違反(1〜11,3seed行)"]
    print("=== 会計的チェックの違反社数(68社、seed0) ===")
    print(out[["案", "段階", *chk]].to_string(index=False))
    print("\n=== 実行可能性・主/副指標 ===")
    cols = ["案", "段階", "infeasible", "動けない", "prox_cost_w", "proximity_l2", "sparsity",
            "総資産変化", "純資産変化/資産", "RE変化/資産(中央値)", "validity_lgbm",
            "恒等式残差", "符号違反(基礎)", "資本金拘束の社(68中)"]
    print(out[cols].to_string(index=False))
    if not args.no_auc:
        print(f"\n=== 識別AUC(参考)。始点自体のAUC: lgbm={base_auc['lgbm']:.3f} "
              f"logreg={base_auc['logreg']:.3f} ===")
        print(out[["案", "段階", "AUC_lgbm", "AUC_logreg"]].to_string(index=False))
    print("\n=== 拘束している制約 ===")
    print(out[["案", "段階", "拘束制約Top"]].to_string(index=False))


if __name__ == "__main__":
    main()
