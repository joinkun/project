#!/usr/bin/env python
"""ステージ16: 設定(最近傍の距離×射影の重み×α×β)ごとの指標(新しい主指標)。

lightgbm/sklearn/scipyのプロセス(torch/cvxpy/POTを使わない)。
run_generation_stage16.py の parquet({tag}__seed*.parquet)を読む。

区分(ステージ16 §D.1): 必須=12項目のチェック違反0 / 主=総資産変化の絶対値・validity(3分類器) /
副=等重みproximity・コスト重み付きproximity / 参考=sparsity・識別AUC(始点自体を併記)。

使い方:
    uv run python scripts/run_metrics_stage16.py --out D --tags nearest_na0,nearestu_na0
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
    RETENTION_ACCOUNTS,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402
from fsgen.models.cost import cost_vectors, load_cost_config  # noqa: E402

_DIR = Path("data/interim/stage16")
_STUCK_MOVE = 0.02  # 始点からの移動(tilde空間のL2)がこれ未満なら「動けない」
_CASH_HIGH = 0.9  # 現金/総資産がこれを超える=ほぼ全資産を現金化
_CASH_ALL = 0.99
_SPARSITY_THR = 0.01  # row_metricsと同じ(総資産比の変化がこれを超えた科目数)
_AUC_REPEATS = 20


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


def _nontrivial_active(
    sub: pd.DataFrame, prefix: str, names: tuple[str, ...], x0: np.ndarray
) -> set[str]:
    """prefixで始まる制約が拘束し、かつ対象科目の始点が正(0>=0の自明な拘束を除く)の社の集合。"""
    out: set[str] = set()
    for i, (d, s) in enumerate(zip(sub["docID"], sub["active"], strict=True)):
        for k in (s.split(";") if s else []):
            key = k.removeprefix("retention:")
            acct = key.split(">=")[0]
            if k.startswith(prefix) and acct in names and x0[i, names.index(acct)] > 1e-6:
                out.add(d)
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", required=True, help="計画タグ(例: nearest_na0,nearestu_na0)")
    ap.add_argument("--configs", default="", help="設定名のカンマ区切り(省略=全設定)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--auc", action="store_true")
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
    i_cash = names.index("jppfs_cor_CashAndDeposits")
    rows: list[dict[str, object]] = []
    base_auc: dict[str, float] = {}
    frames = [pd.read_parquet(f) for t in args.tags.split(",")
              for f in sorted(_DIR.glob(f"{t}__seed*.parquet"))]
    allx = pd.concat(frames, ignore_index=True)
    allx = allx[allx["method"] == "PCFM"]
    configs = args.configs.split(",") if args.configs else sorted(allx["config"].unique())
    for tag, tsub in allx.groupby("tag", sort=False):
        for config in configs:
            sub = tsub[tsub["config"] == config].sort_values(["seed", "docID"])
            sub = sub.reset_index(drop=True)
            if sub.empty:
                continue
            x0, x1 = sub[x0c].to_numpy(), sub[x1c].to_numpy()
            flags = evaluate_checks(x0, x1, spec, names)
            extra = evaluate_extra_checks(x1, x0, spec, names)
            s0 = (sub["seed"] == 0).to_numpy()
            delta = x1 - x0
            move = np.linalg.norm(delta, axis=1)
            a0, a1 = x0 @ m["jppfs_cor_Assets"], x1 @ m["jppfs_cor_Assets"]
            cash1 = x1[:, i_cash] / a1
            row: dict[str, object] = {"tag": tag, "config": config}
            for j, nm in enumerate(CHECK_NAMES):
                row[nm] = int(flags[s0, j].sum())
            for j, nm in enumerate(EXTRA_CHECK_NAMES):
                row[nm] = int(extra[s0, j].sum())
            anyv = flags.any(axis=1) | extra.any(axis=1)
            row["違反(1〜12,seed0)"] = int(anyv[s0].sum())
            row["違反(1〜12,3seed行)"] = int(anyv.sum())
            row["infeasible"] = int((sub["status"] == "infeasible").sum())
            row["動けない"] = int((move < _STUCK_MOVE).sum())
            row["|総資産変化|"] = float(np.median(np.abs((a1 - a0) / a0)))
            row["総資産変化"] = float(np.median((a1 - a0) / a0))
            for key in ("lightgbm", "logreg", "random_forest"):
                row[f"validity_{key}"] = float(np.median(1 - clfs[key].predict_proba_class2(x1)))
            row["proximity_l2"] = float(np.median(move))
            row["prox_cost_w"] = float(
                np.median(np.sum(np.where(delta >= 0, w_plus, w_minus) * np.abs(delta), axis=1))
            )
            row["sparsity"] = float(np.median(np.sum(np.abs(delta) > _SPARSITY_THR, axis=1)))
            row["現金/資産>0.9(社,seed0)"] = int((cash1[s0] > _CASH_HIGH).sum())
            row["現金/資産>0.9(始点,68社)"] = int(
                (x0[s0, i_cash] / a0[s0] > _CASH_HIGH).sum()
            )
            row["現金=総資産の社(>0.99)"] = int((cash1[s0] > _CASH_ALL).sum())
            row["純資産変化/資産"] = float(np.median((x1 @ m["jppfs_cor_NetAssets"]
                                                 - x0 @ m["jppfs_cor_NetAssets"]) / a0))
            row["恒等式残差"] = float(np.max(np.abs(x1 @ c)))
            act = Counter(a for s in sub["active"] for a in (s.split(";") if s else []))
            cap_firms = _nontrivial_active(sub, "jppfs_cor_Capital", names, x0)
            ret_firms = _nontrivial_active(sub, "retention:", names, x0)
            op = [names.index(n) for n in RETENTION_ACCOUNTS]
            op0, op1 = x0[:, op].sum(axis=1), x1[:, op].sum(axis=1)
            row["運転資産<30%に縮小(社,seed0)"] = int(((op0 > 1e-6) & (op1 < 0.3 * op0))[s0].sum())
            row["資本金拘束(社)"] = len(cap_firms)
            row["事業継続拘束(社)"] = len(ret_firms)
            row["拘束Top"] = "; ".join(f"{k}:{v}" for k, v in act.most_common(4))
            if args.auc:
                res = identification_auc_repeated(x1[s0], real0, _AUC_REPEATS)
                row["AUC_lgbm"] = float(res["lightgbm"].mean())
                row["AUC_logreg"] = float(res["logreg"].mean())
                if not base_auc:
                    rb = identification_auc_repeated(x0[s0], real0, _AUC_REPEATS)
                    base_auc = {"lgbm": float(rb["lightgbm"].mean()),
                                "logreg": float(rb["logreg"].mean())}
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(_DIR / f"summary_{args.out}.csv", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.float_format", "{:.3g}".format)
    chk = [*CHECK_NAMES, *EXTRA_CHECK_NAMES, "違反(1〜12,seed0)", "違反(1〜12,3seed行)"]
    print("=== 会計的チェックの違反社数(68社、seed0) ===")
    print(out[["tag", "config", *chk]].to_string(index=False))
    print("\n=== 主指標(総資産変化の絶対値・validity)と副指標 ===")
    cols = ["tag", "config", "infeasible", "動けない", "|総資産変化|", "総資産変化",
            "validity_lightgbm", "validity_logreg", "validity_random_forest", "proximity_l2",
            "prox_cost_w", "sparsity", "純資産変化/資産", "現金/資産>0.9(社,seed0)",
            "現金/資産>0.9(始点,68社)", "現金=総資産の社(>0.99)", "運転資産<30%に縮小(社,seed0)",
            "資本金拘束(社)", "事業継続拘束(社)", "恒等式残差"]
    print(out[cols].to_string(index=False))
    if args.auc:
        print(f"\n=== 識別AUC(参考)。始点自体: lgbm={base_auc['lgbm']:.3f} "
              f"logreg={base_auc['logreg']:.3f} ===")
        print(out[["tag", "config", "AUC_lgbm", "AUC_logreg"]].to_string(index=False))
    print("\n=== 拘束している制約 ===")
    print(out[["tag", "config", "拘束Top"]].to_string(index=False))


if __name__ == "__main__":
    main()
