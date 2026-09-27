#!/usr/bin/env python
"""ステージ24 C: コスト重みの 3 版(+ 旧重み)の生成結果を、企業単位(n=68)のペア比較で比べる。

lightgbm/sklearn/scipy のプロセス(torch/cvxpy/POT なし)。``data/interim/stage24/<版>/`` の
``nearest_na0__seed*.parquet``(最終設定 W1_S5_a05_b03、PCFM)を読む。

- 行ごとの指標: |総資産変化|、validity(LightGBM・RF・LogReg)、等重み proximity(L2)、sparsity
  (総資産比の変化が 0.01 を超えた科目数)、純資産変化 / 総資産。コスト重み付き proximity は使わない
  (重みで選んだ経路を重みで評価するのは自己参照的、ステージ16)
- 13 項目のチェック: 違反した行を社・seed・項目まで出す
- 企業ごとに seed をまたいだ中央値を取り、各版 − 等重み を
  ペア Wilcoxon + ブートストラップ 95% CI で検定
- 科目ごとの動き(|Δx| / 総資産 の中央値)を版ごとに出す(推定ベースで補完した科目との比較用)

    uv run python scripts/run_paired_stage24.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import (  # noqa: E402
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.paired import aggregate_by_company, paired_test  # noqa: E402

_CONFIG = "W1_S5_a05_b03"
_SPARSITY_THR = 0.01
VERSIONS = ("equal", "estimated", "qualitative", "old")
LABELS = {
    "equal": "等重み",
    "estimated": "推定ベース",
    "qualitative": "定性ベース",
    "old": "旧重み(参考)",
}
METRICS = (
    "abs_asset_change",
    "validity_lightgbm",
    "validity_random_forest",
    "validity_logreg",
    "proximity_l2",
    "sparsity",
    "net_asset_change",
)


def load_real(
    names: tuple[str, ...], excluded: frozenset[str] = frozenset()
) -> tuple[np.ndarray, np.ndarray]:
    """分類器の学習データ。``excluded`` の書類を除く(ステージ28)。"""
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    m = m[~m["docID"].isin(excluded)]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    x = x / m["total_assets"].to_numpy(dtype=float)[:, None]
    return x, m["label"].astype(int).to_numpy() == 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage24"))
    ap.add_argument("--out", type=Path, default=Path("results/stage24"))
    ap.add_argument(
        "--estimated-costs", type=Path, default=Path("conf/accounts/costs_estimated.yaml")
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    m = subtotal_members(spec, names)
    x_all, is2 = load_real(names)
    clfs = train_classifiers(x_all, is2, seed=42)
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]
    check_names = [*CHECK_NAMES, *EXTRA_CHECK_NAMES]

    frames = []
    violations = []
    moves = []
    for v in VERSIONS:
        files = sorted(glob.glob(str(args.dir / v / "nearest_na0__seed*.parquet")))
        df = pd.concat([pd.read_parquet(f) for f in files])
        df = df[(df["method"] == "PCFM") & (df["config"] == _CONFIG)].reset_index(drop=True)
        x0, x1 = df[x0c].to_numpy(), df[x1c].to_numpy()
        d = x1 - x0
        a0, a1 = x0 @ m["jppfs_cor_Assets"], x1 @ m["jppfs_cor_Assets"]
        na0, na1 = x0 @ m["jppfs_cor_NetAssets"], x1 @ m["jppfs_cor_NetAssets"]
        flags = np.column_stack(
            [evaluate_checks(x0, x1, spec, names), evaluate_extra_checks(x1, x0, spec, names)]
        )
        for i, j in zip(*np.nonzero(flags), strict=True):
            violations.append(
                {
                    "version": v,
                    "docID": df.at[i, "docID"],
                    "filerName": df.at[i, "filerName"] if "filerName" in df else "",
                    "seed": int(df.at[i, "seed"]),
                    "check": check_names[j],
                }
            )
        frames.append(
            pd.DataFrame(
                {
                    "docID": df["docID"],
                    "seed": df["seed"],
                    "tag": v,
                    "abs_asset_change": np.abs((a1 - a0) / a0),
                    "validity_lightgbm": 1 - clfs["lightgbm"].predict_proba_class2(x1),
                    "validity_random_forest": 1 - clfs["random_forest"].predict_proba_class2(x1),
                    "validity_logreg": 1 - clfs["logreg"].predict_proba_class2(x1),
                    "proximity_l2": np.linalg.norm(d, axis=1),
                    "sparsity": np.sum(np.abs(d) > _SPARSITY_THR, axis=1).astype(float),
                    "net_asset_change": (na1 - na0) / a0,
                }
            )
        )
        rel = np.abs(d) / np.abs(a0)[:, None]
        moves.append(pd.Series(np.median(rel, axis=0), index=names, name=v))

    long = pd.concat(frames, ignore_index=True)
    rows = []
    for metric in METRICS:
        agg = aggregate_by_company(long, metric, group_cols=("tag",))
        for v in VERSIONS[1:]:
            r = paired_test(agg[v].to_numpy(), agg["equal"].to_numpy())
            rows.append(
                {
                    "metric": metric,
                    "version": v,
                    "n": r.n,
                    "median_version": float(agg[v].median()),
                    "median_equal": float(agg["equal"].median()),
                    "median_diff(version-equal)": r.median_diff,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                    "p": r.p_value,
                }
            )
    paired = pd.DataFrame(rows)
    paired.to_csv(args.out / "paired.csv", index=False)
    viol = pd.DataFrame(violations, columns=["version", "docID", "filerName", "seed", "check"])
    viol.to_csv(args.out / "violations.csv", index=False)

    # 科目ごとの動き。推定ベースで補完(1.0)した科目に印を付ける
    est = yaml.safe_load(args.estimated_costs.read_text(encoding="utf-8"))
    filled = {a["name"] for a in est["basis_costs"] if "補完" in a["source"]}
    weight = {a["name"]: a["cost_increase"] for a in est["basis_costs"]}
    mv = pd.concat(moves, axis=1)
    mv.insert(0, "estimated_weight", [weight[n] for n in mv.index])
    mv.insert(0, "filled", [n in filled for n in mv.index])
    mv.to_csv(args.out / "account_moves.csv", index_label="account")

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.4g}".format)
    print("=== 13 項目の違反(3 seed の全行) ===")
    print(viol.groupby(["version", "check"]).size().to_string() if len(viol) else "なし")
    print(viol.to_string(index=False))
    print("\n=== 企業単位(n=68)のペア比較(各版 − 等重み) ===")
    print(paired.to_string(index=False))
    print("\n=== 科目ごとの |Δx|/総資産 の中央値(全行) ===")
    print(mv.to_string())


if __name__ == "__main__":
    main()
