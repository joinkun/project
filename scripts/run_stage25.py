#!/usr/bin/env python
"""ステージ25: 再建の度合いの指標を加えて、ステージ24 の 4 版を再評価する(再生成なし)。

lightgbm/sklearn/scipy のプロセス。``data/interim/stage24/<版>/nearest_na0__seed*.parquet``
(最終設定 W1_S5_a05_b03、PCFM)を読む。

- 必須: 13 項目の違反(3 seed の全行)
- 主(改善): 純資産の改善、債務超過の解消率
- 主(実行可能性): |総資産変化|、sparsity
- 副: validity(3 分類器)、等重み proximity。追加で有利子負債の圧縮、自己資本比率の改善
- 企業ごとに seed をまたいだ中央値(解消は 3 seed のうち解消した割合)を取り、各版 − 等重み を
  ペア Wilcoxon + ブートストラップ 95% CI で検定。全 68 社と、外れ値 2 社を除く 66 社の両方
- 純資産の変化を純資産の部の科目に分解
- 版ごとの重み(純資産の部と主要科目)を並べる

    uv run python scripts/run_stage25.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import sys  # noqa: E402
from collections.abc import Mapping  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from run_paired_stage24 import LABELS, VERSIONS, load_real  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import (  # noqa: E402
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.datasets.exclusion import load_excluded_docs  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.paired import paired_test  # noqa: E402
from fsgen.evaluation.recovery import net_asset_decomposition, recovery_metrics  # noqa: E402

_CONFIG = "W1_S5_a05_b03"  # 既定の設定名。ディレクトリに PCFM の設定が 1 つだけならそれを使う


def select_pcfm(df: pd.DataFrame) -> pd.DataFrame:
    """PCFM の行を返す。設定が 1 つだけならそれを、複数あれば _CONFIG を使う(ステージ27)。"""
    p = df[df["method"] == "PCFM"]
    configs = sorted(p["config"].unique())
    if len(configs) > 1:
        p = p[p["config"] == _CONFIG]
    if p.empty:
        raise ValueError(f"PCFM の行がない(configs={configs})")
    return p
_SPARSITY_THR = 0.01
OUTLIERS = ("S100TSS9", "S100TWFQ")  # オンコセラピー・サイエンス、桃太郎源(HANDOFF §8)
CONTINUOUS = (
    ("net_asset_improvement", "純資産の改善", "大"),
    ("abs_asset_change", "|総資産変化|", "小"),
    ("sparsity", "sparsity", "小"),
    ("validity_lightgbm", "validity LightGBM", "大"),
    ("validity_random_forest", "validity RF", "大"),
    ("validity_logreg", "validity LogReg", "大"),
    ("proximity_l2", "等重み proximity", "小"),
    ("debt_reduction", "有利子負債の圧縮", "大"),
    ("equity_ratio_change", "自己資本比率の改善", "大"),
)
COST_FILES = {
    "equal": "conf/accounts/costs_equal.yaml",
    "estimated": "conf/accounts/costs_estimated.yaml",
    "qualitative": "conf/accounts/costs_qualitative.yaml",
    "old": "conf/accounts/costs.yaml",
}


def rows_for_version(
    directory: Path,
    version: str,
    names: tuple[str, ...],
    members: dict[str, np.ndarray],
    clfs: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    files = sorted(glob.glob(str(directory / version / "nearest_na0__seed*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = select_pcfm(df).reset_index(drop=True)
    x0 = df[[f"x0_{n}" for n in names]].to_numpy()
    x1 = df[[f"x1_{n}" for n in names]].to_numpy()
    d = x1 - x0
    a0, a1 = x0 @ members["jppfs_cor_Assets"], x1 @ members["jppfs_cor_Assets"]
    flags = np.column_stack(
        [evaluate_checks(x0, x1, spec, names), evaluate_extra_checks(x1, x0, spec, names)]
    )
    rec = recovery_metrics(x0, x1, members, names)
    out = pd.DataFrame(
        {
            "docID": df["docID"],
            "filerName": df["filerName"],
            "seed": df["seed"],
            "version": version,
            "violation": flags.any(axis=1),
            "net_asset_improvement": rec.net_asset_improvement,
            "insolvent_start": rec.insolvent_start,
            "insolvency_resolved": rec.insolvency_resolved,
            "debt_reduction": rec.debt_reduction,
            "equity_ratio_change": rec.equity_ratio_change,
            "abs_asset_change": np.abs((a1 - a0) / a0),
            "sparsity": np.sum(np.abs(d) > _SPARSITY_THR, axis=1).astype(float),
            "proximity_l2": np.linalg.norm(d, axis=1),
        }
    )
    for key in ("lightgbm", "random_forest", "logreg"):
        out[f"validity_{key}"] = 1 - clfs[key].predict_proba_class2(x1)
    parts = net_asset_decomposition(x0, x1, members, names)
    dec = pd.DataFrame(parts).assign(docID=df["docID"], seed=df["seed"], version=version)
    return out, dec


def company_level(rows: pd.DataFrame) -> pd.DataFrame:
    """企業 × 版。連続指標は seed をまたいだ中央値、解消は 3 seed のうち解消した割合。"""
    agg = {m: "median" for m, _, _ in CONTINUOUS}
    agg |= {"insolvent_start": "first", "insolvency_resolved": "mean", "violation": "sum"}
    return rows.groupby(["version", "docID"]).agg(agg).reset_index()


def summarize(
    comp: pd.DataFrame,
    exclude: tuple[str, ...],
    versions: tuple[str, ...] = VERSIONS,
    base_name: str = "equal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = comp[~comp["docID"].isin(exclude)]
    wide = {v: g.set_index("docID") for v, g in comp.groupby("version")}
    summary_rows = []
    for v in versions:
        g = wide[v]
        ins = g[g["insolvent_start"]]
        row: dict[str, object] = {
            "version": v,
            "n": len(g),
            "violations(rows)": int(g["violation"].sum()),
            "insolvent_start": len(ins),
            "resolution_rate": float(ins["insolvency_resolved"].mean()),
            "resolved_firms(>=2/3 seeds)": int((ins["insolvency_resolved"] >= 2 / 3).sum()),
        }
        for m, _, _ in CONTINUOUS:
            row[m] = float(g[m].median())
        summary_rows.append(row)
    paired_rows = []
    base = wide[base_name]
    for v in (x for x in versions if x != base_name):
        g = wide[v].reindex(base.index)
        for m, label, better in CONTINUOUS:
            r = paired_test(g[m].to_numpy(), base[m].to_numpy())
            paired_rows.append(
                {
                    "version": v,
                    "metric": label,
                    "better": better,
                    "n": r.n,
                    "median_diff(version-equal)": r.median_diff,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                    "p": r.p_value,
                }
            )
        ins = base["insolvent_start"]
        r = paired_test(
            g.loc[ins, "insolvency_resolved"].to_numpy(),
            base.loc[ins, "insolvency_resolved"].to_numpy(),
        )
        paired_rows.append(
            {
                "version": v,
                "metric": "債務超過の解消率(社ごとの 3 seed 平均)",
                "better": "大",
                "n": r.n,
                "median_diff(version-equal)": r.median_diff,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "p": r.p_value,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage24"))
    ap.add_argument("--out", type=Path, default=Path("results/stage25"))
    ap.add_argument(
        "--versions", default=",".join(VERSIONS), help="比べる版(--dir 直下のディレクトリ名)"
    )
    ap.add_argument("--base", default="equal", help="ペア比較の基準にする版")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="分類器の学習から除く書類の CSV(列 docID、ステージ28)")
    ap.add_argument("--outliers", default=",".join(OUTLIERS),
                    help="評価から除く外れ値の docID(カンマ区切り。ステージ36 で class 1 用に追加)")
    args = ap.parse_args()
    versions = tuple(args.versions.split(","))
    args.out.mkdir(parents=True, exist_ok=True)

    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    x_all, is2 = load_real(names, load_excluded_docs(args.exclude))
    clfs = train_classifiers(x_all, is2, seed=42)

    rows, decs = [], []
    for v in versions:
        r, d = rows_for_version(args.dir, v, names, members, clfs)
        rows.append(r)
        decs.append(d)
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(args.out / "rows.csv", index=False)
    comp = company_level(long)
    n_firms = comp["docID"].nunique()
    outliers = tuple(d for d in args.outliers.split(",") if d)
    n_out = len(set(outliers) & set(comp["docID"]))
    for tag, excl in ((f"all{n_firms}", ()), (f"excl_outliers{n_firms - n_out}", outliers)):
        summary, paired = summarize(comp, excl, versions, args.base)
        summary.to_csv(args.out / f"summary_{tag}.csv", index=False)
        paired.to_csv(args.out / f"paired_{tag}.csv", index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.float_format", "{:.4g}".format)
        print(f"=== {tag}: 版ごとの要約(企業の中央値) ===")
        print(summary.to_string(index=False))
        print(f"=== {tag}: 各版 − 等重み ===")
        print(paired.to_string(index=False))

    # 純資産の変化の分解(企業ごとに seed の中央値 → 版ごとに企業の中央値と平均)
    dec = pd.concat(decs, ignore_index=True)
    parts = [c for c in dec.columns if c not in ("docID", "seed", "version")]
    dec_c = dec.groupby(["version", "docID"])[parts].median().reset_index()
    dec_c["増資(資本金+資本剰余金)"] = (
        dec_c["jppfs_cor_CapitalStock"] + dec_c["jppfs_cor_CapitalSurplus"]
    )
    dec_c["純資産の改善(合計)"] = dec_c[parts].sum(axis=1)
    cols = [*parts, "増資(資本金+資本剰余金)", "純資産の改善(合計)"]
    table = pd.concat(
        {
            "median": dec_c.groupby("version")[cols].median(),
            "mean": dec_c.groupby("version")[cols].mean(),
        }
    ).T
    table.to_csv(args.out / "net_asset_decomposition.csv")
    print("=== 純資産の変化の分解(総資産比、企業の中央値 / 平均) ===")
    print(table.to_string())

    # 版ごとの重み
    weights: dict[str, dict[str, float]] = {}
    for v, path in COST_FILES.items():
        if v not in versions:
            continue
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        for a in doc["basis_costs"]:
            weights.setdefault(a["name"], {})[f"{LABELS[v]} w+"] = a["cost_increase"]
            weights[a["name"]][f"{LABELS[v]} w-"] = a["cost_decrease"]
    w = pd.DataFrame(weights).T
    w.to_csv(args.out / "weights_by_version.csv", index_label="account")


if __name__ == "__main__":
    main()
