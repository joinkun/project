#!/usr/bin/env python
"""ステージ28 B.3・C: 除外後のデータで CAFM を再評価し、3 回の実行の一致を確かめる。

lightgbm/sklearn/scipy のプロセス(torch・POT なし)。``data/interim/stage28/s13_{a,b,c}`` を読む
(同じ設定・同じコードで 3 回生成したもの。ステージ10.5/13 と同じ条件: 旧 costs.yaml の輸送、
W1_uniform・W3_eq_only、seed 0/1/2)。分類器(seed 42)は除外後の実データで学習する。

1. 3 回の実行の一致: 生成された経路、全指標(企業ごとの値)、ペア検定の p 値と CI
2. CAFM − PCFM(validity LightGBM): 差の中央値、95% CI、p、Cliff's delta、対応のある優越

    uv run python scripts/cafm_stage28.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import (  # noqa: E402
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.datasets.exclusion import load_excluded_docs  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.paired import (  # noqa: E402
    aggregate_by_company,
    cliff_magnitude,
    cliffs_delta,
    paired_dominance,
    paired_test,
)

RUNS = ("a", "b", "c")
CONDITIONS = ("W1_uniform", "W3_eq_only")
CAFMS = ("CAFM_scratch", "CAFM_warm")
METHODS = ("FM", "PCFM", "CAFM_scratch", "CAFM_warm")


def load_real(names: tuple[str, ...], excluded: frozenset[str]) -> tuple[np.ndarray, np.ndarray]:
    """分類器の学習データ(総資産で割った値、class 2 か否か)。除外した書類を除く。"""
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


def read_run(directory: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(directory / "*.parquet")))
    if len(files) != len(CONDITIONS) * 3:
        raise FileNotFoundError(f"{directory}: parquet が {len(files)} 個(6 個のはず)")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["condition", "method", "seed", "docID"]).reset_index(drop=True)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage28"))
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/stage28/cafm.md"))
    args = ap.parse_args()

    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    excluded = load_excluded_docs(args.exclude)
    x_real, is2 = load_real(names, excluded)
    clfs = train_classifiers(x_real, is2, seed=42)
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]

    runs = {r: read_run(args.dir / f"s13_{r}") for r in RUNS}
    n_firms = runs["a"]["docID"].nunique()
    leaked = sorted(set(runs["a"]["docID"]) & excluded)
    if leaked:
        raise ValueError(f"除外した書類が生成に残っている: {leaked}")

    # 企業ごとの全指標
    metric_tables: dict[str, pd.DataFrame] = {}
    for r, df in runs.items():
        x0, x1 = df[x0c].to_numpy(), df[x1c].to_numpy()
        out = df[["condition", "method", "seed", "docID"]].copy()
        for key in ("lightgbm", "random_forest", "logreg"):
            out[f"validity_{key}"] = 1 - clfs[key].predict_proba_class2(x1)
        out["proximity"] = np.linalg.norm(x1 - x0, axis=1)
        a0, a1 = x0 @ members["jppfs_cor_Assets"], x1 @ members["jppfs_cor_Assets"]
        out["abs_asset_change"] = np.abs((a1 - a0) / a0)
        flags = np.column_stack(
            [evaluate_checks(x0, x1, spec, names), evaluate_extra_checks(x1, x0, spec, names)]
        )
        out["violation"] = flags.any(axis=1).astype(float)
        metric_tables[r] = out
    metric_cols = [c for c in metric_tables["a"].columns if c not in ("condition", "method",
                                                                       "seed", "docID")]

    lines = [
        "# ステージ28 B.3・C: CAFM の再評価(除外後)", "",
        f"除外後の class 2 は {n_firms} 社。旧 costs.yaml の輸送、seed 0/1/2。"
        "分類器は除外後の実データで学習(seed 42)。差は企業単位(seed の中央値)。", "",
        "## B.3 同じ設定での 3 回の実行の一致", "",
        "| 比較 | 経路の最大差 | 全指標の最大差 | p 値・CI の最大差 |", "|---|---:|---:|---:|",
    ]
    tests: dict[str, dict[tuple[str, str], tuple[float, ...]]] = {}
    for r in RUNS:
        tests[r] = {}
        mt = metric_tables[r]
        for cond in CONDITIONS:
            agg = aggregate_by_company(mt[mt["condition"] == cond], "validity_lightgbm")
            for cafm in CAFMS:
                t = paired_test(agg[cafm].to_numpy(), agg["PCFM"].to_numpy())
                tests[r][(cond, cafm)] = (t.median_diff, t.ci_low, t.ci_high, t.p_value)
    for r in RUNS[1:]:
        dpath = float(np.abs(runs[r][x1c].to_numpy() - runs["a"][x1c].to_numpy()).max())
        dmet = float(
            np.abs(metric_tables[r][metric_cols].to_numpy()
                   - metric_tables["a"][metric_cols].to_numpy()).max()
        )
        dtest = max(
            abs(u - v)
            for key in tests["a"]
            for u, v in zip(tests[r][key], tests["a"][key], strict=True)
        )
        lines.append(f"| a と {r} | {dpath:.3g} | {dmet:.3g} | {dtest:.3g} |")

    lines += [
        "", "## C. CAFM − PCFM(validity LightGBM、実行 a)", "",
        "Cliff's delta は CAFM と PCFM の企業ごとの値を独立な 2 標本とみなした値"
        "(負なら CAFM が低い)。対応のある優越は、同じ企業で CAFM が上回る割合 − 下回る割合。"
        "大きさの目安は "
        "Romano et al. (2006)(|δ| < 0.147 無視できる、< 0.33 小、< 0.474 中)。", "",
        "| 条件 | CAFM | CAFM の中央値 | PCFM の中央値 | 差の中央値 | 95% CI | p | Cliff's δ "
        "| 大きさ | 対応のある優越 |",
        "|---|---|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    mt = metric_tables["a"]
    for cond in CONDITIONS:
        agg = aggregate_by_company(mt[mt["condition"] == cond], "validity_lightgbm")
        for cafm in CAFMS:
            a_v, p_v = agg[cafm].to_numpy(), agg["PCFM"].to_numpy()
            t = paired_test(a_v, p_v)
            d = cliffs_delta(a_v, p_v)
            lines.append(
                f"| {cond} | {cafm} | {np.median(a_v):.4f} | {np.median(p_v):.4f} "
                f"| {t.median_diff:+.4f} | [{t.ci_low:+.4f}, {t.ci_high:+.4f}] | {t.p_value:.2g} "
                f"| {d:+.3f} | {cliff_magnitude(d)} | {paired_dominance(a_v, p_v):+.3f} |"
            )

    lines += [
        "", "## 参考: 方法ごとの指標(実行 a、企業ごとの中央値の中央値)", "",
        "| 条件 | 方法 | validity LGBM | validity RF | validity LogReg | proximity "
        "| \\|総資産変化\\| | 違反した社 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cond in CONDITIONS:
        for meth in METHODS:
            sub = mt[(mt["condition"] == cond) & (mt["method"] == meth)]
            med = sub.groupby("docID")[metric_cols].median()
            viol = int(sub.groupby("docID")["violation"].max().sum())
            lines.append(
                f"| {cond} | {meth} | {med['validity_lightgbm'].median():.4f} "
                f"| {med['validity_random_forest'].median():.4f} "
                f"| {med['validity_logreg'].median():.4f} | {med['proximity'].median():.3f} "
                f"| {med['abs_asset_change'].median():.3f} | {viol} / {n_firms} |"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
