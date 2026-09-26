#!/usr/bin/env python
"""ステージ19 §B: 現行(設計1)と規模制約(0.5, 2.0)(設計3)の trade-off を、外れ値2社の有無で示す。

lightgbm/sklearn/scipyのプロセス(torch/cvxpy/POTなし)。data/interim/stage18 の生成結果を読み、
(a) 68社全体 (b) 始点が極端な外れ値の2社(資本剰余金・欠損が総資産の16〜30倍)を除く66社
のそれぞれで、12+1項目の違反行数、主指標の企業単位ペア比較(n社、95%CI)を出す。

使い方:
    uv run python scripts/run_tradeoff_stage19.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

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
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.paired import aggregate_by_company, paired_test  # noqa: E402

_DIR = Path("data/interim/stage18")
_OUT = Path("results/stage19")
_CONFIG = "W1_S5_a05_b03"
_DESIGNS = {"1_現行": "nearest_na0", "3_規模(0.5,2.0)": "d3_none_0.5_2"}
_OUTLIERS = ("S100TSS9", "S100TWFQ")
_METRICS = ("abs_asset_change", "validity_lightgbm", "validity_random_forest",
            "validity_logreg", "proximity_l2")


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


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    m = subtotal_members(spec, names)
    x_all, is2 = load_real(names)
    clfs = train_classifiers(x_all, is2, seed=42)
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]
    frames = []
    for label, tag in _DESIGNS.items():
        files = sorted(glob.glob(f"{_DIR}/{tag}__seed*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files])
        df = df[(df["method"] == "PCFM") & (df["config"] == _CONFIG)].reset_index(drop=True)
        x0, x1 = df[x0c].to_numpy(), df[x1c].to_numpy()
        flags = np.column_stack([evaluate_checks(x0, x1, spec, names),
                                 evaluate_extra_checks(x1, x0, spec, names)])
        a0, a1 = x0 @ m["jppfs_cor_Assets"], x1 @ m["jppfs_cor_Assets"]
        frames.append(pd.DataFrame({
            "docID": df["docID"], "seed": df["seed"], "tag": label,
            "violation": flags.any(axis=1),
            "abs_asset_change": np.abs((a1 - a0) / a0),
            "validity_lightgbm": 1 - clfs["lightgbm"].predict_proba_class2(x1),
            "validity_random_forest": 1 - clfs["random_forest"].predict_proba_class2(x1),
            "validity_logreg": 1 - clfs["logreg"].predict_proba_class2(x1),
            "proximity_l2": np.linalg.norm(x1 - x0, axis=1),
        }))
    long = pd.concat(frames, ignore_index=True)
    rows = []
    scopes = (("68社", long), ("外れ値2社を除く66社", long[~long["docID"].isin(_OUTLIERS)]))
    for scope, sub in scopes:
        viol = sub.groupby("tag")["violation"].sum()
        for metric in _METRICS:
            agg = aggregate_by_company(sub, metric, group_cols=("tag",))
            a, b = "3_規模(0.5,2.0)", "1_現行"
            r = paired_test(agg[a].to_numpy(), agg[b].to_numpy())
            rows.append({
                "対象": scope, "指標": metric, "n": r.n,
                "現行_中央値": float(agg[b].median()), "規模_中央値": float(agg[a].median()),
                "差(規模−現行)": r.median_diff, "CI_low": r.ci_low, "CI_high": r.ci_high,
                "p": r.p_value, "違反行_現行": int(viol[b]), "違反行_規模": int(viol[a]),
            })
    out = pd.DataFrame(rows)
    _OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(_OUT / "tradeoff_size.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.4g}".format)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
