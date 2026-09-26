#!/usr/bin/env python
"""ステージ18 §B.1: 業種×規模制約の5設計を、企業単位(n=68)のペア検定で現行(制約なし)と比較する。

lightgbm/sklearn/scipyのプロセス(torch/cvxpy/POTなし)。run_generation_stage16.pyのparquet
(data/interim/stage18/)を読む。

使い方:
    uv run python scripts/run_paired_stage18.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import subtotal_members  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.paired import aggregate_by_company, paired_test  # noqa: E402
from fsgen.models.cost import cost_vectors, load_cost_config  # noqa: E402

_DIR = Path("data/interim/stage18")
_BASELINE = "nearest_na0"
_DESIGNS = ("d2_hard_none", "d3_none_0.5_2", "d4_hard_0.5_2", "d5_hard_0.33_3", "d6_soft_0.5_2")
_CONFIG = "W1_S5_a05_b03"
_METRICS = ("abs_asset_change", "validity_lightgbm", "validity_random_forest",
            "validity_logreg", "proximity_l2", "prox_cost_w")


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
    cc = load_cost_config(Path("conf/accounts/costs.yaml"))
    w_plus, w_minus = cost_vectors(cc, names, non_actionable_mode="keep")
    x_all, is2 = load_real(names)
    clfs = train_classifiers(x_all, is2, seed=42)
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]

    frames = []
    for tag in (_BASELINE, *_DESIGNS):
        files = sorted(glob.glob(f"{_DIR}/{tag}__seed*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files])
        df = df[(df["method"] == "PCFM") & (df["config"] == _CONFIG)].reset_index(drop=True)
        x0, x1 = df[x0c].to_numpy(), df[x1c].to_numpy()
        d = x1 - x0
        a0, a1 = x0 @ m["jppfs_cor_Assets"], x1 @ m["jppfs_cor_Assets"]
        frames.append(pd.DataFrame({
            "docID": df["docID"], "seed": df["seed"], "tag": tag,
            "abs_asset_change": np.abs((a1 - a0) / a0),
            "validity_lightgbm": 1 - clfs["lightgbm"].predict_proba_class2(x1),
            "validity_random_forest": 1 - clfs["random_forest"].predict_proba_class2(x1),
            "validity_logreg": 1 - clfs["logreg"].predict_proba_class2(x1),
            "proximity_l2": np.linalg.norm(d, axis=1),
            "prox_cost_w": np.sum(np.where(d >= 0, w_plus, w_minus) * np.abs(d), axis=1),
        }))
    long = pd.concat(frames, ignore_index=True)
    rows = []
    for metric in _METRICS:
        agg = aggregate_by_company(long, metric, group_cols=("tag",))
        for design in _DESIGNS:
            r = paired_test(agg[design].to_numpy(), agg[_BASELINE].to_numpy())
            rows.append({
                "metric": metric, "design": design, "n": r.n,
                "median_diff(design-baseline)": r.median_diff,
                "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p_value,
                "design_median": float(agg[design].median()),
                "baseline_median": float(agg[_BASELINE].median()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(_DIR / "paired_B.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.4g}".format)
    print("=== 企業単位(n=68)のペア比較(design − nearest_na0、最終設定) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
