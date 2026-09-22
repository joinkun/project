#!/usr/bin/env python
"""ステージ15 §C.4: 4設定の比較を、企業単位(n=68)のペア検定+ブートストラップCIで示す。

lightgbm/sklearn/scipyのプロセス(torch/cvxpy/POTなし)。run_projection_stage14.pyのparquet
(data/interim/stage15/)を読み、指定段階の行ごと指標を企業(seedをまたいだ中央値)に集約して
差を検定する。設定名: ①等重み+W1 ②等重み+W3 ③重みあり+W1 ④重みあり+W3。

使い方:
    uv run python scripts/run_paired_stage15.py --stage S5_a1
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import subtotal_members  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.paired import aggregate_by_company, paired_test  # noqa: E402
from fsgen.models.cost import cost_vectors, load_cost_config  # noqa: E402

_DIR = Path("data/interim/stage15")
_FILES = {
    "1_等重み+W1": "nearestu_na0_all",
    "2_等重み+W3": "nearestu_na0_all_W3_eq_only",
    "3_重みあり+W1": "nearest_na0_all",
    "4_重みあり+W3": "nearest_na0_all_W3_eq_only",
}
_PAIRS = (
    ("3_重みあり+W1", "1_等重み+W1"),  # 輸送の距離の効果(W1)
    ("4_重みあり+W3", "2_等重み+W3"),  # 輸送の距離の効果(W3)
    ("2_等重み+W3", "1_等重み+W1"),  # 射影の重みの効果(等重み輸送)
    ("4_重みあり+W3", "3_重みあり+W1"),  # 射影の重みの効果(重みあり輸送)
    ("4_重みあり+W3", "1_等重み+W1"),  # 両方
)
_METRICS = ("prox_cost_w", "proximity_l2", "asset_change", "validity_lgbm", "sparsity")


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="S5_a1")
    args = ap.parse_args()
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
    for label, f in _FILES.items():
        df = pd.read_parquet(_DIR / f"proj_{f}.parquet")
        df = df[df["stage"] == args.stage].reset_index(drop=True)
        x0, x1 = df[x0c].to_numpy(), df[x1c].to_numpy()
        d = x1 - x0
        a0, a1 = x0 @ m["jppfs_cor_Assets"], x1 @ m["jppfs_cor_Assets"]
        frames.append(pd.DataFrame({
            "docID": df["docID"], "seed": df["seed"], "tag": label,
            "prox_cost_w": np.sum(np.where(d >= 0, w_plus, w_minus) * np.abs(d), axis=1),
            "proximity_l2": np.linalg.norm(d, axis=1),
            "asset_change": (a1 - a0) / a0,
            "validity_lgbm": 1 - clfs["lightgbm"].predict_proba_class2(x1),
            "sparsity": np.sum(np.abs(d) > 0.01, axis=1),
        }))
    long = pd.concat(frames, ignore_index=True)
    rows = []
    for metric in _METRICS:
        agg = aggregate_by_company(long, metric, group_cols=("tag",))
        for a, b in _PAIRS:
            r = paired_test(agg[a].to_numpy(), agg[b].to_numpy())
            rows.append({
                "metric": metric, "a": a, "b": b, "n": r.n, "median_diff(a-b)": r.median_diff,
                "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p_value,
                "a_median": float(agg[a].median()), "b_median": float(agg[b].median()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(_DIR / f"paired_{args.stage}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.4g}".format)
    print(f"=== 企業単位(n=68)のペア比較(段階 {args.stage}、a−b) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
