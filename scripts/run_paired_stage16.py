#!/usr/bin/env python
"""ステージ16 §D.2: 5設定の比較を、企業単位(n=68)のペア検定+ブートストラップCIで示す。

lightgbm/sklearn/scipyのプロセス(torch/cvxpy/POTなし)。run_generation_stage16.pyのparquetを読み、
設定ごとの行指標を企業(seedをまたいだ中央値)に集約して差を検定する。
設定: 1=等重み+W1+α0.5 / 2=等重み+W1+α1.0 / 3=コスト重み+W1+α0.5 / 4=コスト重み+W1+α1.0 /
      5=コスト重み+W3+α0.5(毎ステップ射影で再生成)。β(事業継続)は --beta で指定(既定b03)。

使い方:
    uv run python scripts/run_paired_stage16.py --beta b03
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
from fsgen.accounting.plausibility import subtotal_members  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.paired import aggregate_by_company, paired_test  # noqa: E402
from fsgen.models.cost import cost_vectors, load_cost_config  # noqa: E402

_DIR = Path("data/interim/stage16")
_PAIRS = (
    ("3", "1"),  # 輸送の距離(コスト重み − 等重み)、W1・α0.5
    ("4", "2"),  # 輸送の距離、W1・α1.0
    ("2", "1"),  # α(1.0 − 0.5)、等重み
    ("4", "3"),  # α(1.0 − 0.5)、コスト重み
    ("5", "3"),  # 射影の重み(W3 − W1)、コスト重み・α0.5
    ("5", "1"),
)
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beta", default="b03")
    args = ap.parse_args()
    b = args.beta
    settings = {
        "1": ("nearestu_na0", f"W1_S5_a05_{b}"), "2": ("nearestu_na0", f"W1_S5_a10_{b}"),
        "3": ("nearest_na0", f"W1_S5_a05_{b}"), "4": ("nearest_na0", f"W1_S5_a10_{b}"),
        "5": ("nearest_na0", f"W3_S5_a05_{b}"),
    }
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
    for key, (tag, cfg) in settings.items():
        files = sorted(glob.glob(f"{_DIR}/{tag}__seed*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files])
        df = df[(df["method"] == "PCFM") & (df["config"] == cfg)].reset_index(drop=True)
        x0, x1 = df[x0c].to_numpy(), df[x1c].to_numpy()
        d = x1 - x0
        a0, a1 = x0 @ m["jppfs_cor_Assets"], x1 @ m["jppfs_cor_Assets"]
        frames.append(pd.DataFrame({
            "docID": df["docID"], "seed": df["seed"], "tag": key,
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
        for a, bb in _PAIRS:
            r = paired_test(agg[a].to_numpy(), agg[bb].to_numpy())
            rows.append({
                "metric": metric, "a": a, "b": bb, "n": r.n, "median_diff(a-b)": r.median_diff,
                "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p_value,
                "a_median": float(agg[a].median()), "b_median": float(agg[bb].median()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(_DIR / f"paired_{b}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.4g}".format)
    print(f"設定: {settings}")
    print(f"=== 企業単位(n=68)のペア比較(β={b}、a−b) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
