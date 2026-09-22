#!/usr/bin/env python
"""ステージ13 §D 追加: 識別AUCの原因の仮説検証(lightgbm/sklearnのプロセス、torchなし)。

D.1で「__OTHER__*(残余科目)」が識別の主因と出たため、次を測って原因を切り分ける。
 (a) 全科目でのAUC(基準)
 (b) __OTHER__* 科目を除いたAUC(実データでは厳密に0の科目が多いが、生成点は0にならない)
 (c) 参照をRE<0のclass0に限ったAUC(REは固定なので、生成点は始点のRE負を保つ)
 (d) (b)+(c)
 (e) 参照点: 始点そのもの(class2実データ68社)は class0 とどれだけ識別されるか
 (f) 相関のFrobenius距離の標本ノイズの床: class0から68社を無作為抽出→全class0

使い方:
    uv run python scripts/run_diagnose_identification_ablation.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402
from fsgen.evaluation.metrics import correlation_frobenius_distance  # noqa: E402

_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_REPEATS = 20
_SETS = {
    "NN_W1(最近傍, PCFM)": ("data/interim/stage13/nearest_na0__W1_uniform__seed*.parquet", "PCFM"),
    "balanced_W1(PCFM)": ("data/interim/stage13/bal_na0__W1_uniform__seed*.parquet", "PCFM"),
    "11A_convex_qp": ("data/interim/generation_11a/generated_endpoints.parquet", "convex_qp"),
}


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    agg = pd.read_csv(_AGGREGATED_CSV, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS_CSV, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    x = x / m["total_assets"].to_numpy(dtype=float)[:, None]
    lb = m["label"].astype(int).to_numpy()
    real0, real2 = x[lb == 0], x[lb == 2]
    j_re = names.index("jppfs_cor_RetainedEarnings")
    other = [i for i, n in enumerate(names) if n.startswith("__OTHER__")]
    keep = [i for i in range(len(names)) if i not in other]
    real0_reneg = real0[real0[:, j_re] < 0]
    print(f"class0={len(real0)}社(うちRE<0: {len(real0_reneg)}社)、__OTHER__科目={len(other)}個")
    zero_share = np.mean(np.abs(real0[:, other]) < 1e-6, axis=0)
    print("class0で厳密に0の割合(__OTHER__科目): "
          + ", ".join(f"{names[i][14:]}={z:.0%}" for i, z in zip(other, zero_share, strict=True)))

    rows = []

    def auc(gen: np.ndarray, ref: np.ndarray, cols: list[int]) -> tuple[float, float]:
        r = identification_auc_repeated(gen[:, cols], ref[:, cols], _REPEATS)
        return float(r["lightgbm"].mean()), float(r["logreg"].mean())

    all_cols = list(range(len(names)))
    for label, (pattern, method) in _SETS.items():
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pattern))])
        df = df[(df["method"] == method) & (df["seed"] == 0)]
        gen = df[[f"x1_{n}" for n in names]].to_numpy()
        for tag, ref, cols in (
            ("(a)全科目/全class0", real0, all_cols),
            ("(b)__OTHER__除く/全class0", real0, keep),
            ("(c)全科目/RE<0のclass0", real0_reneg, all_cols),
            ("(d)__OTHER__除く/RE<0のclass0", real0_reneg, keep),
        ):
            lg, lr = auc(gen, ref, cols)
            rows.append({"生成点": label, "条件": tag, "AUC_LightGBM": lg, "AUC_LogReg": lr})
    # (e) 始点そのもの(class2実データ)
    for tag, ref, cols in (
        ("(a)全科目/全class0", real0, all_cols),
        ("(b)__OTHER__除く/全class0", real0, keep),
        ("(d)__OTHER__除く/RE<0のclass0", real0_reneg, keep),
    ):
        lg, lr = auc(real2, ref, cols)
        rows.append({"生成点": "始点(class2実データ68社)", "条件": tag,
                     "AUC_LightGBM": lg, "AUC_LogReg": lr})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n=== 識別AUCの切り分け(seed0の68点 vs class0、20回平均) ===")
    print(out.to_string(index=False))
    out.to_csv("data/interim/stage13/diagnosis/ablation_auc.csv", index=False)

    # (f) 相関距離のノイズの床
    rng = np.random.default_rng(0)
    floor = [correlation_frobenius_distance(real0[rng.choice(len(real0), 68, replace=False)], real0)
             for _ in range(200)]
    print("\n=== 相関のFrobenius距離の基準(30科目) ===")
    print(f"  class0から68社を無作為抽出 vs 全class0(標本ノイズの床): "
          f"{np.mean(floor):.2f} ± {np.std(floor):.2f}")
    start_d = correlation_frobenius_distance(real2, real0)
    print(f"  始点(class2実データ68社) vs class0: {start_d:.2f}")
    keep_d = correlation_frobenius_distance(real2[:, keep], real0[:, keep])
    print(f"  始点(class2実データ68社) vs class0(__OTHER__除く): {keep_d:.2f}")


if __name__ == "__main__":
    main()
