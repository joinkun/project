#!/usr/bin/env python
"""ステージ27 B: 識別 AUC(生成点 vs 実データの class 0/1、副指標)を版ごとに計算する。

lightgbm/sklearn のみ(torch を import しない)。計算はステージ16 と同じ
(`identification_auc_repeated`、seed 0 の 68 社、実データの抽出を 20 回変えた平均)。

    uv run python scripts/auc_stage27.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402

_REPEATS = 20
_CONFIG = "W1_S5_a05_b03"


def load_real(names: tuple[str, ...]) -> pd.DataFrame:
    """ステージ16 の `load_real` と同じ実データ(総資産で割った値、class 2 以外)。"""
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    m = m[m["label"].astype(int) != 2]
    x = m[list(names)].fillna(0.0).div(m["total_assets"].to_numpy(dtype=float), axis=0)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage27"))
    ap.add_argument(
        "--versions", default="r1,r2,r3,r3_cap,r1_median,r3_median,r3_median_cap,old_weights"
    )
    ap.add_argument("--out", type=Path, default=Path("results/stage27/metrics/auc.csv"))
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    real0 = load_real(names).to_numpy(dtype=float)
    rows = []
    start_done = False
    for v in args.versions.split(","):
        df = pd.read_parquet(args.dir / v / "nearest_na0__seed0.parquet")
        p = df[df["method"] == "PCFM"]
        if p["config"].nunique() > 1:
            p = p[p["config"] == _CONFIG]
        if len(p) != 68:
            raise ValueError(f"{v}: PCFM の行数が 68 でない({len(p)})")
        if not start_done:
            x0 = p[[f"x0_{n}" for n in names]].to_numpy(dtype=float)
            r = identification_auc_repeated(x0, real0, _REPEATS)
            rows.append({"version": "始点", "AUC_lgbm": r["lightgbm"].mean(),
                         "AUC_logreg": r["logreg"].mean()})
            start_done = True
        x1 = p[[f"x1_{n}" for n in names]].to_numpy(dtype=float)
        r = identification_auc_repeated(x1, real0, _REPEATS)
        rows.append({"version": v, "AUC_lgbm": r["lightgbm"].mean(),
                     "AUC_logreg": r["logreg"].mean()})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
