#!/usr/bin/env python
"""ステージ13 §B.1/B.2: 収束版OTでのFM/PCFM/CAFMの比較と、企業単位のペア検定。

lightgbm/sklearn/scipyのプロセス(torchは使わない)。run_generation_stage13.pyの
parquetを読み、行ごとの指標を計算し、CAFM−PCFMの差を**企業単位に集約したn=68**で
ペアWilcoxon検定する(集約前のp値と、企業ブートストラップ2,000回の95%信頼区間も併記)。
ステージ10.5のデータ(data/interim/flow_matching_w*/metrics.csv)にも同じ検定を
かけ、非収束OT・擬似反復のもとでの結論と対比する。

使い方:
    uv run python scripts/run_metrics_stage13.py --tag balanced
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.paired import (  # noqa: E402
    aggregate_by_company,
    paired_test,
    unaggregated_p_value,
)
from fsgen.evaluation.row_metrics import compute_row_metrics  # noqa: E402
from fsgen.models.cost import cost_vectors, load_cost_config  # noqa: E402

_CANONICAL = Path("conf/accounts/core.yaml")
_COSTS = Path("conf/accounts/costs.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DIR = Path("data/interim/stage13")
_OLD = {
    "W1": Path("data/interim/flow_matching_w1/metrics.csv"),
    "W2": Path("data/interim/flow_matching_w2/metrics.csv"),
    "W3": Path("data/interim/flow_matching/metrics.csv"),
}
_OLD_NAMES = {"PCFM_every": "PCFM", "CAFM_warmstart": "CAFM_warm"}
_METRICS = ("proximity_l2", "validity_lightgbm")


def load_real(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    agg = pd.read_csv(_AGGREGATED_CSV, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS_CSV, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    x = x / m["total_assets"].to_numpy(dtype=float)[:, None]
    return x, m["label"].astype(int).to_numpy() == 2


def paired_table(df: pd.DataFrame, title: str, cafm_methods: tuple[str, ...]) -> pd.DataFrame:
    """CAFM−PCFM を、企業単位に集約したn=68で検定(集約前のp値も併記)。"""
    rows = []
    for cafm in cafm_methods:
        for metric in _METRICS:
            agg = aggregate_by_company(df, metric)
            res = paired_test(agg[cafm].to_numpy(), agg["PCFM"].to_numpy())
            p_raw = unaggregated_p_value(df, metric, cafm, "PCFM")
            rows.append({
                "比較": f"{cafm} − PCFM", "指標": metric, "n": res.n,
                "差の中央値": res.median_diff, "CI95_low": res.ci_low, "CI95_high": res.ci_high,
                "p(企業集約n=68)": res.p_value, "p(集約前)": p_raw,
            })
    out = pd.DataFrame(rows)
    print(f"\n--- {title} ---")
    with pd.option_context("display.width", 220, "display.float_format", "{:.4g}".format):
        print(out.to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="balanced")
    args = ap.parse_args()

    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    cost_config = load_cost_config(_COSTS)
    w_plus, w_minus = cost_vectors(cost_config, names, non_actionable_mode="keep")
    x_all, is2 = load_real(names)
    clfs = train_classifiers(x_all, is2, seed=42)

    files = sorted(_DIR.glob(f"{args.tag}__*.parquet"))
    gen = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"読み込み: {len(files)}ファイル、{len(gen)}行")
    metrics = compute_row_metrics(gen, names, spec, w_plus, w_minus, clfs)
    metrics.to_csv(_DIR / f"metrics_{args.tag}.csv", index=False)

    print("\n=== 手法×条件の中央値(行ごと=68社×3seed) ===")
    cols = ["proximity_l2", "proximity_cost_weighted", "asset_change", "validity_lightgbm",
            "validity_logreg", "validity_random_forest", "cash_zero", "sign_violations"]
    med = metrics.groupby(["condition", "method"])[cols].median()
    cash = metrics.groupby(["condition", "method"])["cash_zero"].mean()
    med["cash_zero"] = cash
    with pd.option_context("display.width", 220, "display.float_format", "{:.4g}".format):
        print(med.to_string())

    print("\n=== B.1/B.2 CAFM − PCFM(企業単位に集約、n=68) ===")
    tables = []
    for cond in ("W1_uniform", "W3_eq_only"):
        sub = metrics[metrics["condition"] == cond]
        t = paired_table(sub, f"収束版OT / {cond}", ("CAFM_scratch", "CAFM_warm"))
        t.insert(0, "条件", cond)
        tables.append(t)
    print("\n=== 対比: ステージ10.5(非収束OT)の同じ検定(W1/W2/W3、企業単位に集約) ===")
    for tag, path in _OLD.items():
        old = pd.read_csv(path)
        old = old[old["method"].isin(["PCFM_every", "CAFM_scratch", "CAFM_warmstart"])].copy()
        old["method"] = old["method"].map(lambda m: _OLD_NAMES.get(m, m))
        t = paired_table(old, f"ステージ10.5 {tag}(非収束OT)", ("CAFM_scratch", "CAFM_warm"))
        t.insert(0, "条件", f"old_{tag}")
        tables.append(t)
    pd.concat(tables).to_csv(_DIR / f"paired_{args.tag}.csv", index=False)
    print(f"\n出力先: {_DIR}")


if __name__ == "__main__":
    main()
