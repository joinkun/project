#!/usr/bin/env python
"""ステージ13 §B.3/C: 輸送設計(コストmode・バランス/不均衡/片側/最近傍・RE扱い)の比較。

lightgbm/sklearn/scipyのプロセス(torchなし)。設計ごと・重み条件ごとに、PCFMの指標
(proximity・総資産変化・現金ゼロ率・validity・識別AUC)と、生成前(FM生出力)の
RE変化、輸送計画の質(実効サポート・目標のRE・総資産比)を1表にまとめる。
最近傍ベースラインとの差は企業単位(n=68)のペア検定+ブートストラップ信頼区間で示す。

使い方:
    uv run python scripts/run_metrics_designs.py --tags bal_na0,bal_na1 --out designsA
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
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402
from fsgen.evaluation.paired import aggregate_by_company, paired_test  # noqa: E402
from fsgen.evaluation.row_metrics import compute_row_metrics  # noqa: E402
from fsgen.models.cost import cost_vectors, load_cost_config  # noqa: E402

_CANONICAL = Path("conf/accounts/core.yaml")
_COSTS = Path("conf/accounts/costs.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DIR = Path("data/interim/stage13")
_REPEATS = 20


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


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", required=True, help="設計(タグ)名のカンマ区切り")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="nearest_na0", help="ペア比較の基準タグ")
    ap.add_argument("--no-auc", action="store_true")
    args = ap.parse_args()

    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    cc = load_cost_config(_COSTS)
    w_plus, w_minus = cost_vectors(cc, names, non_actionable_mode="keep")
    x_all, is2 = load_real(names)
    clfs = train_classifiers(x_all, is2, seed=42)
    real0 = x_all[~is2]

    tags = args.tags.split(",")
    files = [f for t in tags for f in sorted(_DIR.glob(f"{t}__*.parquet"))]
    gen = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    met = compute_row_metrics(gen, names, spec, w_plus, w_minus, clfs)
    met.to_csv(_DIR / f"metrics_{args.out}.csv", index=False)
    print(f"読み込み: {len(files)}ファイル、{len(gen)}行")

    tsum = pd.read_csv(_DIR / "transport_summary.csv").set_index("design")
    rows = []
    for (tag, cond), sub in met.groupby(["tag", "condition"]):
        pc = sub[sub["method"] == "PCFM"]
        fm = sub[sub["method"] == "FM"]
        row: dict[str, object] = {
            "design": tag, "condition": cond,
            "proximity_l2": pc["proximity_l2"].median(),
            "prox_cost_weighted": pc["proximity_cost_weighted"].median(),
            "asset_change": pc["asset_change"].median(),
            "cash_zero_rate": pc["cash_zero"].mean(),
            "validity_lgbm": pc["validity_lightgbm"].median(),
            "validity_logreg": pc["validity_logreg"].median(),
            "validity_rf": pc["validity_random_forest"].median(),
            "sparsity": pc["sparsity"].median(),
            "RE_change_raw_FM": fm["re_change"].median(),
            "RE_change_PCFM": pc["re_change"].median(),
        }
        if not args.no_auc:
            g0 = gen[(gen["tag"] == tag) & (gen["condition"] == cond)
                     & (gen["method"] == "PCFM") & (gen["seed"] == 0)]
            res = identification_auc_repeated(
                g0[[f"x1_{n}" for n in names]].to_numpy(), real0, _REPEATS
            )
            row["auc_lgbm"] = float(res["lightgbm"].mean())
            row["auc_logreg"] = float(res["logreg"].mean())
        if tag in tsum.index:
            for col in ("n_dropped_sources", "support_median", "n_RE_negative_targets",
                        "RE_target_median", "RE_abs_gap_median", "asset_ratio_median"):
                row[col] = tsum.loc[tag, col]
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["condition", "design"])
    summary.to_csv(_DIR / f"summary_{args.out}.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", "{:.3g}".format)
    print("\n=== 設計×条件(PCFM、行=68社×3seedの中央値) ===")
    print(summary.to_string(index=False))

    print(f"\n=== 企業単位(n=68)のペア比較: 各設計 − {args.baseline}(PCFM) ===")
    prow = []
    for cond in ("W1_uniform", "W3_eq_only"):
        sub = met[(met["condition"] == cond) & (met["method"] == "PCFM")]
        if args.baseline not in set(sub["tag"]):
            continue
        for metric in ("proximity_l2", "validity_lightgbm", "asset_change"):
            agg = aggregate_by_company(sub, metric, group_cols=("tag",))
            for tag in tags:
                if tag == args.baseline or tag not in agg.columns:
                    continue
                r = paired_test(agg[tag].to_numpy(), agg[args.baseline].to_numpy())
                prow.append({
                    "condition": cond, "design": tag, "metric": metric, "n": r.n,
                    "median_diff": r.median_diff, "ci_low": r.ci_low, "ci_high": r.ci_high,
                    "p": r.p_value,
                })
    pdf = pd.DataFrame(prow)
    pdf.to_csv(_DIR / f"paired_vs_baseline_{args.out}.csv", index=False)
    print(pdf.to_string(index=False))
    print(f"\n出力先: {_DIR}")


if __name__ == "__main__":
    main()
