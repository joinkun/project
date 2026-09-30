#!/usr/bin/env python
"""ステージ40 A.1: 残余科目(``__OTHER__*``)の実データの分布(canonical に依存しない)。

作業ディレクトリのルートの canonical と集計を使う(D = 32 はプロジェクトのルート、D = 38 は
``data/stage39_root``)。対象は解析成功・金融業でない・ラベルありの書類
(ステージ16 の 4,156 社と同じ母集団)。
値は総資産比。あわせて、評価の始点(class 2・class 1、特殊業種の除外後)で各残余が
``--floor`` を下回る社数を出す。

    uv run python scripts/residuals_stage40.py --out results/stage40/residuals_core.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

_PROJECT = Path(__file__).resolve().parent.parent
FLOOR = -1e-3  # 既存の残余の下限(0 から丸め許容を引いた値)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    ja = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] > 0)].reset_index(drop=True)
    excl = set(pd.read_csv("data/interim/stage38/excluded.csv", dtype={"docID": str})["docID"])
    c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
    use1 = set(c1.loc[c1["reason"].fillna("") == "", "docID"]) - excl
    lab_i = m["label"].astype(int)
    start2 = ((lab_i == 2) & ~m["docID"].isin(excl)).to_numpy()
    start1 = ((lab_i == 1) & m["docID"].isin(use1)).to_numpy()
    residuals = [n for n in names if n.startswith("__OTHER__")]
    rows = []
    for n in residuals:
        v = m[n].fillna(0.0).to_numpy() / m["total_assets"].to_numpy()
        q01, q05, med = np.quantile(v, [0.01, 0.05, 0.5])
        rows.append({
            "account": n, "ja": ja.get(n, n), "n_docs": len(v),
            "share_neg": float(np.mean(v < 0)), "share_below_floor": float(np.mean(v < FLOOR)),
            "min": float(v.min()), "q01": float(q01), "q05": float(q05), "median": float(med),
            "starts2_below_floor": int(np.sum(v[start2] < FLOOR)), "n_starts2": int(start2.sum()),
            "starts1_below_floor": int(np.sum(v[start1] < FLOOR)), "n_starts1": int(start1.sum()),
            "starts2_below_q01": int(np.sum(v[start2] < q01)),
            "starts1_below_q01": int(np.sum(v[start1] < q01)),
        })
    t = pd.DataFrame(rows)
    t.to_csv(args.out.with_suffix(".csv"), index=False)
    lines = [f"# ステージ40 A.1: 残余科目の実データの分布(D = {len(names)})", "",
             f"{len(m):,} 書類(解析成功・金融業でない・ラベルあり)。値は総資産比。"
             f"「下限未満」は {FLOOR}(既存の残余の下限)未満。始点は class 2 "
             f"{int(start2.sum())} 社・class 1 {int(start1.sum())} 社(特殊業種の除外後)。", "",
             "| 残余科目 | 負の割合 | 下限未満 | 最小 | 1% 点 | 5% 点 | 中央値 "
             "| 下限未満の始点 class 2 / class 1 | 1% 点未満の始点 class 2 / class 1 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in t.itertuples():
        lines.append(
            f"| {r.ja} | {r.share_neg:.3f} | {r.share_below_floor:.3f} | {r.min:.4f} "
            f"| {r.q01:.4f} | {r.q05:.4f} | {r.median:.4f} "
            f"| {r.starts2_below_floor} / {r.starts1_below_floor} "
            f"| {r.starts2_below_q01} / {r.starts1_below_q01} |")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
