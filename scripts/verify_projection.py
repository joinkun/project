#!/usr/bin/env python
"""実データで射影の前提条件と基本性質を検証する(ステージ7 §4)。

§4.1: class 2 の実測値 x_i 自身が実行可能集合 C_i に属するか
      (属さない社があれば sign の判定が間違っている)
§4.2: 射影の基本性質(冪等性・実行可能点は動かない・出力が実行可能)

§4.1 は絶対円単位ではなく、各社の総資産に対する相対許容誤差で判定する。
ステージ4.6以降、実データ自体に decimals(開示精度)由来の小さな不整合
(¥1,000〜¥1,000,000程度)が存在することが繰り返し確認されている
(VTA非開示によるフォールバック等)。絶対1円単位で厳密一致を要求すると、
この既知のノイズと真の定義ミスを区別できない。

使い方:
    uv run python scripts/verify_projection.py
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.projection_qp import is_feasible, project_onto_feasible

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_RELATIVE_TOL = 1e-3  # 総資産の0.1%。ステージ6.5で確認済みの最大相対誤差(0.049%)を包含する。
_ABSOLUTE_TOL = 1e-6  # 比較用(素朴な絶対円単位、これでは decimals ノイズを拾ってしまう)。


def load_class2(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    agg = pd.read_csv(aggregated_csv, dtype={"docID": str})
    is_financial = agg["is_financial"].astype(str)
    agg = agg[(agg["parse_status"] == "ok") & (is_financial != "True")]
    labels_df = pd.read_csv(labels_csv, dtype={"docID": str})
    labels_df = labels_df[pd.to_numeric(labels_df["label"], errors="coerce") == 2]
    merged = agg.merge(labels_df[["docID", "filerName"]], on="docID", how="inner")
    x = merged[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = merged["total_assets"].to_numpy(dtype=float)
    return merged["docID"].tolist(), merged["filerName"].tolist(), x, assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    doc_ids, filer_names, x, assets = load_class2(args.aggregated_csv, args.labels_csv, names)
    print(f"class2企業数: {len(doc_ids)}")

    print(f"\n=== §4.1a 実行可能性(絶対{_ABSOLUTE_TOL}円、参考値) ===")
    n_infeasible_abs = sum(
        1 for i in range(len(doc_ids)) if not is_feasible(x[i], spec, tol=_ABSOLUTE_TOL)[0]
    )
    print(
        f"実行可能でない社: {n_infeasible_abs}/{len(doc_ids)}"
        "(decimals由来のノイズを含む、詳細は§4.1b参照)"
    )

    print(f"\n=== §4.1b 実行可能性(総資産の相対{_RELATIVE_TOL:.1%}、採用基準) ===")
    n_infeasible = 0
    for i, doc_id in enumerate(doc_ids):
        tol = assets[i] * _RELATIVE_TOL
        feasible, violations = is_feasible(x[i], spec, tol=tol)
        if not feasible:
            n_infeasible += 1
            print(f"  {doc_id} {filer_names[i]}: 実行可能でない(tol={tol:.0f}円)")
            for name, kind, value in violations:
                print(f"    {name} ({kind}): {value}")
    print(f"実行可能でない社: {n_infeasible}/{len(doc_ids)}")

    print("\n=== §4.2 射影の基本性質(実データ) ===")
    rng = np.random.default_rng(0)
    idempotent_errors = []
    unchanged_errors = []
    unchanged_errors_rel = []
    feasible_output_errors = []
    solve_times = []
    status_counts: Counter[str] = Counter()
    for i in range(len(doc_ids)):
        x_i = x[i]
        # 実行可能点(に近い実測値)は大きくは動かないか
        x_proj_of_feasible, info0 = project_onto_feasible(x_i, spec, x_start=x_i)
        status_counts[info0.status] += 1
        d = float(np.linalg.norm(x_proj_of_feasible - x_i))
        unchanged_errors.append(d)
        unchanged_errors_rel.append(d / assets[i])

        # ランダムに動かした点を射影
        noise = rng.normal(scale=np.abs(x_i).mean() * 0.1 + 1.0, size=x_i.shape)
        y = x_i + noise
        start = time.perf_counter()
        x_proj, info1 = project_onto_feasible(y, spec, x_start=x_i)
        status_counts[info1.status] += 1
        solve_times.append(time.perf_counter() - start)
        feasible, _violations = is_feasible(x_proj, spec, tol=assets[i] * _RELATIVE_TOL)
        feasible_output_errors.append(0 if feasible else 1)

        # 冪等性
        x_proj2, info2 = project_onto_feasible(x_proj, spec, x_start=x_i)
        status_counts[info2.status] += 1
        idempotent_errors.append(float(np.linalg.norm(x_proj2 - x_proj)))

    print(
        f"実行可能点(実測値)への射影移動量 ‖P(x)-x‖: "
        f"絶対最大={max(unchanged_errors):.4e}円, 相対最大={max(unchanged_errors_rel):.4e}"
        f"(§4.1で不一致だった社ぶんだけ、その不一致量とほぼ一致するはず)"
    )
    print(f"冪等性 ‖P(P(y))-P(y)‖ の最大値: {max(idempotent_errors):.4e}円(ソルバ許容誤差の範囲)")
    print(f"射影出力が実行可能でなかった社: {sum(feasible_output_errors)}/{len(doc_ids)}")
    print(f"ソルバstatusの内訳(全{sum(status_counts.values())}回の求解): {dict(status_counts)}")
    print(
        f"射影の求解時間: 平均{np.mean(solve_times) * 1000:.1f}ms, "
        f"最大{np.max(solve_times) * 1000:.1f}ms"
    )


if __name__ == "__main__":
    main()
