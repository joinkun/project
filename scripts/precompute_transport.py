#!/usr/bin/env python
"""ステージ11-A: OT(最適輸送)の事前計算専用スクリプト。

作業中に判明した追加のOpenMP競合リスク: POT(``ot``パッケージ)は
1) ``ot/backend.py``がバックエンド候補として無条件に``import torch``を
   試み(``POT_BACKEND_DISABLE_PYTORCH``という公式の環境変数で無効化できる)、
2) ``ot/lowrank.py``が``import sklearn.cluster``をtry/exceptで試みる
   (公式の無効化フラグは無いが、ImportErrorで優雅にフォールバックする
   設計になっている)。
sklearnはtorchとは別の独自libomp.dylibを同梱しており、何も対策せずに
``import ot`` すると、vmmapでlibomp.dylibが2つ(torch同梱版とsklearn同梱版)
ロードされることを実測で確認した——``ot``をimportするだけで、torch単体の
プロセスのはずが実質torch+sklearn同居プロセスになってしまう。

これを避けるため、本スクリプトの``import ot``(``fsgen.models.transport``
経由)より前に、(1)環境変数でPOTのtorchバックエンド検出を無効化し、
(2)``sys.modules["sklearn"] = None``でsklearnのimportを意図的に失敗させ
(POT自身のImportErrorフォールバック経路に乗せる、コードを一切変更しない
正攻法)、本スクリプトのプロセスにtorch/sklearnのいずれもロードされない
ことを確認済み(vmmapでlibomp.dylibが1つも無いことを確認)。

輸送計画(Sinkhorn)の計算はこのスクリプトで先に済ませ、npz/csvに保存する。
run_generation.py(torch専用)は``ot``を一切importせず、本スクリプトが
書き出したnpzを読み込む。

計算内容:
- 5-fold LOO用のplan_train/plan_test(現行既定=ignoreモードコスト。
  Sinkhornは決定的でRNGを使わないため、3 seedで共通のプランを再利用できる)
- §B.1: 68社→3951社の全体OT対応先(ignore/W1/W3_eq_only の3条件)
- §B.3: 目標企業(現行ignoreモードでの上位10社)の素性

使い方:
    uv run python scripts/precompute_transport.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.models.cost import (  # noqa: E402
    CostConfig,
    cost_vectors,
    load_cost_config,
    weighted_asymmetric_cost_matrix,
)
from fsgen.models.transport import solve_ot_from_cost  # noqa: E402

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_COSTS = Path("conf/accounts/costs.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_OUTPUT_DIR = Path("data/interim/generation_11a")

_EPSILON = 0.05
_N_FOLDS = 5


def load_dataset(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray, list[int]]:
    agg = pd.read_csv(aggregated_csv, dtype={"docID": str, "edinetCode": str})
    is_financial = agg["is_financial"].astype(str)
    agg = agg[(agg["parse_status"] == "ok") & (is_financial != "True")]
    labels_df = pd.read_csv(labels_csv, dtype={"docID": str})
    labels_df = labels_df[pd.to_numeric(labels_df["label"], errors="coerce").notna()]
    merged = agg.merge(labels_df[["docID", "filerName", "label"]], on="docID", how="inner")
    merged = merged[merged["total_assets"].notna() & (merged["total_assets"] != 0)]
    x = merged[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = merged["total_assets"].to_numpy(dtype=float)
    labels = merged["label"].astype(int).tolist()
    return (
        merged["docID"].tolist(),
        merged["filerName"].tolist(),
        merged["edinetCode"].fillna("").tolist(),
        x,
        assets,
        labels,
    )


def make_folds(n2: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n2)
    chunks = np.array_split(order, n_folds)
    folds = []
    for i in range(n_folds):
        test_pos = chunks[i]
        train_pos = np.concatenate([chunks[j] for j in range(n_folds) if j != i])
        folds.append((train_pos, test_pos))
    return folds


@dataclass(frozen=True)
class TransportWeightCond:
    name: str
    w_plus: np.ndarray
    w_minus: np.ndarray


def build_transport_conditions(
    cost_config: CostConfig, names: tuple[str, ...]
) -> list[TransportWeightCond]:
    """§B.1: 輸送コストの重み条件。現行既定(ignore)に加え、W1・W3_eq_onlyも試す。"""
    dim = len(names)
    ignore_plus, ignore_minus = cost_vectors(cost_config, names, non_actionable_mode="ignore")
    keep_plus, keep_minus = cost_vectors(cost_config, names, non_actionable_mode="keep")
    return [
        TransportWeightCond("ignore_current", ignore_plus, ignore_minus),
        TransportWeightCond("W1_uniform", np.ones(dim), np.ones(dim)),
        TransportWeightCond("W3_eq_only", keep_plus, keep_minus),
    ]


def compute_transport_targets(
    doc_ids2: list[str],
    doc_ids0: list[str],
    filer0: list[str],
    x2_tilde: np.ndarray,
    x0_tilde: np.ndarray,
    conditions: list[TransportWeightCond],
) -> pd.DataFrame:
    """§B.1: 68社→3951社のOT対応先(j*)を条件ごとに求める(LOOなし、全体で1回)。"""
    rows = []
    for cond in conditions:
        cost = weighted_asymmetric_cost_matrix(x2_tilde, x0_tilde, cond.w_plus, cond.w_minus)
        plan = solve_ot_from_cost(cost, _EPSILON)
        targets = np.argmax(plan, axis=1)
        for i, doc_id in enumerate(doc_ids2):
            j = int(targets[i])
            rows.append({
                "weight_condition": cond.name,
                "source_docID": doc_id,
                "target_docID": doc_ids0[j],
                "target_filerName": filer0[j],
            })
    return pd.DataFrame(rows)


def main() -> None:
    spec = load_canonical_spec(_DEFAULT_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    build_constraint_matrix(spec, include_subtotals=False)
    cost_config = load_cost_config(_DEFAULT_COSTS)

    out_dir = _DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids, filer_names, edinet_codes, x, assets, labels = load_dataset(
        _DEFAULT_AGGREGATED_CSV, _DEFAULT_LABELS_CSV, names
    )
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    x0_tilde = x_tilde[idx0]
    x2_tilde = x_tilde[idx2]
    doc_ids2 = [doc_ids[i] for i in idx2]
    doc_ids0 = [doc_ids[i] for i in idx0]
    filer0 = [filer_names[i] for i in idx0]
    n2 = len(idx2)
    print(f"class0: {len(idx0)}社, class2: {n2}社")

    # --- §B.1: 輸送先の集中度(3条件、LOOなし、全体で1回) ---
    print("\n=== §B.1: 輸送先(OT対応先j*)の条件別計算 ===")
    transport_conditions = build_transport_conditions(cost_config, names)
    transport_df = compute_transport_targets(
        doc_ids2, doc_ids0, filer0, x2_tilde, x0_tilde, transport_conditions
    )
    transport_df.to_csv(out_dir / "transport_targets.csv", index=False)
    for cond_name in transport_df["weight_condition"].unique():
        sub = transport_df[transport_df["weight_condition"] == cond_name]
        n_unique = sub["target_docID"].nunique()
        top5 = sub["target_docID"].value_counts().head(5)
        print(f"  {cond_name:<16} 異なる対応先企業数={n_unique}/{len(sub)}")
        print(f"    上位5社の被選択数: {top5.to_dict()}")

    # --- 5-foldのOT輸送計画(現行既定=ignoreモード、seed非依存) ---
    print(f"\n=== 5-fold輸送計画の事前計算(ignoreモード、{_N_FOLDS}分割) ===")
    w_plus_ot, w_minus_ot = cost_vectors(cost_config, names, non_actionable_mode="ignore")
    folds = make_folds(n2, _N_FOLDS, seed=0)
    fold_plans: dict[str, np.ndarray] = {}
    for fold_i, (train_pos, test_pos) in enumerate(folds):
        cost_train = weighted_asymmetric_cost_matrix(
            x2_tilde[train_pos], x0_tilde, w_plus_ot, w_minus_ot
        )
        plan_train = solve_ot_from_cost(cost_train, _EPSILON)
        cost_test = weighted_asymmetric_cost_matrix(
            x2_tilde[test_pos], x0_tilde, w_plus_ot, w_minus_ot
        )
        plan_test = solve_ot_from_cost(cost_test, _EPSILON)
        fold_plans[f"plan_train_{fold_i}"] = plan_train
        fold_plans[f"plan_test_{fold_i}"] = plan_test
        fold_plans[f"train_pos_{fold_i}"] = train_pos
        fold_plans[f"test_pos_{fold_i}"] = test_pos
        print(f"  fold={fold_i + 1}/{_N_FOLDS} 完了(train={len(train_pos)}, test={len(test_pos)})")
    np.savez(out_dir / "fold_plans.npz", **fold_plans)  # type: ignore[arg-type]
    print(f"保存: {out_dir / 'fold_plans.npz'}")

    # --- §B.3: 目標企業の素性(現行既定=ignoreモードの対応先) ---
    print("\n=== §B.3: 目標企業の素性(現行既定ignoreモード) ===")
    ignore_targets = transport_df[transport_df["weight_condition"] == "ignore_current"]
    top_targets = ignore_targets["target_docID"].value_counts().head(10)
    b3_rows = []
    index0 = {d: i for i, d in enumerate(doc_ids0)}
    re_idx = names.index("jppfs_cor_RetainedEarnings")
    cash_idx = names.index("jppfs_cor_CashAndDeposits")
    debt_names = ("short_term_loans", "bonds_current", "long_term_loans", "bonds_noncurrent")
    debt_idx = [names.index(n) for n in debt_names]
    net_assets_idx = [i for i, bb in enumerate(spec.basis) if bb.side == "net_assets"]
    for target_doc_id, count in top_targets.items():
        i0 = index0[target_doc_id]
        global_i = idx0[i0]
        x_i = x_tilde[global_i]
        b3_rows.append({
            "filerName": filer_names[global_i],
            "edinetCode": edinet_codes[global_i],
            "docID": target_doc_id,
            "n_selected": int(count),
            "RE_over_assets": float(x_i[re_idx]),
            "net_assets_over_assets": float(np.sum(x_i[net_assets_idx])),
            "cash_over_assets": float(x_i[cash_idx]),
            "debt_over_assets": float(np.sum(x_i[debt_idx])),
            "assets_yen": float(assets[global_i]),
        })
    b3_df = pd.DataFrame(b3_rows)
    b3_df.to_csv(out_dir / "target_company_profiles.csv", index=False)
    print(b3_df.to_string(index=False))

    print(f"\n出力先: {out_dir}")


if __name__ == "__main__":
    main()
