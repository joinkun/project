#!/usr/bin/env python
"""ステージ13: 輸送設計の比較のための輸送計画の事前計算(POT専用プロセス。torch/sklearnは無効化)。

設計名 = ``{kind}_na{w}``:
- kind: bal(バランスOT・対数領域) / unb{τ}(不均衡OT, τ=reg_m) / onesided(片側制約) /
        nearest(最近傍、OTを使わない。距離=costs.yamlの非対称重み付き二乗距離) /
        nearestu(最近傍・重みなし: 全科目w+=w-=1の元空間の二乗距離。ステージ15のC比較用)
- w   : 非操作科目(利益剰余金・評価差額・非支配株主持分)の輸送コスト重み。
        0=ignore(現行) / 1=keep / 1e6=freeze / その他は中間値。

収束・健全性チェック(§A.1, A.2)は全設計で必須: 反復ソルバは収束しなければ例外、
全計画は質量0の行・NaN/infがあれば例外。バランスOTは周辺分布誤差<1e-6も検証。

出力(data/interim/stage13/): plans_{design}.npz(全68社ぶん+5-fold train/test)、
transport_summary.csv(設計ごとの実効サポート・対応先の質・総資産比)、
full_targets_{design}.csv。

使い方:
    uv run python scripts/precompute_stage13_transport.py --designs bal_na0,bal_na1,onesided_na0
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]

import argparse  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import subtotal_members  # noqa: E402
from fsgen.datasets.exclusion import (  # noqa: E402
    exclusion_masks,
    load_excluded_docs,
    subset_folds,
)
from fsgen.models.cost import (  # noqa: E402
    cost_vectors,
    hard_industry_cost,
    industry_mismatch,
    load_cost_config,
    soft_industry_cost,
    weighted_asymmetric_cost_matrix,
)
from fsgen.models.industry import load_industry_config  # noqa: E402
from fsgen.models.target_filter import (  # noqa: E402
    TARGET_FILTERS,
    apply_mask,
    equity_ratio,
    target_mask,
)
from fsgen.models.transport import (  # noqa: E402
    check_plan_health,
    solve_ot_normalized,
    solve_unbalanced_normalized,
)

_CANONICAL = Path("conf/accounts/core.yaml")
_COSTS = Path("conf/accounts/costs.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_FOLDS = Path("data/interim/industry_12/folds.npz")
_OUT_DIR = Path("data/interim/stage13")
_EPSILON = 0.05
_LOG_STOP_THR = 1e-7
_N_FOLDS = 5
_RE = "jppfs_cor_RetainedEarnings"
_INDUSTRY_YAML = Path("conf/accounts/industry.yaml")
_SOFT_LAM = 1.0  # ステージ17 §B: ソフト業種制約のλ(コスト中央値の何倍か)


def parse_design(name: str) -> tuple[str, float, float]:
    """設計名 → (kind, τ(unbのみ), 非操作科目の重みw)。"""
    pat = r"(bal|onesided|nearestu|nearest|unb(?P<tau>[0-9.]+))_na(?P<w>[0-9.e+\-]+)"
    m = re.fullmatch(pat, name)
    if m is None:
        raise ValueError(f"設計名が不正: {name!r}(例: bal_na0, unb1_na0, onesided_na1e+06)")
    kind = m.group(1)
    tau = float(m.group("tau")) if m.group("tau") else 0.0
    return ("unb" if kind.startswith("unb") else kind), tau, float(m.group("w"))


def solve_plan(kind: str, tau: float, cost_n: np.ndarray) -> np.ndarray:
    """1回のOT(始点=cost_nの行)。cost_nは呼び出し側で正規化済み(§2.1、ステージ17で
    業種制約(hard/soft)をここより前に適用できるよう、正規化をrun_design側に移した)。

    健全性チェック済みの計画を返す。
    """
    n0, n1 = cost_n.shape
    a = np.full(n0, 1.0 / n0)
    b = np.full(n1, 1.0 / n1)
    if kind == "bal":
        plan = solve_ot_normalized(
            cost_n, a, b, _EPSILON, log_domain=True, stop_thr=_LOG_STOP_THR
        ).plan  # 未収束なら例外(§A.1)
        check_plan_health(plan, a, b)
    elif kind == "unb":
        # 不均衡OTは外れ値の始点を切り捨てる(質量0の行)。明示的に許可し、件数を報告する。
        plan = solve_unbalanced_normalized(
            cost_n, a, b, _EPSILON, tau, allow_dropped_sources=True
        )
    elif kind == "onesided":
        logits = -cost_n / _EPSILON
        w = np.exp(logits - logits.max(axis=1, keepdims=True))
        plan = w / w.sum(axis=1, keepdims=True) * a[:, None]  # 始点の質量のみ保存
        check_plan_health(plan, a, check_marginals=True)
    elif kind in ("nearest", "nearestu"):
        plan = np.zeros_like(cost_n)
        plan[np.arange(n0), np.argmin(cost_n, axis=1)] = a
        check_plan_health(plan, a, check_marginals=True)
    else:
        raise ValueError(kind)
    return plan


def load_data(
    names: tuple[str, ...],
    excluded: frozenset[str] = frozenset(),
    source_label: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str], np.ndarray]:
    """最後の要素は、除外前の class 2 の並びで残す始点(ステージ28 の除外、fold の絞り込み用)。"""
    agg = pd.read_csv(_AGGREGATED_CSV, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS_CSV, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "filerName", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    row_keep, src_keep = exclusion_masks(
        m["docID"].tolist(), m["label"].astype(int).tolist(), excluded, source_label
    )
    m = m[row_keep]
    assets = m["total_assets"].to_numpy(dtype=float)
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float) / assets[:, None]
    labels = m["label"].astype(int).to_numpy()
    return (
        x, assets, labels, m["docID"].tolist(), m["filerName"].tolist(),
        m["edinetCode"].astype(str).tolist(), src_keep,
    )


def normalized_cost(
    base_cost: np.ndarray, industry: str, mismatch: np.ndarray | None
) -> np.ndarray:
    """§2.1(現行): base_cost/median。ステージ17 §B: hard/soft業種制約を正規化と同時に適用。"""
    if industry == "none":
        result: np.ndarray = base_cost / np.median(base_cost)
        return result
    if mismatch is None:
        raise ValueError("industry!='none' には mismatch が必要")
    if industry == "hard":
        return hard_industry_cost(base_cost, mismatch)
    if industry == "soft":
        return soft_industry_cost(base_cost, mismatch, _SOFT_LAM)
    raise ValueError(f"未対応のindustry: {industry!r}")


def run_design(
    name: str,
    out_dir: Path = _OUT_DIR,
    industry: str = "none",
    costs_path: Path = _COSTS,
    target_filter: str = "baseline",
    abs_threshold: float = 0.3,
    exclude_path: Path | None = None,
    source_label: int = 2,
    target_label: int = 0,
    folds_path: Path = _FOLDS,
) -> tuple[str, dict[str, object]]:
    t0 = time.perf_counter()
    kind, tau, w_na = parse_design(name)
    out_name = name if industry == "none" else f"{name}_ind{industry}"
    if target_filter != "baseline":  # ステージ26: 目標分布の絞り込み
        if industry != "none":
            raise ValueError("target_filter と industry は同時に使わない")
        out_name = f"{name}_tf{target_filter}"
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    cc = load_cost_config(costs_path)
    w_plus, w_minus = cost_vectors(cc, names, non_actionable_mode="keep")
    if kind == "nearestu":  # 重みなし: 全科目1(非操作科目は下でw_naに上書き=重みありと同じ扱い)
        w_plus, w_minus = np.ones(len(names)), np.ones(len(names))
    for i, n in enumerate(names):
        if n in set(cc.non_actionable):
            w_plus[i] = w_minus[i] = w_na
    x, assets, label, doc, filer, edinet, src_keep = load_data(
        names, load_excluded_docs(exclude_path), source_label
    )
    # ステージ38: 始点・目標のラベルを変えられる(既定は class 2 → class 0)。変数名は従来のまま
    idx0, idx2 = np.where(label == target_label)[0], np.where(label == source_label)[0]
    x0, x2 = x[idx0], x[idx2]
    folds = subset_folds(np.load(folds_path), src_keep)  # 除外なしなら元の分割のまま

    mismatch_full = None
    n_no_match = 0
    if industry != "none":
        ind = load_industry_config(_INDUSTRY_YAML)
        src_lab = np.array(ind.labels([edinet[i] for i in idx2]))
        tgt_lab = np.array(ind.labels([edinet[i] for i in idx0]))
        mismatch_full = industry_mismatch(src_lab, tgt_lab)
        n_no_match = int(np.all(mismatch_full, axis=1).sum())  # 同業種のclass0が0社の始点数(§B.3)

    # ステージ26: 自己資本比率による対応先の候補(baseline は全候補)。正規化の後にマスクする
    members = subtotal_members(spec, names)
    src_ratio, tgt_ratio = equity_ratio(x2, members), equity_ratio(x0, members)
    tf = target_mask(target_filter, src_ratio, tgt_ratio, abs_threshold)  # type: ignore[arg-type]

    plans: dict[str, np.ndarray] = {}
    base_full = weighted_asymmetric_cost_matrix(x2, x0, w_plus, w_minus)
    cost_full = apply_mask(normalized_cost(base_full, industry, mismatch_full), tf.allowed)
    plan_full = solve_plan(kind, tau, cost_full)
    plans["plan_full"] = plan_full
    for k in range(_N_FOLDS):
        for tag in ("train", "test"):
            pos = folds[f"{tag}_pos_{k}"]
            base = weighted_asymmetric_cost_matrix(x2[pos], x0, w_plus, w_minus)
            mm = mismatch_full[pos] if mismatch_full is not None else None
            cost = apply_mask(normalized_cost(base, industry, mm), tf.allowed[pos])
            plans[f"plan_{tag}_{k}"] = solve_plan(kind, tau, cost)
    np.savez_compressed(out_dir / f"plans_{out_name}.npz", **plans)  # type: ignore[arg-type]

    # 対応先の質・実効サポート・総資産比(全68社→class0)
    row_mass = plan_full.sum(axis=1)
    alive = row_mass > 1e-12  # 不均衡OTで切り捨てられなかった始点(以降の統計はこれだけ)
    tg = np.argmax(plan_full, axis=1)
    re_t = x0[tg, names.index(_RE)]
    p_row = plan_full / np.where(alive, row_mass, 1.0)[:, None]
    sup = 1.0 / np.sum(p_row**2, axis=1)
    a_ratio = assets[idx0][tg] / assets[idx2]
    mean_log_ratio = p_row @ np.log10(assets[idx0]) - np.log10(assets[idx2])  # 重み付き平均
    re_src = x2[:, names.index(_RE)]
    src_ind = ind.labels([edinet[i] for i in idx2]) if industry != "none" else [""] * len(idx2)
    tgt_ind_full = ind.labels([edinet[i] for i in idx0]) if industry != "none" else [""] * len(idx0)
    tgt_ind = [tgt_ind_full[j] for j in tg]
    match_bool = [s == t for s, t in zip(src_ind, tgt_ind, strict=True)]
    match: list[bool | None] = match_bool if industry != "none" else [None] * len(idx2)
    summary: dict[str, object] = {
        "design": out_name, "kind": kind, "tau": tau, "w_nonactionable": w_na,
        "industry": industry, "n_no_industry_match_sources": n_no_match,
        "industry_match_rate": float(np.mean(match_bool)) if industry != "none" else None,
        "n_dropped_sources": int((~alive).sum()),
        "transported_mass": float(plan_full.sum()),
        "unique_targets": int(len(set(tg[alive].tolist()))),
        "support_median": float(np.median(sup[alive])),
        "support_q1": float(np.percentile(sup[alive], 25)),
        "support_q3": float(np.percentile(sup[alive], 75)),
        "n_RE_negative_targets": int(np.sum(re_t[alive] < 0)),
        "RE_target_median": float(np.median(re_t[alive])),
        "RE_source_median": float(np.median(re_src)),
        "RE_abs_gap_median": float(np.median(np.abs(re_t - re_src)[alive])),
        "asset_ratio_median": float(np.median(a_ratio[alive])),
        "asset_ratio_q1": float(np.percentile(a_ratio[alive], 25)),
        "asset_ratio_q3": float(np.percentile(a_ratio[alive], 75)),
        "share_target_smaller": float(np.mean(a_ratio[alive] < 1.0)),
        "share_target_over_3x": float(np.mean(a_ratio[alive] > 3.0)),
        "share_target_under_third": float(np.mean(a_ratio[alive] < 1 / 3)),
        "weighted_log10_ratio_median": float(np.median(mean_log_ratio[alive])),
        "target_filter": target_filter,
        "target_filter_threshold": tf.threshold,
        "n_candidates_median": float(np.median(tf.n_candidates)),
        "n_candidates_min": int(tf.n_candidates.min()),
        "n_candidates_max": int(tf.n_candidates.max()),
        "n_fallback_sources": int(tf.fallback.sum()),
        "n_class0_eligible": int(tf.allowed[~tf.fallback].any(axis=0).sum())
        if (~tf.fallback).any()
        else 0,
        "seconds": time.perf_counter() - t0,
    }
    pd.DataFrame({
        "design": out_name, "source_docID": [doc[i] for i in idx2],
        "source_name": [filer[i] for i in idx2], "target_docID": [doc[idx0[j]] for j in tg],
        "target_name": [filer[idx0[j]] for j in tg], "target_RE_over_assets": re_t,
        "support": sup, "asset_ratio": a_ratio,
        "source_industry": src_ind, "target_industry": tgt_ind, "industry_match": match,
        "source_equity_ratio": src_ratio, "target_equity_ratio": tgt_ratio[tg],
        "n_candidates": tf.n_candidates, "fallback": tf.fallback,
    }).to_csv(out_dir / f"full_targets_{out_name}.csv", index=False)
    return name, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--designs", required=True)
    ap.add_argument(
        "--costs", type=Path, default=_COSTS, help="コスト重み(ステージ24 で 3 版を切り替える)"
    )
    ap.add_argument(
        "--target-filter", choices=TARGET_FILTERS, default="baseline",
        help="対応先の候補を自己資本比率で絞る(ステージ26)",
    )
    ap.add_argument("--abs-threshold", type=float, default=0.3, help="absolute の閾値")
    ap.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    ap.add_argument("--industry", choices=("none", "hard", "soft"), default="none",
                    help="ステージ17 §B: 業種制約(全designsに一律適用)")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="除外する書類の CSV(列 docID、ステージ28)")
    ap.add_argument("--source-label", type=int, default=2, help="始点のクラス(ステージ38)")
    ap.add_argument("--target-label", type=int, default=0, help="目標のクラス(ステージ38)")
    ap.add_argument("--folds", type=Path, default=_FOLDS,
                    help="始点の 5-fold 分割(除外前の始点の並びでの位置。ステージ38)")
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    designs = args.designs.split(",")
    for d in designs:
        parse_design(d)  # 名前の検証
    with ProcessPoolExecutor(max_workers=min(len(designs), 8)) as ex:
        results = list(ex.map(
            run_design,
            designs,
            [out_dir] * len(designs),
            [args.industry] * len(designs),
            [args.costs] * len(designs),
            [args.target_filter] * len(designs),
            [args.abs_threshold] * len(designs),
            [args.exclude] * len(designs),
            [args.source_label] * len(designs),
            [args.target_label] * len(designs),
            [args.folds] * len(designs),
        ))
    path = out_dir / "transport_summary.csv"
    new = pd.DataFrame([s for _, s in results])
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~old["design"].isin(new["design"])]
    out = pd.concat([old, new], ignore_index=True)
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3g}".format):
        print(new.to_string(index=False))
    print(f"出力先: {out_dir}")


if __name__ == "__main__":
    main()
