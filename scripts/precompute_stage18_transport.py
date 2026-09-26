#!/usr/bin/env python
"""ステージ18 §A: 最近傍の対応先選びに、業種制約と規模制約(総資産比)を併用する。

POT専用プロセス(torch/sklearnは無効化)。最近傍(argmin)は輸送計画を解かず、
候補の絞り込み(マスク)→残った候補の中でコスト最小、という2段階で決める。

候補の絞り込み(§A.2 フォールバックの順序):
    1. 業種一致 かつ 規模範囲内
    2. 業種一致のみ(規模を緩和)
    3. 規模範囲内のみ(業種を緩和)
    4. 制約なし
実際に効くのは、指定した制約(--industry, --size)がハードなものだけ。
industry=soft はマスクではなくコストへの加算(§A.2の対象外、常に全候補が候補)。
industry=none または size=none の軸は、そもそも段階に含めない
(例: --industry hard --size none は 1と3の区別がなく、2段(業種一致→制約なし)になる)。

業種「不明」の始点(§A.3): 「不明」同士を対応づけるのは誤りなので、
業種制約(ハード・ソフトいずれも)を課さない。規模制約は課す。

出力(--out-dir): plans_{tag}.npz、full_targets_{tag}.csv(対応先の業種一致・規模比・
フォールバック段階を列挙)、transport_summary.csv に1行追記。

使い方:
    uv run python scripts/precompute_stage18_transport.py --tag ind_hard_size2 \\
        --industry hard --size 0.5,2.0 --out-dir data/interim/stage18
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.models.cost import (  # noqa: E402
    cost_vectors,
    load_cost_config,
    soft_industry_cost,
    weighted_asymmetric_cost_matrix,
)
from fsgen.models.industry import load_industry_config  # noqa: E402
from fsgen.models.transport import check_plan_health  # noqa: E402

_CANONICAL = Path("conf/accounts/core.yaml")
_COSTS = Path("conf/accounts/costs.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_FOLDS = Path("data/interim/industry_12/folds.npz")
_INDUSTRY_YAML = Path("conf/accounts/industry.yaml")
_RE = "jppfs_cor_RetainedEarnings"
_UNKNOWN = "不明"
_SOFT_LAM = 1.0
_STAGE_NAMES = ("業種一致+規模内", "業種一致のみ", "規模内のみ", "制約なし")


def parse_size(text: str) -> tuple[float, float] | None:
    if text == "none":
        return None
    lo, hi = text.split(",")
    return float(lo), float(hi)


def load_data(
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    agg = pd.read_csv(_AGGREGATED_CSV, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS_CSV, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "filerName", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    assets = m["total_assets"].to_numpy(dtype=float)
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float) / assets[:, None]
    labels = m["label"].astype(int).to_numpy()
    return (
        x, assets, labels, m["docID"].tolist(), m["filerName"].tolist(),
        m["edinetCode"].astype(str).tolist(),
    )


def candidate_ladder(
    industry_ok: np.ndarray,
    size_ok: np.ndarray,
    use_industry_mask: bool,
    use_size_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """(n0, n1)の最終マスクと、各始点が落ち着いた段階(1-4)を返す(§A.2)。

    使わない軸(use_*_mask=False)は「常にTrue」として扱う(その軸では絞り込まない)。
    """
    n0, n1 = industry_ok.shape
    ind = industry_ok if use_industry_mask else np.ones_like(industry_ok)
    size = size_ok if use_size_mask else np.ones_like(size_ok)
    stages = [ind & size, ind, size, np.ones((n0, n1), dtype=bool)]
    final = np.zeros((n0, n1), dtype=bool)
    stage_of = np.full(n0, 4, dtype=int)
    assigned = np.zeros(n0, dtype=bool)
    for s_idx, mask in enumerate(stages, start=1):
        has_any = mask.any(axis=1) & ~assigned
        final[has_any] = mask[has_any]
        stage_of[has_any] = s_idx
        assigned |= has_any
    return final, stage_of


def run(
    tag: str, industry_mode: str, size_range: tuple[float, float] | None, out_dir: Path,
) -> dict[str, object]:
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    cc = load_cost_config(_COSTS)
    w_plus, w_minus = cost_vectors(cc, names, non_actionable_mode="keep")
    for i, n in enumerate(names):
        if n in set(cc.non_actionable):
            w_plus[i] = w_minus[i] = 0.0  # na0(輸送コストでは無視。ステージ9)
    x, assets, label, doc, filer, edinet = load_data(names)
    idx0, idx2 = np.where(label == 0)[0], np.where(label == 2)[0]
    x0, x2 = x[idx0], x[idx2]
    ind = load_industry_config(_INDUSTRY_YAML)
    src_lab = np.array(ind.labels([edinet[i] for i in idx2]))
    tgt_lab = np.array(ind.labels([edinet[i] for i in idx0]))
    unknown_src = src_lab == _UNKNOWN
    industry_ok = src_lab[:, None] == tgt_lab[None, :]
    industry_ok[unknown_src, :] = True  # §A.3: 不明の始点には業種制約を課さない
    mismatch = ~industry_ok
    mismatch[unknown_src, :] = False  # ソフト側でも不明の始点にはペナルティを課さない

    ratio = assets[idx0][None, :] / assets[idx2][:, None]
    size_ok = (
        np.ones_like(industry_ok) if size_range is None
        else (ratio >= size_range[0]) & (ratio <= size_range[1])
    )
    use_industry_mask = industry_mode == "hard"
    use_size_mask = size_range is not None
    final_mask, stage_of = candidate_ladder(industry_ok, size_ok, use_industry_mask, use_size_mask)

    base_full = weighted_asymmetric_cost_matrix(x2, x0, w_plus, w_minus)
    cost_n = base_full / np.median(base_full)
    if industry_mode == "soft":
        cost_n = soft_industry_cost(base_full, mismatch, _SOFT_LAM)
    cost_masked = np.where(final_mask, cost_n, np.inf)
    n0 = len(idx2)
    a = np.full(n0, 1.0 / n0)
    tg = np.argmin(cost_masked, axis=1)
    plan_full = np.zeros_like(cost_n)
    plan_full[np.arange(n0), tg] = a
    check_plan_health(plan_full, a, check_marginals=True)

    folds = np.load(_FOLDS)
    plans: dict[str, np.ndarray] = {"plan_full": plan_full}
    for k in range(5):
        for split in ("train", "test"):
            pos = folds[f"{split}_pos_{k}"]
            base = weighted_asymmetric_cost_matrix(x2[pos], x0, w_plus, w_minus)
            cn = base / np.median(base)
            if industry_mode == "soft":
                cn = soft_industry_cost(base, mismatch[pos], _SOFT_LAM)
            cm = np.where(final_mask[pos], cn, np.inf)
            n0k = len(pos)
            ak = np.full(n0k, 1.0 / n0k)
            plan_k = np.zeros_like(cn)
            plan_k[np.arange(n0k), np.argmin(cm, axis=1)] = ak
            check_plan_health(plan_k, ak, check_marginals=True)
            plans[f"plan_{split}_{k}"] = plan_k
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"plans_{tag}.npz", **plans)  # type: ignore[arg-type]

    tgt_ind = tgt_lab[tg]
    tgt_ratio = ratio[np.arange(n0), tg]
    match = src_lab == tgt_ind
    re_t = x0[tg, names.index(_RE)]
    stage_counts = Counter(stage_of.tolist())
    pd.DataFrame({
        "design": tag, "source_docID": [doc[i] for i in idx2],
        "source_name": [filer[i] for i in idx2], "source_industry": src_lab,
        "target_docID": [doc[idx0[j]] for j in tg], "target_name": [filer[idx0[j]] for j in tg],
        "target_industry": tgt_ind, "industry_match": match, "asset_ratio": tgt_ratio,
        "fallback_stage": stage_of, "target_RE_over_assets": re_t,
    }).to_csv(out_dir / f"full_targets_{tag}.csv", index=False)

    summary = {
        "design": tag, "industry": industry_mode,
        "size_range": "none" if size_range is None else f"{size_range[0]:g}-{size_range[1]:g}",
        "unique_targets": int(len(set(tg.tolist()))),
        "industry_match_rate": float(match.mean()),
        "asset_ratio_median": float(np.median(tgt_ratio)),
        "asset_ratio_q1": float(np.percentile(tgt_ratio, 25)),
        "asset_ratio_q3": float(np.percentile(tgt_ratio, 75)),
        "share_target_over_3x": float(np.mean(tgt_ratio > 3.0)),
        "n_RE_negative_targets": int(np.sum(re_t < 0)),
        **{f"stage{s}_{_STAGE_NAMES[s - 1]}": stage_counts.get(s, 0) for s in (1, 2, 3, 4)},
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--industry", choices=("none", "hard", "soft"), default="none")
    ap.add_argument("--size", default="none", help="'lo,hi' または 'none'")
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim/stage18"))
    args = ap.parse_args()
    size_range = parse_size(args.size)
    summary = run(args.tag, args.industry, size_range, args.out_dir)
    path = args.out_dir / "transport_summary.csv"
    new = pd.DataFrame([summary])
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~old["design"].isin(new["design"])]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(new.to_string(index=False))
    print(f"出力先: {args.out_dir}")


if __name__ == "__main__":
    main()
