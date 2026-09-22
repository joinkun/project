#!/usr/bin/env python
"""ステージ12: 業種を考慮した輸送計画の事前計算(POTのプロセス。torch/lightgbmは使わない)。

ステージ11-Aの3段階構成(プロセス分離)を守る。``ot``(POT)は無効化しないと
torch/sklearnを間接importしてlibomp.dylibを二重ロードさせるため、
precompute_transport.pyと同じ対策(環境変数+sys.modules)を最初に入れる。

**判明した事実(ステージ12)**: 現行(ステージ9〜11-A)のOT(標準=線形領域Sinkhorn)は
68社×3951社で収束せず、68社中9社の行が質量0になる。その行の``argmax``は
任意のindexなので、11-Aの「ΔDelta-Fly Pharmaに9社集中」の一部
(3社)や日高カントリー倶楽部への4社は、この数値アーティファクトだった。
そこで、全ての設計を対数領域Sinkhornで解き直し、「現行(as-run)」は
比較のためだけに旧ソルバで併記する。

設計(§2.1):
- current_asis : 現行そのまま(標準Sinkhorn、業種なし)
- current      : 現行と同じコストを対数領域で解いたもの(=ソフトλ=0)
- soft_l{λ}    : ソフトペナルティ(33業種、λ ∈ {0.5,1,2,5,10}×median)
- hard33       : ハード制約(33業種、フォールバック付き)
- hard_coarse  : ハード制約(粗い分類、フォールバック付き)

出力(data/interim/industry_12/):
- folds.npz, plans_{design}.npz : 5-fold(train/test)の輸送計画
- full_targets.csv               : 68社→class0の対応先(全データ、設計ごと)
- transport_quality.csv          : §3.1の表
- plan_diagnostics.csv           : 各OTの収束診断・所要時間
- industry_report.txt            : §1.2の業種情報の取得状況
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import (  # noqa: E402
    load_canonical_spec,
    state_vector_names,
)
from fsgen.models.cost import (  # noqa: E402
    cost_vectors,
    hard_industry_cost,
    industry_mismatch,
    load_cost_config,
    soft_industry_cost,
    weighted_asymmetric_cost_matrix,
)
from fsgen.models.industry import (  # noqa: E402
    DEFAULT_MIN_TARGETS,
    assign_blocks,
    block_marginals,
    load_industry_config,
)
from fsgen.models.transport import solve_ot_from_cost_verbose, solve_ot_normalized  # noqa: E402

_CANONICAL = Path("conf/accounts/core.yaml")
_COSTS = Path("conf/accounts/costs.yaml")
_INDUSTRY = Path("conf/accounts/industry.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DOCLIST_CSV = Path("data/raw/index/document_list.csv")
_OUT_DIR = Path("data/interim/industry_12")

_EPSILON = 0.05
_N_FOLDS = 5
_LOG_STOP_THR = 1e-7  # 周辺分布のずれ<1e-6(下記converged判定)を満たす程度に緩める
_SOFT_MULTIPLES = (0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class Design:
    name: str
    kind: str  # "legacy" | "soft" | "hard"
    lam: float = 0.0
    level: str = "33"  # hardのみ: "33" | "coarse"


DESIGNS: tuple[Design, ...] = (
    Design("current_asis", "legacy"),
    Design("current", "soft", lam=0.0),
    *(Design(f"soft_l{m:g}", "soft", lam=m) for m in _SOFT_MULTIPLES),
    Design("hard33", "hard", level="33"),
    Design("hard_coarse", "hard", level="coarse"),
)


@dataclass(frozen=True)
class Data:
    docid: np.ndarray
    name: np.ndarray
    edinet: np.ndarray
    label: np.ndarray
    x_tilde: np.ndarray
    assets: np.ndarray
    sec_null: np.ndarray


def load_data(names: tuple[str, ...]) -> Data:
    agg = pd.read_csv(_AGGREGATED_CSV, dtype={"docID": str, "edinetCode": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS_CSV, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "filerName", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    docs = pd.read_csv(_DOCLIST_CSV, dtype=str)[["docID", "secCode"]]
    m = m.merge(docs, on="docID", how="left")
    assets = m["total_assets"].to_numpy(dtype=float)
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    return Data(
        docid=m["docID"].to_numpy(),
        name=m["filerName"].to_numpy(),
        edinet=m["edinetCode"].fillna("").to_numpy(),
        label=m["label"].astype(int).to_numpy(),
        x_tilde=x / assets[:, None],
        assets=assets,
        sec_null=m["secCode"].isna().to_numpy(),
    )


def make_folds(n2: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n2)
    chunks = np.array_split(order, n_folds)
    return [
        (np.concatenate([chunks[j] for j in range(n_folds) if j != i]), chunks[i])
        for i in range(n_folds)
    ]


def solve_design(
    design: Design,
    base_cost: np.ndarray,
    src33: np.ndarray,
    tgt33: np.ndarray,
    coarse_map: dict[str, str],
) -> tuple[np.ndarray, dict[str, object]]:
    """1回のOT呼び出し(始点集合はbase_costの行)。plan と診断値を返す。"""
    n0, n1 = base_cost.shape
    t0 = time.perf_counter()
    info: dict[str, object] = {}
    if design.kind == "legacy":
        # 比較専用: 標準Sinkhorn(非収束)を意図的に許容する(ステージ13 §A.3)
        res = solve_ot_from_cost_verbose(
            base_cost, n0, n1, _EPSILON, log_domain=False, require_converged=False
        )
    elif design.kind == "soft":
        mismatch = industry_mismatch(src33, tgt33)
        cost_n = soft_industry_cost(base_cost, mismatch, design.lam)
        a = np.full(n0, 1.0 / n0)
        b = np.full(n1, 1.0 / n1)
        res = solve_ot_normalized(cost_n, a, b, _EPSILON, log_domain=True, stop_thr=_LOG_STOP_THR)
    else:
        ba = assign_blocks(
            list(src33), list(tgt33), coarse_map, "33" if design.level == "33" else "coarse",
            min_targets=DEFAULT_MIN_TARGETS,
        )
        inc = ba.src_block >= 0  # 業種制約つきの始点(フォールバックで制約なしになった始点は除く)
        mismatch = np.zeros(base_cost.shape, dtype=bool)
        mismatch[inc] = industry_mismatch(ba.src_block[inc], ba.tgt_block)
        cost_n = hard_industry_cost(base_cost, mismatch)  # 正規化は全始点の基本コストの中央値
        a, b = block_marginals(ba.src_block, ba.tgt_block)
        res = solve_ot_normalized(
            cost_n[inc], a[inc], b, _EPSILON, log_domain=True, stop_thr=_LOG_STOP_THR
        )
        plan = np.zeros(base_cost.shape)
        plan[inc] = res.plan
        # 制約なしにフォールバックした始点は、他の始点と独立に、コストのGibbs分布(質量1/n0)で解く
        for i in np.where(~inc)[0]:
            logits = -cost_n[i] / _EPSILON
            w = np.exp(logits - logits.max())
            plan[i] = w / w.sum() / n0
        info["src_level"] = ba.src_level
        info["n_unrestricted"] = int((~inc).sum())
        info.update(
            converged=res.converged, zero_mass_rows=res.n_zero_mass_rows,
            max_marginal_error=res.max_marginal_error, seconds=time.perf_counter() - t0,
        )
        return plan, info
    info.update(
        converged=res.converged, zero_mass_rows=res.n_zero_mass_rows,
        max_marginal_error=res.max_marginal_error, seconds=time.perf_counter() - t0,
    )
    return res.plan, info


def run_design(design: Design) -> tuple[str, list[dict[str, object]], dict[str, np.ndarray]]:
    """1設計ぶん(全データ1回+5-fold×(train,test))を解く。ワーカープロセスで動く。"""
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    cost_config = load_cost_config(_COSTS)
    ind = load_industry_config(_INDUSTRY)
    d = load_data(names)
    idx0 = np.where(d.label == 0)[0]
    idx2 = np.where(d.label == 2)[0]
    x0, x2 = d.x_tilde[idx0], d.x_tilde[idx2]
    lab33 = np.array(ind.labels(list(d.edinet), "33"))
    src33_all, tgt33 = lab33[idx2], lab33[idx0]
    wp, wm = cost_vectors(cost_config, names, non_actionable_mode="ignore")
    folds = make_folds(len(idx2), _N_FOLDS, seed=0)

    diags: list[dict[str, object]] = []
    plans: dict[str, np.ndarray] = {}

    def record(call: str, plan: np.ndarray, info: dict[str, object]) -> None:
        row = {"design": design.name, "call": call, **{k: v for k, v in info.items()
                                                       if k != "src_level"}}
        diags.append(row)

    base_full = weighted_asymmetric_cost_matrix(x2, x0, wp, wm)
    plan_full, info = solve_design(design, base_full, src33_all, tgt33, ind.coarse_map)
    plans["plan_full"] = plan_full
    record("full", plan_full, info)
    if "src_level" in info:
        plans["src_level_full"] = np.array(info["src_level"])
    for k, (train_pos, test_pos) in enumerate(folds):
        for tag, pos in (("train", train_pos), ("test", test_pos)):
            base = weighted_asymmetric_cost_matrix(x2[pos], x0, wp, wm)
            plan, info = solve_design(design, base, src33_all[pos], tgt33, ind.coarse_map)
            plans[f"plan_{tag}_{k}"] = plan
            record(f"fold{k}_{tag}", plan, info)
    return design.name, diags, plans


def transport_quality(
    design: Design,
    plan_full: np.ndarray,
    d: Data,
    lab33: np.ndarray,
    labc: np.ndarray,
    re_idx: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    idx0 = np.where(d.label == 0)[0]
    idx2 = np.where(d.label == 2)[0]
    tg = np.argmax(plan_full, axis=1)
    tgt_global = idx0[tg]
    counts = pd.Series(d.docid[tgt_global]).value_counts()
    re = d.x_tilde[tgt_global, re_idx]
    rows = pd.DataFrame({
        "design": design.name,
        "source_docID": d.docid[idx2], "source_name": d.name[idx2],
        "source_ind33": lab33[idx2], "source_coarse": labc[idx2],
        "target_docID": d.docid[tgt_global], "target_name": d.name[tgt_global],
        "target_ind33": lab33[tgt_global], "target_coarse": labc[tgt_global],
        "target_RE_over_assets": re, "row_mass": plan_full.sum(axis=1),
    })
    summary: dict[str, object] = {
        "design": design.name,
        "unique_targets": int(counts.size),
        "max_selected": int(counts.iloc[0]),
        "most_selected": str(counts.index[0]),
        "match33": float(np.mean(lab33[idx2] == lab33[tgt_global])),
        "match_coarse": float(np.mean(labc[idx2] == labc[tgt_global])),
        "n_RE_negative": int(np.sum(re < 0)),
        "RE_median": float(np.median(re)),
        "zero_mass_rows": int(np.sum(plan_full.sum(axis=1) < 1e-12)),
    }
    return summary, rows


def industry_report(d: Data, lab33: np.ndarray, labc: np.ndarray) -> str:
    idx0 = np.where(d.label == 0)[0]
    idx2 = np.where(d.label == 2)[0]
    lines = ["=== §1.2 業種情報の取得 ==="]
    lines.append(f"対象: {len(d.docid)}社(class0={len(idx0)}, class2={len(idx2)})")
    lines.append("取得元: EDINETコードリスト(金融庁)の提出者業種(東証33業種と同一体系)")
    n_null, n_null2 = int(d.sec_null.sum()), int(d.sec_null[idx2].sum())
    lines.append(
        f"secCodeがnull(非上場)= {n_null}社(うちclass2={n_null2}社)"
        " → 全社にEDINET業種あり(非上場も除外せず、業種を使用)"
    )
    unk = lab33 == "不明"
    lines.append(f"業種を取得できず「不明」= {int(unk.sum())}社(class0={int(unk[idx0].sum())}, "
                 f"class2={int(unk[idx2].sum())})")
    lines.append("\n-- 正例68社の業種分布(33業種) と class0 側の社数 --")
    c2 = pd.Series(lab33[idx2]).value_counts()
    c0 = pd.Series(lab33[idx0]).value_counts()
    for ind, n in c2.items():
        n0c = int(c0.get(ind, 0))
        flag = "  ← class0が10社未満(フォールバック候補)" if n0c < DEFAULT_MIN_TARGETS else ""
        lines.append(f"  {ind:<14} 正例{n:>3}社  class0 {n0c:>4}社{flag}")
    lines.append("\n-- 正例68社の粗い分類 --")
    cc2 = pd.Series(labc[idx2]).value_counts()
    cc0 = pd.Series(labc[idx0]).value_counts()
    for c, n in cc2.items():
        share = n / len(idx2)
        lines.append(f"  {c:<10} 正例{n:>3}社({share:.0%})  class0 {int(cc0.get(c, 0)):>4}社")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="再計算する設計名(カンマ区切り)")
    args = parser.parse_args()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    ind = load_industry_config(_INDUSTRY)
    d = load_data(names)
    idx2 = np.where(d.label == 2)[0]
    lab33 = np.array(ind.labels(list(d.edinet), "33"))
    labc = np.array(ind.labels(list(d.edinet), "coarse"))
    re_idx = names.index("jppfs_cor_RetainedEarnings")

    report = industry_report(d, lab33, labc)
    print(report)
    (_OUT_DIR / "industry_report.txt").write_text(report, encoding="utf-8")

    folds = make_folds(len(idx2), _N_FOLDS, seed=0)
    fold_arrays: dict[str, np.ndarray] = {}
    for k, (tr, te) in enumerate(folds):
        fold_arrays[f"train_pos_{k}"] = tr
        fold_arrays[f"test_pos_{k}"] = te
    np.savez(_OUT_DIR / "folds.npz", **fold_arrays)  # type: ignore[arg-type]

    only = set(args.only.split(",")) if args.only else {d.name for d in DESIGNS}
    todo = [dz for dz in DESIGNS if dz.name in only]
    print(f"\n=== {len(todo)}設計を並列で解く(対数領域Sinkhorn、ε={_EPSILON}) ===")
    with ProcessPoolExecutor(max_workers=min(len(todo), 8)) as ex:
        results = list(ex.map(run_design, todo))
    diag_path = _OUT_DIR / "plan_diagnostics.csv"
    old_diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    if len(old_diag):
        old_diag = old_diag[~old_diag["design"].isin(only)]
    diag_rows: list[dict[str, object]] = []
    for dz, (name, diags, plans) in zip(todo, results, strict=True):
        assert name == dz.name
        diag_rows.extend(diags)
        np.savez_compressed(_OUT_DIR / f"plans_{name}.npz", **plans)  # type: ignore[arg-type]
        bad = [r for r in diags if not r["converged"]]
        print(f"  {name:<14} 完了 未収束={len(bad)}/{len(diags)}")
    pd.concat([old_diag, pd.DataFrame(diag_rows)]).to_csv(diag_path, index=False)

    # 品質表・対応先は保存済みの全設計のnpzから作る(--onlyでも全設計ぶん更新される)
    summaries: list[dict[str, object]] = []
    target_frames: list[pd.DataFrame] = []
    for dz in DESIGNS:
        plans = dict(np.load(_OUT_DIR / f"plans_{dz.name}.npz"))
        summary, frame = transport_quality(dz, plans["plan_full"], d, lab33, labc, re_idx)
        if "src_level_full" in plans:
            levels = pd.Series(plans["src_level_full"]).value_counts().to_dict()
            summary["fallback_levels"] = str(levels)
        summaries.append(summary)
        target_frames.append(frame)
    pd.concat(target_frames).to_csv(_OUT_DIR / "full_targets.csv", index=False)
    qual = pd.DataFrame(summaries)
    qual.to_csv(_OUT_DIR / "transport_quality.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== §3.1 輸送先の質(全68社→class0、全データ1回) ===")
    print(qual.to_string(index=False))

    # §2.1: λ=0 が現行(対数領域)と一致することの確認、および旧ソルバとの差
    p_cur = np.load(_OUT_DIR / "plans_current.npz")["plan_full"]
    p_old = np.load(_OUT_DIR / "plans_current_asis.npz")["plan_full"]
    ok_rows = p_old.sum(axis=1) > 1e-12
    agree = float(np.mean(np.argmax(p_cur[ok_rows], 1) == np.argmax(p_old[ok_rows], 1)))
    n_ok = int(ok_rows.sum())
    print(f"\nλ=0(current)と旧ソルバ(as-run)のargmax一致率(旧で質量>0の{n_ok}行): {agree:.2f}")
    print(f"出力先: {_OUT_DIR}")


if __name__ == "__main__":
    main()
