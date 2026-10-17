#!/usr/bin/env python
"""ステージ52 C: 密度 + 距離で実在の健全企業から対応先を選ぶ(numpy のみ)。

規則は ``results/stage52/criteria_b.md`` §5。j*(i) = argmin_j [ −log q(y_j) + λ d(x_i, y_j) ]、
d は非操作 3 科目を除く 29 科目の二乗距離(現行の ``nearest_na0`` と同じ)。λ = c · s、
s = class 0 の −log q の四分位範囲 / 現行の最近傍の d の中央値、c ∈ {0.1, 0.5, 1, 5, 10}。
d の中央値が現行の 1.5 倍以内に収まる最小の c を採る。C.2 のゼロパターンの一致を c ごとに出す。

出力: ``data/interim/stage52/plans/plans_{indep,pattern}.npz``(現行と同じ形)、
``results/stage52/matching.md``・``matching.csv``。

    uv run python scripts/matching_stage52.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.datasets.exclusion import subset_folds
from fsgen.models.cost import load_cost_config
from fsgen.models.zero_density import IndepZeroGaussian, PatternMixture

_EQUAL = Path("data/interim/stage28/plans_equal/plans_nearest_na0.npz")
_FOLDS = Path("data/interim/industry_12/folds.npz")
_D = Path("data/interim/stage52/density")
C_GRID = (0.1, 0.5, 1.0, 5.0, 10.0)
ROUNDING = ("__OTHER__jppfs_cor_Assets", "__OTHER__jppfs_cor_NoncurrentAssets",
            "__OTHER__jppfs_cor_Liabilities", "__OTHER__jppfs_cor_ShareholdersEquity")
_TS = "jppfs_cor_TreasuryStock"


def load_models() -> dict[str, IndepZeroGaussian | PatternMixture]:
    a = np.load(_D / "indep.npz")
    b = np.load(_D / "pattern.npz")
    m = len(b["pi"])
    return {"indep": IndepZeroGaussian(a["pi0"], a["u"], a["mu"], a["cov"]),
            "pattern": PatternMixture(b["cols"], b["patterns"], b["pi"],
                                      [b[f"mean{j}"] for j in range(m)],
                                      [b[f"cov{j}"] for j in range(m)])}


def nn_plan(cost: np.ndarray) -> np.ndarray:
    plan = np.zeros_like(cost)
    plan[np.arange(len(cost)), np.argmin(cost, axis=1)] = 1.0 / len(cost)
    return plan


def match_stats(xs: np.ndarray, xt: np.ndarray, names: tuple[str, ...]) -> dict[str, int]:
    """C.2: 始点と対応先のゼロパターンの一致。"""
    zs, zt = xs == 0, xt == 0
    keep28 = np.array([n not in ROUNDING for n in names])
    its = names.index(_TS)
    ts0 = zs[:, its]
    return {"exact32": int(np.all(zs == zt, axis=1).sum()),
            "exact28": int(np.all((zs == zt)[:, keep28], axis=1).sum()),
            "target_has_extra": int(np.any(zs & ~zt, axis=1).sum()),
            "ts_src0": int(ts0.sum()), "ts_target_has": int((ts0 & ~zt[:, its]).sum())}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/interim/stage52/plans"))
    ap.add_argument("--results", type=Path, default=Path("results/stage52"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    cc = load_cost_config(Path("conf/accounts/costs_equal.yaml"))
    mask = np.array([0.0 if n in cc.non_actionable else 1.0 for n in names])
    x, label, _docs, src_keep = load_tilde(names)
    x0, x2 = x[label == 0], x[label == 2]
    folds = subset_folds(np.load(_FOLDS), src_keep)
    d = np.zeros((len(x2), len(x0)))
    for a in np.where(mask > 0)[0]:
        d += (x2[:, a][:, None] - x0[:, a][None, :]) ** 2
    nn = np.argmax(np.load(_EQUAL)["plan_full"], axis=1)
    if not np.array_equal(nn, np.argmin(d, axis=1)):
        raise ValueError("現行の最近傍が距離の再計算と一致しない(並びの不一致)")
    d_nn = float(np.median(d[np.arange(len(x2)), nn]))
    rows = [{"version": "現行(最近傍)", "c": np.nan, "lambda": np.nan, "d_median": d_nn,
             "logq_median": np.nan, "changed": 0, "unique": len(set(nn.tolist())),
             **match_stats(x2, x0[nn], names)}]
    lines = ["# ステージ52 C: 密度 + 距離による対応先(実在の健全企業から選ぶ)", "",
             "j* = argmin_j [ −log q(y_j) + λ d(x_i, y_j) ]、全社の計画(class 2 57 社)。λ = c · s、"
             "s = class 0 の −log q の四分位範囲 / 現行の最近傍の d の中央値。"
             "採用 = d の中央値が現行の 1.5 倍以内の最小の c(`criteria_b.md` §5)。", "",
             "C.2 の列: 完全一致 = 始点と対応先のゼロパターンが同じ(32 科目 / 丸め差の残余を除く"
             " 28 科目)、"
             "余分 = 対応先が始点で 0 の科目を非 0 で持つ、自己株式 = 自己株式を持たない始点の"
             "うち対応先が持つ社数。", "",
             "| 版 | c | λ | d の中央値 | 対応先の log q の中央値 | 現行から変わった社数 | 異な"
             "る対応先 "
             "| 完全一致 32 / 28 | 余分あり | 自己株式 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    r0 = rows[0]
    lines.append(f"| 現行(最近傍) | — | — | {d_nn:.4f} | — | — | {r0['unique']} "
                 f"| {r0['exact32']} / {r0['exact28']} | {r0['target_has_extra']} "
                 f"| {r0['ts_target_has']} / {r0['ts_src0']} |")
    notes: list[str] = []
    for vname, model in load_models().items():
        nlq = -model.logpdf(x0)
        s = float(np.subtract(*np.percentile(nlq, [75, 25])) / d_nn)
        chosen = None
        for c in C_GRID:
            lam = c * s
            tg = np.argmin(nlq[None, :] + lam * d, axis=1)
            dm = float(np.median(d[np.arange(len(x2)), tg]))
            st = match_stats(x2, x0[tg], names)
            rows.append({"version": vname, "c": c, "lambda": lam, "d_median": dm,
                         "logq_median": float(np.median(-nlq[tg])),
                         "changed": int((tg != nn).sum()), "unique": len(set(tg.tolist())), **st})
            if chosen is None and dm <= 1.5 * d_nn:
                chosen = c
            lines.append(f"| {vname} | {c} | {lam:.4g} | {dm:.4f} | {np.median(-nlq[tg]):.1f} "
                         f"| {(tg != nn).sum()} | {len(set(tg.tolist()))} "
                         f"| {st['exact32']} / {st['exact28']} | {st['target_has_extra']} "
                         f"| {st['ts_target_has']} / {st['ts_src0']} |")
        if chosen is None:
            raise ValueError(f"{vname}: 1.5 倍以内に収まる c がない")
        lam = chosen * s
        cost = nlq[None, :] + lam * d
        plans = {"plan_full": nn_plan(cost)}
        for k in range(5):
            for part in ("train", "test"):
                pos = folds[f"{part}_pos_{k}"]
                plans[f"plan_{part}_{k}"] = nn_plan(cost[pos])
        np.savez_compressed(args.out / f"plans_{vname}.npz", **plans)  # type: ignore[arg-type]
        notes.append(f"- **{vname}: c = {chosen} を採用**(λ = {lam:.4g}、s = {s:.4g})")
    lines += ["", *notes, ""]
    pd.DataFrame(rows).to_csv(args.results / "matching.csv", index=False)
    (args.results / "matching.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
