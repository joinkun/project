#!/usr/bin/env python
"""ステージ70 A.2: 新しいコストで輸送計画を解き、目標(argmax・M1・M2・L2 重心)を作る。

POT のプロセス。

規則は ``results/stage70/criteria.md`` §2。片側不均衡 OT(τ1 ∈ {0.01, 0.05, 0.1, 0.5}、ε = 0.001)と
バランス OT(対数領域の Sinkhorn、ε = 0.001)。

版(``--variant``):

- ``main``: 72 社の c±、OT では利益剰余金と評価差額金を 0、非支配株主持分は c = 1/s
- ``nci0``: main で非支配株主持分も 0
- ``p57``: class 1 だけ、同じ class の参照(57 社)の c±

出力: ``data/interim/stage70/targets_{cls}_{variant}.npz``、
``results/stage70/ot_{cls}_{variant}.csv``。

    uv run python scripts/stage70_ot.py --cls c1 --variant main
"""

from __future__ import annotations

import os
import sys

# POT の torch バックエンドと sklearn の読み込みを止める(precompute_transport.py と同じ)
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from prep_stage59 import l1_cost  # noqa: E402
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.models.transport import (  # noqa: E402
    effective_support_sizes,
    solve_ot_normalized,
    solve_semi_unbalanced_normalized,
)

DATA = Path("data/interim/stage70")
OUT = Path("results/stage70")
EPS = 0.001
TAUS = (0.01, 0.05, 0.1, 0.5)
_RE, _VAL, _NCI = ("jppfs_cor_RetainedEarnings", "valuation_difference_securities",
                   "non_controlling_interests")


def ot_weights(w: dict[str, list[float]], names: tuple[str, ...], nci: bool
               ) -> tuple[np.ndarray, np.ndarray]:
    cp, cm = np.array(w["c_plus"], dtype=float), np.array(w["c_minus"], dtype=float)
    zero = [_RE, _VAL] + ([] if nci else [_NCI])
    for nm in zero:
        cp[names.index(nm)] = cm[names.index(nm)] = 0.0
    return cp, cm


def weighted_median(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """行ごと・科目ごとの重み付き中央値(下側。累積重みが初めて 0.5 以上になる値)。"""
    out = np.empty((p.shape[0], y.shape[1]))
    order = np.argsort(y, axis=0, kind="stable")
    ys = np.take_along_axis(y, order, axis=0)
    for i in range(p.shape[0]):
        cum = np.cumsum(p[i][order], axis=0)
        k = (cum >= 0.5 - 1e-12).argmax(axis=0)
        out[i] = ys[k, np.arange(y.shape[1])]
    return out


def sparse_bary(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """非ゼロの確率が 0.5 未満の科目は 0、それ以外は非ゼロの企業の中での条件付き重み付き平均。"""
    nz = (y != 0.0).astype(float)
    pi = p @ nz
    num = p @ (y * nz)
    out: np.ndarray = np.where(pi >= 0.5, num / np.where(pi > 0, pi, 1.0), 0.0)
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--variant", choices=("main", "nci0", "p57"), required=True)
    args = ap.parse_args()
    if args.variant == "p57" and args.cls != "c1":
        raise ValueError("p57 は class 1 だけ")
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    wj = json.loads((DATA / "weights.json").read_text(encoding="utf-8"))
    pool = "57" if args.variant == "p57" else "72"
    nci = args.variant != "nci0"
    x, lab, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    m = sources(args.cls, lab, docs_a)
    xs, xt = x[m], x[lab == 0]
    src = [str(d) for d in docs_a[m]]
    src_codes = [code_of.get(d, "") for d in src]
    tgt_codes = np.array([code_of.get(str(d), "") for d in docs_a[lab == 0]])
    cp, cm = ot_weights(wj["pools"][pool], names, nci)
    c = l1_cost(xs, xt, cp, cm)
    for i, code in enumerate(src_codes):  # 自分自身を除いたコスト
        key = f"{pool}:{code}"
        if key in wj["loo"]:
            cpi, cmi = ot_weights(wj["loo"][key], names, nci)
            c[i] = l1_cost(xs[i:i + 1], xt, cpi, cmi)[0]
    mask = np.array([[sc == tc for tc in tgt_codes] for sc in src_codes]) & (
        np.array(src_codes)[:, None] != "")
    ct = c / np.median(c[~mask])
    ct[mask] = 10.0 * ct.max()
    n0, n1 = ct.shape
    a, b = np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1)
    nn = ct.argmin(axis=1)
    print(f"[{args.cls} {args.variant}] n0 = {n0}、n1 = {n1}、除外した組 {int(mask.sum())}",
          flush=True)
    store: dict[str, np.ndarray] = {"source_docs": np.array(src), "nn": nn}
    rows = []
    for tau in (*TAUS, None):
        t0 = time.perf_counter()
        lab_t = "bal" if tau is None else f"t{tau:g}"
        if tau is None:
            r = solve_ot_normalized(ct, a, b, EPS, max_iter=200000)
        else:
            r = solve_semi_unbalanced_normalized(ct, a, b, EPS, tau, max_iter=2_000_000)
        plan = r.plan
        if (plan[mask] > 1e-12).any():
            raise ValueError("除外した組に質量がある")
        p = plan / plan.sum(axis=1, keepdims=True)
        sup = effective_support_sizes(plan)
        am = p.argmax(axis=1)
        store[f"{lab_t}__argmax"] = xt[am]
        store[f"{lab_t}__m1"] = weighted_median(p, xt)
        store[f"{lab_t}__m2"] = sparse_bary(p, xt)
        store[f"{lab_t}__l2"] = p @ xt
        store[f"{lab_t}__support"] = sup
        store[f"{lab_t}__argmax_idx"] = am
        rows.append({"tau1": "∞(バランス)" if tau is None else tau, "key": lab_t,
                     "converged": bool(r.converged),
                     "marg_err": float(r.max_marginal_error),
                     "mass": float(plan.sum()), "support_med": float(np.median(sup)),
                     "support_p10": float(np.percentile(sup, 10)),
                     "support_p90": float(np.percentile(sup, 90)),
                     "nn_match": float((am == nn).mean()),
                     "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  {lab_t}: 実効サポート 中央値 {rows[-1]['support_med']:.2f}、"
              f"最近傍と一致 {rows[-1]['nn_match']:.0%}({rows[-1]['seconds']}s)", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / f"targets_{args.cls}_{args.variant}.npz", **store)  # type: ignore[arg-type]
    pd.DataFrame(rows).to_csv(OUT / f"ot_{args.cls}_{args.variant}.csv", index=False)


if __name__ == "__main__":
    main()
