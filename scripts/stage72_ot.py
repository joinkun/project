#!/usr/bin/env python
"""ステージ72: バランス OT の候補集合と、事前に混ぜる目標・等重み L2 の最近傍(POT のプロセス)。

規則は ``results/stage72/criteria.md``。コストはステージ70・71 と同じ(72 社、c = 1/s、
OT では利益剰余金・評価差額金を 0、自分自身を除く)。

- バランス OT(対数領域の Sinkhorn、ε = 0.001)を解き、S(i)(累積 0.9 まで、最大 10 社)を作る
- τ1 = 0.05 の S(i)(ステージ71 の ``plan_{cls}.npz``)で、事前に混ぜる目標(L2 重心・重み付き中央値)
- 等重み L2(利益剰余金・評価差額金を除く科目)の最近傍(自分自身を除く)

出力: ``data/interim/stage72/plan_{cls}.npz``。

    uv run python scripts/stage72_ot.py --cls c1
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
from prep_stage59 import l1_cost, l2_cost  # noqa: E402
from stage70_ot import ot_weights, weighted_median  # noqa: E402
from stage71_ot import candidates  # noqa: E402
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.models.transport import solve_ot_normalized  # noqa: E402

DATA = Path("data/interim/stage72")
EPS = 0.001
NO_L2 = ("jppfs_cor_RetainedEarnings", "valuation_difference_securities")


def full_rows(idx: np.ndarray, pt: np.ndarray, n1: int) -> np.ndarray:
    """S(i) の重み p̃ を、対応先の全体(n1 列)の行に戻す。"""
    p = np.zeros((idx.shape[0], n1))
    np.put_along_axis(p, idx, pt, axis=1)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    wj = json.loads(Path("data/interim/stage70/weights.json").read_text(encoding="utf-8"))
    x, lab, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    m = sources(args.cls, lab, docs_a)
    xs, xt = x[m], x[lab == 0]
    src = [str(d) for d in docs_a[m]]
    src_codes = [code_of.get(d, "") for d in src]
    tgt_codes = np.array([code_of.get(str(d), "") for d in docs_a[lab == 0]])
    mask = np.array([[sc == tc for tc in tgt_codes] for sc in src_codes]) & (
        np.array(src_codes)[:, None] != "")
    cp, cm = ot_weights(wj["pools"]["72"], names, True)
    c = l1_cost(xs, xt, cp, cm)
    for i, code in enumerate(src_codes):
        key = f"72:{code}"
        if key in wj["loo"]:
            cpi, cmi = ot_weights(wj["loo"][key], names, True)
            c[i] = l1_cost(xs[i:i + 1], xt, cpi, cmi)[0]
    ct = c / np.median(c[~mask])
    ct[mask] = 10.0 * ct.max()
    n0, n1 = ct.shape
    a, b = np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1)
    t0 = time.perf_counter()
    r = solve_ot_normalized(ct, a, b, EPS, max_iter=200000)
    p = r.plan / r.plan.sum(axis=1, keepdims=True)
    if (p[mask] > 1e-12).any():
        raise ValueError("除外した組に質量がある")
    idx, pt, k = candidates(p)
    print(f"[{args.cls}] バランス OT: 候補数の中央値 {np.median(k):.0f}、p̃ の実効サポート "
          f"{np.median(1.0 / (pt ** 2).sum(axis=1)):.2f}、S(i) の質量の中央値 "
          f"{np.median(np.take_along_axis(p, idx, axis=1)[pt > 0].reshape(-1)):.4f}"
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    z71 = np.load(f"data/interim/stage71/plan_{args.cls}.npz", allow_pickle=True)
    if [str(v) for v in z71["source_docs"]] != src:
        raise ValueError("ステージ71 の始点の並びが一致しない")
    idx5, pt5 = np.asarray(z71["t0.05__idx"]), np.asarray(z71["t0.05__pt"])
    p5 = full_rows(idx5, pt5, n1)
    act = np.array([n not in NO_L2 for n in names], dtype=float)
    c2 = l2_cost(xs, xt, act)
    c2[mask] = np.inf
    store = {"source_docs": np.array(src), "bal__idx": idx, "bal__pt": pt, "bal__k": k,
             "pre05_l2": p5 @ xt, "pre05_m1": weighted_median(p5, xt),
             "l2nn": xt[c2.argmin(axis=1)]}
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / f"plan_{args.cls}.npz", **store)


if __name__ == "__main__":
    main()
