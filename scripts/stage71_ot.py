#!/usr/bin/env python
"""ステージ71: 輸送計画の候補集合 S(i) と F3 の目標(POT のプロセス)。

規則は ``results/stage71/criteria.md``。コストはステージ70 の main 版と同じ(72 社、c = 1/s、
OT では利益剰余金・評価差額金を 0、自分自身を除く)。片側不均衡 OT(τ1 ∈ {0.05, 0.5}、
ε = 0.001)を解き直し、行の確率の上位(累積 0.9 まで、最大 10 社)と、F3 のゼロ整合重心
(ハミング距離 ≤ 2 と 0)を保存する。

出力: ``data/interim/stage71/plan_{cls}.npz``。

    uv run python scripts/stage71_ot.py --cls c1
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
from stage70_ot import ot_weights  # noqa: E402
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.models.transport import solve_semi_unbalanced_normalized  # noqa: E402

DATA = Path("data/interim/stage71")
EPS = 0.001
TAUS = (0.05, 0.5)
CUM, KMAX = 0.9, 10


def candidates(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """行ごとの上位(累積 0.9 まで、最大 10 社)。(添字 (n0, 10)、p̃ (n0, 10)、社数 (n0,))。"""
    order = np.argsort(-p, axis=1)[:, :KMAX]
    pk = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(pk, axis=1)
    k = np.minimum((cum < CUM - 1e-12).sum(axis=1) + 1, KMAX)
    keep = np.arange(KMAX)[None, :] < k[:, None]
    pt = np.where(keep, pk, 0.0)
    pt = pt / pt.sum(axis=1, keepdims=True)
    return order, pt, k


def face(xt: np.ndarray, idx: np.ndarray, pt: np.ndarray, signed: np.ndarray, hmax: int
         ) -> tuple[np.ndarray, np.ndarray]:
    """F3: S(i) のうち argmax とゼロのパターンが近い企業の重み付き平均。

    ゼロのパターンは符号一定の科目で見て、argmax とのハミング距離 ≤ hmax の企業だけを使う。
    """
    z = xt[:, signed] == 0.0
    out = np.empty((idx.shape[0], xt.shape[1]))
    used = np.zeros(idx.shape[0], dtype=int)
    for i in range(idx.shape[0]):
        js = idx[i][pt[i] > 0]
        ws = pt[i][pt[i] > 0]
        ham = (z[js] != z[js[0]][None, :]).sum(axis=1)
        sel = ham <= hmax
        out[i] = (ws[sel, None] * xt[js[sel]]).sum(axis=0) / ws[sel].sum()
        used[i] = int(sel.sum())
    return out, used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    signed = np.array([b.sign in ("nonneg", "nonpos") for b in spec.basis])
    if int(signed.sum()) != 20:
        raise ValueError(f"符号一定の科目が 20 でない: {int(signed.sum())}")
    if [b.name for b in spec.basis] != list(names):
        raise ValueError("基礎科目の並びが状態ベクトルと違う")
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
    cp, cm = ot_weights(wj["pools"]["72"], names, True)
    c = l1_cost(xs, xt, cp, cm)
    for i, code in enumerate(src_codes):
        key = f"72:{code}"
        if key in wj["loo"]:
            cpi, cmi = ot_weights(wj["loo"][key], names, True)
            c[i] = l1_cost(xs[i:i + 1], xt, cpi, cmi)[0]
    mask = np.array([[sc == tc for tc in tgt_codes] for sc in src_codes]) & (
        np.array(src_codes)[:, None] != "")
    ct = c / np.median(c[~mask])
    ct[mask] = 10.0 * ct.max()
    n0, n1 = ct.shape
    a, b = np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1)
    store: dict[str, np.ndarray] = {"source_docs": np.array(src)}
    for tau in TAUS:
        t0 = time.perf_counter()
        r = solve_semi_unbalanced_normalized(ct, a, b, EPS, tau, max_iter=2_000_000)
        p = r.plan / r.plan.sum(axis=1, keepdims=True)
        if (p[mask] > 1e-12).any():
            raise ValueError("除外した組に質量がある")
        idx, pt, k = candidates(p)
        lab_t = f"t{tau:g}"
        store[f"{lab_t}__idx"], store[f"{lab_t}__pt"], store[f"{lab_t}__k"] = idx, pt, k
        for h in (2, 0):
            tf, used = face(xt, idx, pt, signed, h)
            store[f"{lab_t}__face{h}"], store[f"{lab_t}__face{h}_used"] = tf, used
        eff = 1.0 / (pt ** 2).sum(axis=1)
        print(f"[{args.cls}] τ1 = {tau:g}: 候補数の中央値 {np.median(k):.0f}、p̃ の実効サポート "
              f"{np.median(eff):.2f}、F3(≤2)の社数 {np.median(store[f'{lab_t}__face2_used']):.0f}、"
              f"F3(0)の社数 {np.median(store[f'{lab_t}__face0_used']):.0f}"
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / f"plan_{args.cls}.npz", **store)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
