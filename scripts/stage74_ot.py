#!/usr/bin/env python
"""ステージ74: 設定ごとのコストで輸送計画を解き、F1 の候補集合 S(i) を作る(POT のプロセス)。

規則は ``results/stage74/criteria.md``。片側不均衡 OT(τ1 = 0.05、ε = 0.001)、地上コストは
始点ごとの係数 c̃_i^± の非対称 L1、C̃ = C / median(C)、自分自身の候補は質量 0。S(i) は累積 0.9
まで(最大 10 社)。

出力: ``data/interim/stage74/plan_{cls}.npz``(キー ``{コストの種類}{プール}__idx`` など)。

    uv run python scripts/stage74_ot.py --cls c1
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
from stage71_ot import candidates  # noqa: E402
from stage74_costs import SETTINGS, entry, firm_costs, pool_of  # noqa: E402
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.models.transport import solve_semi_unbalanced_normalized  # noqa: E402

DATA = Path("data/interim/stage74")
EPS, TAU1 = 0.001, 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    wj = json.loads((DATA / "weights.json").read_text(encoding="utf-8"))
    floor = np.array(wj["floor"], dtype=float)
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
    n0, n1 = len(xs), len(xt)
    a, b = np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1)
    store: dict[str, np.ndarray] = {"source_docs": np.array(src)}
    keys = sorted({(kind, pool_of(st, args.cls)) for st, (kind, _p2, _p1) in SETTINGS.items()})
    for kind, pool in keys:
        t0 = time.perf_counter()
        c = np.empty((n0, n1))
        for i, code in enumerate(src_codes):
            cp, cm = firm_costs(kind, xs[i], entry(wj, pool, code), floor, names)
            c[i] = l1_cost(xs[i:i + 1], xt, cp, cm)[0]
        ct = c / np.median(c[~mask])
        ct[mask] = 10.0 * ct.max()
        r = solve_semi_unbalanced_normalized(ct, a, b, EPS, TAU1, max_iter=2_000_000)
        p = r.plan / r.plan.sum(axis=1, keepdims=True)
        if (p[mask] > 1e-12).any():
            raise ValueError("除外した組に質量がある")
        idx, pt, k = candidates(p)
        key = f"{kind}{pool}"
        store[f"{key}__idx"], store[f"{key}__pt"], store[f"{key}__k"] = idx, pt, k
        store[f"{key}__nn"] = ct.argmin(axis=1)
        print(f"[{args.cls} {key}] 候補数の中央値 {np.median(k):.0f}、p̃ の実効サポート "
              f"{np.median(1.0 / (pt ** 2).sum(axis=1)):.2f}、argmax = 最近傍 "
              f"{(idx[:, 0] == ct.argmin(axis=1)).mean():.0%}({time.perf_counter() - t0:.0f}s)",
              flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / f"plan_{args.cls}.npz", **store)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
