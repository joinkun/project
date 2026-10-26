#!/usr/bin/env python
"""ステージ72 追補: F1 の τ1 の感度のための候補集合 S(i)(τ1 = 0.01・0.1、POT のプロセス)。

規則は ``results/stage72/addendum_criteria.md``。コストはステージ70〜72 と同じ(72 社、
c = 1/s、OT では利益剰余金・評価差額金を 0、自分自身を除く)。片側不均衡 OT(ε = 0.001)。

出力: ``data/interim/stage72/tau_{cls}.npz``。

    uv run python scripts/stage72_tau_ot.py --cls c1
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
from stage71_ot import candidates  # noqa: E402
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.models.transport import solve_semi_unbalanced_normalized  # noqa: E402

DATA = Path("data/interim/stage72")
EPS = 0.001
TAUS = (0.01, 0.1)


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
        print(f"[{args.cls}] τ1 = {tau:g}: 候補数の中央値 {np.median(k):.0f}、p̃ の実効サポート "
              f"{np.median(1.0 / (pt ** 2).sum(axis=1)):.2f}({time.perf_counter() - t0:.0f}s)",
              flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / f"tau_{args.cls}.npz", **store)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
