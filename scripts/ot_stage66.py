#!/usr/bin/env python
"""ステージ66: 頻度加重の c± を地上コストにした不均衡 OT(τ1 = 0.01、ε = 0.001)の最大質量の対応先。

POT のプロセス(torch・sklearn・cvxpy を import しない)。設定 3〜5 の対応先(設定 5 は QP の目的関数で
選び直すので参考)。``data/interim/stage66/targets_{cls}.npz`` に始点の並び・最近傍・argmax を書く。

    uv run python scripts/ot_stage66.py --cls c2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from density_stage51 import load_tilde
from prep_stage59 import OUTLIERS, l1_cost

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.transport import effective_support_sizes, solve_semi_unbalanced_normalized

DATA = Path("data/interim/stage66")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    wt = json.loads((DATA / f"weights_{args.cls}.json").read_text(encoding="utf-8"))
    cp, cm = np.array(wt["c_plus"]), np.array(wt["c_minus"])
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    if args.cls == "c2":
        is_src = (lab == 2) & ~np.isin(docs_a, OUTLIERS)
    else:
        c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
        is_src = np.isin(docs_a, c1.loc[c1["reason"].fillna("") == "", "docID"].to_numpy())
    xs, xt = x[is_src], x[lab == 0]
    ct = l1_cost(xs, xt, cp, cm)
    ct = ct / np.median(ct)
    n0, n1 = ct.shape
    r = solve_semi_unbalanced_normalized(ct, np.full(n0, 1 / n0), np.full(n1, 1 / n1), 0.001,
                                         0.01, max_iter=2_000_000)
    am, nn = r.plan.argmax(axis=1), ct.argmin(axis=1)
    sup = effective_support_sizes(r.plan)
    info = {"support_med": float(np.median(sup)), "nn_match": int((am == nn).sum()), "n0": n0,
            "n_argmax": int(len(set(am))), "mass": float(r.plan.sum()),
            "min_row_mass": float(r.plan.sum(axis=1).min()), "n_iter": r.n_checkpoints}
    print(args.cls, info, flush=True)
    np.savez_compressed(DATA / f"targets_{args.cls}.npz", source_docs=docs_a[is_src],
                        target_docs=docs_a[lab == 0], semi=am, nn=nn)
    (DATA / f"ot_{args.cls}.json").write_text(json.dumps(info), encoding="utf-8")


if __name__ == "__main__":
    main()
