#!/usr/bin/env python
"""ステージ65 B.2: 各手法の目標点 τ と class 0 の実データの識別 AUC(lightgbm・sklearn のプロセス)。

``identification_auc_repeated``(繰り返し 3 回、seed 42)。目標は、等重み L2・対称 L1 の最近傍
(ステージ61 の ``targets_{cls}.npz``)と、不均衡 OT・バランス OT の argmax・重心(ステージ64)。

    uv run python scripts/auc_stage65.py --cls c2
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from density_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    x, lab, _docs, _keep = load_tilde(names)
    xt = x[lab == 0]
    z61 = np.load(f"data/interim/stage61/targets_{args.cls}.npz", allow_pickle=True)
    z64 = np.load(f"data/interim/stage64/targets_{args.cls}.npz", allow_pickle=True)
    taus = {"m3_l2": np.asarray(z61["l2"], dtype=float),
            "m4_l1s": np.asarray(z61["l1s"], dtype=float),
            "m5_prop": xt[z64["semi_t0.01_e0.001__argmax"].astype(int)],
            "m6_balarg": xt[z64["bal_e0.001__argmax"].astype(int)],
            "m7_bary": np.asarray(z64["bal_e0.001__bary"], dtype=float)}
    rows = []
    for k, tau in taus.items():
        r = identification_auc_repeated(tau, xt, 3)
        rows.append({"method": k, "auc_lgbm": float(r["lightgbm"].mean()),
                     "auc_logreg": float(r["logreg"].mean()),
                     "n_distinct": int(len(np.unique(tau.round(12), axis=0)))})
        print(rows[-1], flush=True)
    Path("results/stage65").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path("results/stage65") / f"auc_{args.cls}.csv", index=False)


if __name__ == "__main__":
    main()
