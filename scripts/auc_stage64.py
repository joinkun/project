#!/usr/bin/env python
"""ステージ64 B.4: 目標点 τ と class 0 の実データの識別 AUC(lightgbm・sklearn のプロセス)。

``identification_auc_repeated``(繰り返し 3 回、seed 42)。argmax と最近傍は実在企業、重心は合成点。

    uv run python scripts/auc_stage64.py --cls c2
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

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
    z = np.load(Path("data/interim/stage64") / f"targets_{args.cls}.npz", allow_pickle=True)
    rows = []
    for key in ["nn", *sorted(k for k in z.files if k.endswith(("__argmax", "__bary")))]:
        v = z[key]
        tau = xt[v.astype(int)] if v.ndim == 1 else np.asarray(v, dtype=float)
        r = identification_auc_repeated(tau, xt, 3)
        rows.append({"key": key, "auc_lgbm": float(r["lightgbm"].mean()),
                     "auc_logreg": float(r["logreg"].mean()),
                     "n_distinct": int(len(np.unique(tau.round(12), axis=0)))})
        print(f"  {key}: {rows[-1]['auc_lgbm']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(Path("results/stage64") / f"auc_{args.cls}.csv", index=False)


if __name__ == "__main__":
    main()
