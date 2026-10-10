#!/usr/bin/env python
"""ステージ60 D.2: 目標点 τ と class 0 の実データの識別 AUC(lightgbm・sklearn のプロセス)。

``identification_auc_repeated``(LightGBM・ロジスティック回帰、5-fold CV、class 0 から目標点と同数を
無作為に抽出、繰り返し 3 回、seed 42)。AUC が 1.0 に近いほど、目標点は実在企業と区別できる合成点。

    uv run python scripts/auc_stage60.py --cls c2
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

DATA = Path("data/interim/stage60")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    x, lab, _docs, _keep = load_tilde(names)
    real0 = x[lab == 0]
    tg = np.load(DATA / f"targets_{args.cls}.npz", allow_pickle=True)
    rows = []
    for key in sorted(k for k in tg.files if k.endswith(("__bary", "__argmax"))):
        r = identification_auc_repeated(tg[key], real0, args.repeats)
        setting, kind = key.rsplit("__", 1)
        rows.append({"setting": setting, "target": kind, "auc_lgbm": float(r["lightgbm"].mean()),
                     "auc_logreg": float(r["logreg"].mean())})
        print(f"  {key}: {rows[-1]['auc_lgbm']:.3f} / {rows[-1]['auc_logreg']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(Path("results/stage60") / f"auc_{args.cls}.csv", index=False)


if __name__ == "__main__":
    main()
