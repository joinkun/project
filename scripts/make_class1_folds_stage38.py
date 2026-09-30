#!/usr/bin/env python
"""ステージ38: class 1 を始点とする遷移(T2)のための 5-fold 分割と、ステージ38 の除外リストを作る。

- 分割: 除外前の class 1 の並び(``precompute_stage13_transport.load_data`` と同じ条件)での位置を、
  numpy の seed 0 で並べ替えて 5 つに分ける。class 2 の ``folds.npz`` は作り直さない
- 除外: ステージ28 の除外リスト + class 1 の特殊業種 5 社(ステージ34)+ class 1 の外れ値
  シンバイオ製薬(S100VGDJ、ステージ34 の基準)。ステージ38 では外れ値を始点・目標の両方から除く

    uv run python scripts/make_class1_folds_stage38.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from precompute_stage13_transport import load_data

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

N_FOLDS = 5
OUTLIER_CLASS1 = "S100VGDJ"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim/stage38"))
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    _x, _a, label, _doc, _f, _e, _keep = load_data(names)
    n1 = int((label == 1).sum())
    perm = np.random.default_rng(0).permutation(n1)
    parts = np.array_split(perm, N_FOLDS)
    folds = {}
    for k in range(N_FOLDS):
        folds[f"test_pos_{k}"] = np.sort(parts[k])
        folds[f"train_pos_{k}"] = np.sort(np.concatenate([parts[j] for j in range(N_FOLDS)
                                                          if j != k]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "folds_class1.npz", **folds)  # type: ignore[arg-type]

    ex28 = pd.read_csv("data/interim/stage28/excluded.csv", dtype={"docID": str})
    c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
    c1x = c1[c1["reason"].fillna("") != ""]
    extra = pd.DataFrame({"docID": [OUTLIER_CLASS1], "reason": ["outlier"]})
    ex = pd.concat([ex28[["docID"]].assign(reason=ex28.get("reason", "")),
                    c1x[["docID", "reason"]], extra], ignore_index=True).drop_duplicates("docID")
    ex.to_csv(args.out_dir / "excluded.csv", index=False)
    print(f"class 1(除外前){n1} 社を 5 分割。除外リスト {len(ex)} 書類"
          f"(ステージ28 {len(ex28)} + class 1 の特殊業種 {len(c1x)} + 外れ値 1)")


if __name__ == "__main__":
    main()
