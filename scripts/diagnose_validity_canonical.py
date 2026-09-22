#!/usr/bin/env python
"""ステージ16: canonical変更(30→32科目)でvalidityが下がった原因の切り分け(lightgbm/sklearn)。

ステージ15(旧30科目)のvalidity_lightgbm(S0で0.958、S5で0.978)が、新32科目では0.83〜0.85に下がった。
原因が(a)生成そのものの劣化か、(b)32次元の分類器が貸倒引当金の分布の違いを捉えるためかを切り分ける:
新32科目の生成結果を旧30科目に**畳み込み**(流動資産その他+=貸倒引当金(流動)、投資その他の資産+=
貸倒引当金(固定))、旧データ(canonical_v30)で学習した旧分類器で測る。
また、新分類器で class0 の実データ(留保)と始点自体のvalidityも測る。
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402

_OLD = Path("data/interim/canonical_v30")


def load(agg_path: Path, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    agg = pd.read_csv(agg_path, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    x = m[list(names)].fillna(0.0).to_numpy(float) / m["total_assets"].to_numpy(float)[:, None]
    return x, m["label"].astype(int).to_numpy() == 2


def main() -> None:
    old_names = state_vector_names(load_canonical_spec(_OLD / "core.yaml"), include_subtotals=False)
    new_names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                                   include_subtotals=False)
    xo, is2 = load(_OLD / "aggregated_core.csv", old_names)
    xn, _ = load(Path("data/interim/values/aggregated_core.csv"), new_names)
    clf_old = train_classifiers(xo, is2, seed=42)
    clf_new = train_classifiers(xn, is2, seed=42)

    def fold_to_old(x_new: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x_new), len(old_names)))
        for j, n in enumerate(old_names):
            out[:, j] = x_new[:, new_names.index(n)] if n in new_names else 0.0
        ca, ioa = old_names.index("__OTHER__jppfs_cor_CurrentAssets"), old_names.index(
            "jppfs_cor_InvestmentsAndOtherAssets")
        out[:, ca] += x_new[:, new_names.index("allowance_doubtful_current")]
        out[:, ioa] += x_new[:, new_names.index("allowance_doubtful_noncurrent")]
        return out

    print("=== 分類器の自己確認(実データ) ===")
    for lab, x, c in (("旧30科目", xo, clf_old), ("新32科目", xn, clf_new)):
        p = c["lightgbm"].predict_proba_class2(x)
        print(f"{lab}: class0の平均P(class2)={p[~is2].mean():.3f} class2={p[is2].mean():.3f}")
    print(f"始点68社のvalidity(1-P(class2), 新分類器 lightgbm 中央値): "
          f"{np.median(1 - clf_new['lightgbm'].predict_proba_class2(xn[is2])):.3f}")

    print("\n=== 新32科目の生成結果のvalidity(lightgbm中央値、行=68社×3seed) ===")
    print(f"{'設定':<28}{'新分類器(32)':>14}{'旧分類器(畳込)':>16}")
    for tag, cfg in (("nearest_na0", "W1_S0"), ("nearest_na0", "W1_S5_a05_b00"),
                     ("nearestu_na0", "W1_S0"), ("nearestu_na0", "W1_S5_a05_b00")):
        files = glob.glob(f"data/interim/stage16/{tag}__seed*.parquet")
        d = pd.concat([pd.read_parquet(f) for f in files])
        d = d[(d["method"] == "PCFM") & (d["config"] == cfg)]
        x1 = d[[f"x1_{n}" for n in new_names]].to_numpy()
        vn = np.median(1 - clf_new["lightgbm"].predict_proba_class2(x1))
        vo = np.median(1 - clf_old["lightgbm"].predict_proba_class2(fold_to_old(x1)))
        print(f"{tag + ' ' + cfg:<28}{vn:>14.3f}{vo:>16.3f}")
    print("\n(参考)ステージ15(旧30科目・旧生成): nearest_na0 W1 S0=0.958 / S5=0.978")


if __name__ == "__main__":
    main()
