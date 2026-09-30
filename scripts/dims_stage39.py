#!/usr/bin/env python
"""ステージ39 A.3: 細分化の前後(D = 32・38)の次元、子の出現率、残余の大きさ、符号。

各ルート(``conf/accounts/core.yaml`` と ``data/interim/values/aggregated_core.csv`` を持つ
ディレクトリ)について、評価対象の書類(解析成功・金融業でない・ラベルあり、除外リストを除く)で:

- D、小計の数、rank(C)(基礎科目のみ・小計込み)、実効次元 D − rank(C)
- null(C^T) の座標での PCA の 95%・99% 累積寄与に必要な主成分数(全書類・class 2)
- 基礎科目ごとの非ゼロの割合(全書類・class 2)、負の割合、総資産比の中央値・p90・最小
- 貸借の残差 r1〜r4 の最大(総資産比)

    uv run python scripts/dims_stage39.py --roots . data/stage39_root --out results/stage39/dims.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from estimate_intrinsic_dimension import n_components_for, pca

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    orthonormal_basis,
    state_vector_names,
)


def load_root(root: Path, exclude: Path) -> tuple[pd.DataFrame, tuple[str, ...], list[str]]:
    spec = load_canonical_spec(root / "conf/accounts/core.yaml")
    names = state_vector_names(spec, include_subtotals=False)
    agg = pd.read_csv(root / "data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(root / "data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] > 0)]
    excluded = set(pd.read_csv(exclude, dtype={"docID": str})["docID"])
    m = m[~m["docID"].isin(excluded)].reset_index(drop=True)
    m["label"] = m["label"].astype(int)
    lines = []
    d = len(names)
    c_basis = build_constraint_matrix(spec, include_subtotals=False)
    c_full = build_constraint_matrix(spec, include_subtotals=True)
    r_basis = int(np.linalg.matrix_rank(c_basis))
    r_full = int(np.linalg.matrix_rank(c_full))
    n_full = len(state_vector_names(spec, include_subtotals=True))
    q = orthonormal_basis(c_basis)
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float) / m["total_assets"].to_numpy()[:, None]
    dims = []
    for key, mask in (("全書類", np.ones(len(m), dtype=bool)), ("class 2", m["label"] == 2)):
        cum = pca(x[np.asarray(mask)] @ q).cumulative
        dims.append(f"{key}({int(np.sum(mask)):,} 書類)95% {n_components_for(cum, 0.95)}"
                    f"・99% {n_components_for(cum, 0.99)}")
    res = pd.read_csv(root / "data/interim/values/residuals.csv", dtype={"docID": str})
    res = res[res["docID"].isin(m["docID"])]
    rmax = {r: float((res[r].abs() / res["assets"].abs()).max()) for r in ("r1", "r2", "r3", "r4")}
    lines += [
        f"- D = {d}、小計 {n_full - d}、rank(C) = {r_basis}(基礎科目のみ)・{r_full}(小計込み)、"
        f"実効次元 D − rank = {d - r_basis}",
        f"- PCA(null(C^T) の座標、総資産比): {'、'.join(dims)}",
        "- 貸借の残差の最大(総資産比): "
        + "、".join(f"{k} {v:.1e}" for k, v in rmax.items()),
    ]
    return m, names, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--exclude", type=Path,
                    default=Path("data/interim/stage28/excluded.csv").resolve())
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = ["# ステージ39 A.3: 細分化の前後の次元と科目", "",
           "評価対象の書類(解析成功・金融業でない・ラベルあり)から、特殊業種の除外リスト"
           "(ステージ28)を除いた。", ""]
    for root in args.roots:
        m, names, lines = load_root(root, args.exclude)
        ja = yaml.safe_load((root / "conf/accounts/labels_ja.yaml").read_text(
            encoding="utf-8"))["accounts"]
        out += [f"## {root}(D = {len(names)})", "", *lines, "",
                "| 科目 | 非ゼロ(全書類) | 非ゼロ(class 2) | 負の割合 | 中央値 | p90 | 最小 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        c2 = (m["label"] == 2).to_numpy()
        for n in names:
            v = m[n].fillna(0.0).to_numpy() / m["total_assets"].to_numpy()
            out.append(
                f"| {ja.get(n, n)} | {np.mean(v != 0):.2f} | {np.mean(v[c2] != 0):.2f} "
                f"| {np.mean(v < 0):.3f} | {np.median(v):.4f} | {np.quantile(v, 0.9):.4f} "
                f"| {v.min():.4f} |")
        out.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
