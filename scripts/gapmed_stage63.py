#!/usr/bin/env python
"""ステージ63 A.1: 群間差 d_a の表と、m_a = d_a / median(d) の設計 gapmed。

規則は ``results/stage63/criteria.md`` §1。

    uv run python scripts/gapmed_stage63.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from density_stage51 import load_tilde
from prep_stage59 import OUTLIERS

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config

OUT = Path("results/stage63")
DATA = Path("data/interim/stage63/m")


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    na = set(load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable)
    act = np.array([n not in na for n in names])
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
    src = {"c2": (lab == 2) & ~np.isin(docs_a, OUTLIERS),
           "c1": np.isin(docs_a, c1.loc[c1["reason"].fillna("") == "", "docID"].to_numpy())}
    xt = x[lab == 0]
    DATA.mkdir(parents=True, exist_ok=True)
    g62 = {c: json.loads(Path(f"data/interim/stage62/m/{c}_gap.json").read_text(encoding="utf-8"))
           for c in src}
    lines = ["# ステージ63 A.1: 群間差 d_a と m_a = d_a / median(d)", "",
             "d_a = |生成する class の平均 − class 0 の平均|(x̃)。median は操作できる 29 科目。"
             "gap(ステージ62)は同じ d を幾何平均 1 で正規化したもの(全科目に共通の定数倍の違い"
             "だけ)。", ""]
    for cls, mask in src.items():
        d = np.abs(x[mask].mean(axis=0) - xt.mean(axis=0))
        med = float(np.median(d[act]))
        m = np.ones(len(names))
        m[act] = np.clip(np.maximum(d[act], 0.001) / med, 0.01, 100.0)
        (DATA / f"{cls}_gapmed.json").write_text(
            json.dumps({n: float(v) for n, v in zip(names, m, strict=True)}), encoding="utf-8")
        order = np.argsort(-np.where(act, d, -1))
        rank = {j: r + 1 for r, j in enumerate(order[: int(act.sum())])}
        k = np.array([g62[cls][n] for n in names]) / np.where(m > 0, m, 1)
        ratio = float(np.median(k[act]))
        lines += [f"## {cls}(median(d) = {med:.4f}、gap / gapmed の比 = {ratio:.2f})", "",
                  "| 順位 | 科目 | d_a | d_a / median(d) | m(gapmed) | m(gap、ステージ62) |",
                  "|---:|---|---:|---:|---:|---:|"]
        for j in order[: int(act.sum())]:
            nm = ja.get(names[j], names[j])
            lines.append(f"| {rank[j]} | {nm} | {d[j]:.4f} | {d[j] / med:.2f} "
                         f"| {m[j]:.2f} | {g62[cls][names[j]]:.2f} |")
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gap_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
