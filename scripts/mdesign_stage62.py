#!/usr/bin/env python
"""ステージ62 B.2: 第 1 項の重みの係数 m_a(規則は ``results/stage62/criteria.md`` §2)。

base 以外の 6 設計を class ごとに ``data/interim/stage62/m/<cls>_<設計>.json`` に書き、要約を
``results/stage62/m_designs.md`` に出す。

    uv run python scripts/mdesign_stage62.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from density_stage51 import load_tilde
from prep_stage59 import OUTLIERS, l2_cost

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config

DATA = Path("data/interim/stage62/m")
OUT = Path("results/stage62")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
FLOOR, LO, HI = 0.001, 0.01, 100.0


def normalized(v: np.ndarray, act: np.ndarray) -> np.ndarray:
    """操作できる科目で幾何平均 1、[0.01, 100] で切る。非操作は 1。"""
    m = np.ones(len(v))
    lv = np.log(np.maximum(v[act], FLOOR))
    m[act] = np.clip(np.exp(lv - lv.mean()), LO, HI)
    return m


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
    table = {}
    for cls, mask in src.items():
        xs = x[mask]
        d = np.abs(xs.mean(axis=0) - xt.mean(axis=0))
        j = l2_cost(xs, xt, act.astype(float)).argmin(axis=1)
        sq = (xt[j] - xs) ** 2 * act
        rho = sq.mean(axis=0) / sq.sum(axis=1).mean()
        designs = {f"paid{k}": np.array([float(k) if n in _PAID else 1.0 for n in names])
                   for k in (3, 10, 30)}
        designs["gap"] = normalized(d, act)
        designs["invgap"] = normalized(1.0 / np.maximum(d, FLOOR), act)
        designs["invrho"] = normalized(1.0 / np.maximum(rho, FLOOR), act)
        for k, m in designs.items():
            (DATA / f"{cls}_{k}.json").write_text(
                json.dumps({n: float(v) for n, v in zip(names, m, strict=True)}), encoding="utf-8")
            table[(cls, k)] = m
    lines = ["# ステージ62 B.2: 第 1 項の重みの係数 m_a(操作できる科目、幾何平均 1)", "",
             "| 科目 | " + " | ".join(f"{c} {k}" for c, k in table if not k.startswith("paid"))
             + " |", "|---|" + "---:|" * sum(not k.startswith("paid") for _, k in table)]
    for i, n in enumerate(names):
        if not act[i]:
            continue
        lines.append(f"| {ja.get(n, n)} | " + " | ".join(
            f"{m[i]:.2f}" for (c, k), m in table.items() if not k.startswith("paid")) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m_designs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
