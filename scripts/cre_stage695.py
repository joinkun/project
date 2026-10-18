#!/usr/bin/env python
"""ステージ69.5 B: class 1 用の c_RE を推定し直す。

最終設定の c± と同じ規則(``scales_stage59.scale_table``)を使う。

ステージ67 は、減少の企業数が 5 未満(n = 4)の c_RE⁻ を反対の向きの s で代替した
(c = 3.1)。最終設定の規則は「n ≥ 5 で操作可能な全方向の c の中央値」で代替する。
利益剰余金を操作可能として扱い(コストを与えるので)、残りの非操作 2 科目は 0 のまま。
numpy・pandas だけのプロセス。

出力: ``results/stage69/cre_check.md``、``data/interim/stage695/weights_c1_rule.json``
(ステージ60 の c± に、規則どおりの c_RE± を入れたもの)。

    uv run python scripts/cre_stage695.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from prep_stage59 import N_MIN, deltas
from scales_stage59 import DEFAULT, SCALES, scale_table

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config

OUT = Path("results/stage69")
DATA = Path("data/interim/stage695")
_RE = "jppfs_cor_RetainedEarnings"


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    na = set(load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    d = deltas(pairs[pairs["group"] == "class 2 → 回復"]["edinetCode"].tolist(), names)
    j = names.index(_RE)
    old = json.loads(Path("data/interim/stage67/weights_c1.json").read_text(encoding="utf-8"))
    base = json.loads(Path("data/interim/stage60/meta_c1.json").read_text(encoding="utf-8"))
    rows = []
    for lab, nact_set in (("最終設定の規則(利益剰余金を操作可能として中央値に含める)", na - {_RE}),
                          ("参考: 中央値から利益剰余金を除く(ステージ60 の中央値のまま)", na)):
        nact = np.array([n in nact_set for n in names])
        t = scale_table(d, nact, SCALES[DEFAULT])
        fill = float(t.loc[(t["n"] >= N_MIN) & ~t["nact"], "c_raw"].median())
        re = t[t["j"] == j].set_index("dir")
        if nact[j]:
            # 利益剰余金を非操作のまま表に入れたので、規則を手で当てる
            cp = 1.0 / (float(re.loc["+", "s"]) + 0.001) if re.loc["+", "n"] >= N_MIN else fill
            cm = 1.0 / (float(re.loc["−", "s"]) + 0.001) if re.loc["−", "n"] >= N_MIN else fill
        else:
            cp, cm = float(re.loc["+", "c"]), float(re.loc["−", "c"])
        rows.append((lab, int(re.loc["+", "n"]), int(re.loc["−", "n"]), cp, cm, fill))
    lines = ["# ステージ69.5 B: class 1 用 c_RE の n < 5 の代替(検算)", "",
             "推定に使う群は class 2 → 回復の 15 社(class 1 の生成用、参考値)。s は変化した企業"
             "の |Δ| "
             "の中央値、c = 1 / (s + 0.001)。", "",
             "| 規則 | n+ | n− | c_RE⁺ | c_RE⁻ | 代替に使う中央値 "
             "|", "|---|---:|---:|---:|---:|---:|",
             f"| ステージ67 で使った値(反対の向きの s で代替) | {rows[0][1]} | {rows[0][2]} "
             f"| {old['c_plus'][j]:.2f} | {old['c_minus'][j]:.2f} | — |"]
    for lab, npos, nneg, cp, cm, fill in rows:
        lines.append(f"| {lab} | {npos} | {nneg} | {cp:.2f} | {cm:.2f} | {fill:.2f} |")
    lines.append("")
    cp_new, cm_new = np.array(base["c_plus"]), np.array(base["c_minus"])
    cp_new[j], cm_new[j] = rows[0][3], rows[0][4]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "weights_c1_rule.json").write_text(json.dumps(
        {"c_plus": cp_new.tolist(), "c_minus": cm_new.tolist(), "group": "class 2 → 回復",
         "note": "ステージ69.5: c_RE を scale_table の規則で推定"}), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cre_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
