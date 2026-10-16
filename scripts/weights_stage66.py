#!/usr/bin/env python
"""ステージ66 A.1・B.3: 頻度加重片側中央値の重みと、c−/c+ の符号の検算(生成なし)。

s̄_a^± = (n_a^± / n_r) × median(|Δ| | 向き)、c_a^± = 1 / (s̄_a^± + ε0)、ε0 = 0.001。
n < 5 の方向と非操作 3 科目の扱いは、ステージ59 の中央値の版と同じ(反対の向き → 全科目の中央値、
非操作は 0)。Δ は正規化(各年の総資産で割った比の差、ステージ59 の確定)。名目(Δ / A0)は参考。
重みは class 2 の生成用に D_rec 57 社、class 1 の生成用に 15 社(参考値)。

出力: ``results/stage66/weights.md``・``weights.csv``、
``data/interim/stage66/weights_{c2,c1}.json``。

    uv run python scripts/weights_stage66.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from changes_stage57 import deltas as nominal_deltas
from prep_stage59 import EPS0, N_MIN, deltas

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config

OUT = Path("results/stage66")
DATA = Path("data/interim/stage66")


def table(d: np.ndarray, nact: np.ndarray, freq: bool) -> tuple[np.ndarray, np.ndarray,
                                                                  pd.DataFrame]:
    """(c+, c−, 内訳)。freq=True で頻度加重。"""
    nr = d.shape[0]
    rows = []
    for j in range(d.shape[1]):
        v = d[:, j]
        pos, neg = v[v > 0], -v[v < 0]
        mp = float(np.median(pos)) if len(pos) else float("nan")
        mn = float(np.median(neg)) if len(neg) else float("nan")
        sp = len(pos) / nr * mp if freq else mp
        sm = len(neg) / nr * mn if freq else mn
        rows.append({"n_plus": len(pos), "n_minus": len(neg), "med_plus": mp, "med_minus": mn,
                     "s_plus": sp, "s_minus": sm})
    t = pd.DataFrame(rows)
    ok_p, ok_m = (t["n_plus"] >= N_MIN).to_numpy(), (t["n_minus"] >= N_MIN).to_numpy()
    cp = 1.0 / (t["s_plus"].to_numpy() + EPS0)
    cm = 1.0 / (t["s_minus"].to_numpy() + EPS0)
    for j in range(len(t)):
        if not ok_p[j] and ok_m[j]:
            cp[j] = cm[j]
        elif ok_p[j] and not ok_m[j]:
            cm[j] = cp[j]
    reliable = np.concatenate([cp[ok_p & ~nact], cm[ok_m & ~nact]])
    fill = float(np.median(reliable))
    both_bad = ~ok_p & ~ok_m
    cp[both_bad], cm[both_bad] = fill, fill
    cp[nact], cm[nact] = 0.0, 0.0
    return cp, cm, t


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    na = set(load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable)
    nact = np.array([n in na for n in names])
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    groups = {"c2": "class 1 → class 0", "c1": "class 2 → 回復"}
    lines = ["# ステージ66 A.1: 頻度加重で c−/c+ の符号が反転するか", "",
             "c−/c+ > 1 = 減らすほうが高い(減らしにくい)。現行 = 中央値(ステージ59〜65)、頻度加"
             "重 = "
             "(n / n_r) × 中央値。ε0 = 0.001 を含めた値(式 (n+/n−)(s+/s−) は ε0 を無視した近似)。"
             "n < 5 の方向は代替する(ステージ59 の規則)。", ""]
    allrows = []
    for cls, grp in groups.items():
        codes = pairs[pairs["group"] == grp]["edinetCode"].tolist()
        for kind, d in (("正規化", deltas(codes, names)),
                        ("名目", nominal_deltas(codes, names).to_numpy())):
            cp0, cm0, t = table(d, nact, freq=False)
            cp1, cm1, _ = table(d, nact, freq=True)
            r0 = np.where(cp0 > 0, cm0 / np.where(cp0 > 0, cp0, 1), np.nan)
            r1 = np.where(cp1 > 0, cm1 / np.where(cp1 > 0, cp1, 1), np.nan)
            if kind == "正規化":
                (DATA / f"weights_{cls}.json").write_text(json.dumps(
                    {"c_plus": cp1.tolist(), "c_minus": cm1.tolist(), "group": grp,
                     "n_r": int(d.shape[0])}), encoding="utf-8")
            df = pd.DataFrame({"account": names, "ja": [ja.get(n, n) for n in names], **t,
                               "ratio_current": r0, "ratio_freq": r1, "cls": cls, "kind": kind})
            allrows.append(df)
            lines += [f"## {grp}({d.shape[0]} 社、{kind}、{'class 2' if cls == 'c2' else 'class 1'}"
                      " の生成に使う)", "",
                      "| 科目 | n+ | n− | 中央値+ | 中央値− | 現行 c−/c+ | 頻度加重 c−/c+ | 反転 |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"]
            for r in df.itertuples():
                if np.isnan(r.ratio_current):
                    continue
                flip = "○" if (r.ratio_current - 1) * (r.ratio_freq - 1) < 0 else ""
                bold = "**" if r.account in ("jppfs_cor_CapitalStock",
                                             "jppfs_cor_CapitalSurplus") else ""
                lines.append(f"| {bold}{r.ja}{bold} | {r.n_plus} | {r.n_minus} | {r.med_plus:.4f} "
                             f"| {r.med_minus:.4f} | {r.ratio_current:.2f} |"
                             f" {bold}{r.ratio_freq:.2f}"
                             f"{bold} | {flip} |")
            lines.append("")
    pd.concat(allrows).to_csv(OUT / "weights.csv", index=False)
    (OUT / "weights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
