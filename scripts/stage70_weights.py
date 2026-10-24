#!/usr/bin/env python
"""ステージ70 A.1: 72 社の統合プール・ε0 なし(c = 1/s)のコスト c±(numpy・pandas だけのプロセス)。

規則は ``results/stage70/criteria.md`` §1。残余科目は c = min(1/s, 10³)、n < 5 の方向は
「n ≥ 5 の全方向の c の中央値」で代替、評価差額金だけ c = 0。始点と同じ EDINET コードの
再建企業は、その始点の推定から除く。

出力: ``data/interim/stage70/weights.json``、``results/stage70/precheck_a1.md``。

    uv run python scripts/stage70_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from prep_stage59 import N_MIN, deltas
from targets_stage68 import sources
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

OUT = Path("results/stage70")
DATA = Path("data/interim/stage70")
FROZEN = ("valuation_difference_securities",)
CLIP = 1e3
GROUPS = {"72": ("class 1 → class 0", "class 2 → 回復"), "57": ("class 1 → class 0",),
          "15": ("class 2 → 回復",)}
SHOW = ("jppfs_cor_RetainedEarnings", "non_controlling_interests", "jppfs_cor_CapitalStock",
        "jppfs_cor_CapitalSurplus", "jppfs_cor_CashAndDeposits",
        "jppfs_cor_PropertyPlantAndEquipment", "short_term_loans", "long_term_loans",
        "jppfs_cor_IncomeTaxesPayable", "accounts_payable_other", "__OTHER__jppfs_cor_Assets")


def costs(d: np.ndarray, names: tuple[str, ...], eps0: float = 0.0) -> dict[str, np.ndarray]:
    """科目 × 方向の n・s・c(criteria §1)。eps0 > 0 は従来の比較用。"""
    k = len(names)
    n = np.zeros((2, k), dtype=int)
    s = np.full((2, k), np.nan)
    for j in range(k):
        for r, v in enumerate((d[:, j][d[:, j] > 0], -d[:, j][d[:, j] < 0])):
            n[r, j] = len(v)
            if len(v):
                s[r, j] = float(np.median(v))
    frozen = np.array([nm in FROZEN for nm in names])
    resid = np.array([nm.startswith("__OTHER__") for nm in names])
    with np.errstate(divide="ignore"):
        c = 1.0 / (s + eps0)
    c = np.where(resid[None, :] & np.isfinite(c), np.minimum(c, CLIP), c)
    ok = (n >= N_MIN) & ~frozen[None, :]
    fill = float(np.median(c[ok]))
    c = np.where(ok, c, fill)
    c[:, frozen] = 0.0
    return {"n": n, "s": s, "c": c, "fill": np.array(fill), "n_sub": np.array(
        int(((n < N_MIN) & ~frozen[None, :]).sum()))}


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    pool = {g: pairs[pairs["group"].isin(gs)]["edinetCode"].tolist() for g, gs in GROUPS.items()}
    res = {g: costs(deltas(c, names), names) for g, c in pool.items()}
    old = {g: costs(deltas(c, names), names, eps0=0.001) for g, c in pool.items()}
    # 始点と同じコードの再建企業を除いた推定
    x, lab, docs, _ = load_tilde(names)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    docs_a = np.array(docs)
    loo: dict[str, dict[str, list[float]]] = {}
    for cls in ("c2", "c1"):
        for d_ in docs_a[sources(cls, lab, docs_a)]:
            code = code_of.get(str(d_), "")
            for g in ("72", "57"):
                if code in pool[g]:
                    r = costs(deltas([c for c in pool[g] if c != code], names), names)
                    loo[f"{g}:{code}"] = {"c_plus": r["c"][0].tolist(),
                                          "c_minus": r["c"][1].tolist()}
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / "weights.json").write_text(json.dumps(
        {"names": list(names),
         "pools": {g: {"c_plus": r["c"][0].tolist(), "c_minus": r["c"][1].tolist(),
                       "codes": pool[g]} for g, r in res.items()},
         "loo": loo}, ensure_ascii=False), encoding="utf-8")
    lines = ["# ステージ70 A.1: 新しいコスト c±(ε0 なし、c = 1/s)", "",
             "s± = 変化した企業の |Δ| の中央値(構成の差、2022 → 2024)。c± = 1/s±。残余科目は"
             f" min(1/s, {CLIP:g})。n < 5 の方向は n ≥ 5 の全方向の c の中央値で代替。"
             "評価差額金だけ 0。"
             "「従来」はステージ60 と同じ ε0 = 0.001 の値(同じプール)。", "",
             "## n < 5 で代替が必要な方向の数(評価差額金を除く 31 科目 × 2 方向 = 62 方向)", "",
             "| プール | 社数 | n < 5 の方向 | 代替に使う中央値(新 / 従来) |",
             "|---|---:|---:|---:|"]
    for g in ("15", "57", "72"):
        lines.append(f"| {g} 社 | {len(pool[g])} | {int(res[g]['n_sub'])} "
                     f"| {float(res[g]['fill']):.1f} / {float(old[g]['fill']):.1f} |")
    lines += ["", "(従来の生成は、class 2 用に 57 社、class 1 用に 15 社を使っていた。)", ""]
    for g in ("72", "57"):
        r, o = res[g], old[g]
        lines += [f"## {g} 社のプール(全科目)", "",
                  "| 科目 | n+ | n− | s+ | s− | c+ | c− | c−/c+ | 従来 c+ | 従来 c− |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for j, nm in enumerate(names):
            mark = "**" if nm in SHOW else ""
            sub = "".join("*" if r["n"][i, j] < N_MIN and nm not in FROZEN else ""
                          for i in (0, 1))
            cp, cm = r["c"][0, j], r["c"][1, j]
            lines.append(f"| {mark}{ja.get(nm, nm)}{mark}{sub} | {r['n'][0, j]} | {r['n'][1, j]} "
                         f"| {r['s'][0, j]:.4f} | {r['s'][1, j]:.4f} | {cp:.1f} | {cm:.1f} "
                         f"| {(cm / cp if cp > 0 else float('nan')):.2f} | {o['c'][0, j]:.1f} "
                         f"| {o['c'][1, j]:.1f} |")
        lines += ["", "(* は n < 5 で代替した方向の数。太字は主な科目。)", ""]
    lines += ["## 自分自身を除いた推定", "",
              f"始点と同じ EDINET コードの再建企業: {', '.join(sorted(loo)) or 'なし'}。"
              "この始点だけ、その企業を除いたプールの c± を使う。", ""]
    (OUT / "precheck_a1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))
    for g in ("72",):
        print(f"[{g}]", {nm: (round(res[g]['c'][0, names.index(nm)], 1),
                               round(res[g]['c'][1, names.index(nm)], 1)) for nm in SHOW})


if __name__ == "__main__":
    main()
