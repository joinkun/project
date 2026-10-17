#!/usr/bin/env python
"""ステージ68 B: 遷移ベースの目標 T1・T2 を作り、QP の前に目標そのものの向きを測る。

numpy・pandas だけのプロセス(POT・cvxpy・sklearn・torch を読み込まない)。規則は
``results/stage68/criteria.md``。

- 再建企業の集合 D_rec = ``results/stage55/pairs.csv`` の 72 社(class 1 → class 0 57 社、
  class 2 → 回復 15 社)。2022 年度(始点)と 2024 年度(終点)の BS は ``panel.csv``。
- 対応づけ: 始点 x_i に、非対称 L1(ステージ60 の c±、class ごと)で最も近い再建企業 k*。
  T1 は再建後の構成 x_{k,2024} / A_{k,2024} の中から、T2 は再建前の構成 x_{k,2022} / A_{k,2022}
  の中から選ぶ。同じ EDINET コードの企業は候補から除く(自分自身を除く)。
- T1: τ_i = x_{k*,2024} / A_{k*,2024}(再建後の実在企業の構成)。
- T2: τ_i = x_i + (x_{k*,2024} − x_{k*,2022}) / A_{k*,2022}(実際の変化量を始点に足す。
  総資産の変化も含む)。

出力: ``data/interim/stage68/targets_{c2,c1}.npz``、``results/stage68/targets.md``。

    uv run python scripts/targets_stage68.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from prep_stage59 import OUTLIERS, l1_cost
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

OUT = Path("results/stage68")
DATA = Path("data/interim/stage68")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
_RE = "jppfs_cor_RetainedEarnings"
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
REAL_GROUP = {"c2": "class 2 → 回復", "c1": "class 1 → class 0"}
T0_KEY = "semi_t0.01_e0.001__argmax"


def sources(cls: str, lab: np.ndarray, docs: np.ndarray) -> np.ndarray:
    if cls == "c2":
        out: np.ndarray = (lab == 2) & ~np.isin(docs, OUTLIERS)
        return out
    c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
    out = np.isin(docs, c1.loc[c1["reason"].fillna("") == "", "docID"].to_numpy())
    return out


def pool(names: tuple[str, ...]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """D_rec 72 社の (コード, 2022 の構成, 2024 の構成, 2022 の総資産比の変化量)。"""
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    codes = pairs["edinetCode"].tolist()
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    b = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
    cols = [f"cur_{n}" for n in names]
    va, vb = a[cols].fillna(0).to_numpy(), b[cols].fillna(0).to_numpy()
    ta, tb = a["cur_total_assets"].to_numpy(), b["cur_total_assets"].to_numpy()
    return codes, va / ta[:, None], vb / tb[:, None], (vb - va) / ta[:, None]


def paid_delta(d: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    out: np.ndarray = d[:, names.index(_PAID[0])] + d[:, names.index(_PAID[1])]
    return out


def props(x: np.ndarray, tau: np.ndarray, names: tuple[str, ...], na: np.ndarray
          ) -> dict[str, float]:
    d = tau - x
    dpaid = paid_delta(d, names)
    dna = d @ na
    pos = dna > 1e-6
    r = d[pos, names.index(_RE)] / dna[pos]
    return {"paid_lower": float((dpaid < 0).mean()), "re_share": float(np.median(r)),
            "re_lt1": float((r < 0.99).mean()), "paid_up0": float((dpaid > 0).mean()),
            "paid_up": float((dpaid > 0.005).mean()), "n": float(len(x)),
            "n_pos": float(pos.sum())}


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    na, assets = members["jppfs_cor_NetAssets"], members["jppfs_cor_Assets"]
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    x, lab, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    codes, comp22, comp24, dnom = pool(names)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    grp = dict(zip(pairs["edinetCode"], pairs["group"], strict=True))
    xt = x[lab == 0]
    lines = ["# ステージ68 B: 遷移ベースの目標(QP の前)", "",
             "D_rec = 再建企業 72 社(class 1 → class 0 57 社 + class 2 → 回復 15 社)。対応づけは"
             "非対称 L1(ステージ60 の c±)の最近傍、同じ EDINET コードは除く。", ""]
    ov = ["## B.2 始点と D_rec の企業の重複", "",
          "| class | 始点 | D_rec と同じ EDINET コード | うち 1 → 0 群 | うち 2 → 回復群 "
          "| 該当企業 |",
          "|---|---:|---:|---:|---:|---|"]
    tab = ["## B.3 目標そのものの向き(QP の前)", "",
           "差 = τ − x(総資産比)。払込資本↑ は差 > 0.005(生成を 5 年とみなしたときの年率 0.001)、"
           "括弧内は差 > 0 の割合。実データは再建群の 2022 → 2024 の名目の変化(2 年、"
           "払込資本↑ は年率 > 0.001)。", "",
           "| class | 目標設計 | τ の払込資本 < 始点 | τ での ΔRE/ΔNA 中央値 | ΔRE/ΔNA < 0.99 | "
           "τ での払込資本↑ |",
           "|---|---|---:|---:|---:|---:|"]
    extra = ["## T2 の目標の総資産と符号", "",
             "| class | τ の総資産 p5 | 中央値 | p95 | 最大 "
             "| 負の科目を含む τ(資産・負債の本来正の科目) |",
             "|---|---:|---:|---:|---:|---:|"]
    picks = ["## 対応先の重複(どの再建企業が選ばれたか)", "",
             "| class | 目標設計 | 異なる対応先の数 | 最多の対応先の選ばれた回数 | 1 → 0 群から "
             "| 2 → 回復群から |",
             "|---|---|---:|---:|---:|---:|"]
    paid = pd.read_csv("results/stage56/paidin.csv", dtype={"edinetCode": str})
    for cls in ("c2", "c1"):
        m = sources(cls, lab, docs_a)
        xs, src = x[m], [str(d) for d in docs_a[m]]
        src_codes = [code_of.get(d, "") for d in src]
        same = [c for c in src_codes if c in grp]
        ov.append(f"| {cls} | {len(src)} | {len(same)} | "
                  f"{sum(grp[c] == 'class 1 → class 0' for c in same)} | "
                  f"{sum(grp[c] == 'class 2 → 回復' for c in same)} | {', '.join(same) or '—'} |")
        meta = json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(encoding="utf-8"))
        cp, cm = np.array(meta["c_plus"]), np.array(meta["c_minus"])
        self_mask = np.array([[sc == pc for pc in codes] for sc in src_codes])
        k = {}
        for key, cand in (("t1", comp24), ("t2", comp22)):
            c = l1_cost(xs, cand, cp, cm)
            c[self_mask] = np.inf
            k[key] = c.argmin(axis=1)
        tau1 = comp24[k["t1"]]
        tau2 = xs + dnom[k["t2"]]
        z = np.load(f"data/interim/stage64/targets_{cls}.npz", allow_pickle=True)
        if [str(v) for v in z["source_docs"]] != src:
            raise ValueError("T0 の始点の並びが一致しない")
        tau0 = xt[z[T0_KEY].astype(int)]
        np.savez(DATA / f"targets_{cls}.npz", source_docs=np.array(src), t0=tau0, t1=tau1, t2=tau2,
                 k_t1=np.array([codes[i] for i in k["t1"]]),
                 k_t2=np.array([codes[i] for i in k["t2"]]))
        for lab_t, tau in (("T0(現行 μ1)", tau0), ("T1(再建終点)", tau1), ("T2(遷移加算)", tau2)):
            p_ = props(xs, tau, names, na)
            tab.append(f"| {cls} | {lab_t} | {p_['paid_lower']:.0%} | {p_['re_share']:.2f} "
                       f"| {p_['re_lt1']:.0%} | {p_['paid_up']:.0%}({p_['paid_up0']:.0%}) |")
        rc = pairs[pairs["group"] == REAL_GROUP[cls]]["edinetCode"].tolist()
        ri = [codes.index(c) for c in rc]
        dr = dnom[ri]
        dpaid = paid_delta(dr, names)
        dna = dr @ na
        pos = dna > 1e-6
        r = dr[pos, names.index(_RE)] / dna[pos]
        pp = paid[paid["group"] == ("再建群(class 2 → 回復)" if cls == "c2"
                                    else "再建群(class 1 → class 0)")]["annual_d_paid"]
        tab.append(f"| {cls} | 実データ({REAL_GROUP[cls]}、{len(rc)} 社) "
                   f"| {(dpaid < 0).mean():.0%} "
                   f"| {np.median(r):.2f} | {(r < 0.99).mean():.0%} "
                   f"| {(pp > 0.001).mean():.0%}({(dpaid > 0).mean():.0%}) |")
        ta2 = tau2 @ assets
        pos_accts = [j for j, n in enumerate(names) if not n.startswith("__OTHER__")
                     and n not in (_RE, "jppfs_cor_TreasuryStock",
                                   "valuation_difference_securities", "non_controlling_interests")
                     and not n.startswith("allowance")]
        neg = (tau2[:, pos_accts] < -1e-9).any(axis=1)
        extra.append(f"| {cls} | {np.percentile(ta2, 5):.3f} | {np.median(ta2):.3f} "
                     f"| {np.percentile(ta2, 95):.3f} | {ta2.max():.3f} "
                     f"| {int(neg.sum())} / {len(xs)} |")
        for key, lab_t in (("t1", "T1"), ("t2", "T2")):
            sel = pd.Series([codes[i] for i in k[key]])
            picks.append(f"| {cls} | {lab_t} | {sel.nunique()} | {sel.value_counts().iloc[0]} "
                         f"| {int(sel.map(grp).eq('class 1 → class 0').sum())} "
                         f"| {int(sel.map(grp).eq('class 2 → 回復').sum())} |")
    text = [*lines, *ov, "", *tab, "", *extra, "", *picks, ""]
    (OUT / "targets.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
