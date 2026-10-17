#!/usr/bin/env python
"""ステージ67 A・B.1・C.4・D.2: 非操作科目の棚卸し、利益剰余金の移動コスト、実データの ΔRE / ΔNA、
始点の制約(生成なし)。

c_RE は他の科目と同じ規則(ステージ59 の中央値の尺度、ε0 = 0.001、n < 5 の代替)で、class 2 用は D_rec
57 社、class 1 用は 15 社(参考値)から推定する。他の科目の c± はステージ60 の値のまま。

出力: ``results/stage67/re_cost.md``・``realdata_re.md``・``start_feasibility.md``、
``data/interim/stage67/weights_{c2,c1}.json``。

    uv run python scripts/weights_stage67.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from changes_stage57 import deltas as nominal_deltas
from prep_stage59 import EPS0, N_MIN, OUTLIERS, deltas
from qp_stage60 import s47_final
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import build_linear_constraints, subtotal_members
from fsgen.accounting.projection_qp import is_feasible

OUT = Path("results/stage67")
DATA = Path("data/interim/stage67")
_RE = "jppfs_cor_RetainedEarnings"
SHOW = (_RE, "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus", "jppfs_cor_CashAndDeposits",
        "jppfs_cor_PropertyPlantAndEquipment", "valuation_difference_securities",
        "non_controlling_interests")


def scale(v: np.ndarray) -> tuple[int, int, float, float]:
    pos, neg = v[v > 0], -v[v < 0]
    return (len(pos), len(neg), float(np.median(pos)) if len(pos) else np.nan,
            float(np.median(neg)) if len(neg) else np.nan)


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    groups = {"c2": "class 1 → class 0", "c1": "class 2 → 回復"}
    lines = ["# ステージ67 B.1: 利益剰余金の移動コスト(他の科目と同じ規則)", "",
             "s± = 変化した企業の |Δ| の中央値(正規化 Δx̃、2022 → 2024)、c± = 1 / (s± + ε0)、ε0 "
             "= 0.001。"
             "n < 5 の方向は反対の向きの s で代替。比較のため他の科目(ステージ60 の値)を並べる。"
             "順位 = 操作できる 29 科目 + 利益剰余金の c の小さい順(1 = 最も動かしやすい)。"
             "凍結(…)は、凍結した科目に同じ規則を当てた値(B.2、生成には使わない)。", ""]
    for cls, grp in groups.items():
        codes = pairs[pairs["group"] == grp]["edinetCode"].tolist()
        d = deltas(codes, names)
        meta = json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(encoding="utf-8"))
        cp, cm = np.array(meta["c_plus"]), np.array(meta["c_minus"])
        j = names.index(_RE)
        npos, nneg, sp, sm = scale(d[:, j])
        if npos < N_MIN and nneg >= N_MIN:
            sp = sm
        elif nneg < N_MIN and npos >= N_MIN:
            sm = sp
        cp[j], cm[j] = 1.0 / (sp + EPS0), 1.0 / (sm + EPS0)
        (DATA / f"weights_{cls}.json").write_text(json.dumps(
            {"c_plus": cp.tolist(), "c_minus": cm.tolist(), "group": grp}), encoding="utf-8")
        active = cp > 0
        allc = np.concatenate([cp[active], cm[active]])
        lines += [f"## {grp}({len(codes)} 社、{'class 2' if cls == 'c2' else 'class 1'} の生成"
                  f"に使う)", "",
                  "| 科目 | n+ | n− | s+ | s− | c+ | c− | c−/c+ | c+ の順位 / c− の順位(60 中) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for n in SHOW:
            k = names.index(n)
            a, b, s1, s2 = scale(d[:, k])
            if cp[k] == 0:
                # B.2: 凍結した科目にも同じ規則で c を推定する(生成には使わない)
                if a < N_MIN and b >= N_MIN:
                    s1 = s2
                elif b < N_MIN and a >= N_MIN:
                    s2 = s1
                hp, hm = 1.0 / (s1 + EPS0), 1.0 / (s2 + EPS0)
                lines.append(f"| {ja.get(n, n)} | {a} | {b} | {s1:.4f} | {s2:.4f} | 凍結({hp:.1f}) "
                             f"| 凍結({hm:.1f}) | {hm / hp:.2f} | {int((allc < hp).sum()) + 1} / "
                             f"{int((allc < hm).sum()) + 1} |")
                continue
            rk_p = int((allc < cp[k]).sum()) + 1
            rk_m = int((allc < cm[k]).sum()) + 1
            lines.append(f"| {ja.get(n, n)} | {a} | {b} | {s1:.4f} | {s2:.4f} | {cp[k]:.1f} "
                         f"| {cm[k]:.1f} | {cm[k] / cp[k]:.2f} | {rk_p} / {rk_m} |")
        lines.append("")
    (OUT / "re_cost.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    # C.4: 実データの ΔRE / ΔNA(名目、総資産(2022)比)
    rl = ["# ステージ67 C.4: 実データの ΔRE / ΔNA(2022 → 2024、名目)", "",
          "ΔNA > 0 の企業について、純資産の増加に占める利益剰余金の増加の割合。1 なら利益だけで"
          "改善、"
          "< 1 なら増資など他の科目も寄与。", "",
          "| 群 | 社数 | ΔNA > 0 | ΔRE/ΔNA 中央値 | p25 | p75 | 払込資本の増加/ΔNA 中央値 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for grp in ("class 2 → 回復", "class 1 → class 0"):
        codes = pairs[pairs["group"] == grp]["edinetCode"].tolist()
        dn = nominal_deltas(codes, names).to_numpy()
        dna = dn @ members["jppfs_cor_NetAssets"]
        dre = dn[:, names.index(_RE)]
        dpaid = dn[:, names.index("jppfs_cor_CapitalStock")] + dn[:, names.index(
            "jppfs_cor_CapitalSurplus")]
        pos = dna > 1e-6
        r = dre[pos] / dna[pos]
        rl.append(f"| {grp} | {len(codes)} | {int(pos.sum())} | {np.median(r):.2f} "
                  f"| {np.percentile(r, 25):.2f} | {np.percentile(r, 75):.2f} "
                  f"| {np.median(dpaid[pos] / dna[pos]):.2f} |")
    (OUT / "realdata_re.md").write_text("\n".join(rl) + "\n", encoding="utf-8")
    print("\n".join(rl))
    # D.2: 始点の制約
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
    srcs = {"class 2(55 社)": (lab == 2) & ~np.isin(docs_a, OUTLIERS),
            "class 1(132 社)": np.isin(docs_a, c1.loc[c1["reason"].fillna("") == "",
                                                     "docID"].to_numpy())}
    cfg = dataclasses.replace(s47_final(), cash_floor=0.11732)
    fl = ["# ステージ67 D.2: 始点 x_i が実行可能集合 F_i に入るか(定理の但し書き)", "",
          "第 1 層 = 貸借一致と符号(`is_feasible`、許容 1e-6)。第 2 層 = 最終設定の追加の線形制"
          "約を始点で評価。",
          "", "| class | 貸借一致の残差 > 1e-6 | 残差の最大(総資産比) | 符号の違反 | 第 2 層を"
              "満たさない | 理由 |",
          "|---|---:|---:|---:|---:|---|"]
    cmat = build_constraint_matrix(spec, include_subtotals=False)
    for lab_, m in srcs.items():
        xs = x[m]
        resid = np.abs(xs @ cmat).max(axis=1)
        sign_bad = 0
        l2_bad: list[str] = []
        for xi in xs:
            ok, v = is_feasible(xi, spec, tol=1e-6)
            if any("balance" not in str(t) for t in v):
                sign_bad += 1
            ex = build_linear_constraints(spec, names, xi, cfg)
            l2_bad += [c.name for c in ex if c.a @ xi < c.rhs - 1e-9]
        reasons = pd.Series(l2_bad).value_counts().to_dict() if l2_bad else {}
        fl.append(f"| {lab_} | {int((resid > 1e-6).sum())} | {resid.max():.1e} | {sign_bad} "
                  f"| {len(l2_bad)} | {reasons or '—'} |")
    (OUT / "start_feasibility.md").write_text("\n".join(fl) + "\n", encoding="utf-8")
    print("\n".join(fl))


if __name__ == "__main__":
    main()
