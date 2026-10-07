#!/usr/bin/env python
"""ステージ58 A: 群内分散・全体分散と、それによる重み(集計のみ、生成なし)。

規則は ``results/stage58/criteria.md``(集計の前に記録)。案 2 = 再建群 72 社の 1 年の Δx̃ の標準偏差
(主)、案 2r = その 1.4826 × MAD、案 1 = class 内のプールした分散、案 3 = 業種内のプールした分散、
全体 = 断面の標準偏差。w_a = (σ_現金 / σ_a)²、上限 100、非操作 3 科目は 0。

出力: ``results/stage58/variance.md``・``variance.csv``、``conf/accounts/costs_within_sd.yaml``・
``costs_within_mad.yaml``・``costs_within_class.yaml``・``costs_total.yaml``。

    uv run python scripts/variance_stage58.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from density_stage51 import load_tilde
from scipy import stats

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config
from fsgen.models.industry import load_industry_config

_PANEL = Path("data/interim/stage23/panel/panel.csv")
CASH = "jppfs_cor_CashAndDeposits"
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
CAP = 100.0


def pooled_within(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """群ごとの分散を社数で重み付けて平均した値の平方根(群内 1 社の群は除く)。"""
    num = np.zeros(x.shape[1])
    den = 0
    for g in np.unique(groups):
        xg = x[groups == g]
        if len(xg) < 2:
            continue
        num += len(xg) * xg.var(axis=0)
        den += len(xg)
    out: np.ndarray = np.sqrt(num / den)
    return out


def to_weights(sigma: np.ndarray, icash: int, nact: np.ndarray) -> tuple[np.ndarray, int]:
    w = np.where(sigma > 0, (sigma[icash] / np.where(sigma > 0, sigma, 1.0)) ** 2, CAP)
    capped = int(((w >= CAP) & ~nact).sum())
    w = np.minimum(w, CAP)
    w[nact] = 0.0
    return w, capped


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage58"))
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    cc = load_cost_config(Path("conf/accounts/costs_equal.yaml"))
    nact = np.array([n in cc.non_actionable for n in names])
    icash = names.index(CASH)
    # 案 2: 再建群 72 社の 1 年の Δx̃
    codes = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"]
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    p = p[p["edinetCode"].isin(codes) & p["t"].isin([2023, 2024])]
    cur = (p[[f"cur_{n}" for n in names]].fillna(0).to_numpy()
           / p["cur_total_assets"].to_numpy()[:, None])
    prev = (p[[f"prev_{n}" for n in names]].fillna(0).to_numpy()
            / p["prev_total_assets"].to_numpy()[:, None])
    dx = cur - prev
    s2 = dx.std(axis=0, ddof=1)
    s2r = 1.4826 * np.median(np.abs(dx - np.median(dx, axis=0)), axis=0)
    # 断面: 案 1・3・全体
    x, label, docs, _keep = load_tilde(names)
    s1 = pooled_within(x, label)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    ind = np.array(load_industry_config(Path("conf/accounts/industry.yaml")).labels(
        [code_of.get(d, "") for d in docs]))
    known = ind != "不明"
    s3 = pooled_within(x[known], ind[known])
    st = x.std(axis=0, ddof=1)
    sig = {"2": s2, "2r": s2r, "1": s1, "3": s3, "total": st}
    ws, caps = {}, {}
    for k, s in sig.items():
        ws[k], caps[k] = to_weights(s, icash, nact)
    w57 = pd.read_csv("results/stage57/weights.csv").set_index("account")["w"].reindex(names)
    df = pd.DataFrame({"account": names, "ja": [ja.get(n, n) for n in names],
                       **{f"sigma_{k}": v for k, v in sig.items()},
                       "ratio_1": s1 / st, "ratio_3": s3 / st,
                       **{f"w_{k}": v for k, v in ws.items()}, "w_stage57": w57.to_numpy()})
    df.to_csv(args.out / "variance.csv", index=False)
    paid = x[:, names.index(_CS)] + x[:, names.index(_CSUR)]
    lines = ["# ステージ58 A: 群内分散と全体分散", "",
             f"案 2・2r: 再建群 72 社の 1 年の Δx̃({len(dx)} 組、2022→2023・2023→2024)。"
             f"案 1・3・全体: 生成に使う断面(除外後の class 0 {int((label == 0).sum()):,}、class 1 "
             f"{int((label == 1).sum())}、class 2 {int((label == 2).sum())})。案 3 は 33 業種(不明 "
             f"{int((~known).sum())} 社を除く)。比 = σ_within / σ_total(小さいほど群を分けている"
             f")。", "",
             "**払込資本 / 総資産(断面、中央値)**: class 0 "
             f"{np.median(paid[label == 0]):.3f}、class 1 {np.median(paid[label == 1]):.3f}、"
             f"class 2 {np.median(paid[label == 2]):.3f}(スペックの値は class 2 1.14、class 0 0."
             f"13)。", "",
             "| 科目 | σ 案 2 | σ 案 2r | σ 案 1(class 内) | σ 案 3(業種内) | σ 全体 | 比 案 1 |"
             " 比 案 3 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in df.itertuples():
        lines.append(f"| {r.ja} | {r.sigma_2:.4f} | {r.sigma_2r:.4f} | {r.sigma_1:.4f} |"
                     f" {r.sigma_3:.4f} "
                     f"| {r.sigma_total:.4f} | {r.ratio_1:.3f} | {r.ratio_3:.3f} |")
    act = df[~nact]
    small = act.nsmallest(5, "ratio_1").itertuples()
    lines += ["", "比 案 1 の小さい科目(群を分けている、上位 5): "
              + "、".join(f"{r.ja}({r.ratio_1:.3f})" for r in small),
              "", "## 重み(w = (σ_現金 / σ)²、上限 100、現金 = 1)", "",
              "| 科目 | 案 2 | 案 2r | 案 1 | 案 3 | 全体 | ステージ57 |", "|---|---:|---:|---:|"
                                                               "---:|---:|---:|"]
    for r in act.sort_values("w_2").itertuples():
        lines.append(f"| {r.ja} | {r.w_2:.2f} | {r.w_2r:.2f} | {r.w_1:.2f} | {r.w_3:.2f} |"
                     f" {r.w_total:.2f} "
                     f"| {r.w_stage57:.2f} |")
    lines += ["", "上限に当たった科目の数(29 科目中): "
              + "、".join(f"案 {k} {v}" for k, v in caps.items()), "",
              "## 重みの順位相関(29 科目、Spearman)", "",
              "| | 案 2 | 案 2r | 案 1 | 案 3 | 全体 | ステージ57 |", "|---|---:|---:|---:|---:|"
                                                            "---:|---:|"]
    cols = ["w_2", "w_2r", "w_1", "w_3", "w_total", "w_stage57"]
    labs = ["案 2", "案 2r", "案 1", "案 3", "全体", "ステージ57"]
    for c, lab in zip(cols, labs, strict=True):
        lines.append(f"| {lab} | " + " | ".join(f"{stats.spearmanr(act[c], act[d]).statistic:+.2f}"
                                               for d in cols) + " |")
    ics, icsur = names.index(_CS), names.index(_CSUR)
    lines += ["", "**A.4 払込資本の重み(現金 = 1)**: "
              f"資本金 案 2 {ws['2'][ics]:.2f}・ステージ57 {w57.iloc[ics]:.2f}、"
              f"資本剰余金 案 2 {ws['2'][icsur]:.2f}・ステージ57 {w57.iloc[icsur]:.2f}"
              f"(案 1: 資本金 {ws['1'][ics]:.2f}、資本剰余金 {ws['1'][icsur]:.2f})。", ""]
    (args.out / "variance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    base = yaml.safe_load(Path("conf/accounts/costs_equal.yaml").read_text(encoding="utf-8"))
    for key, fname in (("2", "costs_within_sd"), ("2r", "costs_within_mad"),
                       ("1", "costs_within_class"), ("total", "costs_total")):
        doc = {**base, "name": fname, "basis_costs": []}
        for b in base["basis_costs"]:
            j = names.index(b["name"])
            wv = float(ws[key][j]) if not nact[j] else 1.0
            doc["basis_costs"].append({**b, "cost_increase": wv, "cost_decrease": wv,
                                       "source": f"ステージ58: σ(案 {key})による標準化"})
        Path(f"conf/accounts/{fname}.yaml").write_text(
            "# ステージ58: scripts/variance_stage58.py が生成(手で編集しない)。非操作科目は near"
            "est_na0 で 0。\n"
            + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
