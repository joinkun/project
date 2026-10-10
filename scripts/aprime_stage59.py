#!/usr/bin/env python
"""ステージ59 A': 払込資本の c−/c+・L2 距離の寄与率・払込資本比率の群平均・I_hard の候補(生成なし)。

A'.1 中央値の尺度(scales_stage59.py の既定)で、資本金・資本剰余金・合算した払込資本の c±。
正規化(Δx̃、確定した定義)と名目(Δx / A_2022、参考)、D_rec 57 社と 15 社(class 1 用、参考値)。
A'.2 等重み L2²(現行、非操作 0)の最近傍 j* での科目別の寄与率
ρ_a = mean_i (ỹ_{j*a} − x̃_ia)² / mean_i d(x_i, y_{j*})、と群間差 |mean(class 2) − mean(class 0)|。
A'.3 払込資本 / 総資産の群別の要約(class 2・class 1・class 0・2 つの最近傍の対応先)。
A'.4 D_rec 57 社で、科目ごとに減少した企業の割合と深さ(I_hard の候補、事前の規則で選ぶ)。

出力(``results/stage59/``): ``aprime.md``・``contribution.csv``・``decrease.csv``。

    uv run python scripts/aprime_stage59.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from changes_stage57 import deltas as nominal_deltas
from density_stage51 import load_tilde
from prep_stage59 import EPS0, OUTLIERS, deltas, l1_cost, l2_cost
from scales_stage59 import DEFAULT, SCALES
from scipy import stats

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config

_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
THR = 0.001  # 「減少した」= Δ < −0.001(総資産比、ステージ47・56 と同じ)
HARD_MAX = 0.05  # I_hard: 減少した企業の割合が、正規化・名目・15 社のすべてで 5% 以下


def paid_c(d: np.ndarray, names: tuple[str, ...]) -> list[tuple[str, int, int, float, float]]:
    """(名前, n+, n−, c+, c−)。合算は資本金 + 資本剰余金の Δ から同じ尺度で。"""
    fn = SCALES[DEFAULT]
    out = []
    cols = {"資本金": d[:, names.index(_CS)], "資本剰余金": d[:, names.index(_CSUR)]}
    cols["払込資本(合算)"] = cols["資本金"] + cols["資本剰余金"]
    for k, v in cols.items():
        pos, neg = v[v > 0], -v[v < 0]
        out.append((k, len(pos), len(neg), 1 / (fn(pos) + EPS0), 1 / (fn(neg) + EPS0)))
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage59"))
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    lab_of = [ja.get(n, n) for n in names]
    na = set(load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable)
    nact = np.array([n in na for n in names])
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    c57 = pairs[pairs["group"] == "class 1 → class 0"]["edinetCode"].tolist()
    c15 = pairs[pairs["group"] == "class 2 → 回復"]["edinetCode"].tolist()
    d57, d15 = deltas(c57, names), deltas(c15, names)
    n57 = nominal_deltas(c57, names).to_numpy()
    n15 = nominal_deltas(c15, names).to_numpy()

    # ---------------- A'.1 ----------------
    lines = ["# ステージ59 A': §B の前の積み残し", "",
             "## A'.1 払込資本の c−/c+(中央値の尺度、ε0 = 0.001)", "",
             "c = 1 / (|Δ| の中央値 + ε0)。c−/c+ < 1 = 減らすほうが安い。合算 = (資本金 + 資本"
             "剰余金) の Δ "
             "から同じ尺度で計算した参考値(生成のコストは科目別の c を使う)。", "",
             "| 群 | Δ の定義 | 科目 | n(増加) | n(減少) | c+ | c− | c−/c+ |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for grp, dd in (("D_rec 57 社(class 2 の重み)", {"正規化(確定)": d57, "名目(参考)": n57}),
                    ("15 社(class 1 の重み、参考値)", {"正規化(確定)": d15, "名目(参考)": n15})):
        for dname, d in dd.items():
            for k, npos, nneg, cp, cm in paid_c(d, names):
                lines.append(f"| {grp} | {dname} | {k} | {npos} | {nneg} | {cp:.1f} | {cm:.1f} "
                             f"| **{cm / cp:.2f}** |")

    # ---------------- A'.2 ----------------
    x, lab, docs, _keep = load_tilde(names)
    is_src = (lab == 2) & ~np.isin(docs, OUTLIERS)
    xs, xt = x[is_src], x[lab == 0]
    w_eq = (~nact).astype(float)
    j_l2 = l2_cost(xs, xt, w_eq).argmin(axis=1)
    sq = (xt[j_l2] - xs) ** 2 * w_eq
    rho = sq.mean(axis=0) / sq.sum(axis=1).mean()
    gap = np.abs(xs.mean(axis=0) - xt.mean(axis=0))
    gap_med = np.abs(np.median(xs, axis=0) - np.median(xt, axis=0))
    cdf = pd.DataFrame({"account": names, "ja": lab_of, "rho": rho, "gap_mean": gap,
                        "gap_median": gap_med, "nact": nact})
    act = cdf[~cdf["nact"]].sort_values("rho", ascending=False).reset_index(drop=True)
    act["rank"] = np.arange(1, len(act) + 1)
    act.to_csv(args.out / "contribution.csv", index=False)
    paid_rho = float(act.loc[act["account"].isin([_CS, _CSUR]), "rho"].sum())
    others = act[~act["account"].isin([_CS, _CSUR])]["rho"]
    lines += ["", "## A'.2 等重み L2² の最近傍での寄与率(class 2 55 社)", "",
              "ρ_a = mean_i (ỹ_{j*a} − x̃_ia)² / mean_i d(x_i, y_{j*})(操作できる 29 科目で合計 "
              "1)。"
              "群間差 = |class 2(55 社)の平均 − class 0 の平均|(参考に中央値の差も)。", "",
              "| 順位 | 科目 | 寄与率 ρ_a | 群間差(平均) | 群間差(中央値) |", "|---:|---|---:|--"
                                                            "-:|---:|"]
    for r in act.head(10).itertuples():
        lines.append(f"| {r.rank} | {r.ja} | {r.rho:.1%} | {r.gap_mean:.3f} | {r.gap_median:.3f} |")
    sp_mean = stats.spearmanr(act["rho"], act["gap_mean"])
    sp_med = stats.spearmanr(act["rho"], act["gap_median"])
    lines += ["", f"**払込資本(資本金 + 資本剰余金)の寄与率: {paid_rho:.1%}**。"
              f"他の 27 科目の平均 {others.mean():.2%} の {paid_rho / 2 / others.mean():.1f} 倍("
              f"1 科目あたり)、"
              f"他の 27 科目の合計 {others.sum():.1%}。",
              f"資本金 {act.loc[act['account'] == _CS, 'rho'].iloc[0]:.1%}"
              f"(順位 {int(act.loc[act['account'] == _CS, 'rank'].iloc[0])})、"
              f"資本剰余金 {act.loc[act['account'] == _CSUR, 'rho'].iloc[0]:.1%}"
              f"(順位 {int(act.loc[act['account'] == _CSUR, 'rank'].iloc[0])})。", "",
              f"寄与率と群間差の順位相関(29 科目、Spearman): 平均の差 {sp_mean.statistic:+.2f}"
              f"(p = {sp_mean.pvalue:.1g})、中央値の差 {sp_med.statistic:+.2f}(p ="
              f" {sp_med.pvalue:.1g})。", ""]

    # ---------------- A'.3 ----------------
    w = pd.read_csv(args.out / "weights_v2.csv")
    cp = w[w["dir"] == "+"].set_index("account").reindex(names)[f"c_{DEFAULT}"].to_numpy()
    cm = w[w["dir"] == "−"].set_index("account").reindex(names)[f"c_{DEFAULT}"].to_numpy()
    j_prop = l1_cost(xs, xt, cp, cm).argmin(axis=1)
    paid = x[:, names.index(_CS)] + x[:, names.index(_CSUR)]
    paid_t = paid[lab == 0]
    c1 = pd.read_csv("data/interim/stage34/class1_docs.csv", dtype={"docID": str})
    c1docs = set(c1.loc[c1["reason"].fillna("") == "", "docID"])  # 金融・ゴルフ場を除く(ステージ28)
    is_c1gen = np.array([d in c1docs for d in docs])
    groups = {"class 2(生成の始点、外れ値 2 社を除く)": paid[is_src],
              "class 1(断面、除外後)": paid[lab == 1],
              "class 1(生成の始点、金融・ゴルフ場を除く)": paid[is_c1gen],
              "class 0(断面、除外後)": paid_t,
              "等重み L2² の最近傍(55 社分、重複を含む)": paid_t[j_l2],
              "提案コスト(非対称 L1・中央値)の最近傍(55 社分)": paid_t[j_prop]}
    lines += ["## A'.3 払込資本 / 総資産 の群別の要約", "",
              "| 群 | n | 平均 | 中央値 | 標準偏差 |", "|---|---:|---:|---:|---:|"]
    for k, v in groups.items():
        lines.append(f"| {k} | {len(v):,} | {v.mean():.3f} | {np.median(v):.3f} |"
                     f" {v.std(ddof=1):.3f} |")
    lines += ["", f"異なる対応先: 等重み L2² {len(set(j_l2))}、提案 {len(set(j_prop))}。", ""]

    # ---------------- A'.4 ----------------
    rows = []
    for j, n in enumerate(names):
        r = {"account": n, "ja": lab_of[j], "nact": nact[j]}
        for tag, d in (("norm", d57[:, j]), ("nom", n57[:, j]), ("nom15", n15[:, j])):
            dec = d < -THR
            r[f"share_{tag}"] = dec.mean()
            r[f"n_{tag}"] = int(dec.sum())
            r[f"depth_{tag}"] = float(np.median(-d[dec])) if dec.any() else np.nan
        x0 = x[lab == 0][:, j]
        r["held_c0"] = float((x0 != 0).mean())
        rows.append(r)
    dec_df = pd.DataFrame(rows)
    dec_df["hard"] = (~dec_df["nact"] & (dec_df["share_norm"] <= HARD_MAX)
                      & (dec_df["share_nom"] <= HARD_MAX) & (dec_df["share_nom15"] <= HARD_MAX))
    dec_df.to_csv(args.out / "decrease.csv", index=False)
    lines += ["## A'.4 減少した企業の割合と深さ(I_hard の候補)", "",
              f"減少 = Δ < −{THR}(総資産比)。深さ = 減少した企業の |Δ| の中央値。"
              f"**規則(この表を見る前にスクリプトに書いた)**: 正規化(57 社)・名目(57 社)・名目(c"
              f"lass 2 → 回復の 15 社)の"
              f"すべてで減少した割合が {HARD_MAX:.0%} 以下の科目だけを I_hard に入れる"
              "(「実データを禁止しない」。15 社は class 2 の比較対象なので、そこで起きる減少も"
              "禁止しない)。"
              "保有率 = class 0 で値が 0 でない割合(参考)。", "",
              "| 科目 | 減少 正規化 57 社 | 深さ | 減少 名目 57 社 | 深さ | 減少 名目 15 社 | "
              "保有率(class 0) | I_hard |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in dec_df[~dec_df["nact"]].sort_values("share_nom").itertuples():
        lines.append(f"| {r.ja} | {r.share_norm:.0%}({r.n_norm}) | {r.depth_norm:.4f} "
                     f"| {r.share_nom:.0%}({r.n_nom}) | {r.depth_nom:.4f} "
                     f"| {r.share_nom15:.0%}({r.n_nom15}) | {r.held_c0:.0%} |"
                     f" {'○' if r.hard else ''} |")
    hard = dec_df[dec_df["hard"]]["ja"].tolist()
    lines += ["", f"**I_hard(規則による)**: {'、'.join(hard) or 'なし'}。", ""]
    (args.out / "aprime.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
