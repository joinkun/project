#!/usr/bin/env python
"""ステージ55 A: 実際に再建した企業の遷移の対(正例)と、負例の構築(学習なし)。

正例: 再建群 72 社(class 2 → 回復 15 社、class 1 → class 0 57 社)の (x_2022, x_2024)。
状態は各年度末の総資産で割った 32 科目(推定用パネル、2022 年度と 2024 年度の書類の当期の値)。
期間は 2 年のまま(A.2 の案 1。生成は 5 年)。

負例: 各始点 x_2022 に対し、生成の目標の候補と同じ class 0(除外後 3,768 書類)から、
- 無作為: 乱数(seed 0)で M = 10 社
- 近い企業(hard negative): 現行の等重み距離(非操作 3 科目を除く 29 科目の二乗距離)で近い順に 10 社
どちらも、その企業自身の書類(同じ EDINET コード)は除く。

出力(``results/stage55/``): ``pairs.csv``、``negatives.csv``、``data_a.md``。

    uv run python scripts/pairs_stage55.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config

_PANEL = Path("data/interim/stage23/panel/panel.csv")
M = 10
GROUPS = {"再建群(class 2 → 回復)": "class 2 → 回復",
          "再建群(class 1 → class 0)": "class 1 → class 0"}


def reproduce_groups() -> dict[str, set[str]]:
    """ステージ30(class 2 → 回復)とステージ34・47(class 1 → class 0)の抽出を同じ規則でやり直す。"""
    from evaluate_stage34 import special_reason
    from recovery_stage30 import changes

    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    y = rf[(rf["reason"].fillna("") == "") & rf["in_panel"].astype(bool) & (rf["group"] == "Y")]
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wl = fy.pivot(index="edinetCode", columns="year", values="label")
    c1 = wl[wl[2022] == 1].copy()
    c1["reason"] = special_reason(fy, c1.index.tolist())
    c1 = c1.join(changes(_PANEL.parent, c1.index.tolist(), names), how="left")
    c1 = c1[(c1["reason"] == "") & c1["n_years"].notna()]
    rec1 = c1[wl.loc[c1.index, 2024] == 0]
    return {"class 2 → 回復": set(y["edinetCode"]), "class 1 → class 0": set(rec1.index)}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage55"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    grp = pd.read_csv("results/stage51/asset_change.csv", dtype={"edinetCode": str})
    grp = grp[grp["group"].isin(GROUPS)].assign(g=lambda d: d["group"].map(GROUPS))
    repro = reproduce_groups()
    checks = {g: (set(grp[grp["g"] == g]["edinetCode"]), repro[g]) for g in repro}
    pan = pd.read_csv(_PANEL, dtype={"edinetCode": str, "docID": str})
    a = pan[pan["t"] == 2022].set_index("edinetCode")
    b = pan[pan["t"] == 2024].set_index("edinetCode")
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    name_of = fy.sort_values("year").drop_duplicates("edinetCode", keep="last").set_index(
        "edinetCode")["filerName"]
    rows = []
    for _, r in grp.sort_values(["g", "edinetCode"]).iterrows():
        c = r["edinetCode"]
        xa = a.loc[c, [f"cur_{n}" for n in names]].fillna(0.0).to_numpy(float) / a.loc[c, "cur_t"
                                                                                          "otal_"
                                                                                          "assets"]
        xb = b.loc[c, [f"cur_{n}" for n in names]].fillna(0.0).to_numpy(float) / b.loc[c, "cur_t"
                                                                                          "otal_"
                                                                                          "assets"]
        nm = name_of.get(c, "")
        rows.append({"edinetCode": c, "docID_t": a.loc[c, "docID"], "docID_t2": b.loc[c, "docID"],
                     "filerName": nm, "group": r["g"],
                     **{f"xt_{n}": v for n, v in zip(names, xa, strict=True)},
                     **{f"xt2_{n}": v for n, v in zip(names, xb, strict=True)}})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(args.out / "pairs.csv", index=False)
    xt = pairs[[f"xt_{n}" for n in names]].to_numpy()
    xt2 = pairs[[f"xt2_{n}" for n in names]].to_numpy()

    # 負例
    x, label, docs, _keep = load_tilde(names)
    x0 = x[label == 0]
    docs0 = np.array(docs)[label == 0]
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    codes0 = code_of.reindex(docs0).to_numpy()
    cc = load_cost_config(Path("conf/accounts/costs_equal.yaml"))
    mask = np.array([0.0 if n in cc.non_actionable else 1.0 for n in names])

    def dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        out: np.ndarray = (((q - p) * mask) ** 2).sum(axis=-1)
        return out

    rng = np.random.default_rng(0)
    negs = []
    own_in_pool = 0
    for i, c in enumerate(pairs["edinetCode"]):
        ok = codes0 != c
        own_in_pool += int((~ok).sum() > 0)
        cand = np.where(ok)[0]
        d_all = dist(xt[i], x0[cand])
        for kind, pick in (("無作為", rng.choice(cand, M, replace=False)),
                           ("近い企業", cand[np.argsort(d_all)[:M]])):
            for j in pick:
                negs.append({"edinetCode": c, "kind": kind, "neg_docID": docs0[j],
                             "neg_edinetCode": codes0[j], "d_equal": float(dist(xt[i], x0[j]))})
    neg = pd.DataFrame(negs)
    neg.to_csv(args.out / "negatives.csv", index=False)

    # 記述: 現行の等重み距離での正例と負例(参考。学習・評価の規則はまだない)
    d_pos = dist(xt, xt2)
    lines = ["# ステージ55 A: 正例(実際の遷移)と負例の構築", "",
             "## A.1 正例(`pairs.csv`)", "",
             "- 状態: 各年度末の総資産で割った 32 科目。始点 = 2022 年度の書類の当期、"
             "終点 = 2024 年度の書類の当期(推定用パネル)。",
             f"- 社数: class 2 → 回復 {int((pairs['group'] == 'class 2 → 回復').sum())}、"
             f"class 1 → class 0 {int((pairs['group'] == 'class 1 → class 0').sum())}、"
             f"合計 {len(pairs)}。", "",
             "**再現の確認**(ステージ30・34・47 の規則でやり直した集合との一致):", ""]
    for g, (have, rep) in checks.items():
        same = "完全一致" if have == rep else (
            f"不一致(使用のみ {len(have - rep)}、やり直しのみ {len(rep - have)})")
        lines.append(f"- {g}: 使用 {len(have)} 社、規則でやり直し {len(rep)} 社、{same}")
    lines += ["", "## A.2 期間", "",
              "- 案 1(2 年の遷移をそのまま使う)。生成の経路は 5 年なので、**正解の期間は生成の 2"
              "/5**"
              "(限界として明記する)。", "",
              "## A.3 負例(`negatives.csv`)", "",
              f"- 候補: 生成の目標の候補と同じ class 0(除外後 {len(x0):,} 書類)。その企業自身の"
              f"書類は除く"
              f"(候補にその企業の書類がある始点: {own_in_pool} / {len(pairs)})。",
              f"- 無作為: 乱数 seed 0 で各始点 {M} 社(計 {M * len(pairs)})。",
              f"- 近い企業(hard negative): 現行の等重み距離で近い順に {M} 社(計 {M * len(pairs)}"
              f")。", "",
              "### 参考: 現行の等重み距離での位置(学習の前の記述。判定には使わない)", "",
              "| 量 | 値 |", "|---|---:|",
              f"| 正例(実際の終点)までの距離の中央値 | {np.median(d_pos):.4f} |"]
    for kind in ("無作為", "近い企業"):
        nk = neg[neg["kind"] == kind]
        closer = nk.groupby("edinetCode")["d_equal"].apply(list)
        frac = [np.mean(np.array(closer[c]) < d_pos[i]) for i, c in enumerate(pairs["edinetCode"])]
        lines += [f"| {kind}の負例までの距離の中央値 | {nk['d_equal'].median():.4f} |",
                  f"| 実際の終点より近い{kind}の負例の割合(始点ごとの中央値) |"
                  f" {np.median(frac):.0%} |",
                  f"| 実際の終点がすべての{kind}の負例より近い始点 | "
                  f"{int(np.sum(np.array(frac) == 0))} / {len(pairs)} |"]
    lines.append("")
    (args.out / "data_a.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
