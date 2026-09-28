#!/usr/bin/env python
"""ステージ30 A.3・A.4・B.1: 実際に GC 注記が外れた企業(再建群)の財務変化と、採否の基準。

再建群 Y: 2022 年度に GC 注記あり → 2024 年度に注記なし(class 0 または class 1)。
厳格な定義(2024 年度に class 0)も参考に出す。非再建群 Z: 2022 年度も 2024 年度も注記あり。
遷移は ``transitions_stage30.py`` の ``data/interim/stage30/firm_years.csv``。

特殊業種の除外(ステージ28 の規則)は、各社の最新の書類の「事業の内容」と 33 業種に直接当てる
(ステージ28 の除外リストは 2024 年度の評価データの書類しか含まないため)。

財務変化: ステージ23 のパネル(``panel.csv``)の 2023・2024 年度の書類の、同一書類内の差分 /
前期総資産 を足したもの(2022 年度末 → 2024 年度末の 2 年分)。年率 = 2 年の和 / 2、
5 年換算 = 年率 × 5。分解はステージ29 と同じ(払込資本の減少分を振替 T、P = ΔRE − T)。

基準(B.1、閾値は機械的に決まる。Y の主定義・除外後・パネルにある社で計算):
2. 利益剰余金が増える社の割合: Y の割合 ± 10 ポイント
3. 利益部分の上限 γ = 5 × (Y の P の年率の p90)。負なら 0
4. 振替の規模: 振替のある社の T / A0(5 年)の中央値と p95 が、Y の振替のある社の T × 5 の
   p5〜p95 に入る
5. 増資の規模: 増資のある社(Δ払込資本 / A0 / 5 > 閾値)の Δ払込資本 / A0(5 年)の中央値と p95 が、
   Y の増資のある社の Δ払込資本 × 5 の p5〜p95 に入る

    uv run python scripts/recovery_stage30.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from extract_business_stage28 import business_text
from special_industry_stage28 import classify

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.models.industry import load_industry_config

_RE = "jppfs_cor_RetainedEarnings"
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
THR = 0.001
YEARS = 5.0
_QS = (5, 25, 50, 75, 90, 95)


def classify_groups(fy: pd.DataFrame) -> pd.DataFrame:
    """企業ごとの群(Y / Y_strict / Z)と、ステージ28 の規則による除外理由。"""
    wide = fy.pivot(index="edinetCode", columns="year", values="label")
    gc = wide[wide[2022] == 2]
    out = pd.DataFrame(index=gc.index)
    out["y2023"] = gc[2023]
    out["y2024"] = gc[2024]
    out["group"] = np.where(gc[2024].isin([0, 1]), "Y", np.where(gc[2024] == 2, "Z", "欠測"))
    out["strict"] = gc[2024] == 0
    latest = fy.sort_values("year").drop_duplicates("edinetCode", keep="last").set_index(
        "edinetCode"
    )
    out["filerName"] = latest.loc[out.index, "filerName"]
    out["docID"] = latest.loc[out.index, "docID"]
    ind = load_industry_config(Path("conf/accounts/industry.yaml"))
    out["industry33"] = ind.labels(out.index.tolist())
    out["business"] = [business_text(d) or "" for d in out["docID"]]
    out["reason"] = classify(out[["industry33", "filerName", "business"]])
    return out


def changes(panel_dir: Path, codes: list[str], names: tuple[str, ...]) -> pd.DataFrame:
    """2022 年度末 → 2024 年度末の変化(2 年分の同一書類内の差分 / 前期総資産 の和)。"""
    p = pd.read_csv(panel_dir / "panel.csv", dtype={"edinetCode": str})
    p = p[p["edinetCode"].isin(codes)]
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    na = subtotal_members(spec, names)["jppfs_cor_NetAssets"]
    cur = p[[f"cur_{n}" for n in names]].fillna(0.0).to_numpy()
    prev = p[[f"prev_{n}" for n in names]].fillna(0.0).to_numpy()
    p = p.assign(na_cur=cur @ na, na_prev=prev @ na)
    d = p[p["t"].isin([2023, 2024])]
    da = d["prev_total_assets"]
    rows = pd.DataFrame(
        {
            "edinetCode": d["edinetCode"],
            "d_re": (d[f"cur_{_RE}"].fillna(0) - d[f"prev_{_RE}"].fillna(0)) / da,
            "d_paid": (
                d[f"cur_{_CS}"].fillna(0) + d[f"cur_{_CSUR}"].fillna(0)
                - d[f"prev_{_CS}"].fillna(0) - d[f"prev_{_CSUR}"].fillna(0)
            ) / da,
            "d_na": (d["na_cur"] - d["na_prev"]) / da,
        }
    )
    g = rows.groupby("edinetCode")
    out = g.sum() / g.size().to_numpy()[:, None]  # 年率(2 年の和 / 2)
    out["n_years"] = g.size()
    out["transfer"] = (-out["d_paid"]).clip(lower=0.0)
    out["profit"] = out["d_re"] - out["transfer"]
    na22 = p[p["t"] == 2022].set_index("edinetCode")["na_cur"]
    na24 = p[p["t"] == 2024].set_index("edinetCode")["na_cur"]
    a22 = p[p["t"] == 2022].set_index("edinetCode")["cur_total_assets"]
    out["na_2022"] = na22 / a22
    out["insolvent_2022"] = na22 < 0
    out["resolved"] = (na22 < 0) & (na24 > 0)
    return out


def _q(s: pd.Series) -> str:
    return " | ".join(f"{s.quantile(q / 100):+.4f}" for q in _QS) if len(s) else "—"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--out", type=Path, default=Path("results/stage30"))
    args = ap.parse_args()
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    groups = classify_groups(fy)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    ch = changes(args.panel_dir, groups.index.tolist(), names)
    df = groups.join(ch, how="left")
    df["in_panel"] = df["n_years"].notna()
    args.out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["business"]).to_csv(args.out / "recovery_firms.csv")

    use = df[(df["reason"] == "") & df["in_panel"]]
    y = use[use["group"] == "Y"]
    ys = use[(use["group"] == "Y") & use["strict"]]
    z = use[use["group"] == "Z"]

    def stats(s: pd.DataFrame) -> dict[str, float]:
        up = s["d_re"] > THR
        return {
            "n": len(s),
            "share_re_up": float(up.mean()),
            "n_re_up": int(up.sum()),
            "n_transfer_up": int((up & (s["d_paid"] < -THR)).sum()),
            "n_pure_profit_up": int((up & (s["d_paid"].abs() <= THR)).sum()),
            "n_issue_up": int((up & (s["d_paid"] > THR)).sum()),
            "n_issue": int((s["d_paid"] > THR).sum()),
            "n_transfer": int((s["transfer"] > THR).sum()),
            "n_profit_pos": int((s["profit"] > THR).sum()),
            "n_insolvent_2022": int(s["insolvent_2022"].sum()),
            "n_resolved": int(s["resolved"].sum()),
        }

    sy, ss, sz = stats(y), stats(ys), stats(z)
    t_pos = y.loc[y["transfer"] > THR, "transfer"] * YEARS
    i_pos = y.loc[y["d_paid"] > THR, "d_paid"] * YEARS
    p90 = float(y["profit"].quantile(0.90))
    criteria = {
        "source": "Y(2022 注記あり → 2024 注記なし、ステージ28 の規則で除外後、パネルにある社)",
        "n_y": len(y),
        "share_re_up": sy["share_re_up"],
        "share_re_up_band": [max(0.0, sy["share_re_up"] - 0.10), sy["share_re_up"] + 0.10],
        "profit_annual_p90": p90,
        "gamma": max(0.0, YEARS * p90),
        "n_transfer_firms": len(t_pos),
        "transfer_5y_band": [float(t_pos.quantile(0.05)), float(t_pos.quantile(0.95))]
        if len(t_pos) else None,
        "n_issue_firms": len(i_pos),
        "issue_5y_band": [float(i_pos.quantile(0.05)), float(i_pos.quantile(0.95))]
        if len(i_pos) else None,
        "threshold_annual": THR,
        "years": YEARS,
        "horizon_years_observed": 2,
    }
    (args.out / "criteria.json").write_text(
        json.dumps(criteria, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    head = " | ".join(f"p{q}" for q in _QS)
    lines = [
        "# ステージ30 A.3・A.4: 再建群と非再建群の財務変化", "",
        "2022 年度末 → 2024 年度末(2 年分の同一書類内の差分 / 前期総資産 の和)。年率 = / 2。",
        "ステージ28 の規則で特殊業種を除き、ステージ23 のパネルにある社に限る。", "",
        "## 対象の社数", "",
        "| 群 | 全体 | 特殊業種(除外) | パネルにない | 使う社 |", "|---|---:|---:|---:|---:|",
    ]
    for gname, gmask in (("Y 再建(2024 注記なし)", df["group"] == "Y"),
                         ("うち厳格(2024 class 0)", (df["group"] == "Y") & df["strict"]),
                         ("Z 非再建(2024 も注記あり)", df["group"] == "Z"),
                         ("欠測", df["group"] == "欠測")):
        g = df[gmask]
        lines.append(
            f"| {gname} | {len(g)} | {int((g['reason'] != '').sum())} "
            f"| {int(((g['reason'] == '') & ~g['in_panel']).sum())} "
            f"| {int(((g['reason'] == '') & g['in_panel']).sum())} |"
        )
    excl = df[df["reason"] != ""]
    lines += ["", "特殊業種で除いた社: " + "、".join(
        f"{r['filerName']}({r['reason']}、{r['group']})" for _, r in excl.iterrows())]
    miss = df[(df["reason"] == "") & ~df["in_panel"] & (df["group"] != "欠測")]
    lines += ["パネルにない社(ステージ23 のパネル構築で除かれた): " + "、".join(
        f"{r['filerName']}({r['group']})" for _, r in miss.iterrows())]

    lines += ["", "## A.3 再建群 Y の分布(年率、総資産比)", "",
              f"| 系列 | 社数 | {head} |", "|---|---:|" + "---:|" * len(_QS)]
    for col, cname in (("d_re", "ΔRE"), ("d_paid", "Δ払込資本"), ("transfer", "振替 T"),
                       ("profit", "利益部分 P"), ("d_na", "Δ純資産")):
        lines.append(f"| {cname} | {len(y)} | {_q(y[col])} |")
    lines.append(f"| ΔRE(5 年換算) | {len(y)} | {_q(y['d_re'] * YEARS)} |")
    lines += ["", "## A.3・A.4 再建群と非再建群", "",
              "| 指標 | 再建群 Y | うち厳格 | 非再建群 Z | 差(Y − Z) |",
              "|---|---:|---:|---:|---:|"]
    for key, lab in (("n", "社数"), ("n_re_up", "RE が増えた社"),
                     ("n_transfer_up", "うち振替(Δ払込資本 < 0)"),
                     ("n_pure_profit_up", "うち純粋な利益(Δ払込資本 ≈ 0)"),
                     ("n_issue_up", "うち増資を伴う"), ("n_profit_pos", "利益部分 P > 0 の社"),
                     ("n_issue", "増資のある社(Δ払込資本 > 0)"), ("n_transfer", "振替のある社"),
                     ("n_insolvent_2022", "2022 年度末に債務超過"), ("n_resolved", "うち解消")):
        lines.append(f"| {lab} | {sy[key]} | {ss[key]} | {sz[key]} | {sy[key] - sz[key]:+d} |")
    lines.append(f"| RE が増えた社の割合 | {sy['share_re_up']:.1%} | {ss['share_re_up']:.1%} "
                 f"| {sz['share_re_up']:.1%} | {sy['share_re_up'] - sz['share_re_up']:+.1%} |")
    for col, lab in (("d_re", "ΔRE 年率の中央値"), ("d_paid", "Δ払込資本 年率の中央値"),
                     ("profit", "利益部分 P 年率の中央値"), ("d_na", "Δ純資産 年率の中央値"),
                     ("na_2022", "2022 年度末の純資産 / 総資産 の中央値")):
        a_, b_, c_ = y[col].median(), ys[col].median(), z[col].median()
        lines.append(f"| {lab} | {a_:+.4f} | {b_:+.4f} | {c_:+.4f} | {a_ - c_:+.4f} |")
    lines += ["", "## 再建群 Y の各社(年率)", "",
              "| 企業 | 2023 → 2024 | ΔRE | Δ払込資本 | 振替 | 利益部分 | Δ純資産 "
              "| 2022 債務超過 → 解消 |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    st = {0: "なし", 1: "重要事象のみ", 2: "注記あり"}
    for _, r in y.sort_values("d_re", ascending=False).iterrows():
        s23 = st[int(r["y2023"])] if pd.notna(r["y2023"]) else "欠測"
        lines.append(
            f"| {r['filerName']} | {s23} → {st[int(r['y2024'])]} "
            f"| {r['d_re']:+.4f} | {r['d_paid']:+.4f} "
            f"| {r['transfer']:.4f} | {r['profit']:+.4f} | {r['d_na']:+.4f} "
            f"| {'債務超過' if r['insolvent_2022'] else ''}{' → 解消' if r['resolved'] else ''} |"
        )
    (args.out / "recovery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(criteria, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
