#!/usr/bin/env python
"""ステージ58 B・D: 群内分散による標準化の評価(すべて K = 31、刻み n = 1)。

``evaluate_stage51.py`` の評価と ``evaluate_stage57.py`` の判定を流用する。規則は
``results/stage58/criteria.md``(集計・生成の前に記録)。評価からは、ステージ55 の再建群 72 社と
同じ EDINET コードの始点を除く。設定 1(等重み・K = 31)と現行(等重み・K = 13)はステージ57 の
生成をリンクで使う(ステージ51 の ``k31_nn``・ステージ50 の ``equal`` と同じ設定)。

出力(``results/stage58/``): ``comparison.md``・``realdata.md``・``targets.md``・``order.md``・
``judgment.md`` ほか。

    uv run python scripts/evaluate_stage58.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import evaluate_stage57 as e57
import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.industry import load_industry_config

OUT = Path("results/stage58")
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
YEARS, THR = 5.0, 0.001
MAIN = {"s0_equal_k13": "現行 等重み・K = 13", "s1_equal_k31": "1 等重み・K = 31",
        "s2_sd": "2 案 2(年次変化の SD)", "s3_mad": "3 案 2r(年次変化の MAD)",
        "s4_class": "4 案 1(class 内)", "s5_total": "5 全体の分散(対照)"}
CANDS = ("s2_sd", "s3_mad", "s4_class")  # 優先順(criteria §3)
PLANS = {"s0_equal_k13": "data/interim/stage28/plans_equal",
         "s1_equal_k31": "data/interim/stage28/plans_equal",
         "s2_sd": "data/interim/stage58/plans_within_sd",
         "s3_mad": "data/interim/stage58/plans_within_mad",
         "s4_class": "data/interim/stage58/plans_within_class",
         "s5_total": "data/interim/stage58/plans_total"}
# 現行の値(criteria §3 の材料 4、ステージ56)
CUR_DOWN = {"c2": 0.82, "c1": 0.84}


def configure() -> None:
    e51._M = Path("data/interim/stage58/metrics")
    e51._GEN = Path("data/interim/stage58/gen")
    e51.SETTINGS = MAIN
    e51.BASE = "s1_equal_k31"
    e51.BASE_DESC = "設定 1(等重み・K = 31)"
    e51.CRITERIA = "`criteria.md` §3"
    e51.BASE_COL = "1 等重み K31"
    e51.PAIRS_F2 = (("s5_total", "s2_sd", "B.4: 全体 対 案 2(5 対 2)"),
                    ("s5_total", "s4_class", "B.4: 全体 対 案 1(5 対 4)"),
                    ("s3_mad", "s2_sd", "ロバスト化の効果(3 対 2)"),
                    ("s0_equal_k13", "s1_equal_k31", "K の効果(等重み、現行 対 1)"))
    e51.STAGE = "ステージ58"
    e51.TITLE = "ステージ58 B: 群内分散による標準化の比較(K = 31、刻み n = 1)"
    e51.OUT = {"comparison": "comparison.md", "losses": "losses.md",
               "assets": "assets_extreme.md", "bs": "bs.md", "summary": "summary_b.json"}
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / "s1_equal_k31").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))
        print(f"[{cls}] 再建群と同じ EDINET コードの始点を除く: {len(drop)} 社")


def realdata_rows() -> tuple[list[str], dict[str, dict[str, float]]]:
    th = json.loads(Path("results/stage56/theta.json").read_text(encoding="utf-8"))
    paid = pd.read_csv("results/stage56/paidin.csv", dtype={"edinetCode": str})
    lines = ["# ステージ58 B.3: 払込資本の実データとの比較", "",
             "年率(総資産比 ±0.001、ステージ56・57 と同じ定義)。生成は 5 年 / 5、実データは 2 "
             "年の和 / 2。"
             "減らす深さ = 減らした企業の払込資本の比(終点 / 始点)の中央値。生成は企業ごとの see"
             "d 中央値、"
             "評価から除いた企業を除く。材料 4 = 減らした割合が現行(class 2 82%、class 1 84%)以"
             "下か。", "",
             "| class | 設定 | 増やした | 減らした | \\|減らした − 実データ\\| | 減らす深さ | "
             "材料 4 |",
             "|---|---|---:|---:|---:|---:|---|"]
    down: dict[str, dict[str, float]] = {}
    for cls in ("c2", "c1"):
        g = th["groups"][e57.REAL[cls]]
        real_down = g["down_annual"] / g["n_annual"]
        rr = paid[paid["group"] == e57.REAL[cls]]
        lines.append(f"| {cls} | **実データ({e57.REAL[cls]})** | {g['up_annual']} / {g['n_annual']}"
                     f"({g['up_annual'] / g['n_annual']:.0%}) | {real_down:.0%} | — "
                     f"| {rr[rr['ratio'] < 1]['ratio'].median():.3f} | — |")
        down[cls] = {}
        for s, lab in MAIN.items():
            gg = e51.load_gen(cls, s)
            gg = gg[~gg["docID"].isin(e51.EXCL[cls])]
            p0 = gg[f"x0_{_CS}"] + gg[f"x0_{_CSUR}"]
            p1 = gg[f"x1_{_CS}"] + gg[f"x1_{_CSUR}"]
            df = pd.DataFrame({"docID": gg["docID"], "an": (p1 - p0) / YEARS,
                               "ratio": p1 / p0.where(p0 > 0)})
            f = df.groupby("docID").agg(an=("an", "median"), ratio=("ratio", "median"))
            d = float((f["an"] < -THR).mean())
            down[cls][s] = d
            m4 = "—" if s in ("s0_equal_k13", "s1_equal_k31") else (
                "満たす" if round(d, 2) <= CUR_DOWN[cls] else "満たさない")
            lines.append(f"| {cls} | {lab} | {(f['an'] > THR).mean():.0%} | {d:.0%} "
                         f"| {abs(d - real_down):.2f} | {f[f['an'] < -THR]['ratio'].median():.3f} "
                         f"| {m4} |")
    return lines + [""], down


def targets_rows() -> list[str]:
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    paidr = (agg[_CS].fillna(0) + agg[_CSUR].fillna(0)) / agg["total_assets"]
    ind = load_industry_config(Path("conf/accounts/industry.yaml"))
    base = pd.read_csv(Path(PLANS["s1_equal_k31"]) / "full_targets_nearest_na0.csv",
                       dtype={"source_docID": str, "target_docID": str}).set_index("source_docID")
    lines = ["# ステージ58 B.3: 対応先の性質(全社の計画)", "",
             "| 設定 | 対応先の払込資本比率(中央値) | 異なる対応先 | 等重みと同じ対応先 | 業種"
             "一致率(33 業種)"
             " | 対応先の総資産 / 始点 |", "|---|---:|---:|---:|---:|---:|"]
    seen = set()
    for s, lab in MAIN.items():
        d = PLANS[s]
        if d in seen:
            continue
        seen.add(d)
        t = pd.read_csv(Path(d) / "full_targets_nearest_na0.csv",
                        dtype={"source_docID": str, "target_docID": str})
        si = ind.labels(agg.loc[t["source_docID"], "edinetCode"].astype(str).tolist())
        ti = ind.labels(agg.loc[t["target_docID"], "edinetCode"].astype(str).tolist())
        match = np.mean([a == b and a != "不明" for a, b in zip(si, ti, strict=True)])
        same = np.mean(t["target_docID"].to_numpy()
                       == base.loc[t["source_docID"], "target_docID"].to_numpy())
        lines.append(f"| {lab} | {paidr.loc[t['target_docID']].median():.3f} "
                     f"| {t['target_docID'].nunique()} | {same:.0%} | {match:.1%} "
                     f"| {t['asset_ratio'].median():.2f} |")
    return lines + [""]


def order_rows(names: tuple[str, ...]) -> tuple[list[str], dict[str, bool]]:
    """材料 5(B.4): 設定 5 を設定 2・4 と比べる。差 = 5 − 2(負なら 5 が劣る)。"""
    rng = np.random.default_rng(0)
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    lines = ["# ステージ58 B.4: 全体の分散 対 群内分散(材料 5)", "",
             "企業単位のペアの差の中央値 [95% 区間](順位相関は企業の同時復元抽出)。"
             "記号は 設定 5 − 比べる相手(− = 設定 5 が有意に劣る)。成立 = 両 class で主指標の −"
             " が 1 つ以上、"
             "+ がない。", "",
             "| 比較 | class | 純資産の改善 | 解消率 | 順位相関 | 記号 |", "|---|---|---|---|---"
                                                          "|---|"]
    res: dict[str, bool] = {}
    for other in ("s2_sd", "s4_class"):
        ok_all = True
        for cls in ("c2", "c1"):
            c = ctx[cls].compare("s5_total", other, rng)
            m = [int(c[k][3]) for k in ("純資産の改善", "解消率", "順位相関")]
            ok = any(v == -1 for v in m) and not any(v == 1 for v in m)
            ok_all &= ok
            sym = " ".join({1: "+", -1: "−", 0: "0"}[v] for v in m)
            lines.append(f"| 5 − {MAIN[other]} | {cls} | {e51.fc(c['純資産の改善'])} "
                         f"| {e51.fc(c['解消率'])} | {e51.fc(c['順位相関'])} | {sym} |")
        res[other] = ok_all
    lines += ["", f"**材料 5(5 対 2)**: {'成立' if res['s2_sd'] else '不成立'}。"
              f"参考(5 対 4): {'成立' if res['s4_class'] else '不成立'}。", ""]
    return lines, res


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    configure()
    jl, res = e57.judge(names, CANDS, "s1_equal_k31")
    passed = [s for s in CANDS if res[s]["c2"] and res[s]["c1"]]
    jl0, res0 = e57.judge(names, tuple(passed), "s0_equal_k13") if passed else ([], {})
    final = [s for s in passed if res0[s]["c2"] and res0[s]["c1"]]
    e51.BS_ORDER = (passed[0] if passed else "s2_sd",)
    sys.argv = [sys.argv[0], "--out", str(OUT)]
    e51.main()
    rl, down = realdata_rows()
    (OUT / "realdata.md").write_text("\n".join(rl) + "\n", encoding="utf-8")
    (OUT / "targets.md").write_text("\n".join(targets_rows()) + "\n", encoding="utf-8")
    ol, ores = order_rows(names)
    (OUT / "order.md").write_text("\n".join(ol) + "\n", encoding="utf-8")
    out = ["# ステージ58 D.1: 判定(対 設定 1 = 等重み・K = 31)", "", *jl, "",
           f"基準 1〜3 を満たした設定: {', '.join(MAIN[s] for s in passed) or 'なし'}。", ""]
    if passed:
        out += ["## 現行(等重み・K = 13)に対して", "", *jl0, "",
                f"最終設定を変える設定: {', '.join(MAIN[s] for s in final) or 'なし'}。", ""]
    out += [f"材料 5(B.4、5 対 2): {'成立' if ores['s2_sd'] else '不成立'}。", ""]
    (OUT / "judgment.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {"judge": res, "judge_vs_k13": res0, "passed": passed,
                               "final": final, "order": ores, "down": down}
    (OUT / "summary_main.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
