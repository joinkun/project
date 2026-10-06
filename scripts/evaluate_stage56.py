#!/usr/bin/env python
"""ステージ56 C・D: 払込資本の下限 θ の評価(``evaluate_stage51.py`` の評価を流用)。

規則は ``results/stage56/criteria.md``(θ を含め生成の前に記録)。設定: ``equal``(現行)、``theta``
(θ + α)、``theta_noalpha``(θ、α なし)。基準 2 は、生成で払込資本を減らした企業(年率 < −0.001)の
割合が、対応する再建群の割合の Wilson 95% 区間に入ること。

出力(``results/stage56/``): ``comparison.md``・``assets_extreme.md``・``bs.md``(evaluate_stage51 の
形式)、``realdata.md``(C.3)、``feasibility.md``(C.4)、``judgment.md``、``summary_d.json``。

    uv run python scripts/evaluate_stage56.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

OUT = Path("results/stage56")
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
YEARS, THR = 5.0, 0.001
REAL = {"c2": "再建群(class 2 → 回復)", "c1": "再建群(class 1 → class 0)"}
CANDS = ("theta", "theta_noalpha")
PRIMARY = ("純資産の改善", "解消率", "順位相関")


def configure() -> None:
    e51._M = Path("data/interim/stage56/metrics")
    e51._GEN = Path("data/interim/stage56/gen")
    e51.SETTINGS = {"equal": "1 現行(θ なし、α あり)", "theta": "2 θ + α",
                    "theta_noalpha": "3 θ(α なし)"}
    e51.BASE = "equal"
    e51.PAIRS_F2 = ()
    e51.STAGE = "ステージ56"
    e51.TITLE = "ステージ56 C.2: 払込資本の下限 θ の比較"
    e51.OUT = {"comparison": "comparison.md", "losses": "losses.md", "assets": "assets_extreme.md",
               "bs": "bs.md", "summary": "summary_c.json"}


def firm_paid(cls: str, s: str) -> pd.DataFrame:
    g = e51.load_gen(cls, s)
    p0 = g[f"x0_{_CS}"] + g[f"x0_{_CSUR}"]
    p1 = g[f"x1_{_CS}"] + g[f"x1_{_CSUR}"]
    x0c = [c for c in g.columns if c.startswith("x0_")]
    x1c = [c for c in g.columns if c.startswith("x1_")]
    moved = np.linalg.norm(g[x1c].to_numpy() - g[x0c].to_numpy(), axis=1)
    df = pd.DataFrame({"docID": g["docID"], "annual": (p1 - p0) / YEARS,
                       "ratio": np.where(p0 > 0, p1 / p0.where(p0 > 0, 1.0), np.nan),
                       "transfer": g["re_transfer"] / YEARS,
                       "cs_zero": (g[f"x0_{_CS}"] > 1e-6) & (g[f"x1_{_CS}"].abs() <= 1e-6),
                       "active": g["active"].fillna("").str.contains("paid_in>="),
                       "infeasible": g["status"] == "infeasible",
                       "stuck": moved < 1e-3})
    return df.groupby("docID").agg(annual=("annual", "median"), ratio=("ratio", "median"),
                                   transfer=("transfer", "median"), cs_zero=("cs_zero", "mean"),
                                   active=("active", "mean"), infeasible=("infeasible", "sum"),
                                   stuck=("stuck", "mean"))


def main() -> None:  # noqa: PLR0915
    configure()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    th = json.loads((OUT / "theta.json").read_text(encoding="utf-8"))
    real = pd.read_csv(OUT / "paidin.csv", dtype={"edinetCode": str})
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    rf = rf.set_index("edinetCode")
    rng = np.random.default_rng(0)
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    res: dict[str, Any] = {}
    rl = ["# ステージ56 C.3: 払込資本の動き(生成 対 実際に再建した企業)", "",
          "年率の変化: 生成は (終点 − 始点) / 5、実データは 2 年の和 / 2(総資産比、ステージ47 "
          "と同じ定義)。"
          "減らした = 年率 < −0.001、増やした = 年率 > 0.001。比 = 払込資本(終点)/ 払込資本(始点)"
          "(生成は 5 年、実データは 2 年)。振替あり = 年率の振替 > 0.001(実データはステージ30 "
          "の値、"
          "class 2 → 回復のみ)。生成は企業ごとの seed 中央値。", "",
          "| class | 設定 | 減らした | 増やした | 減らした企業の比の中央値 | 比の最小 | 振替あり |",
          "|---|---|---|---|---:|---:|---|"]
    fl = ["# ステージ56 C.4: 実行可能性", "",
          "θ が拘束 = 最終ステップで払込資本の下限がアクティブな行がある企業(3 seed の平均)。"
          "動けない = 始点からの移動の L2 < 1e-3 の行。資本金を 0 = 始点で資本金 > 0、終点で総"
          "資産比 1e-6 以下。", "",
          "| class | 設定 | 実行不可能(行) | 動けない企業 | θ が拘束した企業 | 資本金を 0 にし"
          "た企業 |",
          "|---|---|---:|---:|---:|---:|"]
    for cls in ("c2", "c1"):
        g = th["groups"][REAL[cls]]
        wd = g["wilson_down"]
        rr = real[real["group"] == REAL[cls]]
        dec_r = rr[rr["ratio"] < 1]["ratio"]
        rl.append(f"| {cls} | **実データ({REAL[cls]}、{g['n_annual']} 社)** "
                  f"| {g['down_annual']} / {g['n_annual']}({g['down_annual'] / g['n_annual']:.0%})"
                  f" [Wilson {wd[0]:.0%}, {wd[1]:.0%}] "
                  f"| {g['up_annual']} / {g['n_annual']}({g['up_annual'] / g['n_annual']:.0%}) "
                  f"| {dec_r.median():.3f} | {rr['ratio'].min():.3f} "
                  + (f"| {int((rf.reindex(rr['edinetCode'])['transfer'] > THR).sum())} /"
                     f" {len(rr)} |"
                     if cls == "c2" else "| — |"))
        for s in ("equal", *CANDS):
            f = firm_paid(cls, s)
            f = f[~f.index.isin(e51.EXCL[cls])]
            n = len(f)
            down = int((f["annual"] < -THR).sum())
            up = int((f["annual"] > THR).sum())
            share = down / n
            ok2 = wd[0] <= share <= wd[1]
            res.setdefault(s, {})[cls] = {"down_share": share, "crit2": bool(ok2),
                                          "cs_zero": int((f["cs_zero"] >= 2 / 3).sum())}
            dec = f[f["annual"] < -THR]["ratio"]
            rl.append(f"| {cls} | {e51.SETTINGS[s]} | {down} / {n}({share:.0%})"
                      f"{' ✓' if ok2 else ''} | {up} / {n}({up / n:.0%}) "
                      f"| {dec.median():.3f} | {f['ratio'].min():.3f} "
                      f"| {int((f['transfer'] > THR).sum())} / {n} |")
            fl.append(f"| {cls} | {e51.SETTINGS[s]} | {int(f['infeasible'].sum())} "
                      f"| {int((f['stuck'] >= 2 / 3).sum())} | {int((f['active'] > 0).sum())} "
                      f"| {int((f['cs_zero'] >= 2 / 3).sum())} |")
    rl += ["", "✓ = 基準 2(減らした割合が実データの Wilson 95% 区間に入る)を満たす。", ""]
    (OUT / "realdata.md").write_text("\n".join(rl) + "\n", encoding="utf-8")
    (OUT / "feasibility.md").write_text("\n".join(fl) + "\n", encoding="utf-8")
    # 判定
    jl = ["# ステージ56 D: 判定", "",
          "| 設定 | class | 違反 | 基準 2(減少の割合) | 主指標(純資産の改善・解消率・順位相関) |"
          " 劣る主指標なし |",
          "|---|---|---:|---|---|---|"]
    passed = []
    for s in CANDS:
        ok_all = True
        for cls in ("c2", "c1"):
            c = ctx[cls].compare(s, "equal", rng)
            nv = sum(ctx[cls].viol_items(s).values())
            marks = [int(c[k][3]) for k in PRIMARY]
            no_worse = not any(m == -1 for m in marks)
            ok = nv == 0 and res[s][cls]["crit2"] and no_worse
            ok_all &= ok
            sym = " ".join({1: "+", -1: "−", 0: "0"}[m] for m in marks)
            jl.append(f"| {e51.SETTINGS[s]} | {cls} | {nv} | "
                      f"{res[s][cls]['down_share']:.0%}"
                      f"({'満たす' if res[s][cls]['crit2'] else '満たさない'}) "
                      f"| {sym} | {'○' if no_worse else '×'} |")
        res[s]["pass"] = bool(ok_all)
        if ok_all:
            passed.append(s)
    adopted: str | None = None
    if "theta_noalpha" in passed and all(res["theta_noalpha"][c]["cs_zero"] == 0
                                         for c in ("c2", "c1")):
        adopted = "theta_noalpha"
    elif "theta" in passed:
        adopted = "theta"
    verdict = (e51.SETTINGS[adopted] + " を採用") if adopted else "採用しない(最終設定を変えない)"
    jl += ["", f"**判定: {verdict}**", ""]
    (OUT / "judgment.md").write_text("\n".join(jl) + "\n", encoding="utf-8")
    e51.BS_ORDER = (adopted,) if adopted else ("theta",)
    sys.argv = [sys.argv[0], "--out", str(OUT)]
    e51.main()
    (OUT / "summary_d.json").write_text(json.dumps({"results": res, "adopted": adopted},
                                                   ensure_ascii=False, indent=2, default=float),
                                        encoding="utf-8")
    print("\n".join(rl + fl + jl))


if __name__ == "__main__":
    main()
