#!/usr/bin/env python
"""ステージ40 B・D: 残余に下限を課した細分化を core (32) と比べ、停止規則を当てる。

入力は ``metrics_stage39.py`` の出力(``data/interim/stage40/metrics/{core,expanded}``)と
生成の parquet。判定の基準は ``results/stage40/criteria.md``(生成の前に記録)。
主指標は純資産の改善 ↑、\\|総資産変化\\| ↓、
符号付き順位相関 ↑(D をまたぐので共通の 32 科目で判定)。解消率は参考。

    uv run python scripts/compare_stage40.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from compare_stage39 import agree, boot, comp_level, corr, fmt_ci, ref_of, verdict, wide
from run_stage25 import CONTINUOUS, OUTLIERS, summarize

from fsgen.accounting.plausibility import CHECK_NAMES, EXTRA_CHECK_NAMES

_M = Path("data/interim/stage40/metrics")
_GEN = {"32": Path("data/interim/stage40/gen"),
        "38": Path("data/stage39_root/data/interim/stage40/gen")}
EXCLUDE = {"c2": OUTLIERS, "c1": ("S100VGDJ",)}
CLS_LABEL = {"c2": "class 2(55 社)", "c1": "class 1(131 社)"}
ROOT_DIR = {"32": "core", "38": "expanded"}
SETTINGS = {1: ("32", "nearest_na0"), 2: ("38", "nearest_na0"), 3: ("38", "pers_nearest_na0")}
SET_LABEL = {1: "core (32)・等重み", 2: "細分化・下限あり・等重み", 3: "細分化・下限あり・持続性"}
COMPARISONS = ((2, 1, "細分化(等重み)対 core"), (3, 1, "細分化(持続性)対 core"),
               (3, 2, "持続性 対 等重み(細分化の中)"))
PRIMARY = (("純資産の改善", "大"), ("|総資産変化|", "小"))
_STUCK = 0.01


def vkey(d: str, s: str) -> str:
    return f"{d}:{s}"


def load(cls: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, dels, refs, viol = [], [], [], []
    for d, sub in ROOT_DIR.items():
        r = pd.read_parquet(_M / sub / f"rows_{cls}.parquet")
        r["version"] = [vkey(d, v) for v in r["version"]]
        rows.append(r)
        x = pd.read_parquet(_M / sub / f"deltas_{cls}.parquet")
        x["setting"] = [vkey(d, v) for v in x["setting"]]
        dels.append(x)
        f = pd.read_parquet(_M / sub / "refs.parquet")
        refs.append(f[f["cls"] == cls].assign(D=d))
        v = pd.read_csv(_M / sub / f"viol_{cls}.csv", dtype={"docID": str})
        v["setting"] = [vkey(d, s) for s in v["setting"]]
        viol.append(v)
    out = EXCLUDE[cls]
    r = pd.concat(rows, ignore_index=True)
    x = pd.concat(dels, ignore_index=True)
    v = pd.concat(viol, ignore_index=True)
    return (r[~r["docID"].isin(out)], x[~x["docID"].isin(out)],
            pd.concat(refs, ignore_index=True), v[~v["docID"].isin(out)])


def active_table(cls: str) -> list[str]:
    """アクティブになった制約(最終ステップ)の行の割合と、実行不可能の件数。"""
    counts: dict[int, pd.Series] = {}
    n_rows: dict[int, int] = {}
    infeasible: dict[int, int] = {}
    for i, (d, s) in SETTINGS.items():
        g = pd.concat([pd.read_parquet(f) for f in
                       sorted((_GEN[d] / cls / s).glob("*__seed*.parquet"))])
        g = g[(g["method"] == "PCFM") & ~g["docID"].isin(EXCLUDE[cls])]
        act = g["active"].fillna("").str.split(";").explode()
        act = act[act != ""]
        counts[i] = act.value_counts() / len(g)
        n_rows[i] = len(g)
        infeasible[i] = int((g["status"] == "infeasible").sum())
    keys = sorted(set().union(*[set(c.index) for c in counts.values()]),
                  key=lambda k: -max(counts[i].get(k, 0.0) for i in SETTINGS))
    lines = ["実行不可能(`infeasible`、行): "
             + "、".join(f"設定 {i} {infeasible[i]} / {n_rows[i]}" for i in SETTINGS), "",
             "| アクティブな制約(最終ステップ、行の割合) | " + " | ".join(
                 f"設定 {i}" for i in SETTINGS) + " |", "|---|" + "---:|" * len(SETTINGS)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(f"{counts[i].get(k, 0.0):.0%}" for i in SETTINGS)
                     + " |")
    return lines


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage40"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    lines = ["# ステージ40 B・D: 残余に下限を課した細分化と core (32) の比較", ""]
    judg = []
    for cls in ("c2", "c1"):
        rows, dels, refs, viol = load(cls)
        comp = comp_level(rows)
        ref_note = ("回復 15 社の 2 年" if cls == "c2" else "class 1 → class 0 の 1 年の遷移 68 組")
        lines += [f"## B.2 {CLS_LABEL[cls]}", "",
                  f"順位相関の実データ: {ref_note}。「32」= 共通の 32 科目に集約(判定に使う)、"
                  "「元」= 各 D の科目。解消率は参考。", "",
                  "| 設定 | 14 項目の違反(行) | 純資産の改善 | \\|総資産変化\\| "
                  "| 順位相関(32) [95%] | 順位相関(元) [95%] | 符号の一致(元) | validity LGBM "
                  "| proximity(32) | sparsity "
                  "| 解消 |",
                  "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|"]
        gens: dict[tuple[int, str], pd.DataFrame] = {}
        rfs: dict[tuple[int, str], pd.Series] = {}
        for i, (d, s) in SETTINGS.items():
            v = vkey(d, s)
            g = comp[comp["version"] == v]
            ins = g[g["insolvent_start"]]
            cells = []
            for space in ("core32", "native"):
                gen, ref = wide(dels, v, space), ref_of(refs, d, space)
                gens[(i, space)], rfs[(i, space)] = gen, ref
                c = corr(gen, ref)
                lo, hi, _, _ = boot(gen, ref, None, None, rng)
                cells.append(f"{c:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            nv = int(viol[viol["setting"] == v]["rows"].sum())
            ga, ra = gens[(i, "native")], rfs[(i, "native")]
            lines.append(
                f"| {i} {SET_LABEL[i]} | {nv} | {g['net_asset_improvement'].median():.3f} "
                f"| {g['abs_asset_change'].median():.3f} | {cells[0]} | {cells[1]} "
                f"| {agree(ga, ra)} / {ga.shape[1]} | {g['validity_lightgbm'].median():.3f} "
                f"| {g['proximity_32'].median():.3f} | {g['sparsity'].median():.0f} "
                f"| {int((ins['insolvency_resolved'] >= 2 / 3).sum())} / {len(ins)} |")
        lines += ["", "比較(差の中央値 [95% CI]、順位相関は差 [95%]、32 科目)。"
                  "判定の記号は 純資産の改善・\\|総資産変化\\|・順位相関 の順"
                  "(+ 有意に優れる、− 有意に劣る、0 どちらでもない)", "",
                  "| 比較 | 純資産の改善 | \\|総資産変化\\| | 順位相関の差 | 解消率(参考) "
                  "| validity(参考) "
                  "| proximity(32)(参考) | 判定 |",
                  "|---|---|---|---|---|---|---|---|"]
        for a, b, name in COMPARISONS:
            va, vb = vkey(*SETTINGS[a]), vkey(*SETTINGS[b])
            _s, pr = summarize(comp, (), (vb, va), vb)
            pr = pr.set_index("metric")
            marks, cells = [], []
            for metric, better in PRIMARY:
                q = pr.loc[metric]
                m, lo, hi = q["median_diff(version-equal)"], q["ci_low"], q["ci_high"]
                cells.append(fmt_ci(m, lo, hi))
                marks.append(verdict(better, m, lo, hi))
            ga = gens[(a, "core32")]
            gb = gens[(b, "core32")].loc[ga.index]
            ca, cb = corr(ga, rfs[(a, "core32")]), corr(gb, rfs[(b, "core32")])
            _, _, dlo, dhi = boot(ga, rfs[(a, "core32")], gb, rfs[(b, "core32")], rng)
            cells.append(fmt_ci(ca - cb, dlo, dhi, 3))
            marks.append(verdict("大", ca - cb, dlo, dhi))
            ref_cells = []
            for metric in ("債務超過の解消率(社ごとの 3 seed 平均)", "validity LightGBM"):
                q = pr.loc[metric]
                ref_cells.append(fmt_ci(q["median_diff(version-equal)"], q["ci_low"],
                                        q["ci_high"]))
            pa = comp[comp["version"] == va].set_index("docID")["proximity_32"]
            pb = comp[comp["version"] == vb].set_index("docID")["proximity_32"].reindex(pa.index)
            from fsgen.evaluation.paired import paired_test
            t = paired_test(pa.to_numpy(), pb.to_numpy())
            ref_cells.append(fmt_ci(t.median_diff, t.ci_low, t.ci_high))
            wins = sum(1 for x in marks if x > 0)
            loses = sum(1 for x in marks if x < 0)
            win = wins >= 2 and loses == 0
            sym = "".join({1: "+", -1: "−", 0: "0"}[x] for x in marks)
            lines.append(f"| {a} − {b}({name}) | {' | '.join(cells)} | {' | '.join(ref_cells)} "
                         f"| {sym} → {'勝ち' if win else '勝ちでない'} |")
            judg.append({"cls": cls, "comparison": f"{a}-{b}", "name": name, "marks": sym,
                         "win": win})
        # B.3 実行可能性
        stuck = {i: int((comp[comp["version"] == vkey(*SETTINGS[i])]["proximity_l2"]
                         < _STUCK).sum()) for i in SETTINGS}
        lines += ["", f"### B.3 実行可能性({CLS_LABEL[cls]})", "",
                  "始点からほとんど動けない企業(seed の中央値で \\|\\|x1 − x0\\|\\|₂ < 0.01): "
                  + "、".join(f"設定 {i} {n} 社" for i, n in stuck.items()), "",
                  *active_table(cls), ""]
        summ, _ = summarize(comp, (), tuple(vkey(*SETTINGS[i]) for i in SETTINGS),
                            vkey(*SETTINGS[1]))
        cols = [m for m, _, _ in CONTINUOUS]
        lines += ["その他の指標(企業ごとの seed 中央値の中央値):", "",
                  "| 設定 | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for i in SETTINGS:
            r = summ.set_index("version").loc[vkey(*SETTINGS[i])]
            lines.append(f"| {i} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " |")
        lines += ["", "14 項目の違反の内訳: " + ("; ".join(
            f"{k[0]} {k[1]} {int(n)}"
            for k, n in viol.groupby(["setting", "check"])["rows"].sum().items()) or "なし"), ""]
    j = pd.DataFrame(judg)
    lines += ["## D.1 停止規則", "", "| 比較 | class 2 | class 1 | 両 class で勝ち |",
              "|---|---|---|---|"]
    for _a, _b, name in COMPARISONS:
        jj = j[j["name"] == name].set_index("cls")
        both = bool(jj.loc["c2", "win"]) and bool(jj.loc["c1", "win"])
        lines.append(f"| {name} | {jj.loc['c2', 'marks']} | {jj.loc['c1', 'marks']} "
                     f"| {'はい' if both else 'いいえ'} |")
    lines.append(f"\nチェックの名前: {', '.join([*CHECK_NAMES, *EXTRA_CHECK_NAMES])}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    j.to_csv(args.out / "judgments.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
