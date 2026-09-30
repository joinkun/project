#!/usr/bin/env python
"""ステージ39 C.3・D・E: D = 32 と 38 の生成を比べる(scipy のみ。torch・POT・lightgbm なし)。

入力は ``metrics_stage39.py`` の出力(``data/interim/stage39/metrics/{core,expanded}``)と、
輸送計画の要約(``transport_summary.csv``)。判定の基準は ``results/stage39/criteria.md``
(結果を見る前に記録)。

- 主指標: 純資産の改善 ↑、債務超過の解消率 ↑、|総資産変化| ↓(企業単位のペア、差の中央値の
  95% ブートストラップ区間)、実データとの符号付き順位相関 ↑(企業の復元抽出、1,000 回、seed 0)
- 「勝つ」= 主指標の 2 つ以上で有意に優れ、有意に劣るものがない。両 class で勝つ場合だけ採用
- D をまたぐ比較(設定 3 − 1)の順位相関は共通の 32 科目への集約で判定する

    uv run python scripts/compare_stage39.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from persistence_tables_stage33 import spearman
from run_stage25 import CONTINUOUS, OUTLIERS, company_level, summarize

_M = Path("data/interim/stage39/metrics")
_B = 1000
_ZERO = 1e-4
EXCLUDE = {"c2": OUTLIERS, "c1": ("S100VGDJ",)}
CLS_LABEL = {"c2": "class 2(55 社)", "c1": "class 1(131 社)"}
SETTINGS = {
    1: ("32", "nearest_na0"), 2: ("32", "unb0.1_na0"), 3: ("38", "nearest_na0"),
    4: ("38", "unb0.1_na0"), 5: ("38", "pers_nearest_na0"), 6: ("38", "pers_unb0.1_na0"),
}
SET_LABEL = {1: "core・等重み・最近傍", 2: "core・等重み・不均衡 OT",
             3: "細分化・等重み・最近傍", 4: "細分化・等重み・不均衡 OT",
             5: "細分化・持続性・最近傍", 6: "細分化・持続性・不均衡 OT"}
# (比較, 基準, 判定の名前)
COMPARISONS = ((2, 1, "E.2 OT(core)"), (3, 1, "E.1 細分化"), (4, 3, "E.2 OT(細分化)"),
               (5, 3, "E.3 重み(最近傍)"), (6, 4, "E.3 重み(OT)"))
# 事前に決めていない参考の比較(判定には使わない)
REFERENCE = ((5, 1, "参考: 細分化 + 持続性 対 現行"),)
DESIGNS = ("nearest_na0", "bal0.01_na0", "bal0.05_na0", "bal0.1_na0", "unb0.1_na0",
           "unb1_na0", "unb10_na0", "onesided_na0")
ROOT_DIR = {"32": "core", "38": "expanded"}
PLANS = {"32": Path("data/interim/stage39/core/plans_equal"),
         "38": Path("data/stage39_root/data/interim/stage39/plans_equal")}
PRIMARY = (("純資産の改善", "大"), ("|総資産変化|", "小"),
           ("債務超過の解消率(社ごとの 3 seed 平均)", "大"))


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
        v = pd.read_csv(_M / sub / f"viol_{cls}.csv")
        v["setting"] = [vkey(d, s) for s in v["setting"]]
        viol.append(v)
    out = EXCLUDE[cls]
    r = pd.concat(rows, ignore_index=True)
    r = r[~r["docID"].isin(out)]
    x = pd.concat(dels, ignore_index=True)
    x = x[~x["docID"].isin(out)]
    v = pd.concat(viol, ignore_index=True)
    return r, x, pd.concat(refs, ignore_index=True), v[~v["docID"].isin(out)]


def wide(dels: pd.DataFrame, version: str, space: str) -> pd.DataFrame:
    s = dels[(dels["setting"] == version) & (dels["space"] == space)]
    return s.pivot(index="docID", columns="account", values="delta")


def ref_of(refs: pd.DataFrame, d: str, space: str) -> pd.Series:
    s = refs[(refs["D"] == d) & (refs["space"] == space)]
    return s.set_index("account")["median"]


def corr(gen: pd.DataFrame, ref: pd.Series) -> float:
    return spearman(gen.median().to_numpy(), ref.loc[gen.columns].to_numpy())


def agree(gen: pd.DataFrame, ref: pd.Series) -> int:
    def sg(v: np.ndarray) -> np.ndarray:
        return np.where(np.abs(v) < _ZERO, 0, np.sign(v))
    return int((sg(gen.median().to_numpy()) == sg(ref.loc[gen.columns].to_numpy())).sum())


def boot(a: pd.DataFrame, ra: pd.Series, b: pd.DataFrame | None, rb: pd.Series | None,
         rng: np.random.Generator) -> tuple[float, float, float, float]:
    """(a の相関の 95% 区間, a − b の差の 95% 区間)。企業を同時に復元抽出する。"""
    ids = a.index.to_numpy()
    vals, diffs = [], []
    for _ in range(_B):
        pick = rng.choice(ids, size=len(ids), replace=True)
        c = corr(a.loc[pick], ra)
        vals.append(c)
        if b is not None and rb is not None:
            diffs.append(c - corr(b.loc[pick], rb))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    dlo, dhi = np.nanpercentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)
    return float(lo), float(hi), float(dlo), float(dhi)


def comp_level(rows: pd.DataFrame) -> pd.DataFrame:
    comp = company_level(rows)
    extra = rows.groupby(["version", "docID"]).agg(
        proximity_32=("proximity_32", "median"), infeasible=("infeasible", "sum")).reset_index()
    return comp.merge(extra, on=["version", "docID"])


def fmt_ci(v: float, lo: float, hi: float, d: int = 4) -> str:
    if pd.isna(lo):
        return f"{v:+.{d}f}(差 0)" if not pd.isna(v) else "差 0"
    return f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def verdict(better: str, v: float, lo: float, hi: float) -> int:
    """+1 = 有意に優れる、−1 = 有意に劣る、0 = どちらでもない。"""
    if pd.isna(lo) or (lo <= 0 <= hi):
        return 0
    good = v > 0 if better == "大" else v < 0
    return 1 if good else -1


def transport_health() -> list[str]:
    lines = ["| D | 設計 | 脱落した始点(全体) | 輸送された質量 | 実効サポート 中央値 [四分位] "
             "| 異なる対応先 | 対応先の RE < 0 | 総資産比 中央値 | 秒 |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    pers = PLANS["38"].parent / "plans_persistence"
    for d, p, designs, pre in (*((d, p, DESIGNS, "") for d, p in PLANS.items()),
                               ("38", pers, ("nearest_na0", "unb0.1_na0"), "持続性 ")):
        t = pd.read_csv(p / "transport_summary.csv").set_index("design")
        for des in designs:
            r = t.loc[des]
            lines.append(
                f"| {d} | {pre}{des} | {int(r['n_dropped_sources'])} | {r['transported_mass']:.3f} "
                f"| {r['support_median']:.1f} [{r['support_q1']:.1f}, {r['support_q3']:.1f}] "
                f"| {int(r['unique_targets'])} | {int(r['n_RE_negative_targets'])} "
                f"| {r['asset_ratio_median']:.2f} | {r['seconds']:.0f} |")
    return lines


def fold_drops() -> list[str]:
    """fold ごとの訓練計画で質量 0 になった始点の数(生成で学習から除いた社数)。"""
    lines = ["| D | 設計 | fold 0〜4 の訓練始点で質量 0 | 同 テスト始点 |", "|---|---|---|---|"]
    for d, p in PLANS.items():
        for des in DESIGNS:
            z = np.load(p / f"plans_{des}.npz")
            tr = [int((z[f"plan_train_{k}"].sum(axis=1) <= 1e-12).sum()) for k in range(5)]
            te = [int((z[f"plan_test_{k}"].sum(axis=1) <= 1e-12).sum()) for k in range(5)]
            if any(tr) or any(te):
                lines.append(f"| {d} | {des} | {'・'.join(map(str, tr))} "
                             f"| {'・'.join(map(str, te))} |")
    return lines


def section_c(rng: np.random.Generator) -> tuple[list[str], pd.DataFrame]:
    rows, dels, refs, viol = load("c2")
    comp = comp_level(rows)
    out = ["## C.3 輸送の方式 × D(class 2、55 社、等重み)", "",
           "指標は企業ごとの seed の中央値の中央値。proximity は各 D の空間の L2、"
           "proximity(32)は共通の 32 科目に集約した空間の L2(D の比較用)。"
           "順位相関は回復 15 社の実データ(2 年)との符号付き順位相関(各 D の科目)。", "",
           "| D | 設計 | 違反(行) | 純資産の改善 | 解消 | \\|総資産変化\\| | proximity "
           "| proximity(32) | sparsity | 順位相関 [95%] |",
           "|---|---|---:|---:|---|---:|---:|---:|---:|---|"]
    table = []
    for d in ROOT_DIR:
        for des in DESIGNS:
            v = vkey(d, des)
            g = comp[comp["version"] == v]
            if g.empty:
                continue
            ins = g[g["insolvent_start"]]
            gen = wide(dels, v, "native")
            ref = ref_of(refs, d, "native")
            c = corr(gen, ref)
            lo, hi, _, _ = boot(gen, ref, None, None, rng)
            nv = int(viol[viol["setting"] == v]["rows"].sum())
            table.append({"D": d, "design": des, "violation_rows": nv, "corr": c})
            out.append(
                f"| {d} | {des} | {nv} | {g['net_asset_improvement'].median():.3f} "
                f"| {int((ins['insolvency_resolved'] >= 2 / 3).sum())} / {len(ins)} "
                f"| {g['abs_asset_change'].median():.3f} | {g['proximity_l2'].median():.3f} "
                f"| {g['proximity_32'].median():.3f} | {g['sparsity'].median():.0f} "
                f"| {c:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    out += ["", "各設計 − 同じ D の最近傍(差の中央値 [95% CI]):", "",
            "| D | 設計 | 純資産の改善 | \\|総資産変化\\| | proximity(32) |",
            "|---|---|---|---|---|"]
    for d in ROOT_DIR:
        for des in DESIGNS[1:]:
            v, base = vkey(d, des), vkey(d, "nearest_na0")
            if v not in set(comp["version"]):
                continue
            out.append(f"| {d} | {des} | " + " | ".join(
                paired_cell(comp, v, base, m) for m in
                ("net_asset_improvement", "abs_asset_change", "proximity_32")) + " |")
    out += ["", "D の効果(同じ設計の D = 38 − D = 32、差の中央値 [95% CI]):", "",
            "| 設計 | 純資産の改善 | \\|総資産変化\\| | proximity(32) |", "|---|---|---|---|"]
    for des in DESIGNS:
        v, base = vkey("38", des), vkey("32", des)
        if not {v, base} <= set(comp["version"]):
            continue
        out.append(f"| {des} | " + " | ".join(
            paired_cell(comp, v, base, m) for m in
            ("net_asset_improvement", "abs_asset_change", "proximity_32")) + " |")
    return out, pd.DataFrame(table)


def paired_cell(comp: pd.DataFrame, v: str, base: str, metric: str) -> str:
    from fsgen.evaluation.paired import paired_test
    a = comp[comp["version"] == v].set_index("docID")[metric]
    b = comp[comp["version"] == base].set_index("docID")[metric].reindex(a.index)
    r = paired_test(a.to_numpy(), b.to_numpy())
    return fmt_ci(r.median_diff, r.ci_low, r.ci_high)


def section_d(rng: np.random.Generator) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    out: list[str] = []
    judg, corr_rows = [], []
    for cls in ("c2", "c1"):
        rows, dels, refs, viol = load(cls)
        comp = comp_level(rows)
        keep = {vkey(*SETTINGS[i]) for i in SETTINGS}
        comp = comp[comp["version"].isin(keep)]
        ref_note = ("回復 15 社の 2 年" if cls == "c2" else "class 1 → class 0 の 1 年の遷移 68 組")
        out += [f"## D. 6 設定 × {CLS_LABEL[cls]}", "",
                f"順位相関の実データ: {ref_note}。「元」= 各 D の科目、"
                "「32」= 共通の 32 科目に集約。",
                "",
                "| 設定 | 違反(行) | 実行不可能 | 純資産の改善 | 解消 | \\|総資産変化\\| "
                "| sparsity | validity LGBM | proximity(32) | 順位相関(元) [95%] "
                "| 順位相関(32) [95%] | 符号の一致(元) |",
                "|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|"]
        gens: dict[tuple[int, str], pd.DataFrame] = {}
        rfs: dict[tuple[int, str], pd.Series] = {}
        for i, (d, s) in SETTINGS.items():
            v = vkey(d, s)
            g = comp[comp["version"] == v]
            ins = g[g["insolvent_start"]]
            cells = []
            for space in ("native", "core32"):
                gen, ref = wide(dels, v, space), ref_of(refs, d, space)
                gens[(i, space)], rfs[(i, space)] = gen, ref
                c = corr(gen, ref)
                lo, hi, _, _ = boot(gen, ref, None, None, rng)
                cells.append(f"{c:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                corr_rows.append({"cls": cls, "setting": i, "space": space, "corr": c,
                                  "lo": lo, "hi": hi, "agree": agree(gen, ref),
                                  "n_accounts": gen.shape[1]})
            nv = int(viol[viol["setting"] == v]["rows"].sum())
            out.append(
                f"| {i} {SET_LABEL[i]} | {nv} | {int(g['infeasible'].sum())} "
                f"| {g['net_asset_improvement'].median():.3f} "
                f"| {int((ins['insolvency_resolved'] >= 2 / 3).sum())} / {len(ins)} "
                f"| {g['abs_asset_change'].median():.3f} | {g['sparsity'].median():.0f} "
                f"| {g['validity_lightgbm'].median():.3f} | {g['proximity_32'].median():.3f} "
                f"| {cells[0]} | {cells[1]} "
                f"| {agree(gens[(i, 'native')], rfs[(i, 'native')])} / "
                f"{gens[(i, 'native')].shape[1]} |")
        r32 = ref_of(refs, "32", "core32")
        r38 = ref_of(refs, "38", "core32").loc[r32.index]
        out += ["", f"32 科目に集約した実データの参照の D による差(最大の絶対差): "
                f"{float((r32 - r38).abs().max()):.2e}", ""]
        out += ["比較(差の中央値 [95% CI]、順位相関は差 [95%])。判定: + 有意に優れる、"
                "− 有意に劣る、0 どちらでもない", "",
                "| 比較 | 純資産の改善 | \\|総資産変化\\| | 解消率 | 順位相関の差 | 判定 |",
                "|---|---|---|---|---|---|"]
        for a, b, name in (*COMPARISONS, *REFERENCE):
            va, vb = vkey(*SETTINGS[a]), vkey(*SETTINGS[b])
            _summ, pr = summarize(comp, (), (vb, va), vb)
            pr = pr.set_index("metric")
            marks, cells = [], []
            for metric, better in PRIMARY:
                q = pr.loc[metric]
                m, lo, hi = q["median_diff(version-equal)"], q["ci_low"], q["ci_high"]
                cells.append(fmt_ci(m, lo, hi))
                marks.append(verdict(better, m, lo, hi))
            space = "core32" if SETTINGS[a][0] != SETTINGS[b][0] else "native"
            ga, gb = gens[(a, space)], gens[(b, space)].loc[gens[(a, space)].index]
            ca, cb = corr(ga, rfs[(a, space)]), corr(gb, rfs[(b, space)])
            _, _, dlo, dhi = boot(ga, rfs[(a, space)], gb, rfs[(b, space)], rng)
            cells.append(fmt_ci(ca - cb, dlo, dhi, 3) + ("(32)" if space == "core32" else ""))
            marks.append(verdict("大", ca - cb, dlo, dhi))
            wins = sum(1 for x in marks if x > 0)
            loses = sum(1 for x in marks if x < 0)
            win = wins >= 2 and loses == 0
            sym = "".join({1: "+", -1: "−", 0: "0"}[x] for x in marks)
            out.append(f"| {a} − {b}({name}) | {' | '.join(cells)} "
                       f"| {sym} → {'勝ち' if win else '勝ちでない'} |")
            judg.append({"cls": cls, "comparison": f"{a}-{b}", "name": name, "marks": sym,
                         "win": win})
        # 違反の内訳
        vv = viol[viol["setting"].isin(keep)].groupby(["setting", "check"])["rows"].sum()
        out += ["", "違反の内訳: " + ("; ".join(
            f"{k[0]} {k[1]} {int(n)}" for k, n in vv.items()) or "なし"), ""]
        # 解消率のペアの補足、他の指標
        summ, _ = summarize(comp, (), tuple(vkey(*SETTINGS[i]) for i in SETTINGS),
                            vkey(*SETTINGS[1]))
        cols = [m for m, _, _ in CONTINUOUS]
        out += ["その他の指標(企業ごとの seed 中央値の中央値):", "",
                "| 設定 | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for i in SETTINGS:
            r = summ.set_index("version").loc[vkey(*SETTINGS[i])]
            out.append(f"| {i} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " |")
        out.append("")
    return out, pd.DataFrame(judg), pd.DataFrame(corr_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage39"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    lines = ["# ステージ39 C・D: 輸送の方式、細分化、重みの比較", "",
             "## C.2 輸送計画の健全性(全 57 社 → class 0、plan_full)", "",
             "すべての方式が収束した(未収束は例外で止まる。バランス OT は周辺分布の誤差 "
             "< 1e-6 も検証)。",
             "", *transport_health(), "",
             "fold ごとの計画で質量 0 になった始点(不均衡 OT は、訓練始点のそれを学習から除き、"
             "テスト始点は全社を生成した):", "", *fold_drops(), ""]
    c_lines, c_table = section_c(rng)
    d_lines, judg, corr_rows = section_d(rng)
    lines += c_lines + [""] + d_lines
    lines += ["## E. 判定(両 class で勝つ場合だけ採用)", "",
              "| 判定 | class 2 | class 1 | 採用 |", "|---|---|---|---|"]
    for _a, _b, name in COMPARISONS:
        j = judg[judg["name"] == name].set_index("cls")
        both = bool(j.loc["c2", "win"]) and bool(j.loc["c1", "win"])
        lines.append(f"| {name} | {j.loc['c2', 'marks']} | {j.loc['c1', 'marks']} "
                     f"| {'採用' if both else '不採用'} |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    judg.to_csv(args.out / "judgments.csv", index=False)
    corr_rows.to_csv(args.out / "signed_corr.csv", index=False)
    c_table.to_csv(args.out / "transport_designs.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
