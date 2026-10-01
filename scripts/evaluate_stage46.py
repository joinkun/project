#!/usr/bin/env python
"""ステージ46: 資本金の下限(α = 0 / 0.00199 / 0.5)の比較と、払込資本の方向の診断。

入力: ``metrics_stage39.py --gen-dir data/interim/stage46/gen`` の出力。判定の基準は
``results/stage46/criteria.md``(生成の前に記録)。

    uv run python scripts/evaluate_stage46.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_stage39 import boot, comp_level, corr, fmt_ci
from criteria_stage31 import wilson
from evaluate_stage34 import special_reason
from evaluate_stage45 import annual, firm_frac
from recovery_stage30 import changes
from run_stage25 import summarize

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

_M = Path("data/interim/stage46/metrics")
_GEN = Path("data/interim/stage46/gen")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
OUTLIERS = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
CLS = {"c2": "class 2(55 社)", "c1": "class 1(131 社)"}
SETS = {"a0": "1 α = 0(現行)", "ap1": "2 α = 0.00199(資本金)", "a05": "3 α = 0.5(両方)"}
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
THR, HOLD, TOL = 0.001, 1e-6, 1e-4
PRIMARY = (("純資産の改善", "大"), ("債務超過の解消率(社ごとの 3 seed 平均)", "大"),
           ("|総資産変化|", "小"), ("sparsity", "小"))


def load_gen(cls: str, s: str) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in
                   sorted((_GEN / cls / s).glob("*__seed*.parquet"))])
    return g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIERS[cls])].reset_index(drop=True)


def depth(g: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """資本金の減らし方(企業、3 seed の 2/3 以上): 半減、1% 未満、0。"""
    x0, x1 = g[f"x0_{_CS}"], g[f"x1_{_CS}"]
    held = x0 > HOLD
    doc = g.loc[held, "docID"]

    def below(r: float) -> pd.Series:
        out: pd.Series = (x1 < r * x0 - np.maximum(r * x0 * TOL, 1e-6))[held]
        return out

    zero = (x1.abs() <= 1e-6)[held]
    return {"half": firm_frac(below(0.5), doc), "lt1": firm_frac(below(0.01), doc),
            "zero": firm_frac(zero, doc)}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage46"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    summary: dict[str, object] = {}
    # 実データ: class 1・2(2022 年度、特殊業種を除く)の 2 年
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wl = fy.pivot(index="edinetCode", columns="year", values="label")
    pan = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a23 = pan[pan["t"] == 2023].set_index("edinetCode")
    b24 = pan[pan["t"] == 2024].set_index("edinetCode")
    both = [c for c in a23.index.intersection(b24.index) if wl.loc[c, 2022] in (1, 2)]
    reason = special_reason(fy, both)
    press = [c for c in both if reason.get(c, "") == ""]
    y0 = a23.loc[press, f"prev_{_CS}"].fillna(0).to_numpy()
    y1 = b24.loc[press, f"cur_{_CS}"].fillna(0).to_numpy()
    h = y0 > 0
    real = {"half": (int((y1[h] < 0.5 * y0[h]).sum()), int(h.sum())),
            "lt1": (int((y1[h] < 0.01 * y0[h]).sum()), int(h.sum())),
            "zero": (int((y1[h] == 0).sum()), int(h.sum()))}

    def pct(k: int, m: int) -> str:
        lo, hi = wilson(k, m)
        return f"{k}/{m}({k / m:.1%}、[{lo:.0%}, {hi:.0%}])"

    lines = ["# ステージ46: 資本金の下限の比較と払込資本の方向", "",
             "設定 1 = ステージ45 の最終設定(α = 0)。設定 2 = 資本金だけに α = 0.00199(実データ"
             "の p1)。"
             "設定 3 = α = 0.5(資本金・資本剰余金、ステージ43 以前の定義)。他は同じ。", ""]
    judg: dict[str, dict[str, object]] = {}
    for cls in ("c2", "c1"):
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[~rows["docID"].isin(OUTLIERS[cls])]
        comp = comp_level(rows)
        viol = pd.read_csv(_M / f"viol_{cls}.csv", dtype={"docID": str})
        viol = viol[~viol["docID"].isin(OUTLIERS[cls])]
        dels = pd.read_parquet(_M / f"deltas_{cls}.parquet")
        dels = dels[(dels["space"] == "native") & ~dels["docID"].isin(OUTLIERS[cls])]
        refs = pd.read_parquet(_M / "refs.parquet")
        ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index(
            "account")["median"]
        summ, _ = summarize(comp, (), tuple(SETS), "a0")
        s_ = summ.set_index("version")
        w = {s: dels[dels["setting"] == s].pivot(index="docID", columns="account",
                                                 values="delta")[list(names)] for s in SETS}
        dep = {s: depth(load_gen(cls, s)) for s in SETS}
        lines += [f"## B.2 {CLS[cls]}", "",
                  "| 設定 | 15 項目の違反 | 純資産の改善 | 解消 | \\|総資産変化\\| | sparsity |"
                  " 順位相関 "
                  "| 半減させた企業 | 1% 未満まで減らした企業 | 0 にした企業 |",
                  "|---|---:|---:|---|---:|---:|---:|---|---|---|"]
        for s, lab in SETS.items():
            r = s_.loc[s]
            d = dep[s]
            lines.append(
                f"| {lab} | {int(viol[viol['setting'] == s]['rows'].sum())} "
                f"| {r['net_asset_improvement']:.3f} "
                f"| {int(r['resolved_firms(>=2/3 seeds)'])}/{int(r['insolvent_start'])} "
                f"| {r['abs_asset_change']:.3f} | {r['sparsity']:.0f} | {corr(w[s], ref):+.3f} "
                f"| {pct(*d['half'])} | {pct(*d['lt1'])} | {pct(*d['zero'])} |")
        lines += [f"| 実データ(class 1・2、2 年) | | | | | | | {pct(*real['half'])} "
                  f"| {pct(*real['lt1'])} | {pct(*real['zero'])} |", "",
                  "設定 − 設定 1(差の中央値 [95% CI]、順位相関は差 [95%]):", "",
                  "| 設定 | 純資産の改善 | 解消率 | \\|総資産変化\\| | sparsity | 順位相関 | 悪"
                  "化した主指標 |",
                  "|---|---|---|---|---|---|---|"]
        for s in ("ap1", "a05"):
            _su, pr = summarize(comp, (), ("a0", s), "a0")
            pr = pr.set_index("metric")
            cells, bad = [], []
            for m, better in PRIMARY:
                q = pr.loc[m]
                v, lo, hi = q["median_diff(version-equal)"], q["ci_low"], q["ci_high"]
                cells.append(fmt_ci(v, lo, hi))
                if not pd.isna(lo) and ((hi < 0) if better == "大" else (lo > 0)):
                    bad.append(m)
            b_ = w["a0"].loc[w[s].index]
            _, _, dlo, dhi = boot(w[s], ref, b_, ref, rng)
            dc = corr(w[s], ref) - corr(b_, ref)
            lines.append(f"| {SETS[s]} | {' | '.join(cells)} | {fmt_ci(dc, dlo, dhi, 3)} "
                         f"| {'、'.join(bad) or 'なし'} |")
            if s == "ap1":
                dev = {}
                for k in ("half", "lt1"):
                    rr = real[k][0] / real[k][1]
                    dev[k] = (abs(dep["a0"][k][0] / dep["a0"][k][1] - rr),
                              abs(dep["ap1"][k][0] / dep["ap1"][k][1] - rr))
                judg[cls] = {
                    "viol": int(viol[viol["setting"] == "ap1"]["rows"].sum()),
                    "depth_ok": (dev["half"][1] <= dev["half"][0] + 1e-12
                                 and dev["lt1"][1] <= dev["lt1"][0] + 1e-12
                                 and dev["lt1"][1] < dev["lt1"][0] - 1e-12),
                    "dev": dev, "primary_worse": bad}
        lines.append("")
    lines += ["## B.3 判定(設定 2)", "",
              "| 基準 | class 2 | class 1 | 満たす |", "|---|---|---|---|"]
    j2, j1 = judg["c2"], judg["c1"]
    ok1 = j2["viol"] == 0 and j1["viol"] == 0
    ok2 = bool(j2["depth_ok"]) and bool(j1["depth_ok"])
    ok3 = not j2["primary_worse"] and not j1["primary_worse"]

    def devs(j: dict[str, object]) -> str:
        d = j["dev"]
        assert isinstance(d, dict)
        return (f"半減 {d['half'][0]:.3f} → {d['half'][1]:.3f}、"
                f"1% 未満 {d['lt1'][0]:.3f} → {d['lt1'][1]:.3f}")

    lines += [f"| 1. 15 項目の違反 0 | {j2['viol']} | {j1['viol']} |"
              f" {'はい' if ok1 else 'いいえ'} |",
              f"| 2. 減資の深さが実データに近づく(実データとの差: 設定 1 → 設定 2) | {devs(j2)} "
              f"| {devs(j1)} | {'はい' if ok2 else 'いいえ'} |",
              f"| 3. 主指標が悪化しない | {'、'.join(j2['primary_worse']) or 'なし'}"  # type: ignore[arg-type]
              f" | {'、'.join(j1['primary_worse']) or 'なし'}"  # type: ignore[arg-type]
              f" | {'はい' if ok3 else 'いいえ'} |", "",
              "結論: " + ("**設定 2(α = 0.00199)を採用する**" if ok1 and ok2 and ok3
                         else "**設定 2 は基準を満たさない。設定 1(α = 0)のまま**"), ""]
    summary["judgment"] = {"ok1": ok1, "ok2": ok2, "ok3": ok3}

    # ---------------- C.2 払込資本の方向 ----------------
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    rec2 = rf[(rf["reason"].fillna("") == "") & rf["in_panel"].astype(bool) & (rf["group"] == "Y")]
    c1 = wl[wl[2022] == 1].copy()
    c1["reason"] = special_reason(fy, c1.index.tolist())
    c1 = c1.join(changes(_PANEL.parent, c1.index.tolist(), names), how="left")
    c1 = c1[(c1["reason"] == "") & c1["n_years"].notna()]
    rec1 = c1[wl.loc[c1.index, 2024] == 0]
    real_paid = pd.concat([rec2["d_paid"], rec1["d_paid"]]).dropna()

    def dirs(v: pd.Series) -> str:
        return (f"{(v > THR).mean():.0%} / {(v < -THR).mean():.0%} / {v.median():+.4f}")

    lines += ["## C.2 払込資本(資本金 + 資本剰余金)の方向(年率)", "",
              "増えた企業の割合 / 減った企業の割合 / 変化の中央値(年率、閾値 0.001)。"
              f"実データは再建群 {len(real_paid)} 社(class 2 → 注記なし"
              f" {rec2['d_paid'].notna().sum()} 社 + "
              f"class 1 → class 0 {rec1['d_paid'].notna().sum()} 社)。", "",
              "| 設定 | class 2 | class 1 |", "|---|---|---|"]
    for s, lab in SETS.items():
        cells = [dirs(annual(load_gen(c, s), _CS) + annual(load_gen(c, s), _CSUR))
                 for c in ("c2", "c1")]
        lines.append(f"| {lab} | {cells[0]} | {cells[1]} |")
    lines += [f"| 実データ(再建群) | {dirs(rec2['d_paid'].dropna())}(15 社) "
              f"| {dirs(rec1['d_paid'].dropna())}(57 社) |",
              f"| 実データ(再建群 合計) | {dirs(real_paid)} | |", ""]

    # ---------------- C.3 原因の検討 ----------------
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str}).set_index(
        "docID")
    paid_ratio = (agg[_CS].fillna(0) + agg[_CSUR].fillna(0)) / agg["total_assets"]
    ft = pd.read_csv("data/interim/stage28/plans_equal/full_targets_nearest_na0.csv",
                     dtype={"source_docID": str, "target_docID": str})
    ft = ft[~ft["source_docID"].isin(OUTLIERS["c2"])]
    src = paid_ratio.reindex(ft["source_docID"]).to_numpy()
    tgt = paid_ratio.reindex(ft["target_docID"]).to_numpy()
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).set_index("docID")
    c0 = [d for d in agg.index if d in lab.index and str(lab.loc[d, "label"]) == "0"]
    c1s = load_gen("c1", "a0").groupby("docID").first()
    c1_ratio = (c1s[f"x0_{_CS}"] + c1s[f"x0_{_CSUR}"]).to_numpy()  # 始点の総資産で割った値
    lines += ["## C.3 原因の検討: 始点と目標の払込資本 / 総資産", "",
              "| 項目 | 中央値 |", "|---|---:|",
              f"| 始点(class 2、{len(src)} 社) | {np.nanmedian(src):.3f} |",
              f"| 目標(各始点の最近傍の class 0 企業) | {np.nanmedian(tgt):.3f} |",
              f"| class 0 全体({len(c0):,} 書類) | {paid_ratio.reindex(c0).median():.3f} |",
              f"| 始点(class 1、131 社) | {np.median(c1_ratio):.3f} |", "",
              f"目標の比率が始点より高い企業: {int(np.sum(tgt > src))} / {len(src)}"
              f"({np.mean(tgt > src):.0%})。", ""]
    summary["c3"] = {"src_median": float(np.nanmedian(src)), "tgt_median": float(np.nanmedian(tgt)),
                     "share_tgt_higher": float(np.mean(tgt > src))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
