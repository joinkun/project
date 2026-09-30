#!/usr/bin/env python
"""ステージ41 B.2・C・D: 不活性制約の比較(scipy のみ)。

入力: ``metrics_stage39.py --gen-dir data/interim/stage41/gen`` の出力
(``data/interim/stage41/metrics``)、生成の parquet、``results/stage41/rates.csv``・
``selection.csv``。
判定の基準は ``results/stage41/criteria.md``
(生成の前に記録)。

    uv run python scripts/compare_stage41.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from compare_stage39 import boot, comp_level, corr, fmt_ci
from run_stage25 import OUTLIERS, summarize

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.evaluation.paired import paired_test

_M = Path("data/interim/stage41/metrics")
_GEN = Path("data/interim/stage41/gen")
_R = Path("results/stage41")
EXCLUDE = {"c2": OUTLIERS, "c1": ("S100VGDJ",)}
CLS_LABEL = {"c2": "class 2(55 社)", "c1": "class 1(131 社)"}
SETTINGS = {"1": "s1_none", "2": "s2_selected", "2b": "s2b_mech", "3": "s3_all"}
SET_LABEL = {"1": "なし(現行)", "2": "規則 S(17 科目)", "2b": "規則 M(4 科目、参考)",
             "3": "全科目(参考)"}
ZERO, NONZERO, STUCK, MANY = 1e-6, 1.001e-3, 0.01, 10
TOL_NA, TOL_DA = -0.02, 0.02


def load_gen(cls: str, setting: str) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in
                   sorted((_GEN / cls / setting).glob("*__seed*.parquet"))])
    return g[(g["method"] == "PCFM") & ~g["docID"].isin(EXCLUDE[cls])].reset_index(drop=True)


def fmt(v: float) -> str:
    return "—" if pd.isna(v) else f"{v:.1%}"


def emergence(g: pd.DataFrame, names: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(行 × 科目の「始点 0」, 行 × 科目の「0 → 非0」)。"""
    x0 = g[[f"x0_{n}" for n in names]].to_numpy()
    x1 = g[[f"x1_{n}" for n in names]].to_numpy()
    z = np.abs(x0) < ZERO
    return (pd.DataFrame(z, columns=list(names)),
            pd.DataFrame(z & (np.abs(x1) > NONZERO), columns=list(names)))


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_R)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    ja = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    rates = pd.read_csv(_R / "rates.csv").set_index("account")
    sel = pd.read_csv(_R / "selection.csv").set_index("account")
    r5 = sel["r5"].fillna(0.0).reindex(list(names)).to_numpy()
    lines = ["# ステージ41 B・C・D: 不活性制約の比較", ""]
    judg: dict[str, dict[str, object]] = {}
    for cls in ("c2", "c1"):
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[~rows["docID"].isin(EXCLUDE[cls])]
        comp = comp_level(rows)
        dels = pd.read_parquet(_M / f"deltas_{cls}.parquet")
        dels = dels[(dels["space"] == "native") & ~dels["docID"].isin(EXCLUDE[cls])]
        refs = pd.read_parquet(_M / "refs.parquet")
        ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index(
            "account")["median"]
        viol = pd.read_csv(_M / f"viol_{cls}.csv", dtype={"docID": str})
        viol = viol[~viol["docID"].isin(EXCLUDE[cls])]
        gens = {k: load_gen(cls, s) for k, s in SETTINGS.items()}
        # ---- 企業ごとの乖離 dev ----
        dev: dict[str, pd.Series] = {}
        em_rate: dict[str, pd.Series] = {}
        for k, g in gens.items():
            z, e = emergence(g, names)
            ecount = pd.Series(e.sum(axis=1).to_numpy(), index=g.index)
            expect = pd.Series(z.to_numpy() @ r5, index=g.index)
            f = pd.DataFrame({"docID": g["docID"], "E": ecount, "Ehat": expect})
            f = f.groupby("docID").median()
            dev[k] = (f["E"] - f["Ehat"]).abs().rename("dev")
            if k == "1":
                ehat = f["Ehat"]
            em_rate[k] = e.sum() / z.sum().where(z.sum() > 0)
            em_rate[f"{k}_n"] = z.sum()
            dev[f"{k}_E"] = f["E"]
        # ---- B.2 実行可能性 ----
        g2 = gens["2"]
        ninact = g2.groupby("docID")["n_inactive"].first()
        prox = comp.set_index(["version", "docID"])["proximity_l2"]
        lines += [f"## {CLS_LABEL[cls]}", "", "### B.2 実行可能性", "",
                  "企業ごとの不活性科目の数(設定 2): 中央値 "
                  f"{ninact.median():.0f}、四分位 [{ninact.quantile(0.25):.0f}, "
                  f"{ninact.quantile(0.75):.0f}]、最大 {ninact.max()}、"
                  f"10 科目以上 {int((ninact >= MANY).sum())} 社。分布: "
                  + "、".join(f"{int(v)}: {int(c)}" for v, c in ninact.value_counts().sort_index()
                              .items()), ""]
        n3 = gens["3"].groupby("docID")["n_inactive"].first()
        lines.append(f"設定 3(全科目)の不活性科目の数: 中央値 {n3.median():.0f}、最大 {n3.max()}。")
        many = ninact[ninact >= MANY].index
        if len(many):
            pm = prox.loc[SETTINGS["2"]].reindex(many)
            p1 = prox.loc[SETTINGS["1"]].reindex(many)
            lines.append(f"10 科目以上の {len(many)} 社の proximity(seed の中央値)の中央値: 設定 2 "
                         f"{pm.median():.3f}(設定 1 {p1.median():.3f})、0.01 未満 "
                         f"{int((pm < STUCK).sum())} 社。")
        lines += ["", "| 設定 | 実行不可能(行) | ほとんど動けない企業 | 15 項目の違反(行) | 内訳 |",
                  "|---|---:|---:|---:|---|"]
        for k, s in SETTINGS.items():
            g = gens[k]
            st = int((prox.loc[s] < STUCK).sum())
            vv = viol[viol["setting"] == s]
            det = "、".join(f"{c} {int(n)}" for c, n in vv.groupby("check")["rows"].sum().items())
            lines.append(f"| {k} {SET_LABEL[k]} | {int((g['status'] == 'infeasible').sum())} "
                         f"/ {len(g)} | {st} | {int(vv['rows'].sum())} | {det or 'なし'} |")
        # アクティブな制約
        acts = {}
        for k, g in gens.items():
            a = g["active"].fillna("").str.split(";").explode()
            acts[k] = a[a != ""].value_counts() / len(g)
        keys = sorted(set().union(*[set(v.index) for v in acts.values()]),
                      key=lambda x: (-max(acts[k].get(x, 0.0) for k in SETTINGS), x))
        lines += ["", "アクティブな制約(最終ステップ、行の割合。固定(等式)は含まない):", "",
                  "| 制約 | " + " | ".join(f"設定 {k}" for k in SETTINGS) + " |",
                  "|---|" + "---:|" * len(SETTINGS)]
        for x in keys:
            lines.append(f"| {x} | " + " | ".join(f"{acts[k].get(x, 0.0):.0%}" for k in SETTINGS)
                         + " |")
        # ---- C.3 科目別 ----
        lines += ["", "### C.3 「0 → 非0」の割合(科目別。生成は 5 年、seed × 企業の行)", "",
                  "| 科目 | 実データ 1 年 class 1+2 | 5 年換算 r5 | 2 年 class 1+2 | 規則 S | "
                  + " | ".join(f"設定 {k}" for k in SETTINGS) + " |",
                  "|---|---:|---:|---:|---|" + "---:|" * len(SETTINGS)]
        for n in names:
            rr = rates.loc[n]
            if em_rate["1_n"][n] == 0:
                continue
            cells = [f"{em_rate[k][n]:.0%}({int(round(em_rate[k][n] * em_rate[f'{k}_n'][n]))}"
                     f"/{int(em_rate[f'{k}_n'][n])})" for k in SETTINGS]
            lines.append(f"| {ja[n]} | {fmt(rr['rate_c12'])} | {fmt(sel.loc[n, 'r5'])} "
                         f"| {fmt(rr['rate_2y_c12'])} | {'課す' if sel.loc[n, 'rule_S'] else ''} | "
                         + " | ".join(cells) + " |")
        e_tot = {k: float(dev[f"{k}_E"].mean()) for k in SETTINGS}
        lines += ["", "企業あたりの「0 → 非0」の科目数(seed の中央値の平均): "
                  + "、".join(f"設定 {k} {e_tot[k]:.2f}" for k in SETTINGS)
                  + f"。実データの 5 年換算の期待値 Ê の平均 {ehat.mean():.2f}。", ""]
        # ---- 主指標・副指標 ----
        lines += ["### 指標(企業ごとの seed 中央値の中央値)", "",
                  "| 設定 | dev(乖離) | 純資産の改善 | 解消 | \\|総資産変化\\| | sparsity "
                  "| validity LGBM | proximity | 順位相関 [95%] |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---|"]
        gw: dict[str, pd.DataFrame] = {}
        for k, s in SETTINGS.items():
            g = comp[comp["version"] == s]
            ins = g[g["insolvent_start"]]
            w = dels[dels["setting"] == s].pivot(index="docID", columns="account",
                                                   values="delta")[list(names)]
            gw[k] = w
            c = corr(w, ref)
            lo, hi, _, _ = boot(w, ref, None, None, rng)
            lines.append(
                f"| {k} {SET_LABEL[k]} | {dev[k].median():.2f} "
                f"| {g['net_asset_improvement'].median():.3f} "
                f"| {int((ins['insolvency_resolved'] >= 2 / 3).sum())} / {len(ins)} "
                f"| {g['abs_asset_change'].median():.3f} | {g['sparsity'].median():.0f} "
                f"| {g['validity_lightgbm'].median():.3f} | {g['proximity_l2'].median():.3f} "
                f"| {c:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        lines += ["", "各設定 − 設定 1(差の中央値 [95% CI]、順位相関は差 [95%]):", "",
                  "| 設定 | dev | 純資産の改善 | 解消率 | \\|総資産変化\\| | sparsity | validity "
                  "| proximity | 順位相関 |", "|---|---|---|---|---|---|---|---|---|"]
        base = SETTINGS["1"]
        for k in ("2", "2b", "3"):
            s = SETTINGS[k]
            _su, pr = summarize(comp, (), (base, s), base)
            pr = pr.set_index("metric")

            def cell(m: str, pr: pd.DataFrame = pr) -> str:
                q = pr.loc[m]
                return fmt_ci(q["median_diff(version-equal)"], q["ci_low"], q["ci_high"])

            d = paired_test(dev[k].to_numpy(), dev["1"].reindex(dev[k].index).to_numpy())
            b = gw["1"].loc[gw[k].index]
            _, _, dlo, dhi = boot(gw[k], ref, b, ref, rng)
            dc = corr(gw[k], ref) - corr(b, ref)
            lines.append(f"| {k} | {fmt_ci(d.median_diff, d.ci_low, d.ci_high, 2)} "
                         f"| {cell('純資産の改善')} "
                         f"| {cell('債務超過の解消率(社ごとの 3 seed 平均)')} "
                         f"| {cell('|総資産変化|')} | {cell('sparsity', )} "
                         f"| {cell('validity LightGBM')} | {cell('等重み proximity')} "
                         f"| {fmt_ci(dc, dlo, dhi, 3)} |")
            q_na = pr.loc["純資産の改善"]
            q_da = pr.loc["|総資産変化|"]
            judg.setdefault(k, {})[cls] = {
                "viol": int(viol[viol["setting"] == s]["rows"].sum()),
                "dev_sig_lower": bool(d.ci_high < 0),
                "na": float(q_na["median_diff(version-equal)"]),
                "da": float(q_da["median_diff(version-equal)"]),
            }
        lines.append("")
    # ---- D.1 ----
    lines += ["## D.1 判定(設定 2)", "",
              "| 基準 | class 2 | class 1 | 満たす |", "|---|---|---|---|"]
    j = judg["2"]
    c2, c1 = j["c2"], j["c1"]
    assert isinstance(c2, dict) and isinstance(c1, dict)
    ok1 = c2["viol"] == 0 and c1["viol"] == 0
    ok2 = bool(c2["dev_sig_lower"]) and bool(c1["dev_sig_lower"])
    ok3 = all(float(x["na"]) >= TOL_NA and float(x["da"]) <= TOL_DA for x in (c2, c1))
    lines += [f"| 1. 15 項目の違反 0 | {c2['viol']} | {c1['viol']} "
              f"| {'はい' if ok1 else 'いいえ'} |",
              f"| 2. dev が有意に下がる | {c2['dev_sig_lower']} | {c1['dev_sig_lower']} "
              f"| {'はい' if ok2 else 'いいえ'} |",
              f"| 3. 純資産の改善の差 ≥ {TOL_NA} かつ \\|総資産変化\\| の差 ≤ +{TOL_DA} "
              f"| {c2['na']:+.4f} / {c2['da']:+.4f} | {c1['na']:+.4f} / {c1['da']:+.4f} "
              f"| {'はい' if ok3 else 'いいえ(判断を保留)'} |",
              "", "結論: " + ("設定 2 を採用" if ok1 and ok2 and ok3
                              else "設定 2 は基準を満たさない")]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
