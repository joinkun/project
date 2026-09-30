#!/usr/bin/env python
"""ステージ41 A.1: 科目別の「始点 0 → 終点 非0」の割合(実データと現行の生成)。

実データは推定用パネル(ステージ23、3,657 社 × 3 遷移、同じ書類の前期末 → 当期末、1 年)。
- 始点 0: |前期末| / 前期総資産 < ``--zero``(1e-6)
- 終点 非0: |当期末| / 当期総資産 > ``--thr``(1.001e-3)。残余の下限 −1e-3 は
  「0 から丸め許容を引いた値」
  なので、生成で下限にちょうど張り付いた値(ソルバ誤差で 1e-3 をわずかに超える)を非0 と数えないよう、
  1e-3 に 0.1% の余裕を加えた。参考に > 1e-6 も出す
- class は始点の年度のラベル(ステージ30 の年度別ラベル。2021 年度のラベルはないので、
  class 別は 2 遷移)
- 2 年: 2022 年度末が 0 で 2024 年度末が非0(2022 年度のラベル別)
- 区間は Wilson の 95%(企業の重複を無視した組の単位)

生成(現行の最終設定: ステージ31 の class 2、ステージ34 の class 1)は、始点が 0 の行(seed × 企業、
外れ値 3 社を除く)のうち終点が ``--thr`` を超える割合(5 年)。

出力: ``--out`` に ``rates.csv``・``rates.md``。

    uv run python scripts/inactive_rates_stage41.py --out results/stage41
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

OUTLIERS = ("S100TSS9", "S100TWFQ", "S100VGDJ")
GEN = {"c2": Path("data/interim/stage31/r3_cap076"), "c1": Path("data/interim/stage34/class1")}
_Z = 1.96


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + _Z**2 / n
    c = (p + _Z**2 / (2 * n)) / den
    h = _Z * np.sqrt(p * (1 - p) / n + _Z**2 / (4 * n * n)) / den
    return float(c - h), float(c + h)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zero", type=float, default=1e-6)
    ap.add_argument("--thr", type=float, default=1.001e-3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    ja = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    p = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})
    p = p[(p["prev_total_assets"] > 0) & (p["cur_total_assets"] > 0)].reset_index(drop=True)
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    lp = np.array([lab.get((c, t - 1), np.nan) for c, t in zip(p["edinetCode"], p["t"],
                                                              strict=True)])
    a = p[p["t"] == 2023].set_index("edinetCode")
    b = p[p["t"] == 2024].set_index("edinetCode")
    both = a.index.intersection(b.index)
    lab22 = np.array([lab.get((c, 2022), np.nan) for c in both])
    gens = {}
    for cls, d in GEN.items():
        g = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("nearest_na0__seed*.parquet"))])
        gens[cls] = g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIERS)]
    rows = []
    for n in names:
        x0 = p[f"prev_{n}"].fillna(0.0).to_numpy() / p["prev_total_assets"].to_numpy()
        x1 = p[f"cur_{n}"].fillna(0.0).to_numpy() / p["cur_total_assets"].to_numpy()
        z = np.abs(x0) < args.zero
        e = z & (np.abs(x1) > args.thr)
        e6 = z & (np.abs(x1) > args.zero)
        row: dict[str, object] = {"account": n, "ja": ja.get(n, n),
                                  "n_zero_all": int(z.sum()), "k_all": int(e.sum()),
                                  "k_all_1e6": int(e6.sum())}
        for key, m in (("c0", lp == 0), ("c1", lp == 1), ("c2", lp == 2),
                       ("c12", np.isin(lp, [1, 2]))):
            row[f"n_zero_{key}"] = int((z & m).sum())
            row[f"k_{key}"] = int((e & m).sum())
        lo, hi = wilson(int(row["k_c12"]), int(row["n_zero_c12"]))  # type: ignore[call-overload]
        row["c12_lo"], row["c12_hi"] = lo, hi
        p0 = a.loc[both, f"prev_{n}"].fillna(0.0) / a.loc[both, "prev_total_assets"]
        c2y = b.loc[both, f"cur_{n}"].fillna(0.0) / b.loc[both, "cur_total_assets"]
        z2 = (np.abs(p0.to_numpy()) < args.zero) & np.isin(lab22, [1, 2])
        row["n_zero_2y_c12"] = int(z2.sum())
        row["k_2y_c12"] = int((z2 & (np.abs(c2y.to_numpy()) > args.thr)).sum())
        for cls, g in gens.items():
            g0 = g[f"x0_{n}"].to_numpy()  # 生成は始点の総資産で割った値(tilde 空間)
            g1 = g[f"x1_{n}"].to_numpy()
            zg = np.abs(g0) < args.zero
            row[f"n_zero_gen_{cls}"] = int(zg.sum())
            row[f"k_gen_{cls}"] = int((zg & (np.abs(g1) > args.thr)).sum())
        rows.append(row)
    t = pd.DataFrame(rows)

    def r(k: str, n: str) -> pd.Series:
        return t[k] / t[n].where(t[n] > 0)

    t["rate_all"] = r("k_all", "n_zero_all")
    for key in ("c0", "c1", "c2", "c12"):
        t[f"rate_{key}"] = r(f"k_{key}", f"n_zero_{key}")
    t["rate_2y_c12"] = r("k_2y_c12", "n_zero_2y_c12")
    t["rate_all_1e6"] = r("k_all_1e6", "n_zero_all")
    for cls in GEN:
        t[f"rate_gen_{cls}"] = r(f"k_gen_{cls}", f"n_zero_gen_{cls}")
    args.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out / "rates.csv", index=False)

    def f(v: float) -> str:
        return "—" if pd.isna(v) else f"{v:.1%}"

    lines = ["# ステージ41 A.1: 科目別の「始点 0 → 終点 非0」の割合", "",
             f"実データ: 推定用パネル {p['edinetCode'].nunique():,} 社・{len(p):,} 組(1 年)。"
             f"始点 0 は \\|x\\|/A < {args.zero:g}、非0 は \\|x\\|/A > {args.thr:g}。"
             "class は始点の年度のラベル(2 遷移)。区間は class 1+2 の Wilson 95%。"
             "生成は現行の最終設定(5 年、seed × 企業の行)。", "",
             "| 科目 | 全体 | class 0 | class 1 | class 2 | class 1+2 [95%] | 2 年 class 1+2 "
             "| 全体(非0 > 1e-6) | 生成 class 2 | 生成 class 1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in t.to_dict("records"):
        lines.append(
            f"| {x['ja']} | {f(x['rate_all'])}({x['k_all']}/{x['n_zero_all']}) "
            f"| {f(x['rate_c0'])}({x['k_c0']}/{x['n_zero_c0']}) "
            f"| {f(x['rate_c1'])}({x['k_c1']}/{x['n_zero_c1']}) "
            f"| {f(x['rate_c2'])}({x['k_c2']}/{x['n_zero_c2']}) "
            f"| {f(x['rate_c12'])}({x['k_c12']}/{x['n_zero_c12']}) "
            f"[{f(x['c12_lo'])}, {f(x['c12_hi'])}] "
            f"| {f(x['rate_2y_c12'])}({x['k_2y_c12']}/{x['n_zero_2y_c12']}) "
            f"| {f(x['rate_all_1e6'])} "
            f"| {f(x['rate_gen_c2'])}({x['k_gen_c2']}/{x['n_zero_gen_c2']}) "
            f"| {f(x['rate_gen_c1'])}({x['k_gen_c1']}/{x['n_zero_gen_c1']}) |")
    (args.out / "rates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
