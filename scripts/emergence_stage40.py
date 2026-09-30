#!/usr/bin/env python
"""ステージ40 A.4・B.4: 「始点で 0 の科目が正になる」割合(科目別)を実データと生成で比べる。

作業ディレクトリのルートの canonical と推定用パネル(``data/interim/stage23/panel``)を使う。

- 実データ 1 年: 同じ書類の前期末が 0 で、当期末が総資産比 ``--thr`` を超える組の割合
  (全体と、前年度のラベルが class 1・2 の組)
- 実データ 2 年: 2022 年度末が 0 で 2024 年度末が ``--thr`` を超える企業の割合(class 1・2)
- 生成: 始点が 0 の行(seed × 企業、外れ値 3 社を除く)のうち、終点が ``--thr`` を超える割合
  (生成は 5 年。実データの 1・2 年とは期間が違う)

``--gen`` は ``名前=ディレクトリ``(``*__seed*.parquet`` の PCFM の行)。負の科目(引当金・自己株式)は
「0 から負」、符号が自由な科目(残余など)は両方向を数える(ステージ41 で修正)。閾値の既定は 1.001e-3
(残余の下限 −1e-3 にちょうど張り付いた値を数えない。ステージ41 で修正)。

    uv run python scripts/emergence_stage40.py \\
        --gen c2_39=data/interim/stage39/gen/c2/nearest_na0 --out results/stage40/emergence.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

OUTLIERS = ("S100TSS9", "S100TWFQ", "S100VGDJ")
_ZERO = 1e-9


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--thr", type=float, default=1.001e-3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    sign = {b.name: b.sign for b in spec.basis}
    ja = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    p = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})
    p = p[(p["prev_total_assets"] > 0) & (p["cur_total_assets"] > 0)].copy()
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    p["label_prev"] = [lab.get((c, t - 1), np.nan) for c, t in zip(p["edinetCode"], p["t"],
                                                                   strict=True)]
    press = p["label_prev"].isin([1, 2]).to_numpy()

    def emerged(prev: np.ndarray, cur: np.ndarray, n: str) -> np.ndarray:
        # ステージ41 で修正: 符号が自由な科目(残余など)は両方向を数える。以前は正の向きだけで、
        # 評価・換算差額等その他の 0 → 負(生成の 85〜98%)を見落としていた
        sg = sign.get(n)
        out: np.ndarray = (-cur > args.thr if sg == "nonpos" else cur > args.thr if sg == "nonneg"
                           else np.abs(cur) > args.thr)
        return out

    gens = {}
    for spec_s in args.gen:
        key, d = spec_s.split("=", 1)
        g = pd.concat([pd.read_parquet(f) for f in sorted(Path(d).glob("*__seed*.parquet"))])
        gens[key] = g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIERS)]
    # 2 年: 2022 年度末(2023 年度の書類の前期末)と 2024 年度末
    a = p[p["t"] == 2023].set_index("edinetCode")
    b = p[p["t"] == 2024].set_index("edinetCode")
    both = a.index.intersection(b.index)
    lab22 = np.array([lab.get((c, 2022), np.nan) for c in both])
    press2 = np.isin(lab22, [1, 2])
    rows = []
    for n in names:
        prev = p[f"prev_{n}"].fillna(0.0).to_numpy()
        cur = p[f"cur_{n}"].fillna(0.0).to_numpy() / p["cur_total_assets"].to_numpy()
        z = np.abs(prev) <= _ZERO
        e = emerged(prev, cur, n) & z
        p0 = a.loc[both, f"prev_{n}"].fillna(0.0).to_numpy()
        c2 = (b.loc[both, f"cur_{n}"].fillna(0.0) / b.loc[both, "cur_total_assets"]).to_numpy()
        z2 = (np.abs(p0) <= _ZERO) & press2
        e2 = emerged(p0, c2, n) & z2
        row: dict[str, object] = {
            "account": n, "ja": ja.get(n, n),
            "real1_all": f"{int(e.sum())}/{int(z.sum())}",
            "real1_all_rate": e.sum() / z.sum() if z.sum() else np.nan,
            "real1_press": f"{int((e & press).sum())}/{int((z & press).sum())}",
            "real1_press_rate": (e & press).sum() / (z & press).sum() if (z & press).sum()
            else np.nan,
            "real2_press": f"{int(e2.sum())}/{int(z2.sum())}",
            "real2_press_rate": e2.sum() / z2.sum() if z2.sum() else np.nan,
        }
        for key, g in gens.items():
            x0 = g[f"x0_{n}"].to_numpy()
            x1 = g[f"x1_{n}"].to_numpy()
            zg = np.abs(x0) <= _ZERO
            eg = emerged(x0, x1, n) & zg
            row[f"gen_{key}"] = f"{int(eg.sum())}/{int(zg.sum())}"
            row[f"gen_{key}_rate"] = eg.sum() / zg.sum() if zg.sum() else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(args.out.with_suffix(".csv"), index=False)

    def pct(v: object) -> str:
        return "—" if pd.isna(v) else f"{float(v):.0%}"  # type: ignore[arg-type]

    keys = list(gens)
    lines = [f"# 始点で 0 の科目が 0 でなくなる割合(D = {len(names)}、閾値 総資産比 {args.thr:g})",
             "", "実データは同じ書類の前期末 → 当期末(1 年)と、2022 → 2024 年度末"
             "(2 年、class 1・2)。"
             "生成は 5 年(t = 1)で、seed × 企業の行。外れ値 3 社を除く。", "",
             "| 科目 | 実データ 1 年 全体 | 1 年 class 1・2 | 2 年 class 1・2 | "
             + " | ".join(f"生成 {k}" for k in keys) + " |",
             "|---|---|---|---|" + "---|" * len(keys)]
    for r in t.to_dict("records"):
        cells = [f"{pct(r['real1_all_rate'])}({r['real1_all']})",
                 f"{pct(r['real1_press_rate'])}({r['real1_press']})",
                 f"{pct(r['real2_press_rate'])}({r['real2_press']})"]
        cells += [f"{pct(r[f'gen_{k}_rate'])}({r[f'gen_{k}']})" for k in keys]
        lines.append(f"| {r['ja']} | " + " | ".join(cells) + " |")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
