#!/usr/bin/env python
"""ステージ39 B: 細分化した科目の持続性と、圧力下との比較、持続性の重み。

作業ディレクトリの canonical(``conf/accounts/core.yaml``)と推定用パネル
(``data/interim/stage23/panel``)を使う。
ステージ39 では細分化用のルート(``data/stage39_root``)で実行する。
枠組みはステージ33・34 と同じ(書類内の前期末 → 当期末、x / 総資産)。

- ρ1(全組)、両年非ゼロの組だけの ρ1、ρ2(年度末をつないだ 2 年ラグ)、ゼロの割合
- ρ1 の 95% 区間(企業単位のブートストラップ、1,000 回、seed 0)
- class 0 と class 1+2(各書類の年度のラベル)の ρ1 と、|Δx|/A の p90 の比
- 重み c = (1 − ρ現金) / (1 − ρ)。推定できない科目(両年ゼロ 5 割超、残余、区間の幅 0.3 超)は 1.0

出力: ``--out`` に ``persistence.csv``・``persistence.md``、``--conf`` に ``costs_equal.yaml``・
``costs_persistence.yaml``(基礎科目の数に合わせて作る)。

    cd data/stage39_root && uv run --project ../.. python ../../scripts/persistence_stage39.py \\
        --out ../../results/stage39
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from persistence_stage32 import load_panel
from persistence_tables_stage33 import boot_ci, spearman

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

_CASH = "jppfs_cor_CashAndDeposits"
ZERO_MAJORITY = 0.5
CI_WIDTH_MAX = 0.3


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--conf", type=Path, default=Path("conf/accounts"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(args.conf / "core.yaml")
    names = state_vector_names(spec, include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        (args.conf / "labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    codes = set(pd.read_csv(args.exclude, dtype={"edinetCode": str})["edinetCode"])
    p = load_panel(args.panel_dir, codes)
    p = p[(p["cur_total_assets"] > 0) & (p["prev_total_assets"] > 0)].reset_index(drop=True)
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    p["label"] = [lab.get((c, t), np.nan) for c, t in zip(p["edinetCode"], p["t"], strict=True)]
    a_cur, a_prev = p["cur_total_assets"].to_numpy(), p["prev_total_assets"].to_numpy()
    firms = p["edinetCode"].to_numpy()
    press = p["label"].isin([1, 2]).to_numpy()
    calm = (p["label"] == 0).to_numpy()

    def year_end(col_cur: str, col_prev: str) -> pd.DataFrame:
        cur = p.pivot_table(index="edinetCode", columns="t", values=col_cur, aggfunc="first")
        cur[2021] = p[p["t"] == 2022].set_index("edinetCode")[col_prev]
        return cur

    rows = []
    for n in names:
        xc = p[f"cur_{n}"].fillna(0.0).to_numpy()
        xp = p[f"prev_{n}"].fillna(0.0).to_numpy()
        rc, rp = xc / a_cur, xp / a_prev
        p["_rc"], p["_rp"] = rc, rp
        ye = year_end("_rc", "_rp")
        rho2 = spearman(np.concatenate([ye[2023].to_numpy(), ye[2024].to_numpy()]),
                        np.concatenate([ye[2021].to_numpy(), ye[2022].to_numpy()]))
        nz = (xc != 0) & (xp != 0)
        lo, hi = boot_ci(rc, rp, firms, rng)
        d = np.abs(xc - xp) / a_prev
        p90_0, p90_p = float(np.quantile(d[calm], 0.9)), float(np.quantile(d[press], 0.9))
        both_zero = float(((xc == 0) & (xp == 0)).mean())
        reasons = []
        if both_zero > ZERO_MAJORITY:
            reasons.append("ゼロが大半")
        if n.startswith("__OTHER__"):
            reasons.append("残余")
        if hi - lo > CI_WIDTH_MAX:
            reasons.append("区間が広い")
        rows.append({
            "account": n, "ja": ja.get(n, n), "rho1": spearman(rc, rp), "ci_lo": lo, "ci_hi": hi,
            "rho1_nonzero": spearman(rc[nz], rp[nz]), "n_both_nonzero": int(nz.sum()),
            "rho2": rho2, "n_pairs": len(xc), "share_zero": float((xc == 0).mean()),
            "share_both_zero": both_zero,
            "rho1_calm": spearman(rc[calm], rp[calm]), "rho1_press": spearman(rc[press], rp[press]),
            "p90_calm": p90_0, "p90_press": p90_p,
            "p90_ratio": p90_p / p90_0 if p90_0 > 0 else np.nan,
            "unestimable": "、".join(reasons),
        })
    t = pd.DataFrame(rows)
    rho_cash = float(t.loc[t["account"] == _CASH, "rho1"].iloc[0])
    t["cost"] = np.where(t["unestimable"] == "", (1 - rho_cash) / (1 - t["rho1"]), 1.0)
    args.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out / "persistence.csv", index=False)

    equal = yaml.safe_load((args.conf / "costs_equal.yaml").read_text(encoding="utf-8"))
    non_actionable = equal["non_actionable"]
    for fname, costs, src in (
        ("costs_equal.yaml", {n: 1.0 for n in names}, "等重み(ベースライン)"),
        ("costs_persistence.yaml", dict(zip(t["account"], t["cost"], strict=True)),
         "持続性(ステージ39、細分化)"),
    ):
        doc = {"name": fname[:-5], "basis_costs": [
            {"name": n, "cost_increase": round(float(costs[n]), 4),
             "cost_decrease": round(float(costs[n]), 4), "source": src} for n in names],
            "non_actionable": non_actionable}
        (args.conf / fname).write_text(
            "# ステージ39: scripts/persistence_stage39.py が生成(手で編集しない)。\n"
            + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def f(v: float, d: int = 3) -> str:
        return "—" if pd.isna(v) else f"{v:.{d}f}"

    lines = ["# ステージ39 B: 細分化した科目の持続性", "",
             f"推定用パネル({p['edinetCode'].nunique():,} 社、{len(p):,} 組)。区間は企業単位の"
             "ブートストラップ(1,000 回)。重み c = (1 − ρ現金)/(1 − ρ1)、"
             "推定できない科目は 1.0。", "",
             "| # | 科目 | ρ1 [95%] | ρ1(両年非ゼロ) | ρ2 | 両年ゼロ | ρ1 class 0 | ρ1 class 1+2 "
             "| p90 class 0 | p90 class 1+2 | 比 | 推定できない理由 | 重み c |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for i, r in t.iterrows():
        lines.append(
            f"| {i + 1} | {r['ja']} | {f(r['rho1'])} [{f(r['ci_lo'])}, {f(r['ci_hi'])}] "
            f"| {f(r['rho1_nonzero'])} | {f(r['rho2'])} | {r['share_both_zero']:.2f} "
            f"| {f(r['rho1_calm'])} | {f(r['rho1_press'])} | {r['p90_calm']:.4f} "
            f"| {r['p90_press']:.4f} | {f(r['p90_ratio'], 1)} | {r['unestimable']} "
            f"| {r['cost']:.2f} |")
    n_un = int((t["unestimable"] != "").sum())
    lines += ["", f"推定できない科目(重み 1.0): {n_un} / {len(t)}。"]
    (args.out / "persistence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
