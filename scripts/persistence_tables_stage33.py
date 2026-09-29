#!/usr/bin/env python
"""ステージ33: 科目別の持続性・非対称性の整理と、「動かせない」か「動かす必要がない」かの切り分け。

新しい生成はしない。入力:
- ステージ23 の推定用パネル(除外後 3,568 社)と、ステージ32 の推定
  (``results/stage32/persistence.csv``)
- ステージ30 の年度別 GC ラベル(``data/interim/stage30/firm_years.csv``)
- ステージ31 の最終設定の生成(``data/interim/stage31/r3_cap076``)

A.1 ρ1(書類内の前期末 → 当期末、3 組 / 社)、ρ2(2 年ラグ。年度末の値を企業でつなぎ、
    2021 → 2023、2022 → 2024 の 2 組 / 社)、var1/var0(ステージ32)、ゼロの割合
A.4 集約(NOA / A、NFO / A、CSE / A、売上高 / A、ATO、FLEV)の ρ1 と構成科目の ρ1
B   ABJ 型回帰の係数と p 値(ステージ32)
C.1 (a) |Δx| / 前期総資産 の分布、(b) class 2 と class 0 で別々の ρ1(企業を単位と
    するブートストラップ 95% 区間、1,000 回、seed 0)。ラベルは各書類の年度のもの
C.2 生成での |Δx| / A0(5 年)の中央値と ρ1 の順位相関

    uv run python scripts/persistence_tables_stage33.py
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from persistence_stage32 import FO, NCI, load_panel
from scipy.stats import norm, spearmanr

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

_P = "jppfs_cor_"
_SIDE = {"assets": "資産", "liabilities": "負債", "net_assets": "純資産"}
MIN_NONZERO = 100
ZERO_MAJORITY = 0.5
_B = 1000
FROZEN = ("valuation_difference_securities", "non_controlling_interests")  # 最終設定で固定


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() >= 3 else float("nan")


def boot_ci(a: np.ndarray, b: np.ndarray, groups: np.ndarray,
            rng: np.random.Generator) -> tuple[float, float]:
    """企業を単位に復元抽出した Spearman の 95% 区間。"""
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(_B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        vals.append(spearman(a[ii], b[ii]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fmt(v: float, d: int = 3) -> str:
    return "—" if pd.isna(v) else f"{v:.{d}f}"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--stage32", type=Path, default=Path("results/stage32/persistence.csv"))
    ap.add_argument("--gen", type=Path, default=Path("data/interim/stage31/r3_cap076"))
    ap.add_argument("--out", type=Path, default=Path("results/stage33"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    core = yaml.safe_load(Path("conf/accounts/core.yaml").read_text(encoding="utf-8"))
    info = {b["name"]: b for b in core["basis"]}
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    codes = set(pd.read_csv(args.exclude, dtype={"edinetCode": str})["edinetCode"])
    p = load_panel(args.panel_dir, codes)
    p = p[(p["cur_total_assets"] > 0) & (p["prev_total_assets"] > 0)].reset_index(drop=True)
    s32 = pd.read_csv(args.stage32).set_index("account")
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    p["label"] = [lab.get((c, t), np.nan) for c, t in zip(p["edinetCode"], p["t"], strict=True)]
    a_cur, a_prev = p["cur_total_assets"].to_numpy(), p["prev_total_assets"].to_numpy()

    # 年度末の値(企業 × 年度): 2021 は 2022 年度の書類の前期末
    def year_end(col_cur: str, col_prev: str) -> pd.DataFrame:
        cur = p.pivot_table(index="edinetCode", columns="t", values=col_cur, aggfunc="first")
        prev22 = p[p["t"] == 2022].set_index("edinetCode")[col_prev]
        cur[2021] = prev22
        return cur

    rows = []
    for i, n in enumerate(names, 1):
        xc = p[f"cur_{n}"].fillna(0.0).to_numpy()
        xp = p[f"prev_{n}"].fillna(0.0).to_numpy()
        rc, rp = xc / a_cur, xp / a_prev
        rho1 = spearman(rc, rp)
        both_nz = (xc != 0) & (xp != 0)
        rho1_nz = spearman(rc[both_nz], rp[both_nz])
        p[f"_r_cur_{n}"] = rc
        p[f"_r_prev_{n}"] = rp
        ye = year_end(f"_r_cur_{n}", f"_r_prev_{n}")
        a2 = np.concatenate([ye[2023].to_numpy(), ye[2024].to_numpy()])
        b2 = np.concatenate([ye[2021].to_numpy(), ye[2022].to_numpy()])
        rho2 = spearman(a2, b2)
        zero = float((xc == 0).mean())
        both_zero = float(((xc == 0) & (xp == 0)).mean())
        nonzero = int(((xc != 0) | (xp != 0)).sum())
        d = np.abs(xc - xp) / a_prev
        dz = float((xc == xp).mean())
        nz = d[xc != xp]
        # C.1 (b)
        cls = {}
        for c in (2, 0):
            m = (p["label"] == c).to_numpy()
            r = spearman(rc[m], rp[m])
            lo, hi = boot_ci(rc[m], rp[m], p.loc[m, "edinetCode"].to_numpy(), rng) if c == 2 \
                else (float("nan"), float("nan"))
            cls[c] = (r, lo, hi, int(m.sum()), float(np.median(d[m])), float(np.quantile(d[m], .9)))
        b = info[n]
        flags = []
        if nonzero < MIN_NONZERO:
            flags.append("観測数不足")
        if both_zero > ZERO_MAJORITY:
            flags.append("ゼロが大半")
        if n.startswith("__OTHER__"):
            flags.append("残余(差額定義、符号自由)")
        rows.append({
            "#": i, "account": n, "ja": ja.get(n, n),
            "parent": ja.get(b["parent_subtotal"], b["parent_subtotal"]),
            "side": _SIDE[b["side"]], "sign": b["sign"],
            "rho1": rho1, "rho1_nonzero": rho1_nz, "n_both_nonzero": int(both_nz.sum()),
            "rho2": rho2, "var1_var0": float(s32.loc[n, "var1_var0"]),
            "n_pairs": len(xc), "n_nonzero_pairs": nonzero, "share_zero": zero,
            "share_both_zero": both_zero, "flags": "、".join(flags),
            "share_no_change": dz, "absd_p50": float(np.median(d)),
            "absd_p90": float(np.quantile(d, 0.9)),
            "absd_nonzero_p50": float(np.median(nz)) if len(nz) else float("nan"),
            "rho1_c2": cls[2][0], "rho1_c2_lo": cls[2][1], "rho1_c2_hi": cls[2][2],
            "n_c2": cls[2][3], "absd_p50_c2": cls[2][4], "absd_p90_c2": cls[2][5],
            "rho1_c0": cls[0][0], "n_c0": cls[0][3], "absd_p50_c0": cls[0][4],
            "absd_p90_c0": cls[0][5],
        })
    t = pd.DataFrame(rows)

    # B: 非対称性(ステージ32 の推定)
    z = 1.959963984540054
    bu, bd = s32["asym_beta_up"], s32["asym_beta_down"]
    su, sd = s32["asym_se_up"], s32["asym_se_down"]
    t["beta_up"] = t["account"].map(bu)
    t["beta_down"] = t["account"].map(bd)
    p_up = pd.Series(2 * (1 - norm.cdf((bu / su).abs())), index=s32.index)
    p_down = pd.Series(2 * (1 - norm.cdf((bd / sd).abs())), index=s32.index)
    t["p_up"] = t["account"].map(p_up)
    t["p_down"] = t["account"].map(p_down)
    t["asym_n"] = t["account"].map(s32["asym_n"])
    t["ratio"] = t["account"].map(s32["asym_ratio"])
    sig = t["account"].map((bu / su > z) & (bd / sd > z))
    t["asym_judge"] = np.where(t["ratio"].isna(), "推定できない",
                               np.where(sig, "有意", "有意でない"))

    # C.2 生成
    files = sorted(glob.glob(str(args.gen / "nearest_na0__seed*.parquet")))
    g = pd.concat([pd.read_parquet(f) for f in files])
    g = g[g["method"] == "PCFM"]
    for n in names:
        g[f"d_{n}"] = (g[f"x1_{n}"] - g[f"x0_{n}"]).abs()
    gen_med = g.groupby("docID")[[f"d_{n}" for n in names]].median().median()
    t["gen_absd_p50"] = [float(gen_med[f"d_{n}"]) for n in names]
    start = g.drop_duplicates("docID")
    t["size_p50"] = [float(start[f"x0_{n}"].abs().median()) for n in names]  # 始点の |x| / A0
    t["frozen"] = t["account"].isin(FROZEN)
    t.to_csv(args.out / "accounts_persistence.csv", index=False)

    # A.4 集約
    na = members["jppfs_cor_NetAssets"]
    agg: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    xcur = p[[f"cur_{n}" for n in names]].fillna(0.0).to_numpy()
    xprev = p[[f"prev_{n}" for n in names]].fillna(0.0).to_numpy()
    fo_idx = [names.index(n) for n in FO]
    cash = names.index(f"{_P}CashAndDeposits")
    nci = names.index(NCI)

    def parts(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nfo = x[:, fo_idx].sum(axis=1) - x[:, cash]
        cse = x @ na - x[:, nci]
        return nfo, cse, nfo + cse + x[:, nci]

    nfo_c, cse_c, noa_c = parts(xcur)
    nfo_p, cse_p, noa_p = parts(xprev)
    s_c, s_p = p["sales_cur"].to_numpy(), p["sales_prev"].to_numpy()
    agg["NOA / A"] = (noa_c / a_cur, noa_p / a_prev)
    agg["NFO / A"] = (nfo_c / a_cur, nfo_p / a_prev)
    agg["CSE / A"] = (cse_c / a_cur, cse_p / a_prev)
    agg["売上高 / A"] = (s_c / a_cur, s_p / a_prev)
    agg["ATO = 売上高 / NOA(NOA > 0)"] = (np.where(noa_c > 0, s_c / noa_c, np.nan),
                                         np.where(noa_p > 0, s_p / noa_p, np.nan))
    agg["FLEV = NFO / CSE(CSE > 0)"] = (np.where(cse_c > 0, nfo_c / cse_c, np.nan),
                                       np.where(cse_p > 0, nfo_p / cse_p, np.nan))
    agg["借入金 + 社債 / A"] = (xcur[:, fo_idx].sum(axis=1) / a_cur,
                            xprev[:, fo_idx].sum(axis=1) / a_prev)
    agg_rows = [(k, spearman(a, b)) for k, (a, b) in agg.items()]
    oper = ["receivables_trade", "inventories", f"{_P}PropertyPlantAndEquipment",
            f"{_P}IntangibleAssets", f"{_P}InvestmentsAndOtherAssets", "payables_trade",
            "accounts_payable_other", f"{_P}OtherCA", f"{_P}OtherCL"]
    fin = [*FO, f"{_P}CashAndDeposits"]

    # 出力
    lines: list[str] = ["# ステージ33: 科目別の持続性の推定結果", ""]
    lines += [f"推定用パネル(除外後 {p['edinetCode'].nunique():,} 社、書類内の組 {len(p):,})。"
          "ρ1 は書類内の前期末 → 当期末(x / 総資産)、ρ2 は年度末の値を企業でつないだ 2 年ラグ "
          "(2021 → 2023、2022 → 2024)。ゼロの割合は当期末が 0 の組の割合。", ""]
    lines += ["## A.1 32 科目の表(core.yaml の順)", "",
          "| # | 科目 | 親小計 | 側 | ρ1 | ρ1(両年非ゼロの組のみ、組数) | ρ2 | var1/var0 "
          "| 組数(うち少なくとも一方が非ゼロ) | ゼロの割合 | 注意 |",
          "|---:|---|---|---|---:|---|---:|---:|---|---:|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['#']} | {r['ja']} | {r['parent']} | {r['side']} | {fmt(r['rho1'])} "
                 f"| {fmt(r['rho1_nonzero'])}({r['n_both_nonzero']:,}) | {fmt(r['rho2'])} "
                 f"| {fmt(r['var1_var0'])} | {r['n_pairs']:,}"
                 f"({r['n_nonzero_pairs']:,}) | {r['share_zero']:.2f} | {r['flags']} |")
    lines += ["", "## A.3 ρ1 の降順", "",
              "| 順位 | 科目 | ρ1 | ρ1(両年非ゼロ) | ρ2 | ゼロの割合 | 注意 |",
              "|---:|---|---:|---:|---:|---:|---|"]
    for k, (_, r) in enumerate(t.sort_values("rho1", ascending=False).iterrows(), 1):
        lines.append(f"| {k} | {r['ja']} | {fmt(r['rho1'])} | {fmt(r['rho1_nonzero'])} "
                     f"| {fmt(r['rho2'])} | {r['share_zero']:.2f} | {r['flags']} |")
    lines += ["", "## A.4 集約比率と構成科目の ρ1", "", "| 集約 | ρ1 |", "|---|---:|"]
    lines += [f"| {k} | {fmt(v)} |" for k, v in agg_rows]
    sub = t.set_index("account")
    lines += ["", "| 構成 | 科目の ρ1(最小〜最大、中央値) |", "|---|---|",
          f"| 営業資産・負債(NOA の主な構成 {len(oper)} 科目) | "
          f"{sub.loc[oper, 'rho1'].min():.3f}〜{sub.loc[oper, 'rho1'].max():.3f}、"
          f"{sub.loc[oper, 'rho1'].median():.3f} |",
          f"| 金融負債・現金(NFO の構成 {len(fin)} 科目) | "
          f"{sub.loc[fin, 'rho1'].min():.3f}〜{sub.loc[fin, 'rho1'].max():.3f}、"
          f"{sub.loc[fin, 'rho1'].median():.3f} |"]
    lines += ["", "## B 非対称性(ステージ32 の ABJ 型回帰)", "",
          "| # | 科目 | 組数 | β1(増加時) | p | β1+β2(減少時) | p | 比 β1/(β1+β2) | 判定 |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['#']} | {r['ja']} | {int(r['asym_n']):,} | {fmt(r['beta_up'])} "
                 f"| {fmt(r['p_up'], 3)} | {fmt(r['beta_down'])} | {fmt(r['p_down'], 3)} "
                 f"| {fmt(r['ratio'], 2)} | {r['asym_judge']} |")
    lines += ["", "## C.1 (a) 変化の大きさ \\|Δx\\| / 前期総資産(1 年)", "",
          "| # | 科目 | ρ1 | 変化ゼロの組 | 両年ゼロの組 | 中央値 | 上位 10%(p90) "
          "| 非ゼロ変化の中央値 |",
          "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['#']} | {r['ja']} | {fmt(r['rho1'])} | {r['share_no_change']:.2f} "
                 f"| {r['share_both_zero']:.2f} | {r['absd_p50']:.4f} | {r['absd_p90']:.4f} "
                 f"| {fmt(r['absd_nonzero_p50'], 4)} |")
    n_c2 = int((p["label"] == 2).sum())
    f_c2 = p.loc[p["label"] == 2, "edinetCode"].nunique()
    n_c0 = int((p["label"] == 0).sum())
    lines += ["", "## C.1 (b) class 2 と class 0 の ρ1", "",
          f"ラベルは各書類の年度のもの。class 2: {n_c2} 組({f_c2} 社)、class 0: {n_c0:,} 組。"
          "class 2 の区間は企業を単位とするブートストラップ(1,000 回)。", "",
          "| # | 科目 | ρ1 class 0 | ρ1 class 2 [95% 区間] "
          "| class 2 が低い(区間の上端 < class 0) "
          "| \\|Δx\\|/A 中央値 class 0 | class 2 | p90 class 0 | class 2 |",
          "|---:|---|---:|---|---|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        low = "○" if r["rho1_c2_hi"] < r["rho1_c0"] else ""
        lines.append(f"| {r['#']} | {r['ja']} | {fmt(r['rho1_c0'])} | {fmt(r['rho1_c2'])} "
                 f"[{fmt(r['rho1_c2_lo'])}, {fmt(r['rho1_c2_hi'])}] | {low} "
                 f"| {r['absd_p50_c0']:.4f} | {r['absd_p50_c2']:.4f} | {r['absd_p90_c0']:.4f} "
                 f"| {r['absd_p90_c2']:.4f} |")
    # C.2
    use = t[~t["frozen"]]
    c_all = spearman(use["rho1"].to_numpy(), use["gen_absd_p50"].to_numpy())
    use2 = use[use["flags"] == ""]
    c_clean = spearman(use2["rho1"].to_numpy(), use2["gen_absd_p50"].to_numpy())
    c_emp = spearman(use["absd_p50_c2"].to_numpy(), use["gen_absd_p50"].to_numpy())
    lines += ["", "## C.2 生成での動きと ρ1", "",
          "最終設定(R3 + 上限 0.757、57 社 × 3 seed)の |x1 − x0| / A0(5 年)を"
          "企業ごとに seed の中央値、その中央値。固定科目(評価差額金・非支配株主持分)は除く。", "",
          "| 科目 | ρ1 | 生成の \\|Δx\\|/A0 中央値(5 年) "
          "| 実データ class 2 の \\|Δx\\|/A 中央値(1 年) |",
          "|---|---:|---:|---:|"]
    for _, r in use.sort_values("rho1", ascending=False).iterrows():
        lines.append(f"| {r['ja']} | {fmt(r['rho1'])} | {r['gen_absd_p50']:.4f} "
                 f"| {r['absd_p50_c2']:.4f} |")
    c_size = spearman(use["rho1"].to_numpy(), use["size_p50"].to_numpy())
    c_emp_rho = spearman(use["rho1"].to_numpy(), use["absd_p50_c2"].to_numpy())
    big = use[use["size_p50"] > 1e-4]
    c_rel = spearman(big["rho1"].to_numpy(),
                     (big["gen_absd_p50"] / big["size_p50"]).to_numpy())
    big2 = use[use["absd_p50_c2"] > 1e-4]
    c_exc = spearman(big2["rho1"].to_numpy(),
                     (big2["gen_absd_p50"] / (5 * big2["absd_p50_c2"])).to_numpy())
    lines += ["", f"- ρ1 と生成の動きの順位相関(固定科目を除く {len(use)} 科目): **{c_all:+.3f}**",
              f"- 注意のない科目だけ({len(use2)} 科目): **{c_clean:+.3f}**",
              f"- 参考: 実データ class 2 の動きと生成の動きの順位相関: {c_emp:+.3f}",
              "", "規模の影響(ρ1 が高い科目は金額も大きく、動く絶対量も大きい):", "",
              f"- ρ1 と科目の大きさ(始点の |x| / A0 の中央値)の順位相関: {c_size:+.3f}",
              f"- ρ1 と実データ class 2 の動き(1 年)の順位相関: {c_emp_rho:+.3f}",
              f"- **大きさで割った生成の動き(|Δx| / 始点の |x| の中央値)と ρ1: {c_rel:+.3f}**"
              f"(大きさが 1e-4 超の {len(big)} 科目)",
              f"- 生成の動き / 実データ class 2 の動きの 5 倍 と ρ1: {c_exc:+.3f}"
              f"(実データの動きが 1e-4 超の {len(big2)} 科目)"]
    (args.out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
