#!/usr/bin/env python
"""ステージ32 B: 科目別の持続性(Nissim & Penman 2001 Table 3 の設計)と、増減の非対称性の推定。

データ: ステージ23 の推定用パネル(パネル A、3,657 社 × 3 年度の書類)。各書類は当期末(cur)と
前期末(prev)の BS を持つので、同一書類内の比較で 1 年ラグの組が 3 × 3,657 = 10,971 できる。
ステージ28 の除外リストの企業(ゴルフ場・金融)は除く。売上高はステージ23.5 の抽出。

B.1 持続性(科目 a ごと): ρ_a1 = Spearman(x_a,t / A_t, x_a,t−1 / A_t−1)(全企業年をプール)。
    vart/var0: 基準年(2021 年度末 = 2022 年度の書類の前期末)に x_a / A で 10 分位に分け、各分位の
    中央値の分散を t = 1, 2, 3 年後(2022・2023・2024 年度末)と比べる(N&P は 7 回の平均、
    ここでは 1 回)。
B.2 検証: N&P の定義で ATO = 売上高 / NOA、FLEV = NFO / CSE を作り、ρ1 を N&P
    (0.947、0.937)と比べる。
    FA = 現金預金(N&P は短期投資と「投資・貸付金その他」も含むが、32 科目では分けられない)、
    FO = 短期借入金 + 長期借入金 + 社債(流動・固定)、CSE = 純資産 − 非支配株主持分、
    NFO = FO − FA、NOA = NFO + CSE + 非支配株主持分。分母が負の比率は除く(N&P と同じ)。
    BS は期首・期末の平均(N&P と同じ。ラグの組は書類をまたぐので 2 組 / 社)と、期末値
    (書類内で 3 組 / 社)の両方。
B.4 非対称性(ABJ 型): log(x_t / x_t−1) = β0 + β1 log(A_t / A_t−1) + β2 D log(A_t / A_t−1) + ε、
    D = 総資産が減った年。企業でクラスタした標準誤差。x_t, x_t−1 > 0 の組だけ使う(符号が ≤ 0 の
    科目は −x を使う)。増加時の弾力性 β1、減少時 β1 + β2、比 r = β1 / (β1 + β2)(ABJ では
    0.55 / 0.35 = 1.57、r > 1 は「減らしにくい」)。

    uv run python scripts/persistence_stage32.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

_P = "jppfs_cor_"
FO = ("short_term_loans", "long_term_loans", "bonds_current", "bonds_noncurrent")
NCI = "non_controlling_interests"
_SALES = ("2022_2023", "2023_2024", "fy2024")
MIN_OBS = 100


def load_panel(panel_dir: Path, excluded_codes: set[str]) -> pd.DataFrame:
    p = pd.read_csv(panel_dir / "panel.csv", dtype={"edinetCode": str, "docID": str})
    p = p[~p["edinetCode"].isin(excluded_codes)].copy()
    s = pd.concat(
        [pd.read_csv(f"data/interim/stage23_5/sales_{k}.csv", dtype={"docID": str}) for k in _SALES]
    ).drop_duplicates("docID")
    return p.merge(s[["docID", "sales_cur", "sales_prev"]], on="docID", how="left")


def np_ratios(p: pd.DataFrame, names: tuple[str, ...], na: np.ndarray) -> pd.DataFrame:
    """書類ごとの ATO・FLEV(期末値と期首期末平均)。"""
    out = pd.DataFrame({"edinetCode": p["edinetCode"], "t": p["t"]})
    parts = {}
    for side in ("cur", "prev"):
        x = p[[f"{side}_{n}" for n in names]].fillna(0.0).to_numpy()
        fo = p[[f"{side}_{n}" for n in FO]].fillna(0.0).sum(axis=1).to_numpy()
        fa = p[f"{side}_{_P}CashAndDeposits"].fillna(0.0).to_numpy()
        mi = p[f"{side}_{NCI}"].fillna(0.0).to_numpy()
        cse = x @ na - mi
        nfo = fo - fa
        parts[side] = {"nfo": nfo, "cse": cse, "noa": nfo + cse + mi}
    for side in ("cur", "prev"):
        noa, nfo, cse = parts[side]["noa"], parts[side]["nfo"], parts[side]["cse"]
        out[f"ato_end_{side}"] = np.where(noa > 0, p[f"sales_{side}"] / noa, np.nan)
        out[f"flev_end_{side}"] = np.where(cse > 0, nfo / cse, np.nan)
    noa_avg = (parts["cur"]["noa"] + parts["prev"]["noa"]) / 2
    nfo_avg = (parts["cur"]["nfo"] + parts["prev"]["nfo"]) / 2
    cse_avg = (parts["cur"]["cse"] + parts["prev"]["cse"]) / 2
    out["ato_avg"] = np.where(noa_avg > 0, p["sales_cur"] / noa_avg, np.nan)
    out["flev_avg"] = np.where(cse_avg > 0, nfo_avg / cse_avg, np.nan)
    return out


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    r = spearmanr(a[m], b[m]).statistic
    return float(r), int(m.sum())


def var_ratio(base: pd.Series, later: dict[int, pd.Series]) -> dict[int, float]:
    """基準年の 10 分位の中央値の分散に対する、t 年後の同じ分位の中央値の分散の比。"""
    b = base.dropna()
    if b.nunique() < 10:
        return {t: float("nan") for t in later}
    dec = pd.qcut(b.rank(method="first"), 10, labels=False)
    v0 = b.groupby(dec).median().var()
    out = {}
    for t, s in later.items():
        med = s.reindex(b.index).groupby(dec).median()
        out[t] = float(med.var() / v0) if v0 > 0 else float("nan")
    return out


def asymmetry(p: pd.DataFrame, name: str, sign: str) -> dict[str, float]:
    import statsmodels.api as sm

    k = -1.0 if sign == "nonpos" else 1.0
    xc = k * p[f"cur_{name}"]
    xp = k * p[f"prev_{name}"]
    ac, ap_ = p["cur_total_assets"], p["prev_total_assets"]
    m = (xc > 0) & (xp > 0) & (ac > 0) & (ap_ > 0)
    n = int(m.sum())
    res = {"n": n, "beta_up": np.nan, "beta_down": np.nan, "se_up": np.nan, "se_down": np.nan,
           "ratio": np.nan}
    if n < MIN_OBS:
        return res
    y = np.log(xc[m] / xp[m]).to_numpy()
    g = np.log(ac[m] / ap_[m]).to_numpy()
    d = (g < 0).astype(float)
    x = sm.add_constant(np.column_stack([g, d * g]))
    groups = p.loc[m, "edinetCode"].to_numpy()
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    b1, b2 = float(fit.params[1]), float(fit.params[2])
    cov = fit.cov_params()
    se_down = float(np.sqrt(cov[1, 1] + cov[2, 2] + 2 * cov[1, 2]))
    res.update(beta_up=b1, beta_down=b1 + b2, se_up=float(np.sqrt(cov[1, 1])), se_down=se_down,
               ratio=b1 / (b1 + b2) if (b1 > 0 and b1 + b2 > 0) else np.nan)
    return res


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/stage32"))
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    signs = {b.name: b.sign for b in spec.basis}
    codes = set(pd.read_csv(args.exclude, dtype={"edinetCode": str})["edinetCode"])
    p = load_panel(args.panel_dir, codes)
    a_cur, a_prev = p["cur_total_assets"], p["prev_total_assets"]
    ok = (a_cur > 0) & (a_prev > 0)
    p = p[ok].reset_index(drop=True)
    a_cur, a_prev = p["cur_total_assets"], p["prev_total_assets"]

    # B.2 N&P の再現
    r = np_ratios(p, names, members["jppfs_cor_NetAssets"])
    r_end = {}
    for key in ("ato", "flev"):
        r_end[key] = _spearman(r[f"{key}_end_cur"], r[f"{key}_end_prev"])
    wide = r.pivot(index="edinetCode", columns="t", values=["ato_avg", "flev_avg"])
    r_avg = {}
    for key in ("ato", "flev"):
        a = pd.concat([wide[(f"{key}_avg", 2023)], wide[(f"{key}_avg", 2024)]], ignore_index=True)
        b = pd.concat([wide[(f"{key}_avg", 2022)], wide[(f"{key}_avg", 2023)]], ignore_index=True)
        r_avg[key] = _spearman(a, b)

    # B.1 科目別
    rows = []
    by_code = p.set_index(["edinetCode", "t"])
    for n in names:
        cur = p[f"cur_{n}"].fillna(0.0) / a_cur
        prev = p[f"prev_{n}"].fillna(0.0) / a_prev
        rho, nobs = _spearman(cur, prev)
        share_zero = float(((p[f"cur_{n}"].fillna(0.0) == 0) & (p[f"prev_{n}"].fillna(0.0) == 0))
                           .mean())
        base = (by_code[f"prev_{n}"].fillna(0.0) / by_code["prev_total_assets"]).xs(
            2022, level="t")
        later = {
            k: (by_code[f"cur_{n}"].fillna(0.0) / by_code["cur_total_assets"]).xs(y, level="t")
            for k, y in ((1, 2022), (2, 2023), (3, 2024))
        }
        vr = var_ratio(base, later)
        asym = asymmetry(p, n, signs[n])
        rows.append({"account": n, "rho1": rho, "n": nobs, "share_both_zero": share_zero,
                     "var1_var0": vr[1], "var2_var0": vr[2], "var3_var0": vr[3],
                     **{f"asym_{k}": v for k, v in asym.items()}})
    tab = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out / "persistence.csv", index=False)

    ja = __import__("yaml").safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    lines = [
        "# ステージ32 B: 持続性と非対称性(推定用パネル)", "",
        f"パネル A から除外対象を除いた {p['edinetCode'].nunique():,} 社、{len(p):,} 組"
        "(書類内の前期末 → 当期末)。",
        "", "## B.2 N&P の集約比率の再現", "",
        "| 比率 | N&P の ρ1(Table 3、1963–1999 年の米国) | 自前: 期首期末平均(書類をまたぐ) "
        "| 自前: 期末値(書類内) |",
        "|---|---:|---:|---:|",
        f"| ATO(売上高 / NOA) | 0.947 | {r_avg['ato'][0]:.3f}(n = {r_avg['ato'][1]:,}) "
        f"| {r_end['ato'][0]:.3f}(n = {r_end['ato'][1]:,}) |",
        f"| FLEV(NFO / CSE) | 0.937 | {r_avg['flev'][0]:.3f}(n = {r_avg['flev'][1]:,}) "
        f"| {r_end['flev'][0]:.3f}(n = {r_end['flev'][1]:,}) |",
        "", "## B.1 科目別の持続性(x / 総資産)と B.4 非対称性", "",
        "| # | 科目 | ρ1 | 両年 0 の割合 | var1/var0 | var2/var0 | var3/var0 | B.4 n | 増加時 β1 "
        "| 減少時 β1+β2 | 比 r = β1/(β1+β2) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in tab.iterrows():
        def f(v: float, d: int = 3) -> str:
            return "—" if pd.isna(v) else f"{v:.{d}f}"
        lines.append(
            f"| {i + 1} | {ja.get(row['account'], row['account'])} | {f(row['rho1'])} "
            f"| {row['share_both_zero']:.2f} | {f(row['var1_var0'])} | {f(row['var2_var0'])} "
            f"| {f(row['var3_var0'])} | {int(row['asym_n']):,} "
            f"| {f(row['asym_beta_up'])}({f(row['asym_se_up'])}) "
            f"| {f(row['asym_beta_down'])}({f(row['asym_se_down'])}) | {f(row['asym_ratio'], 2)} |"
        )
    lines += ["", "β の括弧内は企業でクラスタした標準誤差。— は推定できない"
              f"(正の値の組が {MIN_OBS} 未満、または β1 か β1+β2 が 0 以下で比が定義できない)。"]
    (args.out / "persistence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
