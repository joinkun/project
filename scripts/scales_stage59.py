#!/usr/bin/env python
"""ステージ59 §A 修正: 方向別の尺度(中央値・IQR・p90−p10)・コスト行列・符号・σ_pool(生成なし)。

A.0.2 D_rec(class 1 → class 0 の 57 社)の Δx̃ = x̃(2024) − x̃(2022) から、方向ごとに
MAD(§A の版)・中央値(既定)・IQR・p90 − p10 を計算し、c = 1 / (s + ε0) とする。
A.0.3 n < 5 の方向は、どの尺度でも c = 全科目(操作できる 29 科目、n ≥ 5 の方向)の c の中央値
で代替する。
A.0.4 コスト行列(class 2 55 社 × class 0)の性質を、等重み L2²・対称 L1・非対称 L1 で再計算する。
A.0.5 c−/c+ の向きを、ステージ49 の定義(1 年・全件・log 比の |g| の平均)との間で、母集団・期間・
変化の測り方を 1 つずつ変えて比べる。
A.1 σ_pool(案 1)と σ_total、払込資本の群間分散の分解。

出力(``results/stage59/``): ``scales.md``・``weights_v2.csv``・``cost_matrix_v2.md``・
``sign_check.md``・``pooled_sd_v2.md``。

    uv run python scripts/scales_stage59.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from density_stage51 import load_tilde
from prep_stage59 import EPS0, N_MIN, OUTLIERS, cost_stats, deltas, l1_cost, l2_cost
from variance_stage58 import pooled_within

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config

_PANEL = Path("data/interim/stage23/panel/panel.csv")
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
Scale = Callable[[np.ndarray], float]


def _q(v: np.ndarray, hi: float, lo: float) -> float:
    return float(np.percentile(v, hi) - np.percentile(v, lo))


SCALES: dict[str, Scale] = {
    "MAD": lambda v: float(np.median(np.abs(v - np.median(v)))),
    "中央値": lambda v: float(np.median(v)),
    "IQR": lambda v: _q(v, 75, 25),
    "p90−p10": lambda v: _q(v, 90, 10),
}
DEFAULT = "中央値"


def scale_table(d: np.ndarray, nact: np.ndarray, fn: Scale) -> pd.DataFrame:
    """科目 × 方向の s・c(n < 5 は全科目の中央値で代替、非操作は 0)。"""
    rows = []
    for j in range(d.shape[1]):
        v = d[:, j]
        for sign, sub in (("+", v[v > 0]), ("−", -v[v < 0])):
            s = fn(sub) if len(sub) else float("nan")
            rows.append({"j": j, "dir": sign, "n": len(sub), "s": s})
    t = pd.DataFrame(rows)
    t["nact"] = nact[t["j"]]
    t["c_raw"] = 1.0 / (t["s"] + EPS0)
    ok = (t["n"] >= N_MIN) & ~t["nact"]
    fill = float(t.loc[ok, "c_raw"].median())
    t["c"] = np.where(ok, t["c_raw"], fill)
    t.loc[t["nact"], "c"] = 0.0
    t["substituted"] = (t["n"] < N_MIN) & ~t["nact"]
    t["s_zero"] = (t["s"] == 0) & ~t["nact"]
    return t


def tilde(df: pd.DataFrame, prefix: str, total: str, names: tuple[str, ...]) -> np.ndarray:
    out: np.ndarray = (df[[f"{prefix}{n}" for n in names]].fillna(0).to_numpy()
                       / df[total].to_numpy()[:, None])
    return out


def stage49_scale(xa: np.ndarray, xb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ステージ49 の定義: 同符号・両期非 0 の組で g = log(|x_b| / |x_a|)。(増加側の平均 |g|, 減"
      "少側, 増加の割合)。"""
    up, dn, fr = [], [], []
    for j in range(xa.shape[1]):
        a, b = xa[:, j], xb[:, j]
        m = (a != 0) & (b != 0) & (np.sign(a) == np.sign(b))
        g = np.log(np.abs(b[m]) / np.abs(a[m]))
        g = g[g != 0]
        up.append(g[g > 0].mean() if (g > 0).any() else np.nan)
        dn.append(-g[g < 0].mean() if (g < 0).any() else np.nan)
        fr.append((g > 0).mean() if len(g) else np.nan)
    return np.array(up), np.array(dn), np.array(fr)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage59"))
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    lab_of = [ja.get(n, n) for n in names]
    na = set(load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable)
    nact = np.array([n in na for n in names])
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    codes = pairs[pairs["group"] == "class 1 → class 0"]["edinetCode"].tolist()
    d57 = deltas(codes, names)
    tabs = {k: scale_table(d57, nact, fn) for k, fn in SCALES.items()}

    # ---------------- A.0.2・A.0.3 ----------------
    base = tabs["MAD"][["j", "dir", "n"]].copy()
    for k, t in tabs.items():
        base[f"s_{k}"], base[f"c_{k}"] = t["s"], t["c"]
    base["account"] = [names[j] for j in base["j"]]
    base["ja"] = [lab_of[j] for j in base["j"]]
    base["nact"] = nact[base["j"]]
    base["substituted"] = tabs[DEFAULT]["substituted"]
    base.drop(columns="j").to_csv(args.out / "weights_v2.csv", index=False)
    lines = ["# ステージ59 A.0.2・A.0.3: 方向別の尺度の比較(D_rec 57 社、Δx̃ = x̃(2024) − x̃(202"
             "2))", "",
             f"s は変化した企業(Δ > 0 または Δ < 0)の Δ(減少側は |Δ|)から計算。c = 1 / (s + ε0)"
             f"、ε0 = {EPS0}。"
             f"n < {N_MIN} の方向は、どの尺度でも c = 全科目の c の中央値(操作できる科目の n ≥"
             f" {N_MIN} の方向)"
             "で代替(* 印)。非操作 3 科目は c = 0 で表から除く。", "",
             "| 科目 | 方向 | n | MAD | 中央値 | IQR | p90−p10 | c(MAD) | c(中央値) | c(IQR) | c"
             "(p90−p10) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in base[~base["nact"]].iterrows():
        star = "*" if r["substituted"] else ""
        lines.append(f"| {r['ja']} | {r['dir']} | {r['n']}{star} | "
                     + " | ".join(f"{r[f's_{k}']:.2e}" for k in SCALES) + " | "
                     + " | ".join(f"{r[f'c_{k}']:.1f}" for k in SCALES) + " |")
    lines += ["", "## 尺度ごとの要約(操作できる 29 科目 × 2 方向 = 58 方向)", "",
              "| 尺度 | s = 0 の方向(n ≥ 1) | s < 1e-4 の方向 | c ≥ 900 の方向(1/ε0 付近) | c "
              "の最大 / 最小 | 代替した方向 |",
              "|---|---:|---:|---:|---:|---:|"]
    for k, t in tabs.items():
        a = t[~t["nact"]]
        zero = a[(a["n"] >= 1) & (a["s"] == 0)]
        tiny = a[(a["n"] >= 1) & (a["s"] < 1e-4)]
        zl = "、".join(f"{lab_of[r.j]}{r.dir}(n={r.n})" for r in zero.itertuples()) or "なし"
        lines.append(f"| {k} | {len(zero)}({zl}) | {len(tiny)} | {int((a['c'] >= 900).sum())} "
                     f"| {a['c'].max() / a['c'].min():.1f} "
                     f"| {int(a['substituted'].sum())} |")
    sub = tabs[DEFAULT][tabs[DEFAULT]["substituted"]]
    fill = float(sub["c"].iloc[0]) if len(sub) else float("nan")
    lines += ["", f"**A.0.3 代替した方向(既定の中央値、c = {fill:.1f})**: "
              + "、".join(f"{lab_of[r.j]}{r.dir}(n = {r.n})" for r in sub.itertuples())
              + f"。{sub['j'].nunique()} 科目 {len(sub)} 方向。", ""]
    (args.out / "scales.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------------- A.0.4 ----------------
    x, lab, docs, _keep = load_tilde(names)
    is_src = (lab == 2) & ~np.isin(docs, OUTLIERS)
    xs, xt = x[is_src], x[lab == 0]
    lines = [f"# ステージ59 A.0.4: コスト行列の性質(修正後、class 2 {len(xs)} 社 × class 0"
             f" {len(xt):,} 社)",
             "", "C̃ = C / median(C)。対称 L1 は、方向を分けずに |Δ|(Δ ≠ 0)から同じ尺度で s を"
                 "計算した。"
             "差 = 2 番目の対応先と最近傍の C̃ の差の中央値。", "",
             "| コスト | 最大/中央値 | 行内の分散 | 最近傍の C̃ 中央値 | 差 | 異なる最近傍 | 最"
             "多の被選択数 "
             "| 最近傍の払込資本比率 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    paid_t = xt[:, names.index(_CS)] + xt[:, names.index(_CSUR)]
    costs: dict[str, np.ndarray] = {"等重み L2²(現行)": l2_cost(xs, xt, (~nact).astype(float))}
    for k, t in tabs.items():
        cp = t[t["dir"] == "+"].sort_values("j")["c"].to_numpy()
        cm = t[t["dir"] == "−"].sort_values("j")["c"].to_numpy()
        both = np.abs(d57)
        sym = scale_table(np.where(d57 != 0, both, 0.0), nact, SCALES[k])
        cs = sym[sym["dir"] == "+"].sort_values("j")["c"].to_numpy()
        if k in ("MAD", DEFAULT):
            costs[f"対称 L1({k})"] = l1_cost(xs, xt, cs, cs)
        costs[f"非対称 L1({k})"] = l1_cost(xs, xt, cp, cm)
    for k, c in costs.items():
        s = cost_stats(c)
        nn = c.argmin(axis=1)
        lines.append(f"| {k} | {s['max_over_med']:.1f} | {s['rowvar_med']:.3f} |"
                     f" {s['rowmin_p50']:.3f} "
                     f"| {s['gap_med']:.4f} | {len(set(nn))} | {np.bincount(nn).max()} "
                     f"| {np.median(paid_t[nn]):.3f} |")
    lines += ["", "比は 1,000 を大きく下回るので、重みの上限(A・B)は設けない。", ""]
    (args.out / "cost_matrix_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------------- A.0.5 ----------------
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    cur = tilde(p, "cur_", "cur_total_assets", names)
    prev = tilde(p, "prev_", "prev_total_assets", names)
    ok = np.isfinite(cur).all(axis=1) & np.isfinite(prev).all(axis=1)
    cur, prev = cur[ok], prev[ok]
    a22 = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    a24 = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
    xa = tilde(a22, "cur_", "cur_total_assets", names)
    xb = tilde(a24, "cur_", "cur_total_assets", names)
    in57 = (p.loc[ok, "edinetCode"].isin(codes).to_numpy()
            & p.loc[ok, "t"].isin([2023, 2024]).to_numpy())

    def med_ratio(d: np.ndarray) -> np.ndarray:
        """中央値の尺度での c−/c+ = (s+ + ε0) / (s− + ε0)。"""
        out = []
        for j in range(d.shape[1]):
            v = d[:, j]
            sp = np.median(v[v > 0]) if (v > 0).any() else np.nan
            sm = np.median(-v[v < 0]) if (v < 0).any() else np.nan
            out.append((sp + EPS0) / (sm + EPS0))
        return np.array(out)

    variants = {
        "① ステージ49(全件・1 年・log 比・平均)": None,
        "② 全件・1 年・log 比・中央値": "g_all_med",
        "③ 全件・1 年・差 Δx̃・中央値": med_ratio(cur - prev),
        "④ 57 社・1 年・差 Δx̃・中央値": med_ratio(cur[in57] - prev[in57]),
        "⑤ 57 社・2 年・log 比・平均": "g57",
        "⑥ 57 社・2 年・差 Δx̃・中央値(今回)": med_ratio(d57),
    }
    up49, dn49, fr49 = stage49_scale(prev, cur)
    res: dict[str, np.ndarray] = {}
    for k, v in variants.items():
        if v is None:
            res[k] = up49 / dn49
        elif isinstance(v, str) and v == "g_all_med":
            r = []
            for j in range(len(names)):
                a, b = prev[:, j], cur[:, j]
                m = (a != 0) & (b != 0) & (np.sign(a) == np.sign(b))
                g = np.log(np.abs(b[m]) / np.abs(a[m]))
                both_sides = (g > 0).any() and (g < 0).any()
                r.append(np.median(g[g > 0]) / np.median(-g[g < 0]) if both_sides else np.nan)
            res[k] = np.array(r)
        elif isinstance(v, str):
            u, dd, _ = stage49_scale(xa, xb)
            res[k] = u / dd
        else:
            res[k] = v
    focus = ["jppfs_cor_PropertyPlantAndEquipment", "jppfs_cor_IntangibleAssets",
             "long_term_loans", _CSUR, "jppfs_cor_TreasuryStock", _CS]
    lines = ["# ステージ59 A.0.5: c−/c+ の向きの確認", "",
             "値はすべて「増加側の尺度 / 減少側の尺度」(中央値の尺度では c−/c+ と同じ。> 1 = 増"
             "えるときは大きく"
             "増える = 減らしにくい)。① はステージ49 の `asymmetry.md` の「比」の定義(同符号・"
             "両期非 0 の組の "
             "g = log(|x̃_t| / |x̃_{t−1}|)、|g| の平均)を、このスクリプトで再計算した。"
             "② 以降は母集団・期間・測り方を 1 つずつ変えた。全件 = 推定用パネルの全組。", "",
             "| 科目 | " + " | ".join(variants) + " |", "|---|" + "---:|" * len(variants)]
    for f in focus:
        j = names.index(f)
        vals = " | ".join(f"{res[k][j]:.2f}" for k in variants)
        lines.append(f"| {lab_of[j]} | {vals} |")
    lines += ["", f"①の組の数: {len(cur):,}。①の増加の割合(参考): "
              + "、".join(f"{lab_of[names.index(f)]} {fr49[names.index(f)]:.3f}"
                         for f in focus), ""]
    (args.out / "sign_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------------- A.1 ----------------
    s_tot = x.std(axis=0, ddof=0)
    s1 = pooled_within(x, lab)
    lines = ["# ステージ59 A.1: σ_pool(案 1、class 内の分散を社数で重み付けたプール)", "",
             f"断面 = 除外後の class 0 {int((lab == 0).sum()):,}・class 1 {int((lab == 1).sum())}・"
             f"class 2 {int((lab == 2).sum())}。比 = σ_pool / σ_total(どちらも ddof = 0。"
             "1 − 比² = class で説明される分散の割合)。", "",
             "| 科目 | σ_total | σ_pool | 比 | 1 − 比² | 順位(比の小さい順) |", "|---|---:|---:|"
                                                                    "---:|---:|---:|"]
    ratio = s1 / s_tot
    order = np.argsort(ratio)
    rank = np.empty(len(names), int)
    rank[order] = np.arange(1, len(names) + 1)
    for j in order:
        lines.append(f"| {lab_of[j]} | {s_tot[j]:.4f} | {s1[j]:.4f} | {ratio[j]:.3f} "
                     f"| {1 - ratio[j] ** 2:.3f} | {rank[j]} |")
    paid = x[:, names.index(_CS)] + x[:, names.index(_CSUR)]
    rp = pooled_within(paid[:, None], lab)[0] / paid.std()
    agg = np.sqrt((s1 ** 2).sum() / (s_tot ** 2).sum())
    lines += ["", f"**全科目の合計(Σσ_pool² / Σσ_total² の平方根)**: {agg:.3f}。"
              f"科目ごとの比の中央値 {np.median(ratio):.3f}。", "",
              f"**払込資本(資本金 + 資本剰余金)**: 比 {rp:.3f}(1 − 比² = {1 - rp ** 2:.3f})。", "",
              "| class | 社数 | 平均 | 中央値 | 標準偏差 |", "|---|---:|---:|---:|---:|"]
    for cl in (0, 1, 2):
        v = paid[lab == cl]
        lines.append(f"| {cl} | {len(v):,} | {v.mean():.3f} | {np.median(v):.3f} | {v.std():.3f} |")
    mu = paid.mean()
    between = sum((lab == cl).sum() * (paid[lab == cl].mean() - mu) ** 2
                  for cl in (0, 1, 2)) / len(paid)
    lines += ["", f"群間分散 = {between:.4f}、全体分散 = {paid.var():.4f}(割合"
                  f" {between / paid.var():.3f})。", ""]
    (args.out / "pooled_sd_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
