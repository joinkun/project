#!/usr/bin/env python
"""ステージ59 §A: 方向別 MAD の重み・コスト行列の性質・群内プール分散(生成なし)。

A.1 D_rec(class 1 → class 0 の 57 社)の Δx̃ = x̃(2024) − x̃(2022)(各年の総資産で割る)から、
s_a^± = MAD(Δ | Δ > 0)・MAD(|Δ| | Δ < 0)、c_a^± = 1 / (s_a^± + ε0)。
MAD は中央値からの絶対偏差の中央値(1.4826 を掛けない)。n < 5 の方向は、反対の向きの s(n ≥ 5 なら)
で代替し、両方とも n < 5 なら信頼できる c の中央値で代替する。
非操作 3 科目は c = 0(nearest_na0 と同じ)。
A.2 class 2(外れ値 2 社を除く 55 社)× class 0(除外後)のコスト行列を 3 通り(等重み L2²・
対称 L1・非対称 L1)で作り、性質を比べる。
A.3 σ_total と群内プール分散 3 案(案 1 class 内・案 2 D_rec の Δx̃・案 3 業種内)。

出力: ``results/stage59/weights.csv``・``weights.md``・``cost_matrix.md``・``pooled_sd.csv``・
``pooled_sd.md``。

    uv run python scripts/prep_stage59.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from changes_stage57 import deltas as nominal_deltas
from density_stage51 import load_tilde
from variance_stage58 import pooled_within

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config
from fsgen.models.industry import load_industry_config

_PANEL = Path("data/interim/stage23/panel/panel.csv")
EPS0 = 0.001  # 総資産比 0.1%(ステージ47・56 の「変化した」の閾値と同じ)
N_MIN = 5
OUTLIERS = ("S100TSS9", "S100TWFQ")  # 評価から除く class 2 の外れ値(ステージ44)
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
QS = (0, 1, 25, 50, 75, 99, 100)


def mad(v: np.ndarray) -> float:
    return float(np.median(np.abs(v - np.median(v)))) if len(v) else float("nan")


def deltas(codes: list[str], names: tuple[str, ...]) -> np.ndarray:
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    b = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
    cols = [f"cur_{n}" for n in names]
    xa = a[cols].fillna(0).to_numpy() / a["cur_total_assets"].to_numpy()[:, None]
    xb = b[cols].fillna(0).to_numpy() / b["cur_total_assets"].to_numpy()[:, None]
    out: np.ndarray = xb - xa
    return out


def directional(d: np.ndarray, nact: np.ndarray) -> pd.DataFrame:
    """科目ごとの n±・s±・c±(代替の規則つき)。"""
    rows = []
    for j in range(d.shape[1]):
        v = d[:, j]
        pos, neg = v[v > 0], -v[v < 0]
        both = np.abs(v[v != 0])
        rows.append({"n_plus": len(pos), "n_minus": len(neg), "s_plus_raw": mad(pos),
                     "s_minus_raw": mad(neg), "s_sym_raw": mad(both),
                     "med_plus": float(np.median(pos)) if len(pos) else np.nan,
                     "med_minus": float(np.median(neg)) if len(neg) else np.nan})
    df = pd.DataFrame(rows)
    sp, sm, rule = df["s_plus_raw"].copy(), df["s_minus_raw"].copy(), [""] * len(df)
    ok_p, ok_m = df["n_plus"] >= N_MIN, df["n_minus"] >= N_MIN
    for j in range(len(df)):
        if nact[j]:
            rule[j] = "非操作(c = 0)"
        elif not ok_p[j] and ok_m[j]:
            sp[j], rule[j] = sm[j], "+ を − で代替"
        elif ok_p[j] and not ok_m[j]:
            sm[j], rule[j] = sp[j], "− を + で代替"
        elif not ok_p[j] and not ok_m[j]:
            rule[j] = "両方を中央値で代替"
    cp, cm = 1.0 / (sp + EPS0), 1.0 / (sm + EPS0)
    reliable = np.concatenate([cp[ok_p & ~nact], cm[ok_m & ~nact]])
    fill = float(np.median(reliable))
    both_bad = np.array([r == "両方を中央値で代替" for r in rule])
    cp[both_bad], cm[both_bad] = fill, fill
    ok_s = (df["n_plus"] + df["n_minus"]) >= N_MIN
    cs = 1.0 / (df["s_sym_raw"] + EPS0)
    cs[~ok_s] = fill
    cp[nact], cm[nact], cs[nact] = 0.0, 0.0, 0.0
    df["s_plus"], df["s_minus"], df["c_plus"], df["c_minus"], df["c_sym"] = sp, sm, cp, cm, cs
    df["ratio"] = np.where(cp > 0, cm / np.where(cp > 0, cp, 1.0), np.nan)
    df["rule"] = rule
    return df


def l1_cost(x: np.ndarray, y: np.ndarray, cp: np.ndarray, cm: np.ndarray) -> np.ndarray:
    c = np.zeros((len(x), len(y)))
    for a in range(x.shape[1]):
        if cp[a] == 0 and cm[a] == 0:
            continue
        diff = y[None, :, a] - x[:, None, a]
        c += cp[a] * np.maximum(diff, 0.0) + cm[a] * np.maximum(-diff, 0.0)
    return c


def l2_cost(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    c = np.zeros((len(x), len(y)))
    for a in range(x.shape[1]):
        if w[a] > 0:
            c += w[a] * (y[None, :, a] - x[:, None, a]) ** 2
    return c


def cost_stats(c: np.ndarray) -> dict[str, float]:
    med = float(np.median(c))
    cn = c / med
    rmin = cn.min(axis=1)
    out = {f"p{q}": float(np.percentile(c, q)) for q in QS}
    out |= {"max_over_med": float(c.max() / med), "rowmin_p50": float(np.median(rmin)),
            "rowmin_p5": float(np.percentile(rmin, 5)),
            "rowmin_p95": float(np.percentile(rmin, 95)),
            "rowmin_max": float(rmin.max()),
            "rowvar_med": float(np.median(cn.var(axis=1))),
            "row_cv_med": float(np.median(c.std(axis=1) / c.mean(axis=1))),
            "gap_med": float(np.median(np.sort(cn, axis=1)[:, 1] - rmin)),
            "n_tie": int((np.sort(cn, axis=1)[:, 1] - rmin < 1e-12).sum())}
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage59"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    na = set(load_cost_config(Path("conf/accounts/costs_equal.yaml")).non_actionable)
    nact = np.array([n in na for n in names])
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    grp = {g: pairs[pairs["group"] == g]["edinetCode"].tolist() for g in pairs["group"].unique()}
    d57 = deltas(grp["class 1 → class 0"], names)
    d15 = deltas(grp["class 2 → 回復"], names)
    w = directional(d57, nact)
    w15 = directional(d15, nact)
    w.insert(0, "account", names)
    w.insert(1, "ja", [ja.get(n, n) for n in names])
    wn = directional(nominal_deltas(grp["class 1 → class 0"], names).to_numpy(), nact)
    w["n_plus_nom"], w["n_minus_nom"], w["ratio_nom"] = wn["n_plus"], wn["n_minus"], wn["ratio"]
    w["n_plus_15"], w["n_minus_15"] = w15["n_plus"], w15["n_minus"]
    w["c_plus_15"], w["c_minus_15"], w["rule_15"] = w15["c_plus"], w15["c_minus"], w15["rule"]
    w.to_csv(args.out / "weights.csv", index=False)

    # ---------------- A.1 ----------------
    ws = w.sort_values("ratio", ascending=False, na_position="last")
    lines = ["# ステージ59 A.1: 方向別 MAD と重み(D_rec = class 1 → class 0 の 57 社)", "",
             f"Δx̃ = x̃(2024) − x̃(2022)(各年の総資産で割る)。MAD = 中央値からの絶対偏差の中央値"
             f"(1.4826 を掛けない)。ε0 = {EPS0}(総資産比 0.1%)。c = 1 / (s + ε0)。"
             f"Δ = 0 はどちらの向きにも入れない。n < {N_MIN} の方向の扱いは「代替」の列"
             f"(反対の向きの s、両方とも n < {N_MIN} なら信頼できる c の中央値)。"
             "非操作 3 科目は"
             f" c = 0。"
             "参考: 中央値 = 変化した企業での |Δ| の中央値"
             "(大きさの尺度。MAD はばらつきの尺度)。", "",
             "参考: 名目 = Δx / 総資産(2022)(ステージ57 の定義。総資産の増減で比が動く分を含ま"
             "ない)。", "",
             "| 科目 | n(増加) | n(減少) | s+ | s− | c+ | c− | c−/c+ | 代替 | 参考: 中央値+ | "
             "中央値− "
             "| 名目 n+ / n− | 名目 c−/c+ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|"]
    for r in ws.itertuples():
        lines.append(f"| {r.ja} | {r.n_plus} | {r.n_minus} | {r.s_plus_raw:.5f} |"
                     f" {r.s_minus_raw:.5f} "
                     f"| {r.c_plus:.1f} | {r.c_minus:.1f} | {r.ratio:.2f} | {r.rule} "
                     f"| {r.med_plus:.4f} | {r.med_minus:.4f} | {r.n_plus_nom} / {r.n_minus_nom} "
                     f"| {r.ratio_nom:.2f} |")
    lines += ["", "(s± は代替前の値。c± は代替後。)", "",
              "## class 1 を生成するときの代わりの推定(class 2 → 回復の 15 社、D.3)", "",
              "| 科目 | n(増加) | n(減少) | c+ | c− | 代替 |", "|---|---:|---:|---:|---:|---|"]
    for r in w.itertuples():
        lines.append(f"| {r.ja} | {r.n_plus_15} | {r.n_minus_15} | {r.c_plus_15:.1f} "
                     f"| {r.c_minus_15:.1f} | {r.rule_15} |")
    n_sub = int(((w["n_plus_15"] < N_MIN) | (w["n_minus_15"] < N_MIN))[~nact].sum())
    n_sub57 = int(((w["n_plus"] < N_MIN) | (w["n_minus"] < N_MIN))[~nact].sum())
    lines += ["", f"n < {N_MIN} の方向がある科目(操作できる 29 科目中): 57 社 {n_sub57}、15 社"
                  f" {n_sub}。", ""]
    (args.out / "weights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------------- A.2 ----------------
    x, lab, docs, _keep = load_tilde(names)
    is_src = (lab == 2) & ~np.isin(docs, OUTLIERS)
    xs, xt = x[is_src], x[lab == 0]
    cp, cm, cs = (w[c].to_numpy() for c in ("c_plus", "c_minus", "c_sym"))
    costs = {"等重み L2²(現行、非操作 0)": l2_cost(xs, xt, (~nact).astype(float)),
             "対称 L1(MAD の逆数)": l1_cost(xs, xt, cs, cs),
             "非対称 L1(提案)": l1_cost(xs, xt, cp, cm)}
    st = {k: cost_stats(v) for k, v in costs.items()}
    nn = {k: v.argmin(axis=1) for k, v in costs.items()}
    keys = list(costs)
    lines = [f"# ステージ59 A.2: コスト行列の性質(class 2 {len(xs)} 社 × class 0 {len(xt):,} 社)",
             "",
             "C̃ = C / median(C)。行の最小 = 各始点の最近傍までの C̃。行内の分散 = C̃ の行ごと"
             "の分散の中央値。"
             "差 = 2 番目に近い対応先と最近傍の C̃ の差(行ごと、中央値)。", "",
             "| コスト | 最大/中央値 | 行内の分散 | 行の変動係数 | 行の最小 p5 / p50 / p95 / 最"
             "大 | 差 |",
             "|---|---:|---:|---:|---|---:|"]
    for k in keys:
        s = st[k]
        lines.append(f"| {k} | {s['max_over_med']:.1f} | {s['rowvar_med']:.3f} |"
                     f" {s['row_cv_med']:.3f} "
                     f"| {s['rowmin_p5']:.3f} / {s['rowmin_p50']:.3f} / {s['rowmin_p95']:.3f} / "
                     f"{s['rowmin_max']:.3f} | {s['gap_med']:.4f} |")
    lines += ["", "| コスト | 最小 | p1 | p25 | 中央値 | p75 | p99 | 最大 |", "|---|---:|---:|--"
                                                                    "-:|---:|---:|---:|---:|"]
    for k in keys:
        s = st[k]
        lines.append(f"| {k} | " + " | ".join(f"{s[f'p{q}']:.4g}" for q in QS) + " |")
    lines += ["", "**ε(C̃ の単位)との関係**: 最近傍と 2 番目の差(中央値)が ε より小さいと、質量が"
              "複数の対応先に分かれる。exp(−C̃/ε) の最小値は、対数領域では問題にならない。", "",
              "| コスト | "
              + " | ".join(f"ε = {e}: 最大 C̃/ε" for e in (0.001, 0.01, 0.05, 0.1)) + " |",
              "|---|---:|---:|---:|---:|"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(f"{st[k]['max_over_med'] / e:,.0f}"
                                             for e in (0.001, 0.01, 0.05, 0.1)) + " |")
    lines += ["", "**最近傍の一致**(55 社のうち同じ対応先): "
              + "、".join(f"{keys[i]} 対 {keys[j]} {int((nn[keys[i]] == nn[keys[j]]).sum())}"
                         for i, j in ((0, 1), (0, 2), (1, 2))), "",
              "**異なる最近傍の数**: " + "、".join(f"{k} {len(set(nn[k]))}" for k in keys)
              + "。最も多く選ばれた対応先の被選択数: "
              + "、".join(f"{k} {np.bincount(nn[k]).max()}" for k in keys), ""]
    paid_t = xt[:, names.index(_CS)] + xt[:, names.index(_CSUR)]
    lines += ["**最近傍の払込資本比率(中央値)**: " + "、".join(
        f"{k} {np.median(paid_t[nn[k]]):.3f}" for k in keys), ""]
    (args.out / "cost_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------------- A.3 ----------------
    s_tot = x.std(axis=0, ddof=1)
    s1 = pooled_within(x, lab)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    ind = np.array(load_industry_config(Path("conf/accounts/industry.yaml")).labels(
        [code_of.get(d, "") for d in docs]))
    known = ind != "不明"
    s3 = pooled_within(x[known], ind[known])
    s2 = d57.std(axis=0, ddof=1)
    eq = np.ones(3) / 3
    s1e = np.sqrt(sum(eq[k] * x[lab == k].var(axis=0, ddof=1) for k in (0, 1, 2)))
    sd = pd.DataFrame({"account": names, "ja": [ja.get(n, n) for n in names], "sigma_total": s_tot,
                       "sigma_1": s1, "sigma_1_equal": s1e, "sigma_2": s2, "sigma_3": s3})
    for k in ("1", "1_equal", "2", "3"):
        sd[f"ratio_{k}"] = sd[f"sigma_{k}"] / sd["sigma_total"]
    sd.to_csv(args.out / "pooled_sd.csv", index=False)
    lines = ["# ステージ59 A.3: 群内プール分散", "",
             f"断面 = 除外後の class 0 {int((lab == 0).sum()):,}・class 1 {int((lab == 1).sum())}・"
             f"class 2 {int((lab == 2).sum())}。案 1 = class 内の分散を社数で重み付けたプール"
             "(ステージ58 と同じ)。案 1' = 3 class を等しい重みで平均(参考)。"
             "案 2 = D_rec 57 社の 2 年の Δx̃ の標準偏差。案 3 = 33 業種内のプール(不明を除く)。"
             "比 = σ / σ_total(案 2 は別のデータなので参考)。", "",
             "| 科目 | σ_total | σ 案 1 | σ 案 1' | σ 案 2 | σ 案 3 | 比 案 1 | 比 案 1' | 比 案"
             " 2 | 比 案 3 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sd.sort_values("ratio_1").itertuples():
        lines.append(f"| {r.ja} | {r.sigma_total:.4f} | {r.sigma_1:.4f} | {r.sigma_1_equal:.4f} "
                     f"| {r.sigma_2:.4f} | {r.sigma_3:.4f} | {r.ratio_1:.3f} |"
                     f" {r.ratio_1_equal:.3f} "
                     f"| {r.ratio_2:.3f} | {r.ratio_3:.3f} |")
    c_hi = max(cp[~nact].max(), cm[~nact].max())
    c_lo = min(cp[~nact].min(), cm[~nact].min())
    inv = {k: 1.0 / sd[f"sigma_{k}"].to_numpy()[~nact] ** 2 for k in ("1", "1_equal", "2", "3")}
    lines += ["", "**重み比**(1/σ² の最大 / 最小、操作できる 29 科目): "
              + "、".join(f"案 {k} {v.max() / v.min():,.0f}" for k, v in inv.items())
              + f"。c± の最大 / 最小: {c_hi / c_lo:.1f}。", ""]
    (args.out / "pooled_sd.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
