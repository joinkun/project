#!/usr/bin/env python
"""ステージ49: ``growth_dist_stage49.py`` の CSV を報告用の表(Markdown)にまとめる。

群の比較には、Subbotin のパラメータに加えて |g| の中央値(群ごと)を並べる
(b が小さいと尺度パラメータが
分布の幅を表さないため)。

    uv run python scripts/summarize_stage49.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

_PANEL = Path("data/interim/stage23/panel/panel.csv")


def f3(v: float) -> str:
    return "—" if pd.isna(v) else f"{v:.3f}"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("results/stage49"))
    args = ap.parse_args()
    d = args.dir
    z = pd.read_csv(d / "zeros.csv")
    fits = pd.read_csv(d / "fits.csv")
    asym = pd.read_csv(d / "asymmetry.csv")
    bycls = pd.read_csv(d / "by_class.csv")
    byind = pd.read_csv(d / "by_industry.csv")
    conv = pd.read_csv(d / "convexity.csv")
    diff = pd.read_csv(d / "diff_gennorm.csv")
    # |g| の中央値(群ごと)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    p = p[(p["prev_total_assets"] > 0) & (p["cur_total_assets"] > 0)].reset_index(drop=True)
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    cls = np.array([lab.get((c, t - 1), np.nan) for c, t in zip(p["edinetCode"], p["t"],
                                                               strict=True)])
    med_cls: dict[str, dict[float, float]] = {}
    out_any = np.zeros(len(p), dtype=bool)  # いずれかの科目で Tukey の外れ値になった組
    for n in names:
        x0 = p[f"prev_{n}"].fillna(0.0).to_numpy() / p["prev_total_assets"].to_numpy()
        x1 = p[f"cur_{n}"].fillna(0.0).to_numpy() / p["cur_total_assets"].to_numpy()
        same = (x0 != 0) & (x1 != 0) & (np.sign(x0) == np.sign(x1))
        g = np.abs(np.log(np.abs(x1[same]) / np.abs(x0[same])))
        med_cls[n] = {k: float(np.median(g[cls[same] == k])) if (cls[same] == k).sum() else np.nan
                      for k in (0.0, 1.0, 2.0)}
        q1, q3 = np.percentile(g, [25, 75])
        out_any[np.flatnonzero(same)[g > q3 + 3 * (q3 - q1)]] = True

    # A.3 外れ値の組の特徴
    la = np.log10(p["prev_total_assets"].to_numpy())
    ga = np.abs(np.log(p["cur_total_assets"].to_numpy() / p["prev_total_assets"].to_numpy()))
    o, k_ = out_any, ~out_any
    lines = ["# ステージ49 A.3: 外れ値(Tukey、|g| > Q3 + 3 IQR)の組の特徴", "",
             "いずれかの科目で外れ値になった組と、そうでない組を比べる(推定用パネル、1 年)。"
             "class の割合の残りはラベルなし。", "",
             "| 項目 | 外れ値を含む組 | 含まない組 |", "|---|---:|---:|",
             f"| 組の数 | {int(o.sum())} | {int(k_.sum())} |"]
    for c in (0.0, 1.0, 2.0):
        lines.append(f"| 始点の class {int(c)} の割合 | {np.mean(cls[o] == c):.3f} "
                     f"| {np.mean(cls[k_] == c):.3f} |")
    lines += [f"| 前期の総資産(log10 円、中央値) | {np.median(la[o]):.2f} "
              f"| {np.median(la[k_]):.2f} |",
              f"| 総資産の変化 \\|log(A_t/A_t-1)\\|(中央値) | {np.median(ga[o]):.3f} "
              f"| {np.median(ga[k_]):.3f} |",
              f"| 総資産が 2 倍以上・半分以下に変わった組の割合 | {np.mean(ga[o] > np.log(2)):.3f} "
              f"| {np.mean(ga[k_] > np.log(2)):.3f} |",
              "", "科目ごとの外れ値の数は `asymmetry.md` の最後の列。"]
    (d / "outliers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # A.2・B.4 ゼロの構造
    lines = ["# ステージ49 A.2・B.4: ゼロの構造と不動", "",
             "推定用パネル 10,971 組(1 年)。割合は全組に対する。「符号反転」は両期非 0 で符号が"
             "逆の組"
             "(分布の推定から除く)。不動: 両期とも同符号で非 0 の組のうち、|g| < 0.01(総資産比"
             "の値の変化率)、"
             "名目額の |log 比| < 0.001、名目額がまったく同じ組の割合。", "",
             "| 科目 | 符号 | 両期 0 | 新規計上 | 消滅 | 両期非 0 | 符号反転 | 推定に使う組 "
             "| \\|g\\| < 0.01 | 名目 < 0.001 | 名目が同じ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in z.itertuples():
        lines.append(f"| {r.ja} | {r.sign} | {r.both0:.3f} | {r.new:.3f} | {r.gone:.3f} "
                     f"| {r.both:.3f} | {r.flip:.3f} | {r.n_both} | {f3(r.g_lt001)} "
                     f"| {f3(r.gn_lt0001)} | {f3(r.gn_eq0)} |")
    (d / "zeros.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # B.1・B.2 科目別の分布
    lines = ["# ステージ49 B.1・B.2: 科目別の分布(g = log R)", "",
             "最良 = AIC が最小の分布族。正規との尤度比 = 2(LL_最良 − LL_正規)。KS は最良の分布"
             "族の値。"
             "b = Subbotin(一般化正規)の形状(2 = 正規、1 = Laplace、< 1 はさらに尖る)。", ""]
    for version in ("全件", "外れ値を除く"):
        s = fits[fits["version"] == version]
        best = s[s["best_aic"]].set_index("ja")
        sub = s[s["family"] == "Subbotin"].set_index("ja")
        norm = s[s["family"] == "正規"].set_index("ja")
        lines += [f"## {version}", "",
                  "| 科目 | n | 最良 | AIC(最良) | BIC(最良) | KS(最良) | 正規との尤度比 | Subbo"
                  "tin b "
                  "| Subbotin 尺度 | KS(Subbotin) | KS(正規) |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for ja in sub.index:
            bst = best.loc[ja]
            lines.append(f"| {ja} | {int(bst['n'])} | {bst['family']} | {bst['aic']:.0f} "
                         f"| {bst['bic']:.0f} | {bst['ks']:.3f} | {bst['lr_vs_normal']:.0f} "
                         f"| **{sub.loc[ja, 'p_b']:.3f}** | {sub.loc[ja, 'p_scale']:.4f} "
                         f"| {sub.loc[ja, 'ks']:.3f} | {norm.loc[ja, 'ks']:.3f} |")
        vc = s[s["best_aic"]]["family"].value_counts()
        lines += ["", "AIC で最良の分布族の数: " + "、".join(f"{k} {v}" for k, v in vc.items())
                  + "。BIC で最良: " + "、".join(
                      f"{k} {v}" for k, v in s.loc[s.groupby("account")["bic"].idxmin(),
                                                    "family"].value_counts().items())
                  + f"。Subbotin の b: 中央値 {sub['p_b'].median():.3f}、範囲"
                    f" {sub['p_b'].min():.3f}〜"
                  f"{sub['p_b'].max():.3f}、b < 1 の科目 {int((sub['p_b'] < 1).sum())} /"
                  f" {len(sub)}。", ""]
    (d / "accounts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # B.3 非対称性
    lines = ["# ステージ49 B.3: 非対称性(全件)", "",
             "頻度: g > 0(増加)の割合と二項検定(g = 0 を除く)。大きさ: 増加側・減少側の |g| の"
             "平均(位置 0 の指数分布の"
             "尺度)と、尺度が等しいとする尤度比検定(自由度 1)。歪み Subbotin の右 / 左の尺度の"
             "比と、Subbotin との尤度比検定。"
             "外れ値: Tukey(|g| > Q3 + 3 IQR)で除いた組の数と割合。", "",
             "| 科目 | 増加の割合 | p(頻度) | 増加側の尺度 | 減少側の尺度 | 比 | p(大きさ) "
             "| 歪み Subbotin 右/左 | p | 外れ値 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in asym.itertuples():
        lines.append(f"| {r.ja} | {r.share_pos:.3f} | {r.p_binom:.2g} | {r.scale_pos:.3f} "
                     f"| {r.scale_neg:.3f} | {r.ratio:.2f} | {r.p_scale:.2g} |"
                     f" {r.aep_r_over_l:.2f} "
                     f"| {r.p_aep:.2g} | {r.n_outliers}({r.outlier_share:.1%}) |")
    (d / "asymmetry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # C.1・C.2 群別
    for df, title, fname, groups in (
        (bycls, "C.1 class 別", "by_class.md", (0.0, 1.0, 2.0)),
        (byind, "C.2 業種別(粗い分類 6 区分)", "by_industry.md",
         ("製造", "小売・卸売", "情報通信", "サービス", "不動産・建設", "その他(非金融)")),
    ):
        lines = [f"# ステージ49 {title}: Subbotin の b [Wald 95%] と n(全件)", "",
                 "p = 群ごとに当てはめた場合と、まとめて当てはめた場合の尤度比検定(自由度 3 ×("
                 "群の数 − 1))。"
                 "n < 30 の群は当てはめない。", ""]
        glab = [("class " + str(int(g))) if isinstance(g, float) else str(g) for g in groups]
        lines += ["| 科目 | " + " | ".join(glab) + " | p |",
                  "|---|" + "---|" * len(groups) + "---:|"]
        for ja, sub in df.groupby("ja", sort=False):
            cells = []
            for g in groups:
                row = sub[sub["group"].astype(str) == str(g)]
                if row.empty:
                    cells.append("—")
                else:
                    r = row.iloc[0]
                    cells.append(f"{r['b']:.2f} [{r['b_lo']:.2f}, {r['b_hi']:.2f}](n"
                                 f" {int(r['n'])})")
            lines.append(f"| {ja} | " + " | ".join(cells) + f" | {sub['p_lr'].iloc[0]:.2g} |")
        if fname == "by_class.md":
            lines += ["", "## \\|g\\| の中央値(class 別。変化の大きさの比較)", "",
                      "| 科目 | class 0 | class 1 | class 2 | class 2 / class 0 |", "|---|---:|-"
                                                                                  "--:|---:|---:|"]
            jmap = dict(zip(z["account"], z["ja"], strict=True))
            for n, v in med_cls.items():
                lines.append(f"| {jmap[n]} | {f3(v[0.0])} | {f3(v[1.0])} | {f3(v[2.0])} "
                             f"| {f3(v[2.0] / v[0.0]) if v[0.0] else '—'} |")
        (d / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # D.2 凸性
    lines = ["# ステージ49 D.2: コスト c(Δ) = −log p(log(1 + Δ/x)) の凸性", "",
             "当てはめた Subbotin(全件)の c(u) = (|log(1+u) − μ| / s)^b を、u = Δ/x の等間隔の"
             "格子(R = 1 + u が "
             "0.01〜10、刻み 1e-4)で 2 階差分を取り、最頻値を含む凸な区間を求めた。b < 1 では最"
             "頻値で尖り、両側が凹で"
             "凸な区間を持たない。「観測の割合」= 両期非 0 の組の R がその区間に入る割合。対数"
             "正規(案 A)= b = 2 で同じ判定。", "",
             "| 科目 | b | Subbotin で凸な区間 R | 観測の割合 | 対数正規で凸な区間 R | 観測の割"
             "合 |",
             "|---|---:|---|---:|---|---:|"]
    for r in conv.itertuples():
        sub_ = (f"[{r.R_lo:.2f}, {r.R_hi:.2f}]" if r.convex_at_mode else "なし(非凸)")
        lines.append(f"| {r.ja} | {r.b:.3f} | {sub_} | {r.share_in:.3f} "
                     f"| [{r.R_lo_lognormal:.2f}, {r.R_hi_lognormal:.2f}] "
                     f"| {r.share_in_lognormal:.3f} |")
    lines += ["", f"Subbotin で凸な区間を持つ科目: {int(conv['convex_at_mode'].sum())} /"
                  f" {len(conv)}。",
              "", "補足: 総資産比の差分 d = x̃_t − x̃_{t−1}(両期 0 を除く)に Subbotin を当ては"
                  "めた b: 中央値 "
              f"{diff['b'].median():.3f}、範囲 {diff['b'].min():.3f}〜{diff['b'].max():.3f}、b ≥"
              f" 1 の科目 "
              f"{int((diff['b'] >= 1).sum())} / {len(diff)}。差分で定義しても −log p は Δ につ"
              f"いて凸にならない。"]
    (d / "convexity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
