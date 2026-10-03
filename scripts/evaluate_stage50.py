#!/usr/bin/env python
"""ステージ50: 分布由来のコストによる対応先の評価(C・D)。

入力は ``metrics_stage39.py --gen-dir data/interim/stage50/gen --out data/interim/stage50/metrics``
の出力、生成の parquet、``precompute_stage50_transport.py`` の対応先。判定の基準は
``results/stage50/criteria.md``(生成の前に記録)。

出力(``results/stage50/``): ``comparison.md``(C.2・D.1)、``targets.md``(B.3・A.6・C.3)、
``zeros.md``(C.4・D.2)、``bs.md``(C.5)、``summary.json``。

    uv run python scripts/evaluate_stage50.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_stage39 import boot, comp_level, corr, fmt_ci, verdict
from run_stage25 import CONTINUOUS, OUTLIERS, summarize

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import INACTIVE_ACCOUNTS
from fsgen.models.industry import load_industry_config
from fsgen.viz.balance_sheet import build_bs_lines, load_labels

_M = Path("data/interim/stage50/metrics")
_GEN = Path("data/interim/stage50/gen")
_PLANS = Path("data/interim/stage50/plans")
_EQUAL_TARGETS = Path("data/interim/stage28/plans_equal/full_targets_nearest_na0.csv")
EXCL = {"c2": tuple(OUTLIERS), "c1": ("S100VGDJ",)}
CLS = {"c2": "class 2", "c1": "class 1"}
SETTINGS = ("equal", "sub_all", "sub_trim", "t", "lognorm", "t_unb", "t_pi12")
SET_LABEL = {"equal": "1 等重み最近傍(現行)", "sub_all": "2 尤度・Subbotin 全件",
             "sub_trim": "3 尤度・Subbotin 外れ値除去", "t": "4 尤度・t【主】",
             "lognorm": "5 尤度・対数正規", "t_unb": "6 尤度・t + 不均衡 OT",
             "t_pi12": "7 尤度・t・π は class 1+2(感度)"}
CANDIDATES = ("t", "t_unb")  # 採用の候補(この順に判定、criteria.md §7)
PRIMARY = (("net_asset_improvement", "純資産の改善", "大"),
           ("abs_asset_change", "|総資産変化|", "小"))
ZERO, NONZERO = 1e-6, 1.001e-3  # ステージ41 の新規計上の定義
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
_TS = "jppfs_cor_TreasuryStock"


def slabel(s: str) -> str:
    base = s.removesuffix("_noinact")
    return SET_LABEL[base] + ("(不活性制約なし)" if s != base else "")


def load_gen(cls: str, s: str) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in sorted((_GEN / cls / s).glob("*__seed*.parquet"))])
    return g[(g["method"] == "PCFM") & ~g["docID"].isin(EXCL[cls])].reset_index(drop=True)


def section_c2(rng: np.random.Generator, names: tuple[str, ...], versions: tuple[str, ...]
               ) -> tuple[list[str], dict[str, dict[str, object]]]:
    """C.2 の比較と D.1 の判定(``versions`` の各設定 対 equal)。"""
    labels = {m: lab for m, lab, _ in CONTINUOUS}
    lines: list[str] = []
    judg: dict[str, dict[str, object]] = {}
    for cls in ("c2", "c1"):
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[~rows["docID"].isin(EXCL[cls]) & rows["version"].isin(versions)]
        comp = comp_level(rows)
        viol = pd.read_csv(_M / f"viol_{cls}.csv", dtype={"docID": str})
        viol = viol[~viol["docID"].isin(EXCL[cls])]
        dels = pd.read_parquet(_M / f"deltas_{cls}.parquet")
        dels = dels[(dels["space"] == "native") & ~dels["docID"].isin(EXCL[cls])]
        refs = pd.read_parquet(_M / "refs.parquet")
        ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index(
            "account")["median"]
        wide = {s: dels[dels["setting"] == s].pivot(index="docID", columns="account",
                                                    values="delta")[list(names)]
                for s in versions}
        summ, pr = summarize(comp, (), versions, "equal")
        s_ = summ.set_index("version")
        n = int(s_.loc["equal", "n"])
        lines += [f"## {CLS[cls]}({n} 社)", "",
                  "値は企業ごとの seed 中央値の中央値。差は企業単位のペアの差の中央値 [95% 区間]"
                  "(対 設定 1)。判定: + 有意に優れる、− 有意に劣る、0 どちらでもない。", "",
                  "| 設定 | 違反(行) | 純資産の改善 | 差 | \\|総資産変化\\| | 差 | 順位相関 | 差 "
                  "| 判定(改善・総資産・相関) | 解消 | sparsity | validity | proximity |",
                  "|---|---:|---:|---|---:|---|---:|---|---|---:|---:|---:|---:|"]
        ce = corr(wide["equal"], ref)
        for s in versions:
            nv = int(viol[viol["setting"] == s]["rows"].sum())
            cs = corr(wide[s], ref)
            cells, marks = [], []
            for m, lab, better in PRIMARY:
                v = s_.loc[s, m]
                if s == "equal":
                    cells += [f"{v:.3f}", "—"]
                    continue
                q = pr[(pr["version"] == s) & (pr["metric"] == lab)].iloc[0]
                d, lo, hi = (float(q["median_diff(version-equal)"]), float(q["ci_low"]),
                             float(q["ci_high"]))
                cells += [f"{v:.3f}", fmt_ci(d, lo, hi, 3)]
                marks.append(verdict(better, d, lo, hi))
            if s == "equal":
                ccell = "—"
            else:
                _, _, dlo, dhi = boot(wide[s], ref, wide["equal"].loc[wide[s].index], ref, rng)
                ccell = fmt_ci(cs - ce, dlo, dhi, 3)
                marks.append(verdict("大", cs - ce, dlo, dhi))
            sym = " ".join({1: "+", -1: "−", 0: "0"}[k] for k in marks) if marks else "—"
            res = (f"{int(s_.loc[s, 'resolved_firms(>=2/3 seeds)'])}/"
                   f"{int(s_.loc[s, 'insolvent_start'])}")
            lines.append(
                f"| {slabel(s)} | {nv} / {3 * n} | {cells[0]} | {cells[1]} | {cells[2]} "
                f"| {cells[3]} | {cs:+.3f} | {ccell} | {sym} | {res} "
                f"| {s_.loc[s, 'sparsity']:.0f} | {s_.loc[s, 'validity_lightgbm']:.3f} "
                f"| {s_.loc[s, 'proximity_l2']:.3f} |")
            if s != "equal":
                judg.setdefault(s, {})[cls] = {
                    "violations": nv, "wins": sum(k == 1 for k in marks),
                    "losses": sum(k == -1 for k in marks), "marks": marks,
                    "ok": nv == 0 and sum(k == 1 for k in marks) >= 2
                    and not any(k == -1 for k in marks)}
        lines.append("")
        # 副指標の差(参考)
        lines += [f"### {CLS[cls]}: 副指標の差(対 設定 1、参考)", "",
                  "| 設定 | sparsity | validity LightGBM | proximity | 解消率 |", "|---|---|---|"
                                                                             "---|---|"]
        for s in versions[1:]:
            cl = []
            for lab in (labels["sparsity"], labels["validity_lightgbm"], labels["proximity_l2"],
                        "債務超過の解消率(社ごとの 3 seed 平均)"):
                q = pr[(pr["version"] == s) & (pr["metric"] == lab)].iloc[0]
                cl.append(fmt_ci(float(q["median_diff(version-equal)"]), float(q["ci_low"]),
                                 float(q["ci_high"]), 3))
            lines.append(f"| {slabel(s)} | " + " | ".join(cl) + " |")
        lines.append("")
    return lines, judg


def targets_section(names: tuple[str, ...]) -> tuple[list[str], dict[str, object]]:
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    paid = (agg[_CS].fillna(0) + agg[_CSUR].fillna(0)) / agg["total_assets"]
    ind = load_industry_config(Path("conf/accounts/industry.yaml"))
    code = agg["edinetCode"].astype(str)
    eq = pd.read_csv(_EQUAL_TARGETS, dtype={"source_docID": str, "target_docID": str})
    eq_t = eq.set_index("source_docID")["target_docID"]
    lines = ["## C.3 対応先の変化(全社の計画、class 2 57 社)", "",
             "対応先 = 計画の行の最大の列。払込資本比率 = (資本金 + 資本剰余金)/ 総資産。業種は "
             "33 業種。"
             f"始点の払込資本比率の中央値 {paid.loc[eq['source_docID']].median():.3f}。", "",
             "| 設定 | 対応先が変わった社数(57 社中) | 異なる対応先の企業数 | 対応先の払込資本"
             "比率(中央値)"
             " | 業種一致率 | 対応先の総資産 / 始点(中央値) |", "|---|---:|---:|---:|---:|---:|"]
    out: dict[str, object] = {}
    for s in SETTINGS:
        if s == "equal":
            t = eq.rename(columns={"target_docID": "tgt"})[["source_docID", "tgt", "asset_ratio"]]
        else:
            f = pd.read_csv(_PLANS / f"full_targets_{s}.csv",
                            dtype={"source_docID": str, "target_docID": str})
            t = f.rename(columns={"target_docID": "tgt"})[["source_docID", "tgt", "asset_ratio"]]
        changed = int((t.set_index("source_docID")["tgt"] != eq_t.loc[t["source_docID"]]).sum())
        src_ind = ind.labels(code.loc[t["source_docID"]].tolist())
        tgt_ind = ind.labels(code.loc[t["tgt"]].tolist())
        match = float(np.mean([a == b and a != "不明" for a, b in zip(src_ind, tgt_ind,
                                                                     strict=True)]))
        pr = float(paid.loc[t["tgt"]].median())
        lines.append(f"| {SET_LABEL[s]} | {changed} | {t['tgt'].nunique()} | {pr:.3f} "
                     f"| {match:.1%} | {t['asset_ratio'].median():.2f} |")
        out[s] = {"changed": changed, "unique": int(t["tgt"].nunique()), "paid_ratio": pr,
                  "industry_match": match}
    return lines, out


def zeros_section(names: tuple[str, ...]) -> tuple[list[str], dict[str, object]]:
    rates = pd.read_csv("results/stage41/rates.csv").set_index("account")
    others = [n for n in names if n not in INACTIVE_ACCOUNTS]
    lines = ["# ステージ50 C.4・D.2: ゼロの扱いの効果", "",
             "新規計上 = 始点が |x| / 総資産 < 1e-6 の行のうち、終点が 1.001e-3 超(ステージ41 "
             "の定義)。"
             "行単位(企業 × seed、外れ値を除く)。「不活性制約なし」は `--inactive none` で生成"
             "した版。"
             "チェック 15 = 規則 M の 4 科目のうち始点 0 の科目が 1e-6 を超えて動いた行。", ""]
    out: dict[str, object] = {}
    for cls in ("c2", "c1"):
        lines += [f"## {CLS[cls]}", "",
                  "| 設定 | 不活性制約 | 自己株式 | 規則 M の他の 3 科目 | その他の 28 科目 | "
                  "チェック 15 の違反(行) |",
                  "|---|---|---|---|---|---:|"]
        for s in SETTINGS:
            for suffix, lab in (("", "あり"), ("_noinact", "なし")):
                g = load_gen(cls, s + suffix)
                x0 = g[[f"x0_{n}" for n in names]].to_numpy()
                x1 = g[[f"x1_{n}" for n in names]].to_numpy()
                zero = np.abs(x0) < ZERO
                born = zero & (np.abs(x1) > NONZERO)
                moved = zero & (np.abs(x1 - x0) > ZERO)

                def rate(cols: list[str], z: np.ndarray = zero, b: np.ndarray = born) -> str:
                    idx = [names.index(c) for c in cols]
                    nz, nb = int(z[:, idx].sum()), int(b[:, idx].sum())
                    return f"{nb / nz:.1%}({nb}/{nz})" if nz else "—"

                m3 = [a for a in INACTIVE_ACCOUNTS if a != _TS]
                c15 = int(moved[:, [names.index(a) for a in INACTIVE_ACCOUNTS]].any(axis=1).sum())
                lines.append(f"| {SET_LABEL[s]} | {lab} | {rate([_TS])} | {rate(m3)} "
                             f"| {rate(others)} | {c15} / {len(g)} |")
                out[f"{cls}_{s}{suffix}"] = {
                    "ts_born": int(born[:, names.index(_TS)].sum()),
                    "ts_zero": int(zero[:, names.index(_TS)].sum()), "check15": c15}
        lines.append("")
    r = rates.loc[_TS]
    lines += ["実データ(ステージ41、推定用パネルの class 1+2 の 1 年): 自己株式の新規計上 "
              f"{int(r['k_c12'])} / {int(r['n_zero_c12'])}"
              f"(Wilson 95% {r['c12_lo']:.1%}〜{r['c12_hi']:.1%})。", ""]
    # 対応先の段階: 始点で 0 の科目について、対応先が非 0 の割合(生成の前、計画の値)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    lines += ["## 対応先の段階(生成の前): 始点で 0 の科目の対応先が非 0 の割合", "",
              "全社の計画(class 2 57 社)。0 = 総資産比 < 1e-6。", "",
              "| 設定 | 自己株式 | 規則 M の他の 3 科目 | その他の 28 科目 |", "|---|---|---|---|"]
    for s in SETTINGS:
        path = _EQUAL_TARGETS if s == "equal" else _PLANS / f"full_targets_{s}.csv"
        f = pd.read_csv(path, dtype={"source_docID": str, "target_docID": str})

        def share(cols: list[str], f: pd.DataFrame = f) -> str:
            sa = agg.loc[f["source_docID"]]
            ta = agg.loc[f["target_docID"]]
            xs = sa[cols].fillna(0).to_numpy() / sa["total_assets"].to_numpy()[:, None]
            xt = ta[cols].fillna(0).to_numpy() / ta["total_assets"].to_numpy()[:, None]
            z = np.abs(xs) < ZERO
            k = int((z & (np.abs(xt) >= ZERO)).sum())
            return f"{k / z.sum():.0%}({k}/{int(z.sum())})" if z.sum() else "—"

        m3 = [a for a in INACTIVE_ACCOUNTS if a != _TS]
        lines.append(f"| {SET_LABEL[s]} | {share([_TS])} | {share(m3)} | {share(others)} |")
    lines.append("")
    return lines, out


def mechanism_section() -> list[str]:
    """純資産の改善が下がる仕組み: 利益剰余金の変化の分解と払込資本の変化(総資産(0)比、行の中央"
      "値)。"""
    lines = ["## 仕組み: 利益剰余金の変化の分解と払込資本", "",
             "行(企業 × seed)の中央値、総資産(0)比、t = 1。振替 = 払込資本から利益剰余金への"
             "振替(欠損填補)、利益部分 = それ以外の利益剰余金の増加。", "",
             "| class | 設定 | 振替 | 利益部分 | 利益部分 > 0.001 の行 | Δ払込資本 |",
             "|---|---|---:|---:|---:|---:|"]
    for cls in ("c2", "c1"):
        for s in SETTINGS:
            g = load_gen(cls, s)
            paid = (g[f"x1_{_CS}"] - g[f"x0_{_CS}"]) + (g[f"x1_{_CSUR}"] - g[f"x0_{_CSUR}"])
            lines.append(f"| {CLS[cls]} | {SET_LABEL[s]} | {g['re_transfer'].median():.3f} "
                         f"| {g['re_profit'].median():.3f} | {(g['re_profit'] > 1e-3).mean():.0%} "
                         f"| {paid.median():+.3f} |")
    lines.append("")
    return lines


def bs_section(names: tuple[str, ...], best: str) -> list[str]:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    gens = {}
    for s in ("equal", best):
        (f,) = sorted((_GEN / "c2" / s).glob("*__seed0.parquet"))
        g = pd.read_parquet(f)
        gens[s] = g[g["method"] == "PCFM"].set_index("docID")
    tg = {"equal": pd.read_csv(_EQUAL_TARGETS, dtype={"source_docID": str}),
          best: pd.read_csv(_PLANS / f"full_targets_{best}.csv", dtype={"source_docID": str})}
    pool = sorted(set(gens["equal"].index) - set(EXCL["c2"]))
    docs = [str(d) for d in np.random.default_rng(0).choice(pool, 5, replace=False)]
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    out = ["# ステージ50 C.5: BS 表(class 2、seed 0、百万円、t = 1 は 5 年)", "",
           f"企業は外れ値を除く class 2(55 社)から乱数(seed 0)で選んだ 5 社(結果を見て選ばない)。"
           f"列は始点、設定 1(等重み最近傍)、設定 `{best}` の終点。対応先は全社の計画の値で、"
           "その企業の生成は、その企業を除いた fold の訓練対で学習した速度場による"
           "(対応先そのものへ向かうわけではない)。", ""]
    for d in docs:
        r = gens["equal"].loc[d]
        te = tg["equal"].set_index("source_docID").loc[d, "target_name"]
        tb = tg[best].set_index("source_docID").loc[d, "target_name"]
        cols = [("始点", r[x0c].to_numpy(float)), ("等重み", r[x1c].to_numpy(float)),
                (best, gens[best].loc[d][x1c].to_numpy(float))]
        assets = float(r["assets_start"])
        bl = build_bs_lines(spec, names, [v * assets for _, v in cols], labels)
        out += [f"## {r['filerName']}({d})", "", f"対応先: 等重み = {te}、{best} = {tb}", "",
                "| 科目 | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
        for line in bl:
            if line.values is None:
                out.append(f"| **{line.label}** |" + " |" * len(cols))
                continue
            lab = "&nbsp;&nbsp;" * line.indent + line.label
            bold = "**" if line.kind in ("subtotal", "total", "diff") else ""
            out.append(f"| {bold}{lab}{bold} | "
                       + " | ".join(f"{v / 1e6:,.0f}" for v in line.values) + " |")
        out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage50"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    summary: dict[str, object] = {}
    lines, judg = section_c2(rng, names, SETTINGS)
    jl = ["## D.1 判定(対 設定 1)", "",
          "基準: 15 項目の違反 0、主指標の 2 つ以上で有意に優れ有意に劣るものがない、両 class "
          "で成立。"
          "候補は設定 4、次に 6(`criteria.md` §7)。", "",
          "| 設定 | class 2 | class 1 | 両 class で成立 | 扱い |", "|---|---|---|---|---|"]
    adopted = None
    for s in SETTINGS[1:]:
        j = judg[s]
        both = bool(j["c2"]["ok"]) and bool(j["c1"]["ok"])  # type: ignore[index]
        role = "候補" if s in CANDIDATES else "感度分析(採用しない)"
        if both and s in CANDIDATES and adopted is None:
            adopted = s
        cell = {c: f"違反 {j[c]['violations']}、勝ち {j[c]['wins']}・負け {j[c]['losses']}"  # type: ignore[index]
                for c in ("c2", "c1")}
        jl.append(f"| {SET_LABEL[s]} | {cell['c2']} | {cell['c1']} |"
                  f" {'成立' if both else '不成立'} "
                  f"| {role} |")
    jl += ["", f"**採用: {adopted if adopted else 'なし(最終設定を変えない)'}**", ""]
    head = ["# ステージ50 C.2・D.1: 対応先の選択の比較", "",
            "生成は対応先の選択だけを変え、他は最終設定 `s47_final`(不活性制約あり)。seed 0/1/2。"
            "順位相関の実データ: class 2 は再建群 15 社の 2 年、class 1 は class 1 → class 0 の "
            "1 年の 68 組。"
            "差の区間は企業の復元抽出(順位相関は 1,000 回、seed 0)。", ""]
    (args.out / "comparison.md").write_text("\n".join(head + jl + lines + mechanism_section())
                                            + "\n", encoding="utf-8")
    summary["judgment"] = judg
    summary["adopted"] = adopted

    # D.2: 不活性制約なしの版の判定(参考を含め全設定)
    nl, nj = section_c2(np.random.default_rng(0), names,
                        ("equal", *[s + "_noinact" for s in SETTINGS]))
    summary["judgment_noinact"] = nj

    tl, tsum = targets_section(names)
    checks = pd.read_csv(args.out / "cost_checks.csv")
    period = pd.read_csv(args.out / "period.csv")
    b3 = ["# ステージ50 B.3・A.6・C.3: コストと対応先", "", "## B.3 数値の確認(全社のコスト行列 "
                                               "57 × 3,768)",
          "", "| 版 | 有限 | 最小 | 中央値 | 最大 | (最大 − 最小)/(中央値 − 最小) | コストの科目数 "
          "| クリップした確率 | 計算時間(全社、秒) | 輸送された質量 | 脱落した始点 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in checks.itertuples():
        b3.append(f"| {r.design} | {r.finite} | {r.cost_min:.1f} | {r.cost_median:.1f} "
                  f"| {r.cost_max:.1f} | {r.max_over_median_shifted:.2f} | {r.n_accounts_in_cost} "
                  f"| {'—' if pd.isna(r.n_clipped_probs) else int(r.n_clipped_probs)} "
                  f"| {r.seconds_full_cost:.2f} | {r.transported_mass:.3f} | {r.n_dropped_full} |")
    b3 += ["", "## A.6 期間: 案 1(1 年)と案 2(位置 × 5、尺度 × √5)で対応先が変わった数", "",
           "| 版 | 全社(57 社中) | fold の訓練の行(延べ) |", "|---|---:|---:|"]
    for r in period.itertuples():
        b3.append(f"| {r.version} | {r.changed_full} | {r.changed_train_rows} / {r.train_rows} |")
    b3.append("")
    (args.out / "targets.md").write_text("\n".join(b3 + tl) + "\n", encoding="utf-8")
    summary["targets"] = tsum

    zl, zsum = zeros_section(names)
    zl += ["## D.2 不活性制約なしの版の比較(対 設定 1 = 不活性制約ありの現行)", ""] + nl
    (args.out / "zeros.md").write_text("\n".join(zl) + "\n", encoding="utf-8")
    summary["zeros"] = zsum
    best = adopted if adopted else "t"
    (args.out / "bs.md").write_text("\n".join(bs_section(names, best)) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                      default=str), encoding="utf-8")
    print("\n".join(head + jl))


if __name__ == "__main__":
    main()
