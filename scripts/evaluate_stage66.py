#!/usr/bin/env python
"""ステージ66 C: 5 設定(と η の感度)の比較。λ は class 1 だけで較正し、class 2 には当てはめるだけ。

規則は ``results/stage66/criteria.md``。生成は ``data/interim/stage66/all/<cls>/<設定>__l<λ̃>``
(設定 1 はステージ65 の提案、class 2 は class 1 の λ で生成し直したもの)。事前に
``metrics_stage39.py --gen-dir data/interim/stage66/all --out data/interim/stage66/metrics``
を実行する。

    uv run python scripts/evaluate_stage66.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from compare_stage39 import corr
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from evaluate_stage65 import real_firm_values

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import CHECK_NAMES, EXTRA_CHECK_NAMES, subtotal_members

OUT = Path("results/stage66")
DATA = Path("data/interim/stage66")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
SETTINGS = {"s1": "1 現行(ステージ65)", "s2": "2 + 階層分離(m 廃止)",
            "s3": "3 + 頻度加重", "s4": "**4 + Elastic-Net(提案、η = 1)**",
            "s5": "5 4 + 1 段階統合", "s4e05": "参考: 4 で η = 0.5", "s4e2": "参考: 4 で η = 2"}
STEPS = (("s2", "s1", "階層分離"), ("s3", "s2", "頻度加重"), ("s4", "s3", "Elastic-Net"),
         ("s5", "s4", "1 段階統合"))
CHECKS = (*CHECK_NAMES, *EXTRA_CHECK_NAMES)
BETA = 0.11732
_CS, _CSUR, _CASH = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus",
                     "jppfs_cor_CashAndDeposits")


def excl() -> None:
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / "s4__l1").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))


def extra_cols(cls: str, s: str, names: tuple[str, ...], members: dict[str, np.ndarray]
               ) -> pd.DataFrame:
    g = e51.load_gen(cls, s)
    g = g[g["seed"] == 0].set_index("docID")
    x0 = g[[f"x0_{n}" for n in names]].to_numpy()
    x1 = g[[f"x1_{n}" for n in names]].to_numpy()
    a0 = x0 @ members["jppfs_cor_Assets"]
    pi = [names.index(_CS), names.index(_CSUR)]
    dp = (x1[:, pi].sum(axis=1) - x0[:, pi].sum(axis=1)) / a0 / 5.0
    ic = names.index(_CASH)
    floor = (x0[:, ic] > 0) & (x1[:, ic] <= BETA * x0[:, ic] + 1e-6)
    return pd.DataFrame({"paid_inc": (dp > 0.001).astype(float), "cash_floor": floor.astype(float)},
                        index=g.index)


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    e51._M = DATA / "metrics"
    e51._GEN = DATA / "all"
    excl()
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    rng = np.random.default_rng(0)
    # λ̃*_0 を class 1 で
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    star: dict[str, float | None] = {}
    lines_cal = ["# ステージ66 C.4: λ の較正(class 1 の 128 社だけで選ぶ)", "",
                 "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_0 |",
                 "|---|" + "---:|" * (len(LAMS) + 1)]
    for st in SETTINGS:
        cells: list[str] = []
        best: tuple[float | None, float] = (None, np.inf)
        for lt in LAMS:
            s = f"{st}__l{lt:g}"
            if not (e51._GEN / "c1" / s).exists():
                cells.append("—")
                continue
            f = gen_firms("c1", s, names, members)
            f = f[~f.index.isin(e51.EXCL["c1"])]
            nv = ctx["c1"].viol[ctx["c1"].viol["setting"] == s]["docID"].nunique()
            g = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                 "d_assets": float(f["d_assets"].median())}
            lv = loss(g, r1, w1)
            cells.append(f"{lv:.2f}({nv})")
            if nv == 0 and lv < best[1]:
                best = (lt, lv)
        star[st] = best[0]
        lines_cal.append(f"| {SETTINGS[st]} | " + " | ".join(cells)
                         + f" | **{best[0] if best[0] is None else f'{best[0]:g}'}** |")
    lines_cal += ["", "セルは L(違反した企業数)。class 2 には同じ λ(= λ̃*_0 × s、s は class 1 "
                      "で計算)を"
                  "当てはめる。設定 1 はステージ65 の class 1 の格子(s = 0.3866)。", ""]
    # 指標
    lines = ["# ステージ66 C.2: 5 設定の比較(λ は class 1 で較正)", ""]
    item_lines = ["# 項目別の違反(企業数)", ""]
    real_lines = ["# 提案(設定 4)と実データの比較", "",
                  "差 = 提案 − 実データ。95% 区間は、生成と実データを別々に復元抽出"
                  "(2,000 回)。", ""]
    firms: dict[tuple[str, str], pd.DataFrame] = {}
    summary: dict[str, Any] = {"star": star}
    for cls in ("c2", "c1"):
        r, se, n_real = real_targets(cls, names)
        w = {m: 1.0 / se[m] for m in CAL}
        rf = real_firm_values(cls, names, members)
        paid = pd.read_csv("results/stage56/paidin.csv", dtype={"edinetCode": str})
        pg = "再建群(class 2 → 回復)" if cls == "c2" else "再建群(class 1 → class 0)"
        pp = paid[paid["group"] == pg]["annual_d_paid"]
        lines += [f"## {cls}(実データ: 払込資本↑ {(pp > 0.001).mean():.0%}、↓ {r['paid_dec']:.0%}、"
                  f"RE↑ {r['re_up']:.0%}、変更科目数 {n_real:.0f})", "",
                  "| 設定 | λ̃ | 違反(社) | 払込資本↑ | 払込資本↓ | RE↑ | 変更科目数 | 現金が下"
                  "限に張り付いた社 "
                  "| 純資産の改善 | 解消率 | 順位相関 | L |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        item_lines += [f"## {cls}", "", "| 設定 | " + " | ".join(c.split("_")[0] for c in CHECKS)
                       + " |", "|---|" + "---:|" * len(CHECKS)]
        for st, lab in SETTINGS.items():
            lt_s = star[st]
            if lt_s is None:
                lines.append(f"| {lab} | なし |" + " — |" * 10)
                continue
            s = f"{st}__l{lt_s:g}"
            if not (e51._GEN / cls / s).exists():
                lines.append(f"| {lab} | {lt_s:g} | 生成なし |" + " — |" * 9)
                continue
            f = gen_firms(cls, s, names, members).join(extra_cols(cls, s, names, members))
            f = f[~f.index.isin(e51.EXCL[cls])]
            firms[(cls, st)] = f
            cr = ctx[cls].comp[ctx[cls].comp["version"] == s]
            v = ctx[cls].viol[ctx[cls].viol["setting"] == s].drop_duplicates(["docID", "check"])
            g = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                 "d_assets": float(f["d_assets"].median())}
            res = cr.loc[cr["insolvent_start"].astype(bool), "insolvency_resolved"].mean()
            nviol = v["docID"].nunique()
            lines.append(f"| {lab} | {lt_s:g} | {nviol} | {f['paid_inc'].mean():.0%} "
                         f"| {g['paid_dec']:.0%} | {g['re_up']:.0%} |"
                         f" {f['n_changed'].median():.0f} "
                         f"| {int(f['cash_floor'].sum())} |"
                         f" {cr['net_asset_improvement'].median():.3f} "
                         f"| {res:.2f} | {corr(ctx[cls].wide[s], ctx[cls].ref):+.3f} "
                         f"| {loss(g, r, w):.2f} |")
            item_lines.append(f"| {lab} | " + " | ".join(str(int((v["check"] == c).sum()))
                                                         for c in CHECKS) + " |")
        item_lines.append("")
        lines += ["", "**段階の効果**(差 = 後 − 前。払込資本はブートストラップ 2,000 回、記号 ="
                      " 純資産の改善・"
                  "解消率・順位相関)", "",
                  "| 変更 | 払込資本↑ の差 | 払込資本↓ の差 | 変更科目数(中央値) | 現金が下限の社 "
                  "| 純資産の改善の差 | 記号 |", "|---|---|---|---|---|---|---|"]
        for a, b, lab in STEPS:
            if (cls, a) not in firms or (cls, b) not in firms:
                continue
            fa, fb = firms[(cls, a)], firms[(cls, b)].reindex(firms[(cls, a)].index)
            cells = []
            for col in ("paid_inc", "paid_dec"):
                dd = (fa[col] - fb[col]).to_numpy()
                bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                cells.append(f"{dd.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
            sa, sb = f"{a}__l{star[a]:g}", f"{b}__l{star[b]:g}"
            c = ctx[cls].compare(sa, sb, rng)
            sym = " ".join({1: "+", -1: "−", 0: "0"}[int(c[k][3])]
                           for k in ("純資産の改善", "解消率", "順位相関"))
            lines.append(f"| {lab} | {cells[0]} | {cells[1]} | {fb['n_changed'].median():.0f} → "
                         f"{fa['n_changed'].median():.0f} | {int(fb['cash_floor'].sum())} → "
                         f"{int(fa['cash_floor'].sum())} | {e51.fc(c['純資産の改善'])} | {sym} |")
        lines.append("")
        # 実データとの比較(設定 4)
        if (cls, "s4") in firms:
            prop = firms[(cls, "s4")].copy()
            sp = f"s4__l{star['s4']:g}"
            cp_ = ctx[cls].comp[ctx[cls].comp["version"] == sp].set_index("docID")
            prop["nai_annual"] = cp_["net_asset_improvement"].reindex(prop.index) / 5.0
            rf = rf.assign(paid_inc=(pp.reset_index(drop=True).reindex(range(len(rf))) > 0.001)
                           .astype(float).to_numpy() if len(pp) == len(rf) else np.nan)
            real_lines += [f"## {cls}({len(rf)} 社)", "", "| 項目 | 提案 | 実データ | 差 [95%] |",
                           "|---|---:|---:|---|"]
            real_inc = float((pp > 0.001).mean())
            for lab, col, how in (("純資産の改善(年率、中央値)", "nai_annual", "median"),
                                  ("払込資本を減らした割合", "paid_dec", "mean"),
                                  ("利益剰余金が増えた割合", "re_up", "mean"),
                                  ("変更された科目数(中央値)", "n_changed", "median"),
                                  ("総資産変化(年率、中央値)", "d_assets", "median")):
                a_ = prop[col].dropna().to_numpy()
                b_ = rf[col].dropna().to_numpy()
                fn = np.median if how == "median" else np.mean
                ds = [fn(a_[rng.integers(0, len(a_), len(a_))])
                      - fn(b_[rng.integers(0, len(b_), len(b_))]) for _ in range(2000)]
                lo, hi = np.percentile(ds, [2.5, 97.5])
                real_lines.append(f"| {lab} | {fn(a_):.3f} | {fn(b_):.3f} "
                                  f"| {fn(a_) - fn(b_):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
            real_lines += [f"| 払込資本を増やした割合 | {prop['paid_inc'].mean():.3f} |"
                           f" {real_inc:.3f} "
                           "| (実データは企業ごとの値と組まないので区間なし) |", ""]
    (OUT / "calibration.md").write_text("\n".join(lines_cal) + "\n", encoding="utf-8")
    (OUT / "comparison.md").write_text("\n".join([*lines, *item_lines]) + "\n", encoding="utf-8")
    (OUT / "realdata.md").write_text("\n".join(real_lines) + "\n", encoding="utf-8")
    # 枝刈りの効率(設定 5、λ̃*_0)
    q = pd.concat([pd.read_csv(f) for f in sorted((OUT / "qp").glob("*_s5_*.csv"))])
    pr = ["# B.5 枝刈りの効率(設定 5)", "",
          "| class | λ̃ | 始点 | K 以降で枝刈りされなかった候補 | 追加で解いた QP | 1 社あたり |"
          " 時間(秒) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for row in q.sort_values(["cls", "lam_tilde"]).itertuples():
        n = 55 if row.cls == "c2" else 132
        pr.append(f"| {row.cls} | {row.lam_tilde:g} | {n} | {row.n_unpruned_beyond_K} "
                  f"| {row.n_extra_qp} | {row.n_extra_qp / n:.1f} | {row.seconds} |")
    (OUT / "pruning.md").write_text("\n".join(pr) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for fn_ in ("calibration.md", "comparison.md", "realdata.md", "pruning.md"):
        print((OUT / fn_).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
