#!/usr/bin/env python
"""ステージ62 A・C・D: 違反の診断、λ に依存しない差、科目別の分解、ハード制約の記述。

ステージ61 の生成と指標(``data/interim/stage61``)を読む(生成しない)。規則は
``results/stage62/criteria.md``。出力(``results/stage62/``): ``violations.md``・
``rank_corr.md``・``accounts.md``・``hard.md``。

    uv run python scripts/analysis_stage62.py
"""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
import yaml
from compare_stage39 import _ZERO, boot, corr
from scipy import stats

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import evaluate_checks, subtotal_members

OUT = Path("results/stage62")
S61 = Path("data/interim/stage61")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DISTS = {"l2": "等重み L2", "l1s": "対称 L1", "l1a": "非対称 L1"}
_CASH, _RECV, _INV = "jppfs_cor_CashAndDeposits", "receivables_trade", "inventories"


def setup(names: tuple[str, ...]) -> dict[str, e51.Ctx]:
    e51._M = S61 / "metrics"
    e51._GEN = S61 / "gen"
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / "l2_l1").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))
    return {c: e51.Ctx(c, names) for c in ("c2", "c1")}


def violations(ctx: dict[str, e51.Ctx], names: tuple[str, ...]) -> list[str]:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    m = subtotal_members(spec, names)
    grid = pd.read_csv("results/stage61/grid.csv")
    ic, ir, ii = names.index(_CASH), names.index(_RECV), names.index(_INV)
    lines = ["# ステージ62 A: 違反の診断(ステージ61 の生成)", "",
             "行は seed 0 の企業(QP は決定的で 3 seed とも同じ)。程度: 項目 3 = 終点の現金 / 総"
             "資産(0)"
             "(閾値 0.001)、項目 4 = 現金・売上債権・棚卸資産の最大 / 総資産(0)(閾値 0.001)、"
             "項目 5 = 負債(1) / 負債(0)(閾値 2)。「始点で違反」= 始点 x0 自身が同じ項目に当たる"
             "(evaluate_checks(x0, x0))。", ""]
    for setting in ("l1a_l1", "l1a_l3", "l1s_l0.3"):
        cls = "c2"
        v = ctx[cls].viol[ctx[cls].viol["setting"] == setting]
        g = e51.load_gen(cls, setting)
        g = g[g["seed"] == 0].set_index("docID")
        lines += [f"## class 2、{setting}", "",
                  "| 企業 | docID | 項目 | 程度 | 始点の値 | 始点で違反 | 違反が消える λ̃(同じ"
                  "距離) |",
                  "|---|---|---|---:|---:|---|---|"]
        for doc, grp in v.drop_duplicates(["docID", "check"]).groupby("docID"):
            r = g.loc[doc]
            x0 = r[[f"x0_{n}" for n in names]].to_numpy(dtype=float)[None, :]
            x1 = r[[f"x1_{n}" for n in names]].to_numpy(dtype=float)[None, :]
            a0 = float((x0 @ m["jppfs_cor_Assets"])[0])
            start_flags = evaluate_checks(x0, x0, spec, names)[0]
            d = setting.split("_l")[0]
            gone = [lt for lt in LAMS
                    if doc not in set(ctx[cls].viol[ctx[cls].viol["setting"] == f"{d}_l{lt:g}"]
                                      ["docID"])]
            for item in grp["check"]:
                k = int(item.split("_")[0])
                if k == 3:
                    deg, st = x1[0, ic] / a0, x0[0, ic] / a0
                elif k == 4:
                    deg = max(abs(x1[0, ic]), abs(x1[0, ir]), abs(x1[0, ii])) / a0
                    st = max(abs(x0[0, ic]), abs(x0[0, ir]), abs(x0[0, ii])) / a0
                elif k == 5:
                    l0 = float((x0 @ m["jppfs_cor_Liabilities"])[0])
                    deg, st = float((x1 @ m["jppfs_cor_Liabilities"])[0]) / l0, 1.0
                else:
                    deg, st = float("nan"), float("nan")
                start_v = bool(start_flags[k - 1]) if k <= len(start_flags) else False
                lines.append(f"| {r['filerName']} | {doc} | {item} | {deg:.4g} | {st:.4g} "
                             f"| {'はい' if start_v else 'いいえ'} "
                             f"| {', '.join(f'{x:g}' for x in gone) or 'なし'} |")
        lines.append("")
    lines += ["外れ値 2 社(オンコセラピー・サイエンス S100TSS9、桃太郎源 S100TWFQ)は、ステージ60"
              " 以降の"
              "生成の始点に含めていない(55 社)。", "",
              "## A.2 λ̃ ごとの違反企業数", "",
              "| λ̃ | " + " | ".join(f"{c} {DISTS[d]}" for c in ("c2", "c1") for d in DISTS) + " |",
              "|---:|" + "---:|" * 6]
    for lt in LAMS:
        vals = [int(grid[(grid["cls"] == c) & (grid["dist"] == d) & (grid["lam_tilde"] == lt)]
                    ["viol_firms"].iloc[0]) for c in ("c2", "c1") for d in DISTS]
        lines.append(f"| {lt:g} | " + " | ".join(str(v) for v in vals) + " |")
    lines += ["", "## A.2 違反 0 を制約にした較正 λ̃*_0", "",
              "| class | 距離 | λ̃*(制約なし) | 違反(社) | λ̃*_0(違反 0) | L | 純資産の改善 | "
              "解消率 "
              "| 払込資本↓ | 順位相関 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    star0: dict[str, dict[str, float | None]] = {}
    for c in ("c2", "c1"):
        star0[c] = {}
        for d in DISTS:
            sub = grid[(grid["cls"] == c) & (grid["dist"] == d)]
            free = sub.loc[sub["loss"].idxmin()]
            ok = sub[sub["viol_firms"] == 0]
            if ok.empty:
                star0[c][d] = None
                lines.append(f"| {c} | {DISTS[d]} | {free['lam_tilde']:g} | {free['viol_firms']} "
                             "| **なし** | — | — | — | — | — |")
                continue
            b = ok.loc[ok["loss"].idxmin()]
            star0[c][d] = float(b["lam_tilde"])
            lines.append(f"| {c} | {DISTS[d]} | {free['lam_tilde']:g} | {free['viol_firms']} "
                         f"| **{b['lam_tilde']:g}** | {b['loss']:.2f} "
                         f"| {b['net_asset_improvement']:.3f} | {b['resolved']:.2f} "
                         f"| {b['paid_dec']:.0%} | {b['rank_corr']:+.3f} |")
    lines += ["", "## A.3 クラス間の逆転(違反した行の項目別、λ̃ ごとに合計、seed 0)", "",
              "| class | 距離 | 項目 3 現金≈0 | 項目 4 主要資産 | 項目 5 負債 2 倍 | その他 |",
              "|---|---|---:|---:|---:|---:|"]
    for c in ("c2", "c1"):
        v = ctx[c].viol
        for d in DISTS:
            vv = v[v["setting"].str.startswith(f"{d}_l")]
            vv = vv.drop_duplicates(["docID", "check", "setting"])
            cnt = vv.groupby("check")["docID"].count()
            k3, k4, k5 = (int(sum(cnt.get(k, 0) for k in cnt.index if k.startswith(pre)))
                          for pre in ("3_", "4_", "5_"))
            other = int(cnt.sum()) - k3 - k4 - k5
            lines.append(f"| {c} | {DISTS[d]} | {k3} | {k4} | {k5} | {other} |")
    lines.append("")
    (OUT / "star0.json").write_text(json.dumps(star0), encoding="utf-8")
    return lines


def corr_zeroed(gen: pd.DataFrame, ref: pd.Series) -> float:
    """|中央値| < 1e-4 を 0 にそろえた Spearman(ソルバの残差の符号で順位が付くのを防ぐ)。"""
    g = gen.median().to_numpy()
    r = ref.loc[gen.columns].to_numpy()
    g = np.where(np.abs(g) < _ZERO, 0.0, g)
    r = np.where(np.abs(r) < _ZERO, 0.0, r)
    return float(stats.spearmanr(g, r).statistic)


def boot_zeroed(wa: pd.DataFrame, wb: pd.DataFrame, ref: pd.Series, rng: np.random.Generator
                ) -> tuple[float, float]:
    d = []
    for _ in range(1000):
        i = rng.integers(0, len(wa), len(wa))
        d.append(corr_zeroed(wa.iloc[i], ref) - corr_zeroed(wb.iloc[i], ref))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def rank_corr(ctx: dict[str, e51.Ctx]) -> list[str]:
    rng = np.random.default_rng(0)
    lines = ["# ステージ62 C.1: 符号付き順位相関の λ̃ ごとの比較", "",
             "差 = 非対称 L1 − 等重み L2(企業の同時復元抽出 1,000 回の 95% 区間)。", ""]
    for c in ("c2", "c1"):
        lines += [f"## {c}", "", "| λ̃ | 等重み L2 | 対称 L1 | 非対称 L1 | 最も高い | 差 [95%] |",
                  "|---:|---:|---:|---:|---|---|"]
        wins, sig = 0, 0
        for lt in LAMS:
            vals = {d: corr(ctx[c].wide[f"{d}_l{lt:g}"], ctx[c].ref) for d in DISTS}
            top = max(vals, key=lambda k: vals[k])
            wa, wb = ctx[c].wide[f"l1a_l{lt:g}"], ctx[c].wide[f"l2_l{lt:g}"]
            _, _, lo, hi = boot(wa, ctx[c].ref, wb.loc[wa.index], ctx[c].ref, rng)
            wins += top == "l1a"
            sig += lo > 0
            lines.append(f"| {lt:g} | {vals['l2']:+.3f} | {vals['l1s']:+.3f} | {vals['l1a']:+.3f} "
                         f"| {DISTS[top]} | {vals['l1a'] - vals['l2']:+.3f} [{lo:+.3f},"
                         f" {hi:+.3f}] |")
        lines += ["", f"非対称 L1 が最も高い λ̃: {wins} / 10。"
                      f"非対称 L1 − 等重み L2 が有意に正: {sig} / 10。", "",
                  "**頑健性**: |中央値| < 1e-4 を 0 にそろえた順位相関"
                  "(ソルバの残差の符号を除く)", "",
                  "| λ̃ | 等重み L2 | 対称 L1 | 非対称 L1 | 差(非対称 − 等重み)[95%] |",
                  "|---:|---:|---:|---:|---|"]
        wins_z, sig_z = 0, 0
        for lt in LAMS:
            vz = {d: corr_zeroed(ctx[c].wide[f"{d}_l{lt:g}"], ctx[c].ref) for d in DISTS}
            wa, wb = ctx[c].wide[f"l1a_l{lt:g}"], ctx[c].wide[f"l2_l{lt:g}"]
            zlo, zhi = boot_zeroed(wa, wb.loc[wa.index], ctx[c].ref, rng)
            wins_z += max(vz, key=lambda k: vz[k]) == "l1a"
            sig_z += zlo > 0
            lines.append(f"| {lt:g} | {vz['l2']:+.3f} | {vz['l1s']:+.3f} | {vz['l1a']:+.3f} "
                         f"| {vz['l1a'] - vz['l2']:+.3f} [{zlo:+.3f}, {zhi:+.3f}] |")
        lines += ["", f"(0 にそろえた場合)非対称 L1 が最も高い λ̃: {wins_z} / 10。"
                      f"有意に正: {sig_z} / 10。", ""]
    return lines


def accounts(ctx: dict[str, e51.Ctx], names: tuple[str, ...]) -> list[str]:
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    star = json.loads(Path("results/stage61/summary.json").read_text(encoding="utf-8"))["star"]
    lines = ["# ステージ62 C.2: 科目別の分解(各距離の λ̃*)", "",
             "符号 = 生成の変化の中央値と実データの変化の中央値の符号(|値| < 1e-4 は 0、"
             "`compare_stage39.agree` と同じ)。寄与 = 順位相関(Spearman)の科目ごとの項 "
             "(R_g − R̄)(R_r − R̄) / Σ…(合計が相関になる)。c−/c+ は非対称 L1 の重み。", ""]
    for c in ("c2", "c1"):
        meta = json.loads(Path(f"data/interim/stage60/meta_{c}.json").read_text(encoding="utf-8"))
        cp, cm = np.array(meta["c_plus"]), np.array(meta["c_minus"])
        ref = ctx[c].ref.loc[list(names)].to_numpy()
        sa, sb = f"l2_l{star[c]['l2']:g}", f"l1a_l{star[c]['l1a']:g}"
        ga = ctx[c].wide[sa].median().loc[list(names)].to_numpy()
        gb = ctx[c].wide[sb].median().loc[list(names)].to_numpy()

        def sg(v: np.ndarray) -> np.ndarray:
            return np.where(np.abs(v) < _ZERO, 0, np.sign(v))

        def contrib(g: np.ndarray, ref: np.ndarray = ref) -> np.ndarray:
            rg, rr = stats.rankdata(g), stats.rankdata(ref)
            dg, dr = rg - rg.mean(), rr - rr.mean()
            out: np.ndarray = dg * dr / np.sqrt((dg ** 2).sum() * (dr ** 2).sum())
            return out

        ca, cb = contrib(ga), contrib(gb)
        lines += [f"## {c}({sa} 対 {sb})", "",
                  f"相関: 等重み L2 {ca.sum():+.3f}、非対称 L1 {cb.sum():+.3f}。符号の一致: "
                  f"等重み L2 {int((sg(ga) == sg(ref)).sum())} / 32、非対称 L1 "
                  f"{int((sg(gb) == sg(ref)).sum())} / 32。", "",
                  "| 科目 | 実データ | 等重み L2 | 一致 | 非対称 L1 | 一致 | 寄与の改善 | c−/c+ |",
                  "|---|---:|---:|---|---:|---|---:|---:|"]
        order = np.argsort(-(cb - ca))
        for j in order:
            ratio = cm[j] / cp[j] if cp[j] > 0 else float("nan")
            lines.append(f"| {ja.get(names[j], names[j])} | {ref[j]:+.4f} | {ga[j]:+.4f} "
                         f"| {'○' if sg(ga)[j] == sg(ref)[j] else '×'} | {gb[j]:+.4f} "
                         f"| {'○' if sg(gb)[j] == sg(ref)[j] else '×'} | {cb[j] - ca[j]:+.3f} "
                         f"| {ratio:.2f} |")
        lines.append("")
    return lines


def hard() -> list[str]:
    d = pd.read_csv("results/stage59/decrease.csv")
    d = d[~d["nact"]].sort_values("share_nom")
    lines = ["# ステージ62 D: ハード制約(I_hard)を課さない(スライド30 の書き換え)", "",
             "**方針**: 実データの再建企業に存在する行動は禁止しない。I_hard(その科目を減らさな"
             "い)の候補は、"
             "再建企業で減少がほぼ起きない科目に限る(ステージ59 A'.4 の規則: 正規化 57 社・名目 "
             "57 社・"
             "名目 15 社のすべてで減少の割合 ≤ 5%)。", "",
             "| 科目 | 減少(名目、D_rec 57 社) | 減少(正規化、57 社) | 減少(名目、class 2 → 回復"
             " 15 社) |",
             "|---|---:|---:|---:|"]
    for r in d.head(12).itertuples():
        lines.append(f"| {r.ja} | {r.share_nom:.0%}({r.n_nom} 社) | {r.share_norm:.0%} "
                     f"| {r.share_nom15:.0%} |")
    lines += ["", "- 規則を満たすのは、丸め差の残余科目 3 つ(固定資産その他・負債その他・株主資"
                  "本その他)だけで、"
              "値が総資産比 1e-6 程度なので制約として意味を持たない。",
              "- **社債でも 57 社中 10 社(18%)、15 社中 2 社(13%)が減らしている(償還)。** 社債"
              "を減らせない"
              "とすると、実在する再建の行動を禁止することになる。",
              "- 資本金(名目 18%)、自己株式(5〜14%)も、少数だが実データで減っている。",
              "- **結論: ハード制約は課さない。** 減らしにくさは、移動コストの非対称の重み c−/c+"
              "(ソフトな形)で"
              "表す。", ""]
    return lines


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = setup(names)
    (OUT / "violations.md").write_text("\n".join(violations(ctx, names)) + "\n", encoding="utf-8")
    (OUT / "rank_corr.md").write_text("\n".join(rank_corr(ctx)) + "\n", encoding="utf-8")
    (OUT / "accounts.md").write_text("\n".join(accounts(ctx, names)) + "\n", encoding="utf-8")
    (OUT / "hard.md").write_text("\n".join(hard()) + "\n", encoding="utf-8")
    for f in ("violations.md", "rank_corr.md"):
        print((OUT / f).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
