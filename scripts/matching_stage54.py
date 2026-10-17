#!/usr/bin/env python
"""ステージ54 A・B: ゼロパターンの条件付き遷移と、条件付き密度 + 距離による対応先(numpy のみ)。

規則は ``results/stage54/criteria.md``(生成の前に記録)。コストは
c(x_i, y_j) = −log P(S_y | S_x) − log q_cont(y) + λ d(x_i, y_j)。離散部分だけをステージ52 から
変え(周辺 → 条件付き)、連続部分(版 I の条件付き正規)と λ(327.38)はステージ52 と同一。
遷移は推定用パネル全体(主)、再建群 72 社(2 年)・class 1+2(報告のみ)。

出力: ``data/interim/stage54/plans/plans_cond.npz``、``results/stage54/transitions.md``・
``matching.md``・``matching.csv``。

    uv run python scripts/matching_stage54.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from matching_stage52 import load_models, match_stats, nn_plan
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.datasets.exclusion import subset_folds
from fsgen.models.cost import load_cost_config
from fsgen.models.zero_density import IndepZeroGaussian

_EQUAL = Path("data/interim/stage28/plans_equal/plans_nearest_na0.npz")
_FOLDS = Path("data/interim/industry_12/folds.npz")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
LAMBDA52 = 327.3816542646855  # ステージ52 で採用した λ(c = 5、s = 65.48)
S52 = LAMBDA52 / 5.0
C_GRID = (0.1, 0.5, 1.0, 5.0, 10.0)
CATS = ("both0", "new", "gone", "both", "flip")


def counts_from_pairs(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """(科目, 5) の件数: 両期 0、新規計上、消滅、両期非 0(同符号)、符号反転。"""
    nz0, nz1 = x0 != 0, x1 != 0
    same = nz0 & nz1 & (np.sign(x0) == np.sign(x1))
    return np.column_stack([(~nz0 & ~nz1).sum(0), (~nz0 & nz1).sum(0), (nz0 & ~nz1).sum(0),
                            same.sum(0), (nz0 & nz1 & ~same).sum(0)])


def conditional(cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """件数 → (p(0 → 非 0), p(非 0 → 0)) と、クリップした確率の数。"""
    n = int(cnt[0].sum())
    lo, hi = 1.0 / n, 1.0 - 1.0 / n
    z = cnt[:, 0] + cnt[:, 1]
    nz = cnt[:, 2] + cnt[:, 3] + cnt[:, 4]
    birth = np.where(z > 0, cnt[:, 1] / np.maximum(z, 1), 0.5)
    death = np.where(nz > 0, cnt[:, 2] / np.maximum(nz, 1), 0.5)
    clipped = int(((birth < lo) | (birth > hi) | (death < lo) | (death > hi)).sum())
    return np.clip(birth, lo, hi), np.clip(death, lo, hi), clipped


def trans_logp(zx: np.ndarray, zy: np.ndarray, birth: np.ndarray, death: np.ndarray
               ) -> np.ndarray:
    """log P(S_y | S_x) の行列 (len(zx), len(zy))。z は True = 0。"""
    out = np.zeros((len(zx), len(zy)))
    for a in range(zx.shape[1]):
        xa, ya = zx[:, a][:, None], zy[:, a][None, :]
        out += np.where(xa, np.where(ya, np.log1p(-birth[a]), np.log(birth[a])),
                        np.where(ya, np.log(death[a]), np.log1p(-death[a])))
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/interim/stage54/plans"))
    ap.add_argument("--results", type=Path, default=Path("results/stage54"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]

    # ---------------- A 遷移確率 ----------------
    z49 = pd.read_csv("results/stage49/zeros.csv").set_index("account").loc[list(names)]
    cnt1 = np.column_stack([(z49[c] * z49["n"]).round().astype(int) for c in CATS])
    pan = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    pan = pan[(pan["prev_total_assets"] > 0) & (pan["cur_total_assets"] > 0)]
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    cls = np.array([lab.get((c, t - 1), np.nan) for c, t in zip(pan["edinetCode"], pan["t"],
                                                                 strict=True)])
    p12 = pan[np.isin(cls, [1, 2])]

    def tilde(df: pd.DataFrame, side: str) -> np.ndarray:
        out: np.ndarray = (df[[f"{side}_{n}" for n in names]].fillna(0.0).to_numpy()
                           / df[f"{side}_total_assets"].to_numpy()[:, None])
        return out

    cnt3 = counts_from_pairs(tilde(p12, "prev"), tilde(p12, "cur"))
    grp = pd.read_csv("results/stage51/asset_change.csv", dtype={"edinetCode": str})
    rec = grp[grp["group"].str.startswith("再建群")]["edinetCode"]
    a = pan[(pan["t"] == 2023) & pan["edinetCode"].isin(rec)].set_index("edinetCode")
    b = pan[(pan["t"] == 2024) & pan["edinetCode"].isin(rec)].set_index("edinetCode")
    common = a.index.intersection(b.index)
    cnt2 = counts_from_pairs(tilde(a.loc[common], "prev"), tilde(b.loc[common], "cur"))
    srcs = {"案 1 パネル全体(主)": cnt1, "案 2 再建群(2 年)": cnt2, "案 3 class 1+2": cnt3}
    probs = {k: conditional(v) for k, v in srcs.items()}
    for src in srcs:  # 正規化の確認(クリップ後も (0, 1) の中にある)
        bth, dth, _ = probs[src]
        assert np.all((bth > 0) & (bth < 1) & (dth > 0) & (dth < 1)), src
    lines = ["# ステージ54 A: ゼロの条件付き遷移確率", "",
             "ゼロ = 総資産比がちょうど 0。p(0 → 非 0) = 新規計上 / (両期 0 + 新規計上)、"
             "p(非 0 → 0) = 消滅 / (消滅 + 両期非 0 + 符号反転)。各条件で、残りの確率(0 のまま・"
             "非 0 のまま)との和は 1。[1/N, 1 − 1/N] でクリップ。始点の状態の観測がない科目は 1/"
             "2。", "",
             "| 推定元 | 組の数 | クリップした確率 |", "|---|---:|---:|"]
    for k, v in srcs.items():
        lines.append(f"| {k} | {int(v[0].sum()):,} | {probs[k][2]} |")
    lines += ["", "| 科目 | 始点 0 の組(案 1) | p(0 → 非 0) 案 1 | 案 2 | 案 3 "
              "| p(非 0 → 0) 案 1 | 案 2 | 案 3 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, n in enumerate(names):
        bb = [probs[k][0][i] for k in srcs]
        dd = [probs[k][1][i] for k in srcs]
        lines.append(f"| {ja.get(n, n)} | {cnt1[i, 0] + cnt1[i, 1]:,} | {bb[0]:.3f} | {bb[1]:.3f} "
                     f"| {bb[2]:.3f} | {dd[0]:.4f} | {dd[1]:.4f} | {dd[2]:.4f} |")
    its = names.index("jppfs_cor_TreasuryStock")
    lines += ["", f"**自己株式**(ちょうど 0 か否か): p(0 → 非 0) = 案 1"
                  f" {probs['案 1 パネル全体(主)'][0][its]:.1%}"
              f"(始点 0 の {cnt1[its, 0] + cnt1[its, 1]} 組中 {cnt1[its, 1]})、案 2 "
              f"{cnt2[its, 1]} / {cnt2[its, 0] + cnt2[its, 1]}、案 3 {cnt3[its, 1]} / "
              f"{cnt3[its, 0] + cnt3[its, 1]}。", ""]
    diff = np.abs(probs["案 1 パネル全体(主)"][0] - probs["案 2 再建群(2 年)"][0])
    lines += [f"案 1 と案 2 の p(0 → 非 0) の差の絶対値: 中央値 {np.median(diff):.3f}、最大 "
              f"{diff.max():.3f}({ja.get(names[int(np.argmax(diff))], '')})。", ""]
    (args.results / "transitions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    birth, death, _ = probs["案 1 パネル全体(主)"]

    # ---------------- B 対応先 ----------------
    cc = load_cost_config(Path("conf/accounts/costs_equal.yaml"))
    mask = np.array([0.0 if n in cc.non_actionable else 1.0 for n in names])
    x, label, _docs, src_keep = load_tilde(names)
    x0, x2 = x[label == 0], x[label == 2]
    folds = subset_folds(np.load(_FOLDS), src_keep)
    d = np.zeros((len(x2), len(x0)))
    for ai in np.where(mask > 0)[0]:
        d += (x2[:, ai][:, None] - x0[:, ai][None, :]) ** 2
    nn = np.argmax(np.load(_EQUAL)["plan_full"], axis=1)
    model = load_models()["indep"]
    assert isinstance(model, IndepZeroGaussian)
    cont = -model.cont_logpdf(x0)  # (3,768,) 連続部分
    marg = -model.discrete_logp(x0 == 0)  # ステージ52 の離散部分(周辺)
    disc = -trans_logp(x2 == 0, x0 == 0, birth, death)  # (57, 3,768) 本ステージの離散部分
    lines = ["# ステージ54 B: 条件付き密度 + 距離による対応先", "",
             "c = −log P(S_y | S_x) − log q_cont(y) + λ d。λ = c · s、s = 65.48(ステージ52 と同"
             "じ)。"
             f"**生成に使うのは λ = {LAMBDA52:.2f}(c = 5、ステージ52 と同一)**。"
             "全社の計画(class 2 57 社)。自己株式 = 自己株式を持たない始点 15 社のうち対応先が"
             "持つ社数。", "",
             "| 版 | c | 現行から変わった社数 | 異なる対応先 | 完全一致 32 / 28 | 余分あり | 自"
             "己株式 "
             "| 離散(中央値) | 連続(中央値) | 距離 λd(中央値) |",
             "|---|---:|---:|---:|---|---:|---|---:|---:|---:|"]
    st0 = match_stats(x2, x0[nn], names)
    lines.append(f"| 現行(最近傍) | — | — | {len(set(nn.tolist()))} | {st0['exact32']} / "
                 f"{st0['exact28']} | {st0['target_has_extra']} | {st0['ts_target_has']} / "
                 f"{st0['ts_src0']} | — | — | — |")
    rows: list[dict[str, Any]] = []
    for ver, dterm in (("ステージ52(周辺)", np.broadcast_to(marg, d.shape)), ("本ステージ(条件"
                                                                        "付き)", disc)):
        for cg in C_GRID:
            lam = cg * S52
            cost = dterm + cont[None, :] + lam * d
            tg = np.argmin(cost, axis=1)
            st = match_stats(x2, x0[tg], names)
            ii = np.arange(len(x2))
            dv, cv_, lv = dterm[ii, tg], cont[tg], lam * d[ii, tg]
            rows.append({"version": ver, "c": cg, "lambda": lam, "changed": int((tg != nn).sum()),
                         "unique": len(set(tg.tolist())), **st, "disc": float(np.median(dv)),
                         "cont": float(np.median(cv_)), "dist": float(np.median(lv))})
            mark = " **(生成)**" if ver.startswith("本") and cg == 5.0 else ""
            lines.append(f"| {ver}{mark} | {cg} | {(tg != nn).sum()} | {len(set(tg.tolist()))} "
                         f"| {st['exact32']} / {st['exact28']} | {st['target_has_extra']} "
                         f"| {st['ts_target_has']} / {st['ts_src0']} | {np.median(dv):.1f} "
                         f"| {np.median(cv_):.1f} | {np.median(lv):.1f} |")
            if ver.startswith("本") and cg == 5.0:
                plans = {"plan_full": nn_plan(cost)}
                for fk in range(5):
                    for part in ("train", "test"):
                        pos = folds[f"{part}_pos_{fk}"]
                        plans[f"plan_{part}_{fk}"] = nn_plan(cost[pos])
                np.savez_compressed(args.out / "plans_cond.npz", **plans)  # type: ignore[arg-type]
                chosen = tg
    # B.3: 候補の間での各項の広がり(argmin を決めるのは候補間の差)
    iqr = {lab: float(np.median(np.subtract(*np.percentile(t, [75, 25], axis=1))))
           for lab, t in (("離散(条件付き)", disc), ("離散(周辺、ステージ52)",
                                                   np.broadcast_to(marg, d.shape)),
                          ("連続", np.broadcast_to(cont, d.shape)),
                          ("距離 λd", LAMBDA52 * d))}
    its = names.index("jppfs_cor_TreasuryStock")
    ts_birth = float(-np.log(birth[its]) + np.log1p(-birth[its]))
    ts_marg = float(np.log(1 - model.pi0[its]) - np.log(model.pi0[its]))
    lines += ["", "## B.3 各項の寄与", "",
              "上の表の右 3 列は、選ばれた対応先での各項の中央値。argmin を決めるのは候補の間の"
              "違いなので、"
              "始点ごとに全候補(3,768 社)での各項の四分位範囲を取り、その中央値を示す。", "",
              "| 項 | 候補間の四分位範囲(始点の中央値) |", "|---|---:|"]
    for term, val in iqr.items():
        lines.append(f"| {term} | {val:.2f} |")
    lines += ["", f"- 自己株式を持たない始点で、対応先が自己株式を「持つ」ことの追加コスト: "
              f"条件付き {ts_birth:+.2f} nat(ステージ52 の周辺では {-ts_marg:+.2f} nat、つまり"
              f"持つほうが安かった)",
              f"- 生成に使う λ で選んだ対応先: 現行から変わった社 {int((chosen != nn).sum())}", ""]
    pd.DataFrame(rows).to_csv(args.results / "matching.csv", index=False)
    (args.results / "matching.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(open(args.results / "transitions.md", encoding="utf-8").read()[-900:])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
