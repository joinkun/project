#!/usr/bin/env python
"""ステージ52 B: ゼロを明示的に扱う健全企業の密度の推定と検証(sklearn のプロセス)。

規則は ``results/stage52/criteria_b.md``(推定の前に記録)。版 I = 32 科目・独立ゼロ + 条件付き正規
(Ledoit-Wolf)、版 II = 18 科目(class 0 のゼロ率 < 10%)のパターン混合。B.4.1 の class 別の尤度
(順序 hold-out class 0 > class 1 > class 2、かつ class 2 < class 0 の片側
Mann-Whitney p < 0.05)を判定する。

出力: ``data/interim/stage52/density/{indep,pattern}.npz``、``results/stage52/density_b.md``・
``density_b.json``。

    uv run python scripts/density_stage52.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from density_stage51 import load_tilde
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    orthonormal_basis,
    state_vector_names,
)
from fsgen.models.zero_density import IndepZeroGaussian, PatternMixture

OUTLIERS = ("S100TSS9", "S100TWFQ")
LOW_ZERO = 0.10
COVER = 0.80
MIN_RATIO = 10


def fit_indep(x0: np.ndarray, u: np.ndarray) -> tuple[IndepZeroGaussian, float]:
    n = len(x0)
    pi0 = np.clip((x0 == 0).mean(axis=0), 1.0 / n, 1.0 - 1.0 / n)
    lw = LedoitWolf().fit(x0 @ u)
    return IndepZeroGaussian(pi0, u, lw.location_, lw.covariance_), float(lw.shrinkage_)


def fit_pattern(x0: np.ndarray, cols: np.ndarray) -> tuple[PatternMixture, list[dict[str, Any]]]:
    xs = x0[:, cols]
    zero = xs == 0
    cnt = Counter(map(lambda r: r.tobytes(), zero))
    order = [k for k, _ in cnt.most_common()]
    cum = np.cumsum([cnt[k] for k in order]) / len(xs)
    m = int(np.searchsorted(cum, COVER) + 1)
    pats = [np.frombuffer(k, dtype=bool).copy() for k in order[:m]]
    info: list[dict[str, Any]] = []
    keep = []
    for p in pats:  # 社数 < 10 × 次元 は外す(criteria_b.md §3)
        n_exact = int(np.all(zero == p, axis=1).sum())
        dim = int((~p).sum())
        ok = n_exact >= MIN_RATIO * dim
        info.append({"zeros": int(p.sum()), "n_exact": n_exact, "dim": dim,
                     "ratio": n_exact / dim if dim else float("inf"), "kept": ok})
        if ok:
            keep.append(p)
    patterns = np.array(keep)
    pi_tmp = np.array([np.all(zero == p, axis=1).sum() for p in keep], dtype=float)
    mix0 = PatternMixture(cols, patterns, pi_tmp / pi_tmp.sum(), [], [])
    assigned = mix0.assign(zero)
    pi = np.bincount(assigned, minlength=len(keep)) / len(xs)
    means, covs, shrink = [], [], []
    for p in keep:
        rows = np.all(zero == p, axis=1)
        lw = LedoitWolf().fit(xs[rows][:, ~p])
        means.append(lw.location_)
        covs.append(lw.covariance_)
        shrink.append(float(lw.shrinkage_))
    k = 0
    for r in info:
        if r["kept"]:
            r["n_assigned"] = int((assigned == k).sum())
            r["shrinkage"] = shrink[k]
            r["pi"] = float(pi[k])
            k += 1
    return PatternMixture(cols, patterns, pi, means, covs), info


def summarize(v: np.ndarray) -> str:
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:.1f} [{q1:.1f}, {q3:.1f}]"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/interim/stage52/density"))
    ap.add_argument("--results", type=Path, default=Path("results/stage52"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    u = orthonormal_basis(build_constraint_matrix(spec, include_subtotals=False))
    x, label, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    x0, x1, x2 = x[label == 0], x[label == 1], x[label == 2]
    x2_55 = x[(label == 2) & ~np.isin(docs_a, OUTLIERS)]
    cols = np.where((x0 == 0).mean(axis=0) < LOW_ZERO)[0]
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {}
    lines = ["# ステージ52 B: ゼロを明示的に扱う密度の推定と検証", "",
             f"class 0 {len(x0):,}、class 1 {len(x1)}、class 2 {len(x2)}(外れ値 2 社を除く"
             f" {len(x2_55)})"
             " 書類。log q は 1 社あたり。規則は `criteria_b.md`。", ""]

    def fitter(version: str):  # type: ignore[no-untyped-def]
        if version == "I":
            return lambda xx: fit_indep(xx, u)[0]
        return lambda xx: fit_pattern(xx, cols)[0]

    for version, title in (("I", "版 I: 32 科目・独立ゼロ + 条件付き正規(主)"),
                           ("II", "版 II: 18 科目のパターン混合(比較)")):
        fit = fitter(version)
        model: IndepZeroGaussian | PatternMixture
        if version == "I":
            model, shrink = fit_indep(x0, u)
            lines += [f"## {title}", "",
                      f"- 連続部分: 31 次元(null(Cᵀ))× {len(x0):,} 社(社数 / 次元 ="
                      f" {len(x0) / 31:.0f})。"
                      f"**Ledoit-Wolf の縮小係数 {shrink:.4f}**", ""]
            np.savez_compressed(args.out / "indep.npz", pi0=model.pi0, u=model.u, mu=model.mu,
                                cov=model.cov)
            out["I"] = {"shrinkage": shrink}
        else:
            model, info = fit_pattern(x0, cols)
            lines += [f"## {title}", "",
                      "科目: " + "、".join(ja.get(names[c], names[c]) for c in cols) + "。", "",
                      "| 上位パターン(18 科目で 0 の科目) | ちょうど一致の社数 | 非 0 科目数 = "
                      "次元 "
                      "| 社数 / 次元 | 採用 | 併合後の社数 | π_S | 縮小係数 |",
                      "|---|---:|---:|---:|---|---:|---:|---:|"]
            pats_ja = []
            cnt = Counter(map(lambda r: r.tobytes(), x0[:, cols] == 0))
            for (k, _), r in zip(cnt.most_common(len(info)), info, strict=True):
                p = np.frombuffer(k, dtype=bool)
                pj = "、".join(ja.get(names[cols[i]], names[cols[i]]) for i in np.where(p)[0]) \
                    or "(なし)"
                pats_ja.append(pj)
                lines.append(f"| {pj} | {r['n_exact']} | {r['dim']} | {r['ratio']:.0f} "
                             f"| {'○' if r['kept'] else '× 併合'} | {r.get('n_assigned', '—')} "
                             f"| {r.get('pi', float('nan')):.3f} |"
                             f" {r.get('shrinkage', float('nan')):.4f} |")
            lines.append("")
            assert isinstance(model, PatternMixture)
            save = {"cols": cols, "patterns": model.patterns, "pi": model.pi}
            for j in range(len(model.pi)):
                save[f"mean{j}"] = model.means[j]
                save[f"cov{j}"] = model.covs[j]
            np.savez_compressed(args.out / "pattern.npz", **save)
            out["II"] = {"patterns": info}
        # B.4.1
        lp_tr = model.logpdf(x0)
        cv = np.empty(len(x0))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(x0):
            cv[te] = fit(x0[tr]).logpdf(x0[te])
        hold = rng.permutation(len(x0))[:500]
        rest = np.setdiff1d(np.arange(len(x0)), hold)
        lp_h500 = fit(x0[rest]).logpdf(x0[hold])
        lp1, lp2, lp2_55 = model.logpdf(x1), model.logpdf(x2), model.logpdf(x2_55)
        mw = stats.mannwhitneyu(lp2_55, cv, alternative="less")
        order_ok = np.median(cv) > np.median(lp1) > np.median(lp2_55)
        passed = bool(order_ok and mw.pvalue < 0.05)
        lines += ["### B.4.1 class 別の尤度(1 社あたりの log q、中央値 [四分位])", "",
                  "| 集団 | 社数 | log q |", "|---|---:|---|",
                  f"| class 0(学習に使った全社) | {len(x0):,} | {summarize(lp_tr)} |",
                  f"| class 0(5 分割交差検証の hold-out) | {len(x0):,} | {summarize(cv)} |",
                  f"| class 0(検証用 500 社、残りで推定) | 500 | {summarize(lp_h500)} |",
                  f"| class 1 | {len(x1)} | {summarize(lp1)} |",
                  f"| class 2 | {len(x2)} | {summarize(lp2)} |",
                  f"| class 2(外れ値を除く) | {len(x2_55)} | {summarize(lp2_55)} |", "",
                  f"- 順序(hold-out class 0 > class 1 > class 2(55)): "
                  f"{'成立' if order_ok else '**成立しない**'}",
                  f"- class 2(55)< hold-out class 0 の片側 Mann-Whitney: p = {mw.pvalue:.2g}",
                  f"- **判定: {'通過(C 以降へ)' if passed else '通過しない(C 以降は実施しない)'}"
                  f"**", ""]
        out[version] |= {"passed": passed, "order_ok": bool(order_ok), "mw_p": float(mw.pvalue),
                         "median": {"train": float(np.median(lp_tr)), "cv": float(np.median(cv)),
                                    "h500": float(np.median(lp_h500)), "c1": float(np.median(lp1)),
                                    "c2": float(np.median(lp2)),
                                    "c2_55": float(np.median(lp2_55))}}
        # 合成サンプル(判定には使わない)
        syn = model.sample(len(x0), rng)
        real = x0 if version == "I" else x0[:, cols]
        zr = np.abs((syn == 0).mean(axis=0) - (real == 0).mean(axis=0))
        qd = []
        for a in range(real.shape[1]):
            r_nz, s_nz = real[real[:, a] != 0, a], syn[syn[:, a] != 0, a]
            if len(r_nz) > 10 and len(s_nz) > 10:
                iqr = np.subtract(*np.percentile(r_nz, [75, 25])) or 1.0
                qd.append(np.max(np.abs(np.percentile(s_nz, [25, 50, 75])
                                        - np.percentile(r_nz, [25, 50, 75]))) / iqr)
        cr = np.corrcoef(real, rowvar=False)
        cs = np.corrcoef(syn, rowvar=False)
        ok = np.isfinite(cr) & np.isfinite(cs)
        corr_mad = float(np.mean(np.abs(cr - cs)[ok]))
        neg = float(np.mean(syn < -1e-9)) if version == "I" else float("nan")
        lines += ["### 合成サンプルと実データ(参考)", "",
                  f"- ゼロ率の差の最大: {zr.max():.3f}",
                  f"- 非 0 の値の四分位(25・50・75%)のずれ / 実データの四分位範囲: 中央値"
                  f" {np.median(qd):.2f}、"
                  f"最大 {np.max(qd):.2f}",
                  f"- 相関行列の差の絶対値の平均: {corr_mad:.3f}"]
        if version == "I":
            lines.append(f"- 合成サンプルで負の値をとる要素の割合: {neg:.1%}(正規なので非負の科"
                         f"目も負になりうる)")
        lines.append("")
        # B.4.2
        if version == "I":
            assert isinstance(model, IndepZeroGaussian)
            pis = np.exp(model.discrete_logp(x2_55 == 0))
            pis0 = np.exp(model.discrete_logp(x0 == 0))
            lines += ["### B.4.2 パターンの稀さ(版 I: π_S = Π_a)", "",
                      f"- class 2(55)の π_S: 中央値 {np.median(pis):.2e}、最小 {pis.min():.2e}",
                      f"- 参考 class 0 の π_S: 中央値 {np.median(pis0):.2e}",
                      f"- class 2 の π_S < class 0 の片側 Mann-Whitney: "
                      f"p = {stats.mannwhitneyu(pis, pis0, alternative='less').pvalue:.2g}", ""]
        else:
            assert isinstance(model, PatternMixture)
            z2 = x2_55[:, cols] == 0
            k2 = model.assign(z2)
            exact = np.array([np.all(z2[i] == model.patterns[k2[i]]) for i in range(len(z2))])
            dist2 = (z2 != model.patterns[k2]).sum(axis=1)
            z1 = x[(label == 1)][:, cols] == 0
            exact1 = np.array([np.all(r == model.patterns[k]) for r, k in zip(z1, model.assign(z1),
                                                                               strict=True)])
            lines += ["### A'・B.4.2 始点のパターン(版 II、18 科目)", "",
                      f"- class 2(55): 上位パターンにちょうど一致 {int(exact.sum())} 社、"
                      f"ハミング距離で併合 {int((~exact).sum())} 社"
                      f"(距離 {dict(Counter(dist2[~exact].tolist()))})",
                      f"- class 1({len(z1)}): ちょうど一致 {int(exact1.sum())} 社、併合"
                      f" {int((~exact1).sum())} 社",
                      f"- class 2 の併合後の π_S: 中央値 {np.median(model.pi[k2]):.3f}、最小 "
                      f"{model.pi[k2].min():.3f}", ""]
    (args.results / "density_b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.results / "density_b.json").write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                                            default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
