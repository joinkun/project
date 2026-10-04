#!/usr/bin/env python
"""ステージ51 D.2・D.3: 密度モデルによる目標 x*_i を求める(torch/cvxpy のプロセス)。

規則は ``results/stage51/criteria_cd.md`` §5(計算の前に記録)。始点 i(class 2、57 社)ごとに

    x*_i = argmax_{x ∈ C_i} [ log q(Bᵀx) − λ d(x_i, x) ]

を、ガウス混合の成分ごとの凸 QP(min ½(Bᵀx − μ_k)ᵀΣ_k⁻¹(Bᵀx − μ_k) + λ d)で解き、
log π_k + log N(Bᵀx_k; μ_k, Σ_k) − λ d が最大の成分の解を採る。C_i は最終設定 ``s47_final`` の
射影と同じ実行可能集合。d は非操作 3 科目を除く 29 科目の二乗距離(``nearest_na0`` と同じ)。

λ = c · s。s は現行の最近傍の対応先への交換比率の中央値、c ∈ {0.1, 0.5, 1, 5, 10} を全体の文脈で
すべて計算し(トレードオフ曲線)、d の中央値が現行の最近傍に対数で最も近い c を採る。

出力: ``data/interim/stage51/targets/targets_K{K}.npz``(キー ``full``・``fold0``〜``fold4``)、
``results/stage51/targets_K{K}.md``・``.csv``(曲線と選んだ c)。

    uv run python scripts/targets_stage51.py --k 13 \\
        --density data/interim/stage51/density/gmm_K13_mfa.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from run_generation_stage16 import (
    inactive_targets,
    load_dataset,
    make_configs,
    reduced_basis,
)

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import (
    ConstraintLayers,
    LinearConstraint,
    build_linear_constraints,
    inactive_accounts,
)
from fsgen.datasets.exclusion import load_excluded_docs, subset_folds
from fsgen.errors import SolverNotConvergedError
from fsgen.models.cost import load_cost_config

_GAMMA = 0.7565192502545388
_EQUAL = Path("data/interim/stage28/plans_equal/plans_nearest_na0.npz")
_FOLDS = Path("data/interim/industry_12/folds.npz")
C_GRID = (0.1, 0.5, 1.0, 5.0, 10.0)


class Mixture:
    def __init__(self, w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> None:
        self.logw = np.log(w)
        self.mu = mu
        self.prec_sqrt = []  # Σ⁻¹ = P Pᵀ
        self.logdet = []
        for c in cov:
            ev, evec = np.linalg.eigh(c)
            if ev.min() <= 0:
                raise ValueError("共分散が正定値でない")
            self.prec_sqrt.append(evec / np.sqrt(ev))
            self.logdet.append(float(np.sum(np.log(ev))))
        self.dim = mu.shape[1]

    def comp_logpdf(self, z: np.ndarray, k: int) -> np.ndarray:
        u = (np.atleast_2d(z) - self.mu[k]) @ self.prec_sqrt[k]
        out: np.ndarray = -0.5 * np.sum(u * u, axis=1) - 0.5 * self.logdet[k] \
            - 0.5 * self.dim * np.log(2 * np.pi)
        return out

    def logpdf(self, z: np.ndarray) -> np.ndarray:
        lp = np.column_stack([self.logw[k] + self.comp_logpdf(z, k) for k in range(len(self.mu))])
        mx = lp.max(axis=1, keepdims=True)
        out: np.ndarray = mx[:, 0] + np.log(np.exp(lp - mx).sum(axis=1))
        return out


def solve_target(x_i: np.ndarray, b: np.ndarray, mix: Mixture, lam: float, mask: np.ndarray,
                 c: np.ndarray, signs: list[tuple[int, str]], frozen_idx: list[int],
                 extra: tuple[LinearConstraint, ...]) -> tuple[np.ndarray, int, float]:
    """成分ごとの QP を解き、(目標, 成分, 目的関数の値) を返す。"""
    best: tuple[np.ndarray, int, float] | None = None
    for k in range(len(mix.mu)):
        x = cp.Variable(len(x_i))
        quad = cp.sum_squares(mix.prec_sqrt[k].T @ (b.T @ x - mix.mu[k]))
        dist = cp.sum_squares(cp.multiply(mask, x - x_i))
        cons = [c.T @ x == 0]
        cons += [x[i] >= 0 if s == "nonneg" else x[i] <= 0 for i, s in signs]
        cons += [x[i] == x_i[i] for i in frozen_idx]
        cons += [lc.a @ x >= lc.rhs for lc in extra]
        prob = cp.Problem(cp.Minimize(0.5 * quad + lam * dist), cons)
        try:
            prob.solve(solver=cp.OSQP, max_iter=50000, eps_abs=1e-9, eps_rel=1e-9)
        except cp.error.SolverError:
            pass
        if prob.status != "optimal":
            prob.solve(solver=cp.CLARABEL)
        if prob.status != "optimal" or x.value is None:
            raise SolverNotConvergedError(f"目標の QP が最適でない: 成分 {k}、status={prob.status}")
        xv = np.asarray(x.value, dtype=float)
        viol = max(float(np.abs(c.T @ xv).max()),
                   max((float(lc.rhs - lc.a @ xv) for lc in extra), default=0.0))
        if viol > 1e-6:
            raise SolverNotConvergedError(f"目標の QP の制約違反 {viol:.2e}")
        val = float(mix.logw[k] + mix.comp_logpdf(b.T @ xv, k)[0]
                    - lam * np.sum((mask * (xv - x_i)) ** 2))
        if best is None or val > best[2]:
            best = (xv, k, val)
    assert best is not None
    return best


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, required=True, choices=(13, 31))
    ap.add_argument("--density", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/interim/stage51/targets"))
    ap.add_argument("--results", type=Path, default=Path("results/stage51"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    cc = load_cost_config(Path("conf/accounts/costs_equal.yaml"))
    mask = np.array([0.0 if n in cc.non_actionable else 1.0 for n in names])
    frozen = tuple(n for n in cc.non_actionable if n != "jppfs_cor_RetainedEarnings")
    _cond, cfg = make_configs(ConstraintLayers(), _GAMMA)["s47_final"]
    targets_inact = inactive_targets("selected", names)
    signs = [(names.index(bb.name), bb.sign) for bb in spec.basis
             if bb.sign in ("nonneg", "nonpos")]
    doc_ids, _f, x, assets, labels, src_keep = load_dataset(
        names, load_excluded_docs(Path("data/interim/stage28/excluded.csv")))
    xt = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    dens = np.load(args.density)
    if list(dens["docs2"]) != [doc_ids[i] for i in idx2]:
        raise ValueError("密度の推定と始点の並びが一致しない")
    folds = subset_folds(np.load(_FOLDS), src_keep)
    basis_kind = "pca" if args.k == 13 else "null"
    ctx_basis = {"full": reduced_basis(c, xt[idx0 + idx2], 13, basis_kind)}
    for f in range(5):
        pool = idx0 + [idx2[p] for p in folds[f"train_pos_{f}"]]
        ctx_basis[f"fold{f}"] = (reduced_basis(c, xt[pool], 13, basis_kind) if args.k == 13
                                 else ctx_basis["full"])
    for ctx, b in ctx_basis.items():
        key = ctx if args.k == 13 else "full"
        if not np.allclose(b, dens[f"{key}_basis"], atol=1e-10):
            raise ValueError(f"基底が密度の推定と一致しない: {ctx}")

    def mixture(ctx: str) -> Mixture:
        key = ctx if args.k == 13 else "full"
        return Mixture(dens[f"{key}_weights"], dens[f"{key}_means"], dens[f"{key}_covs"])

    def constraints_for(p: int) -> tuple[list[int], tuple[LinearConstraint, ...]]:
        x_i = xt[idx2[p]]
        inact = inactive_accounts(x_i, names, targets_inact)
        fr = [names.index(n) for n in (*frozen, *inact)]
        return fr, build_linear_constraints(spec, names, x_i, cfg)

    # 交換比率 s(全体の文脈、現行の最近傍の対応先)
    mix_full, b_full = mixture("full"), ctx_basis["full"]
    tg_nn = np.argmax(np.load(_EQUAL)["plan_full"], axis=1)
    x2 = xt[idx2]
    y_nn = xt[idx0][tg_nn]
    d_nn = np.sum((mask * (y_nn - x2)) ** 2, axis=1)
    gain = mix_full.logpdf(y_nn @ b_full) - mix_full.logpdf(x2 @ b_full)
    ratio = gain / d_nn
    s = float(np.median(ratio))
    s_note = "中央値"
    if s <= 0:
        s, s_note = float(np.median(np.abs(ratio))), "絶対値の中央値(中央値が正でないため)"
    t0 = time.perf_counter()
    curve, full_by_c = [], {}
    cons = {p: constraints_for(p) for p in range(len(idx2))}
    for cg in C_GRID:
        lam = cg * s
        res = [solve_target(x2[p], b_full, mix_full, lam, mask, c, signs, *cons[p])
               for p in range(len(idx2))]
        tgt = np.array([r[0] for r in res])
        d = np.sum((mask * (tgt - x2)) ** 2, axis=1)
        lq = mix_full.logpdf(tgt @ b_full)
        full_by_c[cg] = tgt
        curve.append({"c": cg, "lambda": lam, "d_median": float(np.median(d)),
                      "logq_median": float(np.median(lq)),
                      "n_components_used": len({r[1] for r in res}),
                      "n_distinct": len({tuple(np.round(t, 6)) for t in tgt})})
        print(curve[-1], flush=True)
    d_nn_med = float(np.median(d_nn))
    chosen = min(curve, key=lambda r: abs(np.log(r["d_median"]) - np.log(d_nn_med)))
    cg = float(chosen["c"])
    out = {"full": full_by_c[cg]}
    for f in range(5):
        if args.k == 31:
            out[f"fold{f}"] = full_by_c[cg]
            continue
        mix_f, b_f = mixture(f"fold{f}"), ctx_basis[f"fold{f}"]
        out[f"fold{f}"] = np.array([solve_target(x2[p], b_f, mix_f, cg * s, mask, c, signs,
                                                 *cons[p])[0] for p in range(len(idx2))])
    np.savez_compressed(args.out / f"targets_K{args.k}.npz", **out)  # type: ignore[arg-type]
    pd.DataFrame(curve).to_csv(args.results / f"targets_K{args.k}.csv", index=False)
    lq_x, lq_nn = mix_full.logpdf(x2 @ b_full), mix_full.logpdf(y_nn @ b_full)
    lines = [f"# ステージ51 D.2: 密度モデルの目標(K = {args.k})", "",
             f"交換比率 s = {s:.4g}({s_note}、(log q(対応先) − log q(始点)) / d、現行の最近傍)。"
             f"始点の log q の中央値 {np.median(lq_x):.1f}、現行の対応先 {np.median(lq_nn):.1f}、"
             f"現行の d の中央値 {d_nn_med:.4f}。計算時間 {time.perf_counter() - t0:.0f} 秒。", "",
             "| c | λ | d の中央値 | log q の中央値 | 使われた成分の数 | 異なる目標の数 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in curve:
        mark = " **(採用)**" if r["c"] == cg else ""
        lines.append(f"| {r['c']}{mark} | {r['lambda']:.4g} | {r['d_median']:.4f} "
                     f"| {r['logq_median']:.1f} | {r['n_components_used']} | {r['n_distinct']} |")
    lines += ["", f"採用: c = {cg}(d の中央値が現行の最近傍 {d_nn_med:.4f} に対数で最も近い)。", ""]
    (args.results / f"targets_K{args.k}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
