#!/usr/bin/env python
"""ステージ51 D.1: 健全企業(class 0)の密度をガウス混合で推定する(sklearn のプロセス)。

規則は ``results/stage51/criteria_cd.md`` §4(推定の前に記録)。z = Bᵀx̃(総資産比)で、
K = 13 は基底の文脈(fold 0〜4 と全体)ごと、K = 31 は null(Cᵀ) の基底 1 つ。成分数 1〜10 を
BIC で選び、乱数 0・1・2 で当てはめ、「推定できた」の 4 条件を判定する。

出力: ``data/interim/stage51/density/gmm_K{13,31}.npz``(文脈ごとの重み・平均・共分散・基底)、
``results/stage51/density.md``・``density.json``。

    uv run python scripts/density_stage51.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    orthonormal_basis,
    state_vector_names,
)
from fsgen.accounting.subspace import build_reduced_basis
from fsgen.datasets.exclusion import subset_folds

_FOLDS = Path("data/interim/industry_12/folds.npz")
K_MAX = 10
SEEDS = (0, 1, 2)
MIN_WEIGHT = 0.01


class MFA:
    """因子分析の混合(因子 q 個 + 成分ごとの対角)。手法 2(criteria_cd.md §4)。

    E ステップは成分ごとの正規 N(μ_k, Λ_kΛ_kᵀ + Ψ_k)、M ステップは重み付き共分散 S_k に対する
    因子分析の EM を 1 回(ECM)。対数尤度の増分が 1e-7 × |対数尤度| 未満で収束とする。
    """

    def __init__(self, k: int, q: int = 5, seed: int = 0, n_init: int = 3,
                 max_iter: int = 1000, reg: float = 1e-6) -> None:
        self.k, self.q, self.seed, self.n_init, self.max_iter, self.reg = k, q, seed, n_init, \
            max_iter, reg
        self.converged_ = False

    def _logpdf(self, z: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        ch = np.linalg.cholesky(cov)
        sol = np.linalg.solve(ch, (z - mu).T)
        out: np.ndarray = (-0.5 * np.sum(sol**2, axis=0) - np.sum(np.log(np.diag(ch)))
                           - 0.5 * z.shape[1] * np.log(2 * np.pi))
        return out

    def _log_resp(self, z: np.ndarray, w: np.ndarray, mu: np.ndarray, cov: np.ndarray
                  ) -> tuple[np.ndarray, float]:
        lp = np.column_stack([np.log(w[j]) + self._logpdf(z, mu[j], cov[j])
                              for j in range(len(w))])
        mx = lp.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(axis=1))
        return lp - lse[:, None], float(lse.sum())

    def _run(self, z: np.ndarray, seed: int) -> tuple[float, bool, tuple[np.ndarray, ...]]:
        from sklearn.cluster import KMeans

        n, d = z.shape
        lab = KMeans(self.k, n_init=1, random_state=seed).fit(z).labels_
        w = np.array([max(np.mean(lab == j), 1e-3) for j in range(self.k)])
        w /= w.sum()
        mu = np.array([z[lab == j].mean(axis=0) if (lab == j).any() else z.mean(axis=0)
                       for j in range(self.k)])
        lam = np.zeros((self.k, d, self.q))
        psi = np.zeros((self.k, d))
        for j in range(self.k):
            zj = z[lab == j] if (lab == j).sum() > self.q else z
            s = np.cov(zj.T) + self.reg * np.eye(d)
            ev, evec = np.linalg.eigh(s)
            lam[j] = evec[:, -self.q:] * np.sqrt(np.maximum(ev[-self.q:], self.reg))
            psi[j] = np.maximum(np.diag(s - lam[j] @ lam[j].T), self.reg)
        prev, conv = -np.inf, False
        ll = -np.inf
        for _ in range(self.max_iter):
            cov = np.einsum("kdq,keq->kde", lam, lam) + np.einsum("kd,de->kde", psi, np.eye(d))
            log_r, ll = self._log_resp(z, w, mu, cov)
            if ll - prev < 1e-7 * abs(ll):
                conv = True
                break
            prev = ll
            r = np.exp(log_r)
            nk = r.sum(axis=0) + 1e-12
            w = nk / n
            for j in range(self.k):
                mu[j] = r[:, j] @ z / nk[j]
                zc = z - mu[j]
                s = (r[:, j, None] * zc).T @ zc / nk[j] + self.reg * np.eye(d)
                beta = lam[j].T @ np.linalg.inv(cov[j])
                m = np.eye(self.q) - beta @ lam[j] + beta @ s @ beta.T
                lam[j] = s @ beta.T @ np.linalg.inv(m)
                psi[j] = np.maximum(np.diag(s - lam[j] @ beta @ s), self.reg)
        cov = np.einsum("kdq,keq->kde", lam, lam) + np.einsum("kd,de->kde", psi, np.eye(d))
        return ll, conv, (w.copy(), mu.copy(), cov)

    def fit(self, z: np.ndarray) -> MFA:
        best: tuple[float, bool, tuple[np.ndarray, ...]] | None = None
        for i in range(self.n_init):
            res = self._run(z, self.seed * 100 + i)
            if best is None or res[0] > best[0]:
                best = res
        assert best is not None
        self.converged_ = best[1]
        self.weights_, self.means_, self.covariances_ = best[2]
        return self

    def score(self, z: np.ndarray) -> float:
        _, ll = self._log_resp(z, self.weights_, self.means_, self.covariances_)
        return ll / len(z)

    def bic(self, z: np.ndarray) -> float:
        d = z.shape[1]
        n_par = self.k * (d + d * self.q - self.q * (self.q - 1) / 2 + d) + self.k - 1
        return float(-2 * self.score(z) * len(z) + n_par * np.log(len(z)))


METHOD = {"name": "gmm"}


def fit(z: np.ndarray, k: int, seed: int) -> GaussianMixture | MFA:
    if METHOD["name"] == "mfa":
        return MFA(k, seed=seed).fit(z)
    g = GaussianMixture(k, covariance_type="full", reg_covar=1e-6, n_init=3, max_iter=1000,
                        random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        g.fit(z)
    return g


def assess(z: np.ndarray) -> tuple[int, dict[str, Any], list[str]]:
    """成分数の選択と「推定できた」の 4 条件(criteria_cd.md §4)。"""
    bic = {s: [fit(z, k, s).bic(z) for k in range(1, K_MAX + 1)] for s in SEEDS}
    chosen = {s: int(np.argmin(bic[s])) + 1 for s in SEEDS}
    k = chosen[0]
    models = {s: fit(z, k, s) for s in SEEDS}
    c1 = all(bool(m.converged_) for m in models.values())
    c2 = len(set(chosen.values())) == 1
    cv_k, cv_1 = [], []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(z):
        cv_k.append(float(fit(z[tr], k, 0).score(z[te])))
        cv_1.append(float(fit(z[tr], 1, 0).score(z[te])))
    c3 = bool(np.all(np.isfinite(cv_k)) and np.mean(cv_k) > np.mean(cv_1))
    min_w = float(models[0].weights_.min())
    c4 = min_w >= MIN_WEIGHT
    info = {"bic": bic, "chosen": chosen, "converged": c1, "same_k": c2,
            "cv_loglik_k": cv_k, "cv_loglik_1": cv_1, "cv_ok": c3, "min_weight": min_w,
            "weight_ok": c4, "ok": bool(c1 and c2 and c3 and c4)}
    lines = [f"- 成分数(BIC 最小): seed 0 → {chosen[0]}、seed 1 → {chosen[1]}、"
             f"seed 2 → {chosen[2]}({'一致' if c2 else '**不一致**'})",
             f"- 収束(選んだ成分数、3 seed): {'すべて収束' if c1 else '**未収束あり**'}",
             f"- 5 分割の検証対数尤度(1 点あたり): {k} 成分 {np.mean(cv_k):.2f}、1 成分 "
             f"{np.mean(cv_1):.2f}({'良い' if c3 else '**良くない**'})",
             f"- 最小の成分の重み: {min_w:.3f}({'1% 以上' if c4 else '**1% 未満**'})",
             f"- **判定: {'推定できた' if info['ok'] else '推定できなかった'}**"]
    return k, info, lines


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/interim/stage51/density"))
    ap.add_argument("--results", type=Path, default=Path("results/stage51"))
    ap.add_argument("--method", choices=("gmm", "mfa"), default="gmm",
                    help="gmm: 手法 1(全共分散)、mfa: 手法 2(因子 5 個 + 対角の混合)")
    ap.add_argument("--dims", default="13,31")
    args = ap.parse_args()
    METHOD["name"] = args.method
    tag = "" if args.method == "gmm" else "_mfa"
    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    x, label, docs, src_keep = load_tilde(names)
    idx0 = np.where(label == 0)[0]
    idx2 = np.where(label == 2)[0]
    folds = subset_folds(np.load(_FOLDS), src_keep)
    title = "ガウス混合、全共分散" if args.method == "gmm" else "因子分析の混合(因子 5 + 対角)"
    lines = [f"# ステージ51 D.1: 健全企業の密度({title})", "",
             f"class 0 {len(idx0)} 書類の z = Bᵀx̃。成分数 1〜{K_MAX} を BIC で選ぶ。"
             "判定の 4 条件は `criteria_cd.md` §4。", ""]
    summary: dict[str, Any] = {}
    for kdim in (int(v) for v in args.dims.split(",")):
        contexts: dict[str, np.ndarray] = {}
        if kdim == 13:
            contexts["full"] = build_reduced_basis(c, x[np.r_[idx0, idx2]], k=13)
            for f in range(5):
                pool = np.r_[idx0, idx2[folds[f"train_pos_{f}"]]]
                contexts[f"fold{f}"] = build_reduced_basis(c, x[pool], k=13)
        else:
            contexts["full"] = orthonormal_basis(c)
        z_full = x[idx0] @ contexts["full"]
        k, info, ls = assess(z_full)
        lines += [f"## K = {kdim}", "", *ls, ""]
        lines += ["| 成分数 | " + " | ".join(f"BIC seed {s}" for s in SEEDS) + " |",
                  "|---:|" + "---:|" * len(SEEDS)]
        for j in range(K_MAX):
            lines.append(f"| {j + 1} | " + " | ".join(f"{info['bic'][s][j]:,.0f}" for s in SEEDS)
                         + " |")
        lines.append("")
        summary[f"K{kdim}"] = info
        save: dict[str, np.ndarray] = {"docs0": np.array([docs[i] for i in idx0]),
                                       "docs2": np.array([docs[i] for i in idx2])}
        if info["ok"]:
            for ctx, b in contexts.items():
                g = fit(x[idx0] @ b, k, 0)
                if not g.converged_:
                    raise RuntimeError(f"K={kdim} {ctx}: ガウス混合が収束しない")
                save |= {f"{ctx}_weights": g.weights_, f"{ctx}_means": g.means_,
                         f"{ctx}_covs": g.covariances_, f"{ctx}_basis": b}
                lines.append(f"- 文脈 {ctx}: 成分の重み "
                             + ", ".join(f"{w:.3f}" for w in np.sort(g.weights_)[::-1]))
            lines.append("")
        np.savez_compressed(args.out / f"gmm_K{kdim}{tag}.npz", **save)  # type: ignore[arg-type]
    (args.results / f"density{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.results / f"density{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
