#!/usr/bin/env python
"""ステージ49: 勘定科目ごとの変化率の分布推定(生成には使わない)。

推定用パネル(ステージ23、3,657 社 × 3 遷移、同じ書類の前期末 → 当期末)で、総資産で正規化した値
x̃ = x / A の変化率 R = x̃_t / x̃_{t−1}、g = log R を科目ごとに求め、分布族を最尤法で当てはめる。

- 符号: 同符号の組は絶対値の比 |x̃_t| / |x̃_{t−1}| で定義する(自己株式・貸倒引当金などの負の科目、
  利益剰余金などの符号が自由な科目も)。符号が反転した組は分布の推定から除き、数える。
- ゼロ: 値が厳密に 0(欠測を 0 とみなす)。両期 0 / 新規計上 / 消滅 / 両期非 0 を数える。
- 外れ値: |g| > Q3(|g|) + 3 IQR(|g|)(Tukey)。含めた版と除いた版の両方で推定する。
- 分布族: 正規、Laplace(閉じた形)、Subbotin(一般化正規)、歪み Subbotin(左右で尺度が違う)、t。
  後の 3 つは ``scipy.optimize.minimize`` で負の対数尤度を最小化し、収束しなければ例外。
- 群別: class(始点の年度のラベル)と粗い業種(6 区分)で Subbotin を当てはめ、尤度比検定と
  Wald の 95% 区間。
- 凸性: 当てはめた Subbotin で c(Δ) = −log p(log(1 + Δ/x)) を Δ/x の格子で数値的に調べる。

numpy / scipy / matplotlib のみ(torch・POT・lightgbm なし)。

    uv run python scripts/growth_dist_stage49.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy import optimize, special, stats

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.industry import load_industry_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PANEL = Path("data/interim/stage23/panel/panel.csv")
MIN_N = 30
INDUSTRIES = ("製造", "小売・卸売", "情報通信", "サービス", "不動産・建設", "その他(非金融)")


class FitError(RuntimeError):
    """最尤推定が収束しなかった。"""


@dataclass
class Fit:
    family: str
    params: dict[str, float]
    ll: float
    k: int
    n: int

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.ll

    @property
    def bic(self) -> float:
        return float(self.k * np.log(self.n) - 2 * self.ll)


def _minimize(nll: Callable[[np.ndarray], float], x0: np.ndarray) -> np.ndarray:
    """Nelder-Mead → Powell の順に試し、収束しなければ例外。"""
    for method in ("Nelder-Mead", "Powell"):
        res = optimize.minimize(nll, x0, method=method,
                                options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10}
                                if method == "Nelder-Mead" else {"maxiter": 20000})
        if res.success and np.isfinite(res.fun):
            out: np.ndarray = res.x
            return out
        x0 = res.x
    raise FitError(f"最尤推定が収束しない: {res.message}")


def fit_normal(g: np.ndarray) -> Fit:
    mu, sd = float(g.mean()), float(g.std())
    return Fit("正規", {"loc": mu, "scale": sd}, float(stats.norm.logpdf(g, mu, sd).sum()), 2,
               len(g))


def fit_laplace(g: np.ndarray) -> Fit:
    mu = float(np.median(g))
    s = float(np.mean(np.abs(g - mu)))
    return Fit("Laplace", {"loc": mu, "scale": s}, float(stats.laplace.logpdf(g, mu, s).sum()),
               2, len(g))


def gennorm_nll(theta: np.ndarray, g: np.ndarray) -> float:
    b, loc, s = np.exp(theta[0]), theta[1], np.exp(theta[2])
    return float(-stats.gennorm.logpdf(g, b, loc, s).sum())


def fit_gennorm(g: np.ndarray) -> Fit:
    b0, l0, s0 = stats.gennorm.fit(g)
    th = _minimize(lambda t: gennorm_nll(t, g), np.array([np.log(b0), l0, np.log(s0)]))
    return Fit("Subbotin", {"b": float(np.exp(th[0])), "loc": float(th[1]),
                            "scale": float(np.exp(th[2]))}, -gennorm_nll(th, g), 3, len(g))


def aep_logpdf(x: np.ndarray, b: float, loc: float, sl: float, sr: float) -> np.ndarray:
    """歪み Subbotin(asymmetric exponential power): 左右で尺度 sl・sr が違う。"""
    s = np.where(x < loc, sl, sr)
    out: np.ndarray = (np.log(b) - np.log(sl + sr) - special.gammaln(1 / b)
                       - (np.abs(x - loc) / s) ** b)
    return out


def aep_cdf(x: np.ndarray, b: float, loc: float, sl: float, sr: float) -> np.ndarray:
    wl = sl / (sl + sr)
    left = wl * special.gammaincc(1 / b, (np.abs(loc - x) / sl) ** b)
    right = wl + (1 - wl) * special.gammainc(1 / b, (np.abs(x - loc) / sr) ** b)
    out: np.ndarray = np.where(x < loc, left, right)
    return out


def fit_aep(g: np.ndarray, start: Fit) -> Fit:
    p = start.params

    def nll(t: np.ndarray) -> float:
        return float(-aep_logpdf(g, np.exp(t[0]), t[1], np.exp(t[2]), np.exp(t[3])).sum())

    th = _minimize(nll, np.array([np.log(p["b"]), p["loc"], np.log(p["scale"]),
                                  np.log(p["scale"])]))
    return Fit("歪み Subbotin", {"b": float(np.exp(th[0])), "loc": float(th[1]),
                                "scale_l": float(np.exp(th[2])), "scale_r": float(np.exp(th[3]))},
               -nll(th), 4, len(g))


def fit_t(g: np.ndarray) -> Fit:
    d0, l0, s0 = stats.t.fit(g)

    def nll(t: np.ndarray) -> float:
        return float(-stats.t.logpdf(g, np.exp(t[0]), t[1], np.exp(t[2])).sum())

    th = _minimize(nll, np.array([np.log(max(d0, 0.1)), l0, np.log(s0)]))
    return Fit("t", {"df": float(np.exp(th[0])), "loc": float(th[1]),
                     "scale": float(np.exp(th[2]))}, -nll(th), 3, len(g))


def ks(g: np.ndarray, f: Fit) -> float:
    p = f.params
    cdf: Callable[[np.ndarray], np.ndarray]
    if f.family == "正規":
        cdf = lambda x: stats.norm.cdf(x, p["loc"], p["scale"])  # noqa: E731
    elif f.family == "Laplace":
        cdf = lambda x: stats.laplace.cdf(x, p["loc"], p["scale"])  # noqa: E731
    elif f.family == "Subbotin":
        cdf = lambda x: stats.gennorm.cdf(x, p["b"], p["loc"], p["scale"])  # noqa: E731
    elif f.family == "t":
        cdf = lambda x: stats.t.cdf(x, p["df"], p["loc"], p["scale"])  # noqa: E731
    else:
        cdf = lambda x: aep_cdf(x, p["b"], p["loc"], p["scale_l"], p["scale_r"])  # noqa: E731
    return float(stats.kstest(g, cdf).statistic)


def gennorm_wald(g: np.ndarray, f: Fit) -> dict[str, tuple[float, float]]:
    """(b, scale) の Wald 95% 区間(対数のパラメータで数値ヘッセ行列)。"""
    th = np.array([np.log(f.params["b"]), f.params["loc"], np.log(f.params["scale"])])
    h = np.zeros((3, 3))
    eps = 1e-4
    for i in range(3):
        for j in range(3):
            e_i, e_j = np.eye(3)[i] * eps, np.eye(3)[j] * eps
            h[i, j] = (gennorm_nll(th + e_i + e_j, g) - gennorm_nll(th + e_i - e_j, g)
                       - gennorm_nll(th - e_i + e_j, g) + gennorm_nll(th - e_i - e_j, g)) / (
                4 * eps * eps)
    try:
        cov = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return {"b": (np.nan, np.nan), "scale": (np.nan, np.nan)}
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"b": (float(np.exp(th[0] - 1.96 * se[0])), float(np.exp(th[0] + 1.96 * se[0]))),
            "scale": (float(np.exp(th[2] - 1.96 * se[2])), float(np.exp(th[2] + 1.96 * se[2])))}


def convex_region(f: Fit, u: np.ndarray) -> tuple[float, float, bool]:
    """c(u) = ((|log(1+u) − loc|)/scale)^b を u = Δ/x の**等間隔**の格子で調べ、2 階微分が非負の、
    最頻値を含む最大の区間 [R_lo, R_hi](R = 1 + u)を返す。

    第 3 要素は凸な区間が幅を持つか(b < 1 では最頻値で尖り、両側が凹なので、区間は 1 点に縮む)。
    """
    p = f.params
    h = float(u[1] - u[0])
    c = (np.abs(np.log1p(u) - p["loc"]) / p["scale"]) ** p["b"]
    d2 = np.diff(c, 2) / (h * h)
    ok = d2 >= -1e-6 * (1.0 + np.abs(c[1:-1]))
    mode = int(np.argmin(np.abs(np.log1p(u) - p["loc"])))
    i = min(max(mode - 1, 0), len(ok) - 1)
    if not ok[i]:
        return float("nan"), float("nan"), False
    lo, hi = i, i
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    if hi - lo < 2:
        return float("nan"), float("nan"), False
    return float(1 + u[lo + 1]), float(1 + u[hi + 1]), True


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage49"))
    args = ap.parse_args()
    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    sign = {b.name: b.sign for b in spec.basis}
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    p = p[(p["prev_total_assets"] > 0) & (p["cur_total_assets"] > 0)].reset_index(drop=True)
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    cls = np.array([lab.get((c, t - 1), np.nan) for c, t in zip(p["edinetCode"], p["t"],
                                                               strict=True)])
    ind = np.array(load_industry_config(Path("conf/accounts/industry.yaml")).labels(
        p["edinetCode"].tolist(), level="coarse"))
    u_grid = np.linspace(-0.99, 9.0, 99901)  # Δ/x の等間隔の格子(R = 0.01〜10、刻み 1e-4)
    rows: list[dict[str, object]] = []
    zero_rows: list[dict[str, object]] = []
    asym_rows: list[dict[str, object]] = []
    cls_rows: list[dict[str, object]] = []
    ind_rows: list[dict[str, object]] = []
    conv_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    qq: dict[str, tuple[np.ndarray, Fit, Fit]] = {}
    for n in names:
        x0 = p[f"prev_{n}"].fillna(0.0).to_numpy() / p["prev_total_assets"].to_numpy()
        x1 = p[f"cur_{n}"].fillna(0.0).to_numpy() / p["cur_total_assets"].to_numpy()
        n0 = p[f"prev_{n}"].fillna(0.0).to_numpy()
        n1 = p[f"cur_{n}"].fillna(0.0).to_numpy()
        z0, z1 = x0 == 0, x1 == 0
        same = ~z0 & ~z1 & (np.sign(x0) == np.sign(x1))
        flip = ~z0 & ~z1 & (np.sign(x0) != np.sign(x1))
        g = np.log(np.abs(x1[same]) / np.abs(x0[same]))
        gn = np.log(np.abs(n1[same]) / np.abs(n0[same]))  # 名目額の変化率(不動の確認用)
        total = len(x0)
        zero_rows.append({
            "account": n, "ja": ja[n], "sign": sign[n], "n": total,
            "both0": float((z0 & z1).mean()), "new": float((z0 & ~z1).mean()),
            "gone": float((~z0 & z1).mean()), "both": float(same.mean()),
            "flip": float(flip.mean()), "n_both": int(same.sum()),
            "g_lt001": float(np.mean(np.abs(g) < 0.01)) if len(g) else np.nan,
            "gn_lt0001": float(np.mean(np.abs(gn) < 0.001)) if len(gn) else np.nan,
            "gn_eq0": float(np.mean(n1[same] == n0[same])) if len(gn) else np.nan})
        # 差分(総資産比の変化)の分布: 両期 0 でない組(凸性の比較用、補足)
        dmask = ~(z0 & z1)
        if dmask.sum() >= MIN_N:
            fd = fit_gennorm(x1[dmask] - x0[dmask])
            diff_rows.append({"account": n, "ja": ja[n], "n": int(dmask.sum()),
                              "b": fd.params["b"], "scale": fd.params["scale"]})
        if len(g) < MIN_N:
            continue
        q1, q3 = np.percentile(np.abs(g), [25, 75])
        keep = np.abs(g) <= q3 + 3 * (q3 - q1)
        for version, gg in (("全件", g), ("外れ値を除く", g[keep])):
            fits = [fit_normal(gg), fit_laplace(gg)]
            fg = fit_gennorm(gg)
            fits += [fg, fit_aep(gg, fg), fit_t(gg)]
            best = min(fits, key=lambda f: f.aic)
            for f in fits:
                rows.append({"account": n, "ja": ja[n], "version": version, "family": f.family,
                             "n": f.n, "ll": f.ll, "aic": f.aic, "bic": f.bic, "ks": ks(gg, f),
                             "best_aic": f.family == best.family,
                             "lr_vs_normal": 2 * (f.ll - fits[0].ll),
                             **{f"p_{k}": v for k, v in f.params.items()}})
            if version == "全件":
                qq[n] = (gg, fits[0], fg)
                n_out = int((~keep).sum())
                # B.3 非対称性: 増加・減少の頻度(二項検定)と大きさ
                # (位置 0 の指数分布の尺度の比、尤度比)
                pos, neg = g[g > 0], -g[g < 0]
                mp, mn, ma = pos.mean(), neg.mean(), np.concatenate([pos, neg]).mean()
                lr = 2 * (len(pos) * np.log(ma / mp) + len(neg) * np.log(ma / mn))
                aep = next(f for f in fits if f.family == "歪み Subbotin")
                lr_aep = 2 * (aep.ll - fg.ll)
                asym_rows.append({
                    "account": n, "ja": ja[n], "n_pos": len(pos), "n_neg": len(neg),
                    "share_pos": len(pos) / (len(pos) + len(neg)),
                    "p_binom": float(stats.binomtest(len(pos), len(pos) + len(neg)).pvalue),
                    "scale_pos": float(mp), "scale_neg": float(mn), "ratio": float(mp / mn),
                    "p_scale": float(stats.chi2.sf(lr, 1)),
                    "aep_r_over_l": aep.params["scale_r"] / aep.params["scale_l"],
                    "p_aep": float(stats.chi2.sf(lr_aep, 1)), "n_outliers": n_out,
                    "outlier_share": n_out / len(g)})
                # 凸性(全件の Subbotin)
                r_lo, r_hi, ok = convex_region(fg, u_grid)
                rr = np.exp(g)
                conv_rows.append({
                    "account": n, "ja": ja[n], "b": fg.params["b"], "convex_at_mode": ok,
                    "R_lo": r_lo, "R_hi": r_hi,
                    "share_in": float(np.mean((rr >= r_lo) & (rr <= r_hi))) if ok else 0.0})
                # 正規(対数正規)でも同じ判定
                fn = Fit("Subbotin", {"b": 2.0, "loc": fits[0].params["loc"],
                                      "scale": fits[0].params["scale"]}, 0.0, 0, 1)
                n_lo, n_hi, n_ok = convex_region(fn, u_grid)
                conv_rows[-1].update({"R_lo_lognormal": n_lo, "R_hi_lognormal": n_hi,
                                      "share_in_lognormal": float(np.mean(
                                          (rr >= n_lo) & (rr <= n_hi))) if n_ok else 0.0})
                # C.1・C.2 群別(全件、Subbotin)
                for group_rows, col, levels in ((cls_rows, cls[same], (0.0, 1.0, 2.0)),
                                                (ind_rows, ind[same], INDUSTRIES)):
                    fits_g, ll_sep, n_g = {}, 0.0, 0
                    for lv in levels:
                        gg2 = g[col == lv]
                        if len(gg2) < MIN_N:
                            continue
                        f2 = fit_gennorm(gg2)
                        fits_g[lv] = (f2, gennorm_wald(gg2, f2))
                        ll_sep += f2.ll
                        n_g += len(gg2)
                    if len(fits_g) >= 2:
                        pooled = fit_gennorm(np.concatenate([g[col == lv] for lv in fits_g]))
                        lr_g = 2 * (ll_sep - pooled.ll)
                        pval = float(stats.chi2.sf(lr_g, 3 * (len(fits_g) - 1)))
                    else:
                        pval = float("nan")
                    for lv, (f2, ci) in fits_g.items():
                        group_rows.append({"account": n, "ja": ja[n], "group": lv, "n": f2.n,
                                           "b": f2.params["b"], "b_lo": ci["b"][0],
                                           "b_hi": ci["b"][1], "loc": f2.params["loc"],
                                           "scale": f2.params["scale"],
                                           "scale_lo": ci["scale"][0],
                                           "scale_hi": ci["scale"][1], "p_lr": pval})
    fits_df = pd.DataFrame(rows)
    fits_df.to_csv(out / "fits.csv", index=False)
    pd.DataFrame(zero_rows).to_csv(out / "zeros.csv", index=False)
    pd.DataFrame(asym_rows).to_csv(out / "asymmetry.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(out / "by_class.csv", index=False)
    pd.DataFrame(ind_rows).to_csv(out / "by_industry.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(out / "convexity.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(out / "diff_gennorm.csv", index=False)
    # QQ プロット
    for kind in ("Subbotin", "normal"):
        fig, axes = plt.subplots(6, 6, figsize=(18, 18))
        for ax, (n, (gg, fnorm, fgen)) in zip(axes.ravel(), qq.items(), strict=False):
            q = (np.arange(1, len(gg) + 1) - 0.5) / len(gg)
            pr = fgen.params if kind == "Subbotin" else fnorm.params
            theo = (stats.gennorm.ppf(q, pr["b"], pr["loc"], pr["scale"]) if kind == "Subbotin"
                    else stats.norm.ppf(q, pr["loc"], pr["scale"]))
            ax.plot(theo, np.sort(gg), ".", ms=1.5)
            lim = np.percentile(np.abs(gg), 99.5) * 1.2
            ax.plot([-lim, lim], [-lim, lim], "r-", lw=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            title = n.replace("jppfs_cor_", "")
            ax.set_title(f"{title[:28]} (b={fgen.params['b']:.2f})" if kind == "Subbotin"
                         else title[:28], fontsize=8)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(qq):]:
            ax.axis("off")
        fig.suptitle(f"QQ plot of g = log R (all obs) vs fitted {kind}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"qq_{kind.lower()}.png", dpi=110)
        plt.close(fig)
    summary = {"n_accounts_fitted": len(qq), "n_pairs": int(len(p))}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
