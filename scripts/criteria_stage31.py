#!/usr/bin/env python
"""ステージ31 A: 再建群 15 社から作る基準に標本誤差を入れ、γ の候補を出す(生成の前に実行)。

入力はステージ30 の ``results/stage30/recovery_firms.csv``(再建群 Y: 2022 注記あり → 2024 注記なし、
特殊業種を除き、パネルにある 15 社)。年率 × 5 を 5 年換算とする(ステージ29・30 と同じ)。

- 基準 2(RE が増える社の割合): Wilson 95% 区間(参考に Clopper–Pearson)
- 基準 3(利益部分の上限): 5 × p90 のブートストラップ 95% 区間(企業を復元抽出、10,000 回、seed 0)
- 基準 4・5(振替・増資の規模): 該当する社の p5 と p95 のブートストラップ区間を求め、
  帯を [p5 の区間の下端, p95 の区間の上端] に広げる
- γ の候補: p90(全社)、p90(Tukey の外れ値 Q3 + 1.5 IQR を除く)、中央値の 2 倍・3 倍

    uv run python scripts/criteria_stage31.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

THR = 0.001
YEARS = 5.0
_B = 10_000
_Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    denom = 1 + _Z**2 / n
    center = (p + _Z**2 / (2 * n)) / denom
    half = _Z * np.sqrt(p * (1 - p) / n + _Z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def boot_quantile(x: np.ndarray, q: float, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(_B, len(x)))
    qs = np.quantile(x[idx], q, axis=1)
    lo, hi = np.percentile(qs, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--firms", type=Path, default=Path("results/stage30/recovery_firms.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/stage31"))
    args = ap.parse_args()
    df = pd.read_csv(args.firms, dtype={"edinetCode": str})
    y = df[(df["group"] == "Y") & (df["reason"].fillna("") == "") & df["in_panel"]]
    n = len(y)
    rng = np.random.default_rng(0)

    k_up = int((y["d_re"] > THR).sum())
    wl, wh = wilson(k_up, n)
    cl, ch = clopper_pearson(k_up, n)

    p = y["profit"].to_numpy()
    p90 = float(np.quantile(p, 0.9))
    g_lo, g_hi = boot_quantile(p, 0.9, rng)
    q1, q3 = np.quantile(p, [0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    outliers = y.loc[y["profit"] > fence, "filerName"].tolist()
    p_clean = p[p <= fence]
    p90_clean = float(np.quantile(p_clean, 0.9))
    med = float(np.median(p))
    d_re = y["d_re"].to_numpy()
    q1r, q3r = np.quantile(d_re, [0.25, 0.75])
    dre_clean = d_re[d_re <= q3r + 1.5 * (q3r - q1r)]

    t = y.loc[y["transfer"] > THR, "transfer"].to_numpy() * YEARS
    i = y.loc[y["d_paid"] > THR, "d_paid"].to_numpy() * YEARS

    def band(x: np.ndarray) -> dict[str, float]:
        p5_lo, _ = boot_quantile(x, 0.05, rng)
        _, p95_hi = boot_quantile(x, 0.95, rng)
        return {
            "n": len(x), "p5": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95)),
            "band_lo": p5_lo, "band_hi": p95_hi, "min": float(x.min()), "max": float(x.max()),
        }

    bt, bi = band(t), band(i)
    gamma = YEARS * p90_clean
    criteria = {
        "n_y": n,
        "k_re_up": k_up,
        "share_re_up": k_up / n,
        "share_band_wilson": [wl, wh],
        "share_band_clopper_pearson": [cl, ch],
        "gamma_5y_p90_all": YEARS * p90,
        "gamma_5y_p90_all_boot95": [YEARS * g_lo, YEARS * g_hi],
        "tukey_fence_annual": float(fence),
        "tukey_outliers": outliers,
        "gamma_5y_p90_excl_outliers": gamma,
        "gamma_5y_median_x2": YEARS * 2 * med,
        "gamma_5y_median_x3": YEARS * 3 * med,
        "dre_5y_p90_excl_outliers": YEARS * float(np.quantile(dre_clean, 0.9)),
        "gamma": gamma,
        "gamma_rule": "5 × p90(利益部分の年率、Tukey の外れ値を除く)",
        "transfer_5y": bt,
        "issue_5y": bi,
        "threshold_annual": THR,
        "years": YEARS,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "criteria.json").write_text(
        json.dumps(criteria, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(criteria, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
