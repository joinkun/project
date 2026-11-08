#!/usr/bin/env python
"""ステージ79: 「移動量 ÷ 区間面積」のコスト A1・A2(numpy・pandas だけのプロセス)。

規則は ``results/stage79/criteria.md`` §1・§2。プールと fold はステージ75 と同じ(57 社、乱数 0 の
5 分割、``data/interim/stage75/weights.json`` の ``folds``)。

- A1: c± = min(1/(P± W±), 10³)、P± = n±/N、W± = mean(min(Z, m±))(= ∫_0^m (1 − F))。
- A2: 小さい区間の傾き c₁ = A1、大きい区間の傾き c₂ = min(1/(P± W₂±), 10³)、
  W₂± = mean((Z − m±)₊)(= ∫_m^∞ (1 − F))。区切りは m±(規格化した幅 Z の中央値)。

Z = |Δ| / (|x₀| + σ) は、向き ± に動いた企業だけで計算する。n± = 0 の向きは c = 10³。評価差額金は
c = 0(凍結)。利益剰余金の 0 は、輸送・QP のスクリプトの側で入れる(ステージ75 と同じ)。

出力: ``data/interim/stage79/weights.json``、``results/stage79/costs.md``。

    uv run python scripts/stage79_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from prep_stage59 import deltas
from stage70_weights import FROZEN

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

OUT = Path("results/stage79")
DATA = Path("data/interim/stage79")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
CAP = 1e3
SIGMA_FLOOR = 1e-3  # qp_stage60.SIGMA_FLOOR と同じ(cvxpy を読み込まないよう、ここに置く)
SHOW = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus", "jppfs_cor_RetainedEarnings",
        "jppfs_cor_CashAndDeposits", "short_term_loans", "long_term_loans", "receivables_trade",
        "inventories", "jppfs_cor_IncomeTaxesPayable")


def area_costs(codes: list[str], names: tuple[str, ...], sigma: np.ndarray
               ) -> dict[str, np.ndarray]:
    """A1・A2 の係数。各配列は (2, D)、行 0 = +(増やす)、行 1 = −(減らす)。"""
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    x0 = a[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float) / a[
        "cur_total_assets"].to_numpy(float)[:, None]
    d = deltas(codes, names)
    z = np.abs(d) / (np.abs(x0) + sigma[None, :])
    n_firms, dim = d.shape
    c1, c2, m, n = (np.full((2, dim), CAP), np.full((2, dim), CAP), np.zeros((2, dim)),
                    np.zeros((2, dim), dtype=int))
    for r, sg in enumerate((1.0, -1.0)):
        for j in range(dim):
            mv = sg * d[:, j] > 0
            n[r, j] = int(mv.sum())
            if not mv.any():
                continue
            zz = z[mv, j]
            med = float(np.median(zz))
            prob = mv.sum() / n_firms
            w1 = float(np.minimum(zz, med).mean())
            w2 = float(np.maximum(zz - med, 0.0).mean())
            m[r, j] = med
            c1[r, j] = min(1.0 / (prob * w1), CAP) if w1 > 0 else CAP
            c2[r, j] = min(1.0 / (prob * w2), CAP) if w2 > 0 else CAP
    frozen = np.array([nm in FROZEN for nm in names])
    c1[:, frozen] = c2[:, frozen] = 0.0
    return {"c1": c1, "c2": c2, "m": m, "n": n}


def main() -> None:  # noqa: PLR0915
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    sigma = np.maximum(sd["sigma_1"].to_numpy(float), SIGMA_FLOOR)
    w75 = json.loads(Path("data/interim/stage75/weights.json").read_text(encoding="utf-8"))
    folds = w75["folds"]
    p57 = sorted(c for f in folds.values() for c in f)
    cfgs = {f: [c for c in p57 if c not in set(codes)] for f, codes in folds.items()}
    cfgs["full"] = p57
    out: dict[str, object] = {"folds": folds, "names": list(names), "configs": {}}
    diag: dict[str, np.ndarray] = {}
    for cfg, codes in cfgs.items():
        ac = area_costs(codes, names, sigma)
        r1 = w75["configs"][cfg]["R1"]
        out["configs"][cfg] = {  # type: ignore[index]
            "R1": r1,
            "A1": {"plus": ac["c1"][0].tolist(), "minus": ac["c1"][1].tolist()},
            "A2": {"c1_plus": ac["c1"][0].tolist(), "c1_minus": ac["c1"][1].tolist(),
                   "c2_plus": ac["c2"][0].tolist(), "c2_minus": ac["c2"][1].tolist(),
                   "m_plus": ac["m"][0].tolist(), "m_minus": ac["m"][1].tolist()},
            "n_pool": len(codes)}
        if cfg == "full":
            diag = {**ac, "R1": np.array([r1["plus"], r1["minus"]], dtype=float)}
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / "weights.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    act = (diag["n"] > 0) & (diag["c1"] < CAP) & (diag["c2"] < CAP) & (diag["c1"] > 0)
    ratio = diag["c1"][act] / diag["R1"][act]
    concave = diag["c2"][act] < diag["c1"][act]
    corr_log = float(np.corrcoef(np.log(diag["c1"][act]), np.log(diag["R1"][act]))[0, 1])
    lines = ["# ステージ79: コスト A1・A2 の診断(57 社の全体)", "",
             "c⁺ = 増やすコスト、c⁻ = 減らすコスト。"
             "A2 の c₁ = 小さい区間(A1 と同じ)、c₂ = 中央値より大きい区間。"
             "n± = 動いた企業数(57 社中)。", "",
             "| 科目 | n+ / n− | R1 c⁺ / c⁻ | A1 c⁺ / c⁻ | A1 / R1(+ / −) | A2 c₂⁺ / c₂⁻ "
             "| c₂ / c₁(+ / −) |",
             "|---|---|---|---|---|---|---|"]
    for nm in SHOW:
        j = names.index(nm)
        r1, a1, a2 = diag["R1"][:, j], diag["c1"][:, j], diag["c2"][:, j]
        lines.append(f"| {ja.get(nm, nm)} | {diag['n'][0, j]} / {diag['n'][1, j]} "
                     f"| {r1[0]:.1f} / {r1[1]:.1f} | {a1[0]:.1f} / {a1[1]:.1f} "
                     f"| {a1[0] / r1[0]:.2f} / {a1[1] / r1[1]:.2f} | {a2[0]:.1f} / {a2[1]:.1f} "
                     f"| {a2[0] / a1[0]:.2f} / {a2[1] / a1[1]:.2f} |")
    lines += ["", f"- 上限に当たらない {int(act.sum())} 方向で、"
              f"A1 / R1 の中央値 {np.median(ratio):.3f}"
              f"(四分位 {np.quantile(ratio, 0.25):.3f}〜{np.quantile(ratio, 0.75):.3f})。"
              f"指数分布なら 2 ln 2 = {2 * np.log(2):.3f} で一定(criteria §2)。",
              f"- A2 で c₂ < c₁(大きく動かすほうが単位あたり安い = 凹)の方向: "
              f"{int(concave.sum())} / {int(act.sum())}。",
              f"- log(A1) と log(R1) の相関 {corr_log:.3f}。",
              ""]
    (OUT / "costs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
