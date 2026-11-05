#!/usr/bin/env python
"""ステージ75: 57 社(と fold ごとの約 45 社)の比率コスト abs・R1・R2・R3。

numpy・pandas だけのプロセス。

規則は ``results/stage75/criteria.md`` §1・§2。57 社を乱数 0 で 5 分割し、fold f のコストは
57 社から fold f を除いた企業で推定する(``f0``〜``f4``)。``full`` は 57 社の全体(D2 と診断表)。

出力: ``data/interim/stage75/weights.json``、``results/stage75/costs.md``。

R3 の定義(criteria.md §1、計算の前に記録): θ± = (1/N) Σ_k (±Δ^nom_k)₊ / (|x_{k,0}| + σ)。
これは、規格化した変化 z_k = Δ^nom_k / (|x_{k,0}| + σ) にモデル A(増加分・減少分を独立な指数分布)を
当てはめたときの尺度の最尤推定である。``--r3-compare`` は、規格化しない θ± = (1/N) Σ_k (±Δ^nom_k)₊
(Δ^nom の単位でのモデル A の最尤推定)と並べる診断だけを出し、weights.json は変えない。

    uv run python scripts/stage75_weights.py
    uv run python scripts/stage75_weights.py --r3-compare results/stage78/r3_compare.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from prep_stage59 import deltas
from qp_stage60 import SIGMA_FLOOR
from stage70_weights import FROZEN, costs
from stage74_weights import log_costs

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

OUT = Path("results/stage75")
DATA = Path("data/interim/stage75")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
CAP = 1e3
KINDS = ("abs", "R1", "R2", "R3")
SHOW = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus", "jppfs_cor_RetainedEarnings",
        "jppfs_cor_CashAndDeposits", "short_term_loans", "long_term_loans", "receivables_trade",
        "inventories", "jppfs_cor_IncomeTaxesPayable")


def folds() -> tuple[list[str], list[list[str]]]:
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    p57 = sorted(pairs[pairs["group"] == "class 1 → class 0"]["edinetCode"])
    perm = list(np.random.default_rng(0).permutation(p57))
    return p57, [list(map(str, f)) for f in np.array_split(perm, 5)]


def ratio_costs(codes: list[str], names: tuple[str, ...], sigma: np.ndarray
                ) -> dict[str, np.ndarray]:
    """R1・R2・R3 の c(行 0 = +、行 1 = −)と、n±。"""
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    b = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
    xa = a[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float)
    xb = b[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float)
    ta = a["cur_total_assets"].to_numpy(float)
    x0 = xa / ta[:, None]
    d = deltas(codes, names)
    dnom = (xb - xa) / ta[:, None]
    n_firms = len(codes)
    den = np.abs(x0) + sigma[None, :]
    out: dict[str, np.ndarray] = {}
    n = np.stack([(d > 0).sum(axis=0), (d < 0).sum(axis=0)])
    s = {k: np.zeros((2, len(names))) for k in ("R1", "R2", "R3")}
    for r, sg in enumerate((1.0, -1.0)):
        for j in range(len(names)):
            m = sg * d[:, j] > 0
            if m.any():
                s["R1"][r, j] = m.mean() * float(np.median(np.abs(d[m, j]) / den[m, j]))
                s["R2"][r, j] = m.mean() * float(np.median(np.abs(d[m, j]))) / sigma[j]
            s["R3"][r, j] = float((np.maximum(sg * dnom[:, j], 0.0) / den[:, j]).sum() / n_firms)
    frozen = np.array([nm in FROZEN for nm in names])
    for k, v in s.items():
        with np.errstate(divide="ignore"):
            c = np.where(v > 0, 1.0 / np.where(v > 0, v, 1.0), CAP)
        c = np.minimum(c, CAP)
        c[:, frozen] = 0.0
        out[k] = c
    out["n"] = n
    return out


def r3_compare(out: Path, names: tuple[str, ...], sigma: np.ndarray) -> None:
    """R3 の (A) 記録した定義(規格化あり)と (B) 規格化なしの c± を、57 社の全体で並べる。"""
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    p57, _ = folds()
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2022].set_index("edinetCode").loc[p57]
    b = p[p["t"] == 2024].set_index("edinetCode").loc[p57]
    xa = a[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float)
    xb = b[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float)
    ta = a["cur_total_assets"].to_numpy(float)
    dnom = (xb - xa) / ta[:, None]
    den = np.abs(xa / ta[:, None]) + sigma[None, :]
    frozen = np.array([nm in FROZEN for nm in names])

    def cost(theta: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            c = np.where(theta > 0, 1.0 / np.where(theta > 0, theta, 1.0), CAP)
        c = np.minimum(c, CAP)
        c[:, frozen] = 0.0
        return c
    ca = cost(np.stack([(np.maximum(sg * dnom, 0.0) / den).mean(axis=0) for sg in (1.0, -1.0)]))
    cb = cost(np.stack([np.maximum(sg * dnom, 0.0).mean(axis=0) for sg in (1.0, -1.0)]))
    rec = ratio_costs(p57, names, sigma)["R3"]
    if not np.allclose(ca, rec):
        raise ValueError("(A) が weights.json の R3 と一致しない")
    lines = ["# R3 の 2 つの定義(57 社の全体)", "",
             "(A) 記録した定義(criteria.md §1、ステージ75 の計算に使ったもの): "
             "θ± = (1/N) Σ (±Δ^nom)₊ / (|x₀| + σ)。(B) 規格化なし: θ± = (1/N) Σ (±Δ^nom)₊。"
             "c = min(1/θ, 10³)。", "",
             "| 科目 | (A) c⁺ / c⁻(比) | (B) c⁺ / c⁻(比) |", "|---|---|---|"]
    for nm in SHOW:
        j = names.index(nm)
        lines.append(f"| {ja.get(nm, nm)} | {ca[0, j]:.1f} / {ca[1, j]:.1f}"
                     f"({ca[1, j] / ca[0, j]:.2f}) | {cb[0, j]:.1f} / {cb[1, j]:.1f}"
                     f"({cb[1, j] / cb[0, j]:.2f}) |")
    act = ~frozen
    rat_a = np.log(ca[1, act] / ca[0, act])
    rat_b = np.log(cb[1, act] / cb[0, act])
    lines += ["", f"凍結以外の {int(act.sum())} 科目で、log(c⁻/c⁺) の (A) と (B) の相関 "
              f"{np.corrcoef(rat_a, rat_b)[0, 1]:.3f}、符号の一致 "
              f"{int((np.sign(rat_a) == np.sign(rat_b)).sum())} / {int(act.sum())}。"
              f"(B) で c = 10³ の上限に当たる方向 {int((cb[:, act] >= CAP).sum())}、"
              f"(A) では {int((ca[:, act] >= CAP).sum())}。", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r3-compare", type=Path, default=None,
                    help="R3 の 2 つの定義の比較だけを書き出す(weights.json は変えない)")
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    if args.r3_compare is not None:
        sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
        r3_compare(args.r3_compare, names, np.maximum(sd["sigma_1"].to_numpy(float), SIGMA_FLOOR))
        return
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    sigma = np.maximum(sd["sigma_1"].to_numpy(float), SIGMA_FLOOR)
    p57, fl = folds()
    cfgs = {f"f{i}": [c for c in p57 if c not in set(f)] for i, f in enumerate(fl)}
    cfgs["full"] = p57
    out: dict[str, object] = {"folds": {f"f{i}": f for i, f in enumerate(fl)}, "configs": {}}
    diag: dict[str, np.ndarray] = {}
    for cfg, codes in cfgs.items():
        ab = costs(deltas(codes, names), names)
        rc = ratio_costs(codes, names, sigma)
        entry = {"abs": {"plus": ab["c"][0].tolist(), "minus": ab["c"][1].tolist()},
                 **{k: {"plus": rc[k][0].tolist(), "minus": rc[k][1].tolist()}
                    for k in ("R1", "R2", "R3")}, "n_pool": len(codes)}
        out["configs"][cfg] = entry  # type: ignore[index]
        if cfg == "full":
            diag = {"abs": ab["c"], "log": log_costs(codes, names)["c"], "R1": rc["R1"],
                    "R2": rc["R2"], "R3": rc["R3"], "n": rc["n"]}
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / "weights.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    lines = ["# ステージ75: コストの診断表(57 社の全体)", "",
             "c⁺ = 増やすコスト、c⁻ = 減らすコスト。比 = c⁻/c⁺(1 より小さいと減らすほうが安い)。"
             "対数 = ステージ74 の名目の対数変化率(c⁺ = 絶対値の拡大、正の科目では増加)。"
             "n± = 総資産比の構成の差 Δ で増えた / 減った企業数(57 社中)。", "",
             "| 科目 | n+ / n− | abs c⁺ / c⁻(比) | 対数 c⁺ / c⁻(比) | R1 c⁺ / c⁻(比) "
             "| R2 c⁺ / c⁻(比) | R3 c⁺ / c⁻(比) |", "|---|---|---|---|---|---|---|"]

    def cell(c: np.ndarray, j: int) -> str:
        return f"{c[0, j]:.1f} / {c[1, j]:.1f}({c[1, j] / c[0, j]:.2f})"
    for nm in SHOW:
        j = names.index(nm)
        cells = " | ".join(cell(diag[k], j) for k in ("abs", "log", "R1", "R2", "R3"))
        lines.append(f"| {ja.get(nm, nm)} | {diag['n'][0, j]} / {diag['n'][1, j]} | {cells} |")
    lines += ["", "R1〜R3 は min(1/s, 10³)、n < 5 の代替なし。abs は従来の規則(n < 5 は"
              "中央値で代替、"
              "残余は 10³ で上限)。fold の分け方は乱数 0(各 fold 11〜12 社)。", ""]
    (OUT / "costs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
