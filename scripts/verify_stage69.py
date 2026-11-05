#!/usr/bin/env python
"""ステージ69 B: 命題 1・2 の検算(POT のプロセス。生成はしない)。

- 命題 1: 一様な周辺分布のバランス輸送の厳密解(``ot.emd``、ステージ64 と同じコスト
  C̃ = C / median(C))で、行ごとの非ゼロ要素数・実効サポート 1/Σp² と、下界 n1/n0、
  頂点解の上界 (n0 + n1 − 1)/n0 を比べる。
- 命題 2: 健全企業(class 0)の「ちょうど 0」の科目数と、2 社の平均・バランス OT の重心
  (ステージ64 の ``bal_e0.001__bary``)での数を比べる。

命題 1 はコストを 3 通りで解く(ステージ78 で追加): ステージ60 のコストで同一企業のマスクなし
(ステージ69 の記録)、ステージ60 のコスト + 同一企業のマスク、R1(57 社の全体、ステージ75)+ 同一
企業のマスク。マスクは ``stage75_ot.py`` と同じ(C̃ = C / median(C[~mask])、C̃[mask] = 10 max C̃)。
命題 2 の 2 社の平均は、異なる 2 社のちょうど 10,000 組(ステージ69 の記録は i1 = i2 を除いた 9,998
組)。ゼロは「ちょうど 0」と |x| < 10⁻⁶ の両方で数える。

出力: ``--out``(既定 ``results/stage69``)の ``verify.md``・``verify.json``。

    uv run python scripts/verify_stage69.py --out results/stage78/verify
"""

from __future__ import annotations

import os
import sys

# POT の torch バックエンドと sklearn の読み込みを止める(precompute_transport.py と同じ)
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import ot  # noqa: E402
import pandas as pd  # noqa: E402
from prep_stage59 import l1_cost  # noqa: E402
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.errors import SolverNotConvergedError  # noqa: E402

OUT = Path("results/stage69")
_ZERO = 0.0  # 「ちょうど 0」(総資産比の値が厳密に 0)
_ZERO_TOL = 1e-6  # QP の終点と同じ判定(stage72_evaluate.ZERO)
_ZERO_OT = ("jppfs_cor_RetainedEarnings", "valuation_difference_securities")
COSTS = (("stage60", False), ("stage60", True), ("R1", True))


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    x, lab, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    tgt_codes = np.array([code_of.get(str(d), "") for d in docs_a[lab == 0]])
    r1 = json.loads(Path("data/interim/stage75/weights.json").read_text(
        encoding="utf-8"))["configs"]["full"]["R1"]
    res: dict[str, dict[str, float]] = {}
    lines = ["# ステージ69 B: 命題 1・2 の検算", "",
             "## 命題 1: バランス輸送の厳密解のサポート", "",
             "一様な周辺分布 a = 1/n0、b = 1/n1。コストは非対称 L1 を中央値で割ったもの(ステー"
             "ジ64 と"
             "同じ)。非ゼロは π_ij > 1e-15。実効サポートは行で正規化した p_ij = n0 π_ij に対す"
             "る 1/Σp²。",
             "", "| class | コスト | マスク | n0 | n1 | n1/n0 | ⌈n1/n0⌉ | 行の非ゼロ数 最小 / 平均 "
                 "/ 最大 | 非ゼロの総数 "
             "(上界 n0 + n1 − 1) | 頂点の上界 (n0+n1−1)/n0 | 実効サポート 最小 / 中央値 / 平均 "
             "| max π_ij × n1 |",
             "|---|---|---|---:|---:|---:|---:|---|---|---:|---|---:|"]
    for cls, (cost, use_mask) in [(c, v) for c in ("c2", "c1") for v in COSTS]:
        if cost == "stage60":
            meta = json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(
                encoding="utf-8"))
            cp, cm = np.array(meta["c_plus"]), np.array(meta["c_minus"])
        else:
            cp, cm = np.array(r1["plus"], dtype=float), np.array(r1["minus"], dtype=float)
            for nm in _ZERO_OT:
                cp[names.index(nm)] = cm[names.index(nm)] = 0.0
        src = docs_a[sources(cls, lab, docs_a)]
        xs = x[sources(cls, lab, docs_a)]
        ct = l1_cost(xs, xt, cp, cm)
        src_codes = np.array([code_of.get(str(d), "") for d in src])
        mask = (src_codes[:, None] == tgt_codes[None, :]) & (src_codes[:, None] != "")
        if use_mask:
            ct = ct / np.median(ct[~mask])
            ct[mask] = 10.0 * ct.max()
        else:
            ct = ct / np.median(ct)
        n0, n1 = ct.shape
        a, b = np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1)
        p, log = ot.emd(a, b, ct, numItermax=10_000_000, log=True)
        if log.get("warning"):
            raise SolverNotConvergedError(f"ot.emd: {log['warning']}")
        nz = (p > 1e-15).sum(axis=1)
        q = p / p.sum(axis=1, keepdims=True)
        eff = 1.0 / (q ** 2).sum(axis=1)
        key = f"{cls}|{cost}|{'mask' if use_mask else 'nomask'}"
        res[key] = {"n0": n0, "n1": n1, "ratio": n1 / n0, "ceil": math.ceil(n1 / n0),
                    "n_masked_pairs": int(mask.sum()),
                    "mass_on_masked": float(p[mask].sum() * n0),
                    "nz_min": int(nz.min()), "nz_mean": float(nz.mean()), "nz_max": int(nz.max()),
                    "nz_total": int(nz.sum()), "vertex_ub": (n0 + n1 - 1) / n0,
                    "eff_min": float(eff.min()), "eff_med": float(np.median(eff)),
                    "eff_mean": float(eff.mean()), "max_pn1": float(p.max() * n1)}
        r = res[key]
        lines.append(f"| {cls} | {cost} | {'あり' if use_mask else 'なし'} | {n0} | {n1} "
                     f"| {r['ratio']:.3f} | {r['ceil']} | {r['nz_min']} / "
                     f"{r['nz_mean']:.3f} / {r['nz_max']} | {r['nz_total']}({n0 + n1 - 1}) "
                     f"| {r['vertex_ub']:.3f} | {r['eff_min']:.3f} / {r['eff_med']:.3f} / "
                     f"{r['eff_mean']:.3f} | {r['max_pn1']:.6f} |")
    lines += ["", "## 命題 2: 凸結合によるゼロの構造の消失", "",
              "「ちょうど 0」の科目数(32 科目中)。2 社の平均は class 0 から無作為に 10,000 組("
              "乱数 0)。"
              "重心はステージ64 の `bal_e0.001__bary`(生成の目標に使ったもの)。", "",
              "| 対象 | 件数 | ちょうど 0 の科目数 平均 | 中央値 | \\|x\\| < 10⁻⁶ の科目数 平均 "
              "| 未払金が 0 の割合 |",
              "|---|---:|---:|---:|---:|---:|"]
    j_ap = names.index("accounts_payable_other")
    rng = np.random.default_rng(0)
    # 異なる 2 社をちょうど 10,000 組: i2 は i1 以外の n − 1 社から一様に選ぶ
    i1 = rng.integers(0, len(xt), 10_000)
    i2 = (i1 + rng.integers(1, len(xt), 10_000)) % len(xt)
    if (i1 == i2).any():
        raise ValueError("同じ企業の組がある")
    pairs = 0.5 * (xt[i1] + xt[i2])
    rows = [("健全企業(class 0、実在)", xt), ("2 社の平均", pairs)]
    for cls in ("c2", "c1"):
        z = np.load(f"data/interim/stage64/targets_{cls}.npz", allow_pickle=True)
        rows.append((f"バランス OT の重心({cls})", np.asarray(z["bal_e0.001__bary"], dtype=float)))
        rows.append((f"提案の対応先({cls}、argmax)",
                     xt[np.asarray(z["semi_t0.01_e0.001__argmax"]).astype(int)]))
    for lab_t, v in rows:
        zc = (v == _ZERO).sum(axis=1)
        zt = (np.abs(v) < _ZERO_TOL).sum(axis=1)
        res[lab_t] = {"n": len(v), "zero_mean": float(zc.mean()), "zero_med": float(np.median(zc)),
                      "zero_tol_mean": float(zt.mean()),
                      "ap_zero": float((v[:, j_ap] == _ZERO).mean())}
        lines.append(f"| {lab_t} | {len(v)} | {zc.mean():.2f} | {np.median(zc):.0f} "
                     f"| {zt.mean():.2f} | {(v[:, j_ap] == _ZERO).mean():.0%} |")
    lines.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verify.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
