#!/usr/bin/env python
"""ステージ69 B: 命題 1・2 の検算(POT のプロセス。生成はしない)。

- 命題 1: 一様な周辺分布のバランス輸送の厳密解(``ot.emd``、ステージ64 と同じコスト
  C̃ = C / median(C))で、行ごとの非ゼロ要素数・実効サポート 1/Σp² と、下界 n1/n0、
  頂点解の上界 (n0 + n1 − 1)/n0 を比べる。
- 命題 2: 健全企業(class 0)の「ちょうど 0」の科目数と、2 社の平均・バランス OT の重心
  (ステージ64 の ``bal_e0.001__bary``)での数を比べる。

出力: ``results/stage69/verify.md``・``verify.json``。

    uv run python scripts/verify_stage69.py
"""

from __future__ import annotations

import os
import sys

# POT の torch バックエンドと sklearn の読み込みを止める(precompute_transport.py と同じ)
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import ot  # noqa: E402
from prep_stage59 import l1_cost  # noqa: E402
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.errors import SolverNotConvergedError  # noqa: E402

OUT = Path("results/stage69")
_ZERO = 0.0  # 「ちょうど 0」(総資産比の値が厳密に 0)


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    x, lab, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    xt = x[lab == 0]
    res: dict[str, dict[str, float]] = {}
    lines = ["# ステージ69 B: 命題 1・2 の検算", "",
             "## 命題 1: バランス輸送の厳密解のサポート", "",
             "一様な周辺分布 a = 1/n0、b = 1/n1。コストは非対称 L1 を中央値で割ったもの(ステー"
             "ジ64 と"
             "同じ)。非ゼロは π_ij > 1e-15。実効サポートは行で正規化した p_ij = n0 π_ij に対す"
             "る 1/Σp²。",
             "", "| class | n0 | n1 | n1/n0 | ⌈n1/n0⌉ | 行の非ゼロ数 最小 / 平均 / 最大 | 非ゼ"
                 "ロの総数 "
             "(上界 n0 + n1 − 1) | 頂点の上界 (n0+n1−1)/n0 | 実効サポート 最小 / 中央値 / 平均 "
             "| max π_ij × n1 |",
             "|---|---:|---:|---:|---:|---|---|---:|---|---:|"]
    for cls in ("c2", "c1"):
        meta = json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(encoding="utf-8"))
        cp, cm = np.array(meta["c_plus"]), np.array(meta["c_minus"])
        xs = x[sources(cls, lab, docs_a)]
        ct = l1_cost(xs, xt, cp, cm)
        ct = ct / np.median(ct)
        n0, n1 = ct.shape
        a, b = np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1)
        p, log = ot.emd(a, b, ct, numItermax=10_000_000, log=True)
        if log.get("warning"):
            raise SolverNotConvergedError(f"ot.emd: {log['warning']}")
        nz = (p > 1e-15).sum(axis=1)
        q = p / p.sum(axis=1, keepdims=True)
        eff = 1.0 / (q ** 2).sum(axis=1)
        res[cls] = {"n0": n0, "n1": n1, "ratio": n1 / n0, "ceil": math.ceil(n1 / n0),
                    "nz_min": int(nz.min()), "nz_mean": float(nz.mean()), "nz_max": int(nz.max()),
                    "nz_total": int(nz.sum()), "vertex_ub": (n0 + n1 - 1) / n0,
                    "eff_min": float(eff.min()), "eff_med": float(np.median(eff)),
                    "eff_mean": float(eff.mean()), "max_pn1": float(p.max() * n1)}
        r = res[cls]
        lines.append(f"| {cls} | {n0} | {n1} | {r['ratio']:.3f} | {r['ceil']} | {r['nz_min']} / "
                     f"{r['nz_mean']:.3f} / {r['nz_max']} | {r['nz_total']}({n0 + n1 - 1}) "
                     f"| {r['vertex_ub']:.3f} | {r['eff_min']:.3f} / {r['eff_med']:.3f} / "
                     f"{r['eff_mean']:.3f} | {r['max_pn1']:.6f} |")
    lines += ["", "## 命題 2: 凸結合によるゼロの構造の消失", "",
              "「ちょうど 0」の科目数(32 科目中)。2 社の平均は class 0 から無作為に 10,000 組("
              "乱数 0)。"
              "重心はステージ64 の `bal_e0.001__bary`(生成の目標に使ったもの)。", "",
              "| 対象 | 件数 | ちょうど 0 の科目数 平均 | 中央値 | 未払金が 0 の割合 |",
              "|---|---:|---:|---:|---:|"]
    j_ap = names.index("accounts_payable_other")
    rng = np.random.default_rng(0)
    i1, i2 = rng.integers(0, len(xt), 10_000), rng.integers(0, len(xt), 10_000)
    keep = i1 != i2
    pairs = 0.5 * (xt[i1[keep]] + xt[i2[keep]])
    rows = [("健全企業(class 0、実在)", xt), ("2 社の平均", pairs)]
    for cls in ("c2", "c1"):
        z = np.load(f"data/interim/stage64/targets_{cls}.npz", allow_pickle=True)
        rows.append((f"バランス OT の重心({cls})", np.asarray(z["bal_e0.001__bary"], dtype=float)))
        rows.append((f"提案の対応先({cls}、argmax)",
                     xt[np.asarray(z["semi_t0.01_e0.001__argmax"]).astype(int)]))
    for lab_t, v in rows:
        zc = (v == _ZERO).sum(axis=1)
        res[lab_t] = {"n": len(v), "zero_mean": float(zc.mean()), "zero_med": float(np.median(zc)),
                      "ap_zero": float((v[:, j_ap] == _ZERO).mean())}
        lines.append(f"| {lab_t} | {len(v)} | {zc.mean():.2f} | {np.median(zc):.0f} "
                     f"| {(v[:, j_ap] == _ZERO).mean():.0%} |")
    lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
