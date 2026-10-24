#!/usr/bin/env python
"""ステージ70 A.2 の診断(結果を見てから追加した。判定には使わない。lightgbm・sklearn のプロセス)。

M1・M2 の目標は、ゼロの構造を保つのに識別 AUC が高い。原因が「科目ごとに中央値・平均を
取ると、BS の全体の整合(貸借一致・総資産 = 1)が崩れること」かを見る。目標を (1) 貸借一致へ
最小の L2 補正 τ − C (Cᵀτ) / (CᵀC)、(2) さらに総資産で割って 1 にそろえた後で、識別 AUC と
ちょうど 0 の科目数を測る。補正は 0 の科目を動かしうる(全科目に残差を配る)。

出力: ``results/stage70/diag_m1.md``。

    uv run python scripts/stage70_diag.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import subtotal_members
from fsgen.evaluation.identification import identification_auc_repeated

OUT = Path("results/stage70")
DATA = Path("data/interim/stage70")
KEYS = (("t0.05__m1", "M1、τ1 = 0.05"), ("t0.1__m1", "M1、τ1 = 0.1"), ("bal__m1", "M1、バランス"),
        ("t0.05__m2", "M2、τ1 = 0.05"), ("t0.1__m2", "M2、τ1 = 0.1"))


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)[:, 0]
    assets = subtotal_members(spec, names)["jppfs_cor_Assets"]
    x, lab, _d, _ = load_tilde(names)
    xt = x[lab == 0]
    lines = ["# ステージ70 A.2 の診断: M1・M2 の目標の全体の整合(結果を見てから追加、判定に"
             "使わない)",
             "", "補正 1 = 貸借一致への最小の L2 補正。補正 2 = 補正 1 の後に総資産で割って 1 に"
             "そろえる。"
             "残差は |Cᵀτ|(総資産比)の中央値。", "",
             "| class | 目標 | 残差 | 総資産の中央値 | AUC(元) | AUC(補正 1) | AUC(補正 2) "
             "| AUC(ゼロの指標だけ) | ちょうど 0(元 / 補正 2) |",
             "|---|---|---:|---:|---|---|---|---|---|"]
    for cls in ("c2", "c1"):
        z = np.load(DATA / f"targets_{cls}_main.npz", allow_pickle=True)
        for key, lab_t in KEYS:
            t = np.asarray(z[key], dtype=float)
            t1 = t - np.outer(t @ c, c) / float(c @ c)
            t2 = t1 / (t1 @ assets)[:, None]
            aucs = []
            for v, ref in ((t, xt), (t1, xt), (t2, xt),
                           ((t == 0).astype(float), (xt == 0).astype(float))):
                r = identification_auc_repeated(v, ref, 3)
                aucs.append(f"{r['lightgbm'].mean():.3f} / {r['logreg'].mean():.3f}")
            lines.append(f"| {cls} | {lab_t} | {np.median(np.abs(t @ c)):.3f} "
                         f"| {np.median(t @ assets):.3f} | {aucs[0]} | {aucs[1]} | {aucs[2]} "
                         f"| {aucs[3]} "
                         f"| {(t == 0).sum(axis=1).mean():.2f} / "
                         f"{(t2 == 0).sum(axis=1).mean():.2f} |")
            print(lines[-1], flush=True)
    lines += ["", "AUC は LightGBM / ロジスティック回帰。「ゼロの指標だけ」は、各科目が"
              "ちょうど 0 か"
              "どうか(0/1)だけを特徴量にして、class 0 の実在企業と比べたもの。", ""]
    (OUT / "diag_m1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
