#!/usr/bin/env python
"""ステージ70 A.2: 目標のゼロの構造と識別 AUC(lightgbm・sklearn のプロセス)。

``stage70_ot.py`` の目標について、ちょうど 0 の科目数・未払金が 0 の割合・識別 AUC(LightGBM /
ロジスティック回帰、``identification_auc_repeated(τ, class 0, 3)``)を計算し、A.1 と合わせて
``results/stage70/precheck.md`` を書く。

    uv run python scripts/stage70_auc.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.evaluation.identification import identification_auc_repeated

OUT = Path("results/stage70")
DATA = Path("data/interim/stage70")
KEYS = ("t0.01", "t0.05", "t0.1", "t0.5", "bal")
TAU_LAB = {"t0.01": "0.01", "t0.05": "0.05", "t0.1": "0.1", "t0.5": "0.5", "bal": "∞(バランス)"}
METHODS = {"argmax": "argmax(1 社)", "m1": "M1 重み付き中央値", "m2": "M2 スパース重心",
           "l2": "L2 重心(対照)"}
VARIANTS = {"main": "主(72 社、非支配株主持分にコスト)", "nci0": "非支配株主持分 0",
            "p57": "class 1 は 57 社(同じ class の参照)"}


def main() -> None:  # noqa: PLR0915
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    x, lab, _docs, _ = load_tilde(names)
    xt = x[lab == 0]
    j_ap = names.index("accounts_payable_other")
    rows = []
    for cls in ("c2", "c1"):
        rng = np.random.default_rng(0)
        n_src = 55 if cls == "c2" else 132
        ref = {"class 0 の実在企業(無作為)": xt[rng.choice(len(xt), n_src, replace=False)]}
        z64 = np.load(f"data/interim/stage64/targets_{cls}.npz", allow_pickle=True)
        ref["ステージ69 の最終設定の目標(τ1 = 0.01 の argmax)"] = xt[
            np.asarray(z64["semi_t0.01_e0.001__argmax"]).astype(int)]
        for lab_r, tau in ref.items():
            r = identification_auc_repeated(tau, xt, 3)
            rows.append({"cls": cls, "variant": "参照", "tau1": "—", "method": lab_r,
                         "support": np.nan, "nn_match": np.nan,
                         "zero": float((tau == 0.0).sum(axis=1).mean()),
                         "ap_zero": float((tau[:, j_ap] == 0.0).mean()),
                         "auc_lgbm": float(r["lightgbm"].mean()),
                         "auc_logreg": float(r["logreg"].mean())})
        for var in VARIANTS:
            path = DATA / f"targets_{cls}_{var}.npz"
            if not path.exists():
                continue
            z = np.load(path, allow_pickle=True)
            ot = pd.read_csv(OUT / f"ot_{cls}_{var}.csv").set_index("key")
            for k in KEYS:
                for meth in METHODS:
                    tau = np.asarray(z[f"{k}__{meth}"], dtype=float)
                    r = identification_auc_repeated(tau, xt, 3)
                    rows.append({"cls": cls, "variant": var, "tau1": TAU_LAB[k], "method": meth,
                                 "support": float(ot.loc[k, "support_med"]),
                                 "nn_match": float(ot.loc[k, "nn_match"]),
                                 "zero": float((tau == 0.0).sum(axis=1).mean()),
                                 "ap_zero": float((tau[:, j_ap] == 0.0).mean()),
                                 "auc_lgbm": float(r["lightgbm"].mean()),
                                 "auc_logreg": float(r["logreg"].mean())})
                    print(f"  {cls} {var} {k} {meth}: AUC {rows[-1]['auc_lgbm']:.3f}",
                          flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "precheck_a2.csv", index=False)
    zero0 = float((xt == 0.0).sum(axis=1).mean())
    ap0 = float((xt[:, j_ap] == 0.0).mean())
    a1 = (OUT / "precheck_a1.md").read_text(encoding="utf-8").split("\n", 1)[1]
    lines = ["# ステージ70 A: 事前検算(QP を解く前)", "", "# A.1 新しいコスト", a1,
             "# A.2 目標のゼロの構造と識別 AUC", "",
             f"健全企業(class 0、3,768 社)のちょうど 0 の科目数の平均は {zero0:.2f}、未払金が 0 の"
             f"割合は {ap0:.0%}。実効サポートは輸送計画の行の 1/Σp² の中央値。「最近傍と一致」は"
             "計画の argmax が最近傍(argmin C̃)と同じ行の割合(計画の性質。M1・M2・L2 の目標は"
             "実在企業ではない)。識別 AUC は目標と class 0 の実在企業を見分ける"
             "(5-fold、3 回の平均、seed 42)。", ""]
    for var, vlab in VARIANTS.items():
        sub = df[df["variant"] == var]
        if sub.empty:
            continue
        lines += [f"## {vlab}", "",
                  "| class | τ1 | 実効サポート | 最近傍と一致 | 目標 | ちょうど 0 の科目数 "
                  "| 未払金が 0 "
                  "| AUC(LightGBM) | AUC(LogReg) |",
                  "|---|---|---:|---:|---|---:|---:|---:|---:|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r['cls']} | {r['tau1']} | {r['support']:.2f} | {r['nn_match']:.0%} "
                         f"| {METHODS[r['method']]} | {r['zero']:.2f} | {r['ap_zero']:.0%} "
                         f"| {r['auc_lgbm']:.3f} | {r['auc_logreg']:.3f} |")
        lines.append("")
    lines += ["## 参照", "", "| class | 目標 | ちょうど 0 の科目数 | 未払金が 0 | AUC(LightGBM) "
              "| AUC(LogReg) |", "|---|---|---:|---:|---:|---:|"]
    for _, r in df[df["variant"] == "参照"].iterrows():
        lines.append(f"| {r['cls']} | {r['method']} | {r['zero']:.2f} | {r['ap_zero']:.0%} "
                     f"| {r['auc_lgbm']:.3f} | {r['auc_logreg']:.3f} |")
    lines.append("")
    (OUT / "precheck.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
