#!/usr/bin/env python
"""ステージ68 B の診断(結果を見てから追加した。判定には使わない)。

T2 は λ̃ を小さくすると違反が出て、違反 0 になるのは動かない λ̃ = 30 だけだった。その原因を見るため、
15 項目のチェックを (1) 目標そのもの(x → τ)と (2) 実データの再建企業の実際の変化に当て、
(3) T2 の生成の違反を λ̃ ごとに項目別に数える。numpy・pandas だけのプロセス。

出力: ``results/stage68/diagnosis.md``。

    uv run python scripts/diag_stage68.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from targets_stage68 import pool, sources
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    evaluate_checks,
    evaluate_extra_checks,
)

OUT = Path("results/stage68")
DATA = Path("data/interim/stage68")
CHECKS = (*CHECK_NAMES, *EXTRA_CHECK_NAMES)
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def flags(x0: np.ndarray, x1: np.ndarray, spec: object, names: tuple[str, ...]) -> np.ndarray:
    out: np.ndarray = np.column_stack([evaluate_checks(x0, x1, spec, names),  # type: ignore[arg-type]
                                       evaluate_extra_checks(x1, x0, spec, names)])  # type: ignore[arg-type]
    return out


def row(lab: str, fl: np.ndarray) -> str:
    return (f"| {lab} | {int(fl.any(axis=1).sum())} / {len(fl)} | "
            + " | ".join(str(int(v)) for v in fl.sum(axis=0)) + " |")


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    head = "| 対象 | 違反した企業 | " + " | ".join(c.split("_")[0] for c in CHECKS) + " |"
    sep = "|---|---:|" + "---:|" * len(CHECKS)
    lines = ["# ステージ68 B の診断(結果を見てから追加、判定には使わない)", "",
             "15 項目のチェック(`evaluate_checks`・`evaluate_extra_checks`)を、生成の終点の"
             "代わりに次のものに当てた。列は項目番号ごとの違反した企業数。", "",
             "項目: " + "、".join(CHECKS), "",
             "## (1) 実データの再建企業の実際の変化(2022 → 2024、始点の総資産比)", "", head, sep]
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    codes, comp22, _comp24, dnom = pool(names)
    for grp in ("class 2 → 回復", "class 1 → class 0"):
        idx = [i for i, c in enumerate(codes)
               if pairs.set_index("edinetCode").loc[c, "group"] == grp]
        x0 = comp22[idx]
        lines.append(row(f"{grp}({len(idx)} 社)", flags(x0, x0 + dnom[idx], spec, names)))
    lines += ["", "## (2) 目標そのもの(始点 → τ)", "", head, sep]
    x, lab, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    for cls in ("c2", "c1"):
        m = sources(cls, lab, docs_a)
        xs = x[m]
        z = np.load(DATA / f"targets_{cls}.npz", allow_pickle=True)
        for key in ("t0", "t1", "t2"):
            lines.append(row(f"{cls} {key.upper()}", flags(xs, np.asarray(z[key]), spec, names)))
    lines += ["", "## (3) T2(設定 4)の生成の違反(λ̃ ごと、class 1 の 128 社)", "",
              "| λ̃ | 違反した企業 | " + " | ".join(c.split("_")[0] for c in CHECKS) + " |", sep]
    v = pd.read_csv(DATA / "metrics" / "viol_c1.csv", dtype={"docID": str})
    excl = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    v = v[~v["docID"].map(lambda d: code_of.get(d, "") in excl) & (v["docID"] != "S100VGDJ")]
    for lt in LAMS:
        vs = v[v["setting"] == f"t2m1__l{lt:g}"].drop_duplicates(["docID", "check"])
        lines.append(f"| {lt:g} | {vs['docID'].nunique()} | "
                     + " | ".join(str(int((vs["check"] == c).sum())) for c in CHECKS) + " |")
    (OUT / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
