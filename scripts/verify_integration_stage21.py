#!/usr/bin/env python
"""ステージ21-C B.3・B.4: canonical の統合前後を比べる。

統合前は ``data/interim/canonical_v32_pre21/core.yaml`` と
``data/interim/stage21/<set>/``、統合後は ``conf/accounts/core.yaml`` と
``data/interim/stage21/integrated/<set>/``(いずれも ``verify_prior_stage21.py`` の出力)。
対象は非金融・解析成功の社。

- B.3: D、rank(C)、恒等式残差、__OTHER__ 各科目の総資産比、グループ科目の出現率、
  値が変わった社数(当期・前期、提出期間別)
- B.4: 統合した要素の、期ごとの出現社数(統合前の ``leaves.csv`` から)

    uv run python scripts/verify_integration_stage21.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    compute_rank,
    load_canonical_spec,
)

_SETS = ("fy2024", "2023_2024")
_PERIODS = ("current", "prior")
_OTHER = "__OTHER__jppfs_cor_"


def _load(interim: Path, set_name: str, period: str) -> pd.DataFrame:
    df = pd.read_csv(interim / set_name / f"values_{period}.csv", dtype={"docID": str})
    ok = (df["parse_status"] == "ok") & (df["is_financial"].astype(str) != "True")
    df = df[ok & df["total_assets"].notna() & (df["total_assets"] != 0)]
    return df.set_index("docID")


def _added_members(pre_core: Path, post_core: Path) -> dict[str, list[str]]:
    pre = {b["name"]: b["members"] for b in yaml.safe_load(pre_core.read_text())["basis"]}
    post = {b["name"]: b["members"] for b in yaml.safe_load(post_core.read_text())["basis"]}
    return {n: [m for m in ms if m not in pre[n]] for n, ms in post.items() if ms != pre[n]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-core", type=Path, default=Path("data/interim/canonical_v32_pre21/core.yaml")
    )
    parser.add_argument("--post-core", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--pre", type=Path, default=Path("data/interim/stage21"))
    parser.add_argument("--post", type=Path, default=Path("data/interim/stage21/integrated"))
    parser.add_argument("--pre-results", type=Path, default=Path("results/stage21"))
    parser.add_argument("--post-results", type=Path, default=Path("results/stage21/integrated"))
    parser.add_argument("--title", default="ステージ21-C B.3・B.4: canonical の統合前後の比較")
    parser.add_argument("--leaks", type=Path, default=Path("data/interim/stage21/leaks"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/stage21/integration_verification.md")
    )
    args = parser.parse_args()

    lines = [f"# {args.title}", ""]

    # 1-2. D と rank(C)
    lines += ["## 次元と制約", ""]
    for label, core in (("統合前", args.pre_core), ("統合後", args.post_core)):
        spec = load_canonical_spec(core)
        c = build_constraint_matrix(spec, include_subtotals=False)
        lines.append(f"- {label}: D = {len(spec.basis)}、rank(C) = {compute_rank(c)}")

    names = [b.name for b in load_canonical_spec(args.post_core).basis]
    groups = [n for n in names if not n.startswith("__OTHER__")]
    others = [n for n in names if n.startswith("__OTHER__")]

    # 2. 恒等式残差(verify_prior_stage21 の residuals.csv)
    lines += ["", "## 恒等式残差(0 でない社数 / 評価社数)", ""]
    lines += ["| 提出期間 | 残差 | 期 | 統合前 | 統合後 |", "|---|---|---|---|---|"]
    for s in _SETS:
        pre = pd.read_csv(args.pre_results / s / "residuals.csv")
        post = pd.read_csv(args.post_results / s / "residuals.csv")
        for (_, a), (_, b) in zip(pre.iterrows(), post.iterrows(), strict=True):
            lines.append(
                f"| {s} | {a['residual']} | {a['period']} | {a['n_nonzero']} / {a['n_evaluated']} "
                f"| {b['n_nonzero']} / {b['n_evaluated']} |"
            )

    # 3-5. __OTHER__ の規模、出現率、値が変わった社
    other_rows = []
    group_rows = []
    changed_rows = []
    for s in _SETS:
        for period in _PERIODS:
            pre = _load(args.pre, s, period)
            post = _load(args.post, s, period).reindex(pre.index)
            assets = pre["total_assets"].abs()
            for n in others:
                a = (pre[n].fillna(0).abs() / assets).median()
                b = (post[n].fillna(0).abs() / assets).median()
                other_rows.append((s, period, n, a, b))
            for n in groups:
                group_rows.append((s, period, n, pre[n].notna().mean(), post[n].notna().mean()))
            diff = (pre[names].fillna(0) != post[names].fillna(0)).any(axis=1)
            changed_rows.append((s, period, len(pre), int(diff.sum())))

    lines += [
        "",
        "## __OTHER__ 各科目の総資産比(|値| / 総資産 の中央値)",
        "",
        "| 提出期間 | 期 | 科目 | 統合前 | 統合後 | 変化 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for s, period, n, a, b in other_rows:
        lines.append(
            f"| {s} | {period} | {n.removeprefix(_OTHER)} | {a:.4f} | {b:.4f} | {b - a:+.4f} |"
        )
    lines += [
        "",
        "## グループ科目の出現率(値がある社の割合)",
        "",
        "| 提出期間 | 期 | 科目 | 統合前 | 統合後 |",
        "|---|---|---|---:|---:|",
    ]
    for s, period, n, a, b in group_rows:
        if not np.isclose(a, b):
            lines.append(f"| {s} | {period} | {n} | {a:.4f} | {b:.4f} |")
    lines.append("\n(統合前後で出現率が変わらない科目は省略)")
    lines += ["", "## 値が変わった社数", "", "| 提出期間 | 期 | 社数 | 変わった社 |"]
    lines.append("|---|---|---:|---:|")
    for s, period, n_docs, k in changed_rows:
        lines.append(f"| {s} | {period} | {n_docs} | {k}({k / n_docs:.1%}) |")

    # B.4 統合した要素の期ごとの出現社数(統合前の leaves.csv = すべて __OTHER__ に落ちていた)
    added = _added_members(args.pre_core, args.post_core)
    period_dirs = ["2023_2024_prior", "2023_2024_current", "fy2024_prior", "fy2024_current"]
    counts = {
        d: pd.read_csv(args.leaks / d / "leaves.csv", usecols=["docID", "element"])
        .groupby("element")["docID"]
        .nunique()
        for d in period_dirs
    }
    lines += [
        "",
        "## B.4 統合した要素の期ごとの出現社数(金融業を含む全書類)",
        "",
        "期: 2023_2024_prior ≈ 2022 年決算、2023_2024_current と fy2024_prior ≈ 2023 年決算、"
        "fy2024_current ≈ 2024 年決算。",
        "",
        "| グループ | 要素 | " + " | ".join(period_dirs) + " |",
        "|---|---|" + "---:|" * len(period_dirs),
    ]
    only_some = []
    for group, elements in added.items():
        for e in elements:
            row = [int(counts[d].get(e, 0)) for d in period_dirs]
            if 0 in row:
                only_some.append(e)
            lines.append(
                f"| {group} | {e.removeprefix('jppfs_cor_')} | " + " | ".join(map(str, row)) + " |"
            )
    lines += ["", f"出現しない期がある要素: {only_some or 'なし'}"]

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
