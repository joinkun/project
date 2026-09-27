#!/usr/bin/env python
"""ステージ21-D A.3・A.4: 拡張要素の局所名マッピングを 4 期の全書類で検証する。

各書類の選択した BS ツリーの葉(値があるもの)について、``find_bucket`` と同じ順で
祖先を辿り、局所名が members と完全一致する拡張要素に当たったら、その要素・写像先・
cal 上で最も近い小計・親小計の一致を記録する。親小計が一致しない場合は写像されない。

出力:
- ``results/stage21/extension_mapping.csv``: 拡張要素 ID、局所名、写像先、cal 上の親
  (最も近い小計)、親の一致、期ごとの社数(非金融)
- ``results/stage21/extension_mapping.md``: 要約(写像した要素数・社数、親の不一致、
  写像されずに残る拡張要素の規模)
- ``data/interim/stage21/extension/<period>.csv``: 葉ごとの記録

    uv run python scripts/survey_extension_mapping_stage21.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import pandas as pd
from lxml import etree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from inspect_values import (  # noqa: E402
    _OTHER_PREFIX,
    _load_calc_link_config,
    find_bucket,
    load_canonical,
)
from survey_group_leaks_stage21 import keyword_group  # noqa: E402

from fsgen.parsing.calculation_link import parse_calculation_link_bytes, select_tree  # noqa: E402
from fsgen.parsing.extension import extension_local_name, nearest_subtotal  # noqa: E402
from fsgen.parsing.instance import Period, context_id_for_role, read_fact  # noqa: E402
from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl  # noqa: E402

# (期の名前, 提出期間の接尾辞, 期)
PERIODS: tuple[tuple[str, str, Period], ...] = (
    ("fy2024_current", "", "current"),
    ("fy2024_prior", "", "prior"),
    ("2023_2024_current", "_2023_2024", "current"),
    ("2023_2024_prior", "_2023_2024", "prior"),
)


def survey(raw_dir: Path, suffix: str, period: Period, canonical_path: Path, out: Path) -> None:
    role_map, prefixes = _load_calc_link_config()
    canonical = load_canonical(canonical_path)
    rule = canonical.extension_rule
    if rule is None:
        raise ValueError("canonical で extension_local_name_mapping が有効になっていない")
    member_to_name = canonical.member_to_name
    subtotal_ids = canonical.subtotal_ids
    assets_id = canonical.totals["assets"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "docID",
                "kind",  # candidate(局所名が一致) / unmapped_other(写像されず __OTHER__ に残る)
                "element",
                "local",
                "removed_suffix",  # "" = 完全一致、それ以外は除去した業種略号
                "group",
                "nearest_subtotal",
                "expected_parent",
                "parent_match",
                "bucket",
                "value",
                "assets",
            ]
        )
        for zip_path in tqdm(sorted((raw_dir / f"documents{suffix}").glob("*.zip"))):
            doc_id = zip_path.stem
            try:
                trees = parse_calculation_link_bytes(
                    read_cal_xml(zip_path), role_map, prefixes, source_label=doc_id
                )
                tree, used_role = select_tree(trees)
                root = etree.fromstring(read_instance_xbrl(zip_path))
                context_id = context_id_for_role(used_role, period)
                assets = read_fact(root, assets_id, context_id).value
                leaves = [(leaf, read_fact(root, leaf, context_id).value) for leaf in tree.leaves()]
            except Exception:  # noqa: BLE001 - 1社の失敗で全体を止めない
                continue
            if assets is None:
                continue
            for leaf, value in leaves:
                if not value:
                    continue
                bucket = find_bucket(leaf, tree, member_to_name, subtotal_ids, rule) or ""
                # find_bucket と同じ順で祖先を辿り、局所名が一致する最初の拡張要素を探す
                u = leaf
                while u not in member_to_name and u not in subtotal_ids:
                    matched = rule.match(u)
                    if matched is not None:
                        group, removed = matched
                        near = nearest_subtotal(u, tree, subtotal_ids)
                        expected = rule.group_parent.get(group)
                        writer.writerow(
                            [
                                doc_id,
                                "candidate",
                                u,
                                extension_local_name(u),
                                removed,
                                group,
                                near,
                                expected,
                                near == expected,
                                bucket,
                                value,
                                assets,
                            ]
                        )
                        if near == expected:
                            break
                    parents = tree.parents_of(u)
                    if not parents:
                        break
                    u = parents[0].parent
                if bucket.startswith(_OTHER_PREFIX) and extension_local_name(leaf) is not None:
                    writer.writerow(
                        [
                            doc_id,
                            "unmapped_other",
                            leaf,
                            extension_local_name(leaf),
                            "",
                            "",
                            "",
                            "",
                            "",
                            bucket,
                            value,
                            assets,
                        ]
                    )


def summarize(interim: Path, financial: set[str], out_csv: Path, out_md: Path) -> None:
    frames = []
    for name, _, _ in PERIODS:
        df = pd.read_csv(interim / f"{name}.csv", dtype={"docID": str})
        df["period"] = name
        frames.append(df[~df["docID"].isin(financial)])
    data = pd.concat(frames)
    data["parent_match"] = data["parent_match"].astype(str) == "True"
    cand = data[data["kind"] == "candidate"]
    table = (
        cand.groupby(
            [
                "element",
                "local",
                "removed_suffix",
                "group",
                "nearest_subtotal",
                "parent_match",
                "period",
            ],
            dropna=False,
        )["docID"]
        .nunique()
        .unstack("period", fill_value=0)
        .reset_index()
    )
    table.to_csv(out_csv, index=False)

    lines = ["# ステージ21-D A: 拡張要素の局所名マッピング(非金融)", ""]
    lines += [
        "| 期 | 局所名が一致した要素 | 社 | うち親小計が一致(写像) | 社 "
        "| 親小計が不一致(除外) | 社 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _, _ in PERIODS:
        c = cand[cand["period"] == name]
        ok = c[c["parent_match"]]
        ng = c[~c["parent_match"]]
        lines.append(
            f"| {name} | {c['element'].nunique()} | {c['docID'].nunique()} "
            f"| {ok['element'].nunique()} | {ok['docID'].nunique()} "
            f"| {ng['element'].nunique()} | {ng['docID'].nunique()} |"
        )
    ng = cand[~cand["parent_match"]]
    lines += ["", "## 親小計が一致しなかった例(写像から除外)", ""]
    if ng.empty:
        lines.append("なし")
    else:
        lines += ["| 局所名 | 写像先候補 | cal 上の最も近い小計 | 期待する親小計 | 件数 |"]
        lines.append("|---|---|---|---|---:|")
        for (local, group, near, exp), g in ng.groupby(
            ["local", "group", "nearest_subtotal", "expected_parent"], dropna=False
        ):
            lines.append(f"| {local} | {group} | {near} | {exp} | {len(g)} |")

    lines += ["", "## 写像先グループ別(fy2024 当期)", ""]
    cur = cand[(cand["period"] == "fy2024_current") & cand["parent_match"]]
    for group, g in cur.groupby("group"):
        rel = (g["value"].abs() / g["assets"].abs()).tolist()
        lines.append(
            f"- {group}: {g['element'].nunique()} 要素・{g['docID'].nunique()} 社、"
            f"総資産比の中央値 {statistics.median(rel):.4f}"
        )

    rest = data[(data["kind"] == "unmapped_other") & (data["period"] == "fy2024_current")].copy()
    rest["keyword"] = rest["element"].map(keyword_group)
    lines += ["", "## 写像されずに __OTHER__ に残る拡張要素(fy2024 当期)", ""]
    for label, sub in (
        ("すべて", rest),
        ("既存グループのキーワードに該当", rest[rest["keyword"] != ""]),
    ):
        per_doc = sub.groupby("docID").apply(
            lambda g: g["value"].abs().sum() / abs(g["assets"].iloc[0]), include_groups=False
        )
        lines.append(
            f"- {label}: {sub['element'].nunique()} 要素・{sub['docID'].nunique()} 社、"
            f"社ごとの合計 / 総資産 中央値 {per_doc.median():.4f}、p90 {per_doc.quantile(0.9):.4f}"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--canonical", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--interim", type=Path, default=Path("data/interim/stage21/extension"))
    parser.add_argument("--results", type=Path, default=Path("results/stage21"))
    parser.add_argument(
        "--values",
        type=Path,
        nargs="+",
        default=[
            Path("data/interim/stage21/fy2024/values_current.csv"),
            Path("data/interim/stage21/2023_2024/values_current.csv"),
        ],
        help="is_financial を引くための values_current.csv",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if not args.summarize_only:
        for name, suffix, period in PERIODS:
            survey(args.raw_dir, suffix, period, args.canonical, args.interim / f"{name}.csv")
    financial: set[str] = set()
    for v in args.values:
        df = pd.read_csv(v, usecols=["docID", "is_financial"], dtype={"docID": str})
        financial |= set(df.loc[df["is_financial"].astype(str) == "True", "docID"])
    summarize(
        args.interim,
        financial,
        args.results / "extension_mapping.csv",
        args.results / "extension_mapping.md",
    )


if __name__ == "__main__":
    main()
