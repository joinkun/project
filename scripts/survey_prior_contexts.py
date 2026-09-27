#!/usr/bin/env python
"""ステージ21-A B.2: 貸借対照表の科目に使われているコンテキスト ID を全社で集計する。

各書類について、calculation linkbase の BS ツリー(連結・単体の両方)に現れる要素の
ファクトを拾い、その contextRef の集合を記録する。コンテキスト ID ごとに
「その ID を BS の科目に 1 回以上使った社数」を数える。

前期(``Prior1YearInstant`` 系)の命名に揺れが無いかを確認するのが目的。
取得途中でも、存在する ZIP だけを対象に実行できる。

    uv run python scripts/survey_prior_contexts.py --suffix "" \\
        --out results/stage21/bs_contexts_fy2024.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from lxml import etree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from inspect_values import _load_calc_link_config  # noqa: E402

from fsgen.parsing.calculation_link import parse_calculation_link_bytes  # noqa: E402
from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl  # noqa: E402


def is_plain(context_id: str) -> bool:
    """連結/単体の区別以外のディメンション(SS の構成要素メンバー等)を持たないか。"""
    return "Member" not in context_id.replace("_NonConsolidatedMember", "")


def bs_contexts(zip_path: Path, role_map: dict[str, str], prefixes: tuple[str, ...]) -> set[str]:
    """BS ツリーの要素に付いたファクトの contextRef の集合。"""
    trees = parse_calculation_link_bytes(
        read_cal_xml(zip_path), role_map, prefixes, source_label=zip_path.stem
    )
    bs_elements = {e for tree in trees.values() for e in tree.elements}
    root = etree.fromstring(read_instance_xbrl(zip_path))
    uri_to_prefix = {uri: p for p, uri in root.nsmap.items() if p is not None}

    contexts: set[str] = set()
    for el in root.iter():
        ctx = el.get("contextRef")
        if ctx is None or not isinstance(el.tag, str):
            continue
        qname = etree.QName(el)
        prefix = uri_to_prefix.get(qname.namespace or "")
        if prefix is not None and f"{prefix}_{qname.localname}" in bs_elements:
            contexts.add(ctx)
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    role_map, prefixes = _load_calc_link_config()
    documents_dir = args.raw_dir / f"documents{args.suffix}"
    zips = sorted(documents_dir.glob("*.zip"))

    counts: Counter[str] = Counter()
    failed: list[str] = []
    for zip_path in tqdm(zips, desc="bs contexts"):
        try:
            counts.update(bs_contexts(zip_path, role_map, prefixes))
        except Exception as exc:  # noqa: BLE001 - 1社の失敗で全体を止めない
            failed.append(f"{zip_path.stem}: {type(exc).__name__}: {exc}")

    n_ok = len(zips) - len(failed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["context_id", "plain", "n_companies", "share"])
        for ctx, n in counts.most_common():
            writer.writerow([ctx, is_plain(ctx), n, f"{n / n_ok:.4f}" if n_ok else ""])

    print(f"対象 {len(zips)} 書類(解析成功 {n_ok} / 失敗 {len(failed)})")
    for line in failed[:10]:
        print(f"  失敗: {line}")
    print(f"\n'Prior' を含むコンテキスト(上位 {args.top})")
    prior = [(c, n) for c, n in counts.most_common() if "Prior" in c]
    for ctx, n in prior[: args.top]:
        print(f"  {n:>6}  {ctx}")
    print(f"  (異なる ID の総数: {len(prior)})")
    print("\nディメンション無し(連結/単体の区別のみ)のコンテキスト(全件)")
    for ctx, n in [(c, n) for c, n in counts.most_common() if is_plain(c)]:
        print(f"  {n:>6}  {ctx}")
    print("\n'Current' を含むコンテキスト(上位 10、比較用)")
    for ctx, n in [(c, n) for c, n in counts.most_common() if "Current" in c][:10]:
        print(f"  {n:>6}  {ctx}")


if __name__ == "__main__":
    main()
