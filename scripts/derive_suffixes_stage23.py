#!/usr/bin/env python
"""ステージ23 E: 標準タクソノミ(jppfs_cor)の要素名から接尾辞の一覧を機械的に導出する。

設定規約書には表示区分(CA・CL 等)の接尾辞の一覧が無い(ステージ22)。そこで、
jppfs_cor のスキーマに定義された全要素名から、末尾に連続する大文字(2 文字以上)を取り出し、
出現する要素数が ``--min-count``(既定 50)以上のものだけを接尾辞として採用する。
業種略号(設定規約書 図表 1-4-6)に該当するかも併記する。

スキーマは EDINET から取得して ``data/raw/taxonomy/`` に置く:
    http://disclosure.edinet-fsa.go.jp/taxonomy/jppfs/2023-12-01/jppfs_cor_2023-12-01.xsd

    uv run python scripts/derive_suffixes_stage23.py
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from lxml import etree

from fsgen.parsing.extension import EDINET_INDUSTRY_SUFFIXES

_XSD = "{http://www.w3.org/2001/XMLSchema}"
_TRAILING_UPPER = re.compile(r"(?<=[a-z0-9])([A-Z]{2,})$")
_SOURCE = "http://disclosure.edinet-fsa.go.jp/taxonomy/jppfs/2023-12-01/jppfs_cor_2023-12-01.xsd"


def element_names(xsd: Path) -> list[str]:
    root = etree.parse(str(xsd)).getroot()
    return [el.get("name") or "" for el in root.iter(f"{_XSD}element") if el.get("name")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xsd", type=Path, default=Path("data/raw/taxonomy/jppfs_cor_2023-12-01.xsd")
    )
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/stage23/suffixes.md"))
    args = parser.parse_args()

    names = element_names(args.xsd)
    counts: Counter[str] = Counter()
    for name in names:
        m = _TRAILING_UPPER.search(name)
        if m:
            counts[m.group(1)] += 1
    adopted = [(s, n) for s, n in counts.most_common() if n >= args.min_count]
    rejected = [(s, n) for s, n in counts.most_common() if n < args.min_count]
    lines = [
        "# jppfs_cor の要素名から導出した接尾辞(ステージ23 E)",
        "",
        f"出典: {_SOURCE}(要素 {len(names)} 個)。",
        "規則: 要素名の末尾に連続する大文字(2 文字以上、直前が小文字か数字)を取り出し、"
        f"{args.min_count} 要素以上に現れるものを採用する。",
        "",
        "| 接尾辞 | 要素数 | 業種略号(規約書 図表 1-4-6) |",
        "|---|---:|---|",
    ]
    for suffix, n in adopted:
        industry = "○" if suffix in EDINET_INDUSTRY_SUFFIXES else ""
        lines.append(f"| {suffix} | {n} | {industry} |")
    lines += [
        "",
        f"採用 {len(adopted)} 個。{args.min_count} 要素未満で採用しなかったもの(上位 15): "
        + "、".join(f"{s}({n})" for s, n in rejected[:15]),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
