#!/usr/bin/env python
"""全社のインスタンスから GoingConcern を含む要素名を横断的に集計する(§B.4)。

継続企業の前提に関する注記(GC注記)の要素IDを推測せず実データから確定する
ための調査スクリプト。全社を横断すれば GC注記ありの会社が必ず含まれるため、
要素の局所名(local name)に "GoingConcern" を含むものを機械的に収集すれば
実際に使われている要素IDが判明する。

要素が存在するだけでは「実質的な記載があるか」までは分からないため、
テキスト長が一定以上かつ否定表現(「該当事項はありません」等)を含まない
場合のみ「実質的な記載あり」としてカウントする。

ZIP は展開しない。

使い方:
    uv run python scripts/survey_gc_notes.py
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

from lxml import etree
from tqdm import tqdm

from fsgen.parsing.zip_reader import read_instance_xbrl

log = logging.getLogger(__name__)

_DEFAULT_DOCUMENT_LIST_CSV = Path("data/raw/index/document_list.csv")
_DEFAULT_DOCUMENTS_DIR = Path("data/raw/documents")
_DEFAULT_OUT = Path("data/interim/survey/gc_note_elements.csv")

_SEARCH_TERM = "GoingConcern"
_MIN_SUBSTANTIVE_LENGTH = 50
_NEGATION_PHRASES = (
    "該当事項はありません",
    "該当事項なし",
    "該当なし",
    "該当する事項はありません",
    "該当するものはありません",
    "特に記載すべき事項はありません",
    "重要事象等は存在しません",
    "重要事象等はありません",
    "生じておりません",
)


def _is_substantive(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) <= _MIN_SUBSTANTIVE_LENGTH:
        return False
    return not any(phrase in stripped for phrase in _NEGATION_PHRASES)


def scan_company(zip_path: Path) -> dict[str, bool]:
    """{element_id: このインスタンス内で実質的な記載があるか} を返す。

    解析に失敗した場合は例外を送出する(1社の失敗で全体を止めないよう、
    呼び出し側で捕捉する)。
    """
    xbrl_bytes = read_instance_xbrl(zip_path)
    root = etree.fromstring(xbrl_bytes)

    ns_by_uri: dict[str, str] = {}
    for prefix, uri in root.nsmap.items():
        if prefix is None:
            continue
        ns_by_uri.setdefault(uri, prefix)

    found: dict[str, bool] = {}
    for el in root.iter():
        tag = el.tag
        if not isinstance(tag, str) or "}" not in tag:
            continue
        uri, local_name = tag[1:].split("}", 1)
        if _SEARCH_TERM not in local_name:
            continue
        prefix = ns_by_uri.get(uri)
        if prefix is None:
            continue
        element_id = f"{prefix}_{local_name}"
        substantive = _is_substantive(el.text or "")
        found[element_id] = found.get(element_id, False) or substantive
    return found


def run(
    document_list_csv: Path, documents_dir: Path, show_progress: bool = True
) -> tuple[int, int, Counter[str], Counter[str]]:
    with document_list_csv.open(newline="", encoding="utf-8") as f:
        index = {row["docID"]: row for row in csv.DictReader(f)}

    n_ok = 0
    n_failed = 0
    presence: Counter[str] = Counter()
    substantive: Counter[str] = Counter()

    for doc_id in tqdm(index, desc="survey_gc_notes", disable=not show_progress):
        zip_path = documents_dir / f"{doc_id}.zip"
        try:
            found = scan_company(zip_path)
        except Exception as exc:  # noqa: BLE001 - 1社の失敗で全体を止めない
            log.warning("failed to scan %s: %s: %s", doc_id, type(exc).__name__, exc)
            n_failed += 1
            continue
        n_ok += 1
        for element_id, is_substantive in found.items():
            presence[element_id] += 1
            if is_substantive:
                substantive[element_id] += 1

    return n_ok, n_failed, presence, substantive


def write_csv(
    n_ok: int, presence: Counter[str], substantive: Counter[str], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["element_id", "n_companies", "ratio", "n_substantive", "substantive_ratio"]
        )
        for element_id, n_companies in sorted(presence.items(), key=lambda kv: -kv[1]):
            n_sub = substantive.get(element_id, 0)
            writer.writerow(
                [
                    element_id,
                    n_companies,
                    f"{n_companies / n_ok:.6f}" if n_ok else "",
                    n_sub,
                    f"{n_sub / n_companies:.6f}" if n_companies else "",
                ]
            )


def print_summary(
    n_ok: int, n_failed: int, presence: Counter[str], substantive: Counter[str]
) -> None:
    print(f"処理: {n_ok + n_failed}社(成功 {n_ok} / 失敗 {n_failed})\n")
    print(
        f"{'element_id':<70} {'n_companies':>11} {'ratio':>7} "
        f"{'n_substantive':>13} {'sub_ratio':>9}"
    )
    for element_id, n_companies in sorted(presence.items(), key=lambda kv: -kv[1]):
        n_sub = substantive.get(element_id, 0)
        ratio = n_companies / n_ok if n_ok else 0.0
        sub_ratio = n_sub / n_companies if n_companies else 0.0
        print(f"{element_id:<70} {n_companies:>11} {ratio:>7.4f} {n_sub:>13} {sub_ratio:>9.4f}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-list-csv", type=Path, default=_DEFAULT_DOCUMENT_LIST_CSV)
    parser.add_argument("--documents-dir", type=Path, default=_DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    n_ok, n_failed, presence, substantive = run(args.document_list_csv, args.documents_dir)
    write_csv(n_ok, presence, substantive, args.out)
    print_summary(n_ok, n_failed, presence, substantive)


if __name__ == "__main__":
    main()
