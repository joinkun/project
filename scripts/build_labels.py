#!/usr/bin/env python
"""GC注記・重要事象の実質判定から3クラスラベルを生成する(§E)。

判定は排他的に以下の順で行う(§E.3):

    class 2: GC注記あり(単体 or 連結、実質判定つき)
    class 1: class 2 でなく、重要事象あり(実質判定つき)
    class 0: それ以外

テキストブロック要素は生のHTMLマークアップ(タグ・style属性)をテキスト
ノードにエスケープして格納している(ステージ4.6で判明)。そのままでは
タグの分だけ文字数が水増しされるため、`lxml.html` の `text_content()` で
タグを除去し、空白を正規化してから長さ判定・否定表現判定を行う
(ステージ5 §A.2)。「実質的な記載があるか」は、除去後の文字列の長さが
50字を超え、かつ「該当事項はありません」等の否定表現を含まないことで
判定する(§E.2で妥当性を検証)。

is_financial は事前に `inspect_values.py --canonical conf/accounts/core.yaml`
で生成した `data/interim/values/aggregated_core.csv` から docID で結合する。

ZIP は展開しない。

使い方:
    uv run python scripts/build_labels.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from lxml import etree, html
from tqdm import tqdm

from fsgen.parsing.calculation_link import parse_calculation_link_bytes, select_tree
from fsgen.parsing.instance import resolve_element_tag
from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[1] / "conf" / "config.yaml"
_DEFAULT_DOCUMENT_LIST_CSV = Path("data/raw/index/document_list.csv")
_DEFAULT_DOCUMENTS_DIR = Path("data/raw/documents")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_OUT = Path("data/interim/labels/labels.csv")

_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_XSI_NIL = f"{{{_XSI_NS}}}nil"

_GC_CONSOLIDATED_ELEMENT = (
    "jpcrp_cor_NotesUncertaintiesOfEntitysAbilityToContinueAsGoingConcern"
    "ConsolidatedFinancialStatementsTextBlock"
)
_GC_NONCONSOLIDATED_ELEMENT = (
    "jpcrp_cor_NotesUncertaintiesOfEntitysAbilityToContinueAsGoingConcern"
    "FinancialStatementsTextBlock"
)
_MATERIAL_MATTERS_ELEMENT = (
    "jpcrp_cor_MaterialMattersRelatingToGoingConcernEtcBusinessRisksTextBlock"
)

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


def _load_calc_link_config() -> tuple[dict[str, str], tuple[str, ...]]:
    with _CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)["calculation_link"]
    return config["balance_sheet_roles"], tuple(config["standard_prefixes"])


def matched_negation_phrases(text: str) -> tuple[str, ...]:
    return tuple(p for p in _NEGATION_PHRASES if p in text)


def is_substantive(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) <= _MIN_SUBSTANTIVE_LENGTH:
        return False
    return not matched_negation_phrases(stripped)


_WHITESPACE_RE = re.compile(r"[\s　\xa0]+")


def strip_html(text: str) -> str:
    """HTMLタグを除去し、地の文だけを取り出す(ステージ5 §A.2)。

    テキストブロック要素は生のHTML(タグ・style属性込み)をエスケープして
    テキストノードに格納しているため、タグ込みの文字数では実質的な記載の
    長さを測れない。`lxml.html.fromstring` でパースして `text_content()`
    を取り、空白(全角スペース・nbsp含む)を1つの半角スペースに正規化する。
    タグを含まない(パースしても意味のある構造にならない)プレーンな
    テキストの場合は、そのまま空白正規化だけを行う。
    """
    if not text.strip():
        return ""
    try:
        fragment = html.fromstring(f"<div>{text}</div>")
        content = fragment.text_content()
    except Exception:  # noqa: BLE001 - パース失敗時は元のテキストで代用する
        content = text
    return _WHITESPACE_RE.sub(" ", content).strip()


def read_text(root: etree._Element, element_id: str) -> str:
    """element_id の全出現のうち最長のテキストを返す(無ければ空文字列)。"""
    tag = resolve_element_tag(root, element_id)
    if tag is None:
        return ""
    texts = [
        (el.text or "").strip()
        for el in root.iter(tag)
        if el.get(_XSI_NIL) != "true"
    ]
    return max(texts, key=len, default="")


def _label_from_flags(has_c: bool, has_n: bool, has_m: bool) -> int:
    if has_c or has_n:
        return 2
    if has_m:
        return 1
    return 0


@dataclass(frozen=True)
class LabelRow:
    doc_id: str
    edinet_code: str
    filer_name: str
    parse_status: str
    label: int | None  # HTML除去後(§A.2)の判定
    has_gc_note_consolidated: bool | None
    has_gc_note_nonconsolidated: bool | None
    has_material_matters: bool | None
    used_role: str | None
    text_consolidated: str  # HTML除去・空白正規化後
    text_nonconsolidated: str
    text_material_matters: str
    old_label: int | None  # 比較用: HTML除去前(生テキストの長さ)での判定
    raw_len_consolidated: int
    raw_len_nonconsolidated: int
    raw_len_material_matters: int


def _process_company_inner(
    doc_id: str,
    edinet_code: str,
    filer_name: str,
    documents_dir: Path,
    role_map: dict[str, str],
    standard_prefixes: tuple[str, ...],
) -> LabelRow:
    zip_path = documents_dir / f"{doc_id}.zip"
    cal_bytes = read_cal_xml(zip_path)
    trees = parse_calculation_link_bytes(
        cal_bytes, role_map, standard_prefixes, source_label=doc_id
    )
    _tree, used_role = select_tree(trees)

    xbrl_bytes = read_instance_xbrl(zip_path)
    root = etree.fromstring(xbrl_bytes)

    raw_c = read_text(root, _GC_CONSOLIDATED_ELEMENT)
    raw_n = read_text(root, _GC_NONCONSOLIDATED_ELEMENT)
    raw_m = read_text(root, _MATERIAL_MATTERS_ELEMENT)

    text_c = strip_html(raw_c)
    text_n = strip_html(raw_n)
    text_m = strip_html(raw_m)

    has_c = is_substantive(text_c)
    has_n = is_substantive(text_n)
    has_m = is_substantive(text_m)
    label = _label_from_flags(has_c, has_n, has_m)

    old_label = _label_from_flags(
        is_substantive(raw_c), is_substantive(raw_n), is_substantive(raw_m)
    )

    return LabelRow(
        doc_id=doc_id,
        edinet_code=edinet_code,
        filer_name=filer_name,
        parse_status="ok",
        label=label,
        has_gc_note_consolidated=has_c,
        has_gc_note_nonconsolidated=has_n,
        has_material_matters=has_m,
        used_role=used_role,
        text_consolidated=text_c,
        text_nonconsolidated=text_n,
        text_material_matters=text_m,
        old_label=old_label,
        raw_len_consolidated=len(raw_c),
        raw_len_nonconsolidated=len(raw_n),
        raw_len_material_matters=len(raw_m),
    )


def process_company(
    doc_id: str,
    edinet_code: str,
    filer_name: str,
    documents_dir: Path,
    role_map: dict[str, str],
    standard_prefixes: tuple[str, ...],
) -> LabelRow:
    try:
        return _process_company_inner(
            doc_id, edinet_code, filer_name, documents_dir, role_map, standard_prefixes
        )
    except Exception as exc:  # noqa: BLE001 - 1社の失敗で全体を止めない
        return LabelRow(
            doc_id=doc_id,
            edinet_code=edinet_code,
            filer_name=filer_name,
            parse_status=f"failed: {type(exc).__name__}: {exc}",
            label=None,
            has_gc_note_consolidated=None,
            has_gc_note_nonconsolidated=None,
            has_material_matters=None,
            used_role=None,
            text_consolidated="",
            text_nonconsolidated="",
            text_material_matters="",
            old_label=None,
            raw_len_consolidated=0,
            raw_len_nonconsolidated=0,
            raw_len_material_matters=0,
        )


def run(
    document_list_csv: Path, documents_dir: Path, show_progress: bool = True
) -> list[LabelRow]:
    role_map, standard_prefixes = _load_calc_link_config()
    with document_list_csv.open(newline="", encoding="utf-8") as f:
        index = {row["docID"]: row for row in csv.DictReader(f)}

    results: list[LabelRow] = []
    for doc_id, row in tqdm(index.items(), desc="build_labels", disable=not show_progress):
        results.append(
            process_company(
                doc_id,
                row.get("edinetCode", ""),
                row.get("filerName", ""),
                documents_dir,
                role_map,
                standard_prefixes,
            )
        )
    return results


def load_is_financial(aggregated_csv: Path) -> dict[str, str]:
    if not aggregated_csv.exists():
        log.warning("aggregated csv not found: %s (is_financial will be blank)", aggregated_csv)
        return {}
    with aggregated_csv.open(newline="", encoding="utf-8") as f:
        return {row["docID"]: row.get("is_financial", "") for row in csv.DictReader(f)}


def write_labels_csv(results: list[LabelRow], is_financial: dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "docID",
                "edinetCode",
                "filerName",
                "label",
                "has_gc_note_consolidated",
                "has_gc_note_nonconsolidated",
                "has_material_matters",
                "used_role",
                "is_financial",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.doc_id,
                    r.edinet_code,
                    r.filer_name,
                    r.label if r.label is not None else "",
                    r.has_gc_note_consolidated if r.has_gc_note_consolidated is not None else "",
                    r.has_gc_note_nonconsolidated
                    if r.has_gc_note_nonconsolidated is not None
                    else "",
                    r.has_material_matters if r.has_material_matters is not None else "",
                    r.used_role or "",
                    is_financial.get(r.doc_id, ""),
                ]
            )


def print_summary(results: list[LabelRow]) -> None:
    ok = [r for r in results if r.parse_status == "ok"]
    failed = [r for r in results if r.parse_status != "ok"]
    print(f"処理: {len(results)}社(成功 {len(ok)} / 失敗 {len(failed)})\n")

    # --- §A.2: HTMLタグ除去によるラベル変化 ---
    print("=== §A.2 HTMLタグ除去によるラベル変化 ===")
    changed = [r for r in ok if r.label != r.old_label]
    print(f"ラベルが変わった社数: {len(changed)}/{len(ok)}社")
    for r in changed:
        print(
            f"  {r.doc_id} {r.filer_name}: old_label={r.old_label} -> label={r.label} "
            f"(raw_len c={r.raw_len_consolidated} n={r.raw_len_nonconsolidated} "
            f"m={r.raw_len_material_matters}; "
            f"cleaned_len c={len(r.text_consolidated)} n={len(r.text_nonconsolidated)} "
            f"m={len(r.text_material_matters)})"
        )

    # --- E.1: 連結/単体GC注記の重なり ---
    print("=== E.1 連結GC注記×単体GC注記の重なり ===")
    counts = Counter((r.has_gc_note_consolidated, r.has_gc_note_nonconsolidated) for r in ok)
    c_yes_n_yes = counts.get((True, True), 0)
    c_yes_n_no = counts.get((True, False), 0)
    c_no_n_yes = counts.get((False, True), 0)
    c_no_n_no = counts.get((False, False), 0)
    print(f"{'':16}単体GC注記あり     なし")
    print(f"連結GC注記あり  {c_yes_n_yes:>10} {c_yes_n_no:>10}")
    print(f"なし            {c_no_n_yes:>10} {c_no_n_no:>10}")
    union = c_yes_n_yes + c_yes_n_no + c_no_n_yes
    print(f"和集合(class 2 の正例数): {union}社")

    # --- E.3: 3クラスの社数 ---
    print("\n=== E.3 3クラスの社数 ===")
    label_counts = Counter(r.label for r in ok)
    for label in (2, 1, 0):
        print(f"  class {label}: {label_counts.get(label, 0)}社")

    # --- E.2: 実質判定の妥当性検証 ---
    print("\n=== E.2 実質判定の妥当性検証 ===")
    all_lengths = [
        len(t)
        for r in ok
        for t in (r.text_consolidated, r.text_nonconsolidated)
        if t
    ]
    print(f"テキスト長の分布(要素が存在する{len(all_lengths)}件、連結・単体プール):")
    if all_lengths:
        deciles = statistics.quantiles(all_lengths, n=10)
        for i, d in enumerate(deciles, start=1):
            print(f"  {i*10:>3}%tile: {d:.0f}字")
        print(f"  最大: {max(all_lengths)}字")

    negation_counter: Counter[str] = Counter()
    n_negation_matched_companies = 0
    for r in ok:
        matched_any = False
        for t in (r.text_consolidated, r.text_nonconsolidated):
            if not t:
                continue
            phrases = matched_negation_phrases(t)
            if phrases:
                matched_any = True
                negation_counter.update(phrases)
        if matched_any:
            n_negation_matched_companies += 1
    print(f"\n否定表現にマッチした社数: {n_negation_matched_companies}社")
    print("マッチした文字列の内訳:")
    for phrase, count in negation_counter.most_common():
        print(f"  {phrase!r}: {count}件")

    # 境界サンプル: テキスト長50〜200字の社を5社(再現性のため固定シード)
    boundary_candidates = [
        (r.doc_id, r.filer_name, t)
        for r in ok
        for t in (r.text_consolidated, r.text_nonconsolidated)
        if 50 < len(t) <= 200
    ]
    print(f"\n境界サンプル(テキスト長50〜200字、該当{len(boundary_candidates)}件から5件抽出):")
    rng = random.Random(0)
    sample = rng.sample(boundary_candidates, k=min(5, len(boundary_candidates)))
    for doc_id, filer_name, text in sample:
        print(f"  [{doc_id} {filer_name}] (len={len(text)}, substantive={is_substantive(text)})")
        print(f"    {text[:200]!r}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-list-csv", type=Path, default=_DEFAULT_DOCUMENT_LIST_CSV)
    parser.add_argument("--documents-dir", type=Path, default=_DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    results = run(args.document_list_csv, args.documents_dir)
    is_financial = load_is_financial(args.aggregated_csv)
    write_labels_csv(results, is_financial, args.out)
    print_summary(results)


if __name__ == "__main__":
    main()
