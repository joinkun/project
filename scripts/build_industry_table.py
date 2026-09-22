#!/usr/bin/env python
"""ステージ12 §1.2: EDINETコードリストから業種対応表 conf/accounts/industry.yaml を作る。

secCodeは銘柄コード+チェックディジットのみで、東証33業種を機械的に引けない
(ユーザーと相談のうえ、代替として金融庁公開のEDINETコードリストを採用)。
EDINETコードリスト(Edinetcode.zip)の「提出者業種」は東証33業種と同一の
体系で、非上場の提出者にも付与されている。

このスクリプトは業種マスタを作る「一度だけ」の取得ステップで、以降の実験は
リポジトリ内のyamlだけを読む(外部APIに依存しない)。

使い方:
    uv run python scripts/build_industry_table.py [--zip Edinetcode.zip]
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_OUTPUT = Path("conf/accounts/industry.yaml")

# 粗い分類(§1.2)。33業種→6区分。金融は本データセットでは除外済み(is_financial)
# だが、対応表としては完全性のため区分を持たせる。
_COARSE: dict[str, list[str]] = {
    "製造": [
        "食料品", "繊維製品", "パルプ・紙", "化学", "医薬品", "石油・石炭製品",
        "ゴム製品", "ガラス・土石製品", "鉄鋼", "非鉄金属", "金属製品", "機械",
        "電気機器", "輸送用機器", "精密機器", "その他製品",
    ],
    "小売・卸売": ["小売業", "卸売業"],
    "サービス": ["サービス業"],
    "不動産・建設": ["不動産業", "建設業"],
    "情報通信": ["情報・通信業"],
    "その他(非金融)": [
        "水産・農林業", "鉱業", "電気・ガス業", "陸運業", "海運業", "空運業", "倉庫・運輸関連",
    ],
    "金融": ["銀行業", "証券、商品先物取引業", "保険業", "その他金融業"],
    "不明": ["不明"],
}
_UNKNOWN = "不明"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=None, help="取得済みのEdinetcode.zip")
    args = parser.parse_args()

    if args.zip is not None:
        raw = args.zip.read_bytes()
    else:
        with urllib.request.urlopen(_URL, timeout=60) as resp:  # noqa: S310
            raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(csv_name).decode("cp932")
    lines = text.splitlines()
    header_line = lines[0]
    ed = pd.read_csv(io.StringIO("\n".join(lines[1:])), dtype=str)
    ed = ed.rename(columns={"ＥＤＩＮＥＴコード": "edinetCode", "提出者業種": "industry"})

    codes = pd.read_csv(_AGGREGATED_CSV, dtype=str)["edinetCode"].dropna().unique()
    table = ed[ed["edinetCode"].isin(codes)].set_index("edinetCode")["industry"]
    missing = sorted(set(codes) - set(table.index))

    coarse_of = {ind: c for c, inds in _COARSE.items() for ind in inds}
    # 「内国法人・組合(有価証券報告書等の提出義務者以外)」等、業種ではない値は「不明」とする。
    pseudo = sorted(set(table.unique()) - set(coarse_of))
    table = table.where(table.isin(coarse_of), _UNKNOWN)
    print(f"業種ではない値を「不明」に置換: {pseudo}({int((table == _UNKNOWN).sum())}社)")

    doc = {
        "source": "EDINETコードリスト(金融庁) 提出者業種。東証33業種と同一の体系。",
        "url": _URL,
        "retrieved": date.today().isoformat(),
        "codelist_header": header_line,
        "n_codes": len(table),
        "codes_without_industry": missing,
        "n_unknown": int((table == _UNKNOWN).sum()),
        "coarse_map": {ind: c for ind, c in coarse_of.items()},
        "edinet_industry": {str(k): str(v) for k, v in sorted(table.items())},
    }
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    print(f"書き出し: {_OUTPUT}({len(table)}社、業種不明={len(missing)}社)")


if __name__ == "__main__":
    main()
