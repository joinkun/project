"""EDINET の XBRL 一式 ZIP から、展開せずに個別ファイルを bytes で取り出す。

ZIP 内のパス区切りは常に ``/`` (ZIP仕様上そう決まっている)。
``os.path`` は使わず文字列演算で扱うこと。
"""

from __future__ import annotations

import fnmatch
import zipfile
from pathlib import Path

_PUBLIC_DOC_PREFIX = "XBRL/PublicDoc/"


def _find_one(zip_path: Path, names: list[str], pattern: str, label: str) -> str:
    matches = [
        n
        for n in names
        if n.startswith(_PUBLIC_DOC_PREFIX)
        and fnmatch.fnmatchcase(n[len(_PUBLIC_DOC_PREFIX) :], pattern)
    ]
    if not matches:
        raise FileNotFoundError(f"no {label} found in {zip_path} (pattern={pattern!r})")
    if len(matches) > 1:
        raise ValueError(
            f"expected exactly one {label} in {zip_path}, found {len(matches)}: {matches}"
        )
    return matches[0]


def read_cal_xml(zip_path: Path) -> bytes:
    """ZIP内の ``XBRL/PublicDoc/*_cal.xml`` を bytes で返す。"""
    with zipfile.ZipFile(zip_path) as zf:
        name = _find_one(zip_path, zf.namelist(), "*_cal.xml", "*_cal.xml")
        return zf.read(name)


def read_instance_xbrl(zip_path: Path) -> bytes:
    """ZIP内の ``XBRL/PublicDoc/jpcrp*-asr-*.xbrl`` を bytes で返す(後段で使用)。"""
    with zipfile.ZipFile(zip_path) as zf:
        name = _find_one(zip_path, zf.namelist(), "jpcrp*-asr-*.xbrl", "instance xbrl")
        return zf.read(name)
