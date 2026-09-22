"""zip_reader.py のテスト。合成ZIPを作って検証する(実データ非依存)。"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl


def _make_zip(tmp_path: Path, name: str, entries: dict[str, str]) -> Path:
    zip_path = tmp_path / name
    with zipfile.ZipFile(zip_path, "w") as zf:
        for entry_name, content in entries.items():
            zf.writestr(entry_name, content)
    return zip_path


def test_read_cal_xml_returns_bytes(tmp_path: Path) -> None:
    zip_path = _make_zip(
        tmp_path,
        "S1000001.zip",
        {
            "XBRL/PublicDoc/jpcrp030000-asr-001_E00001-000_2024-03-31_01_2024-06-25_cal.xml": (
                "<link:linkbase/>"
            ),
            "XBRL/PublicDoc/jpcrp030000-asr-001_E00001-000_2024-03-31_01_2024-06-25.xbrl": (
                "<xbrl/>"
            ),
        },
    )
    content = read_cal_xml(zip_path)
    assert content == b"<link:linkbase/>"


def test_read_cal_xml_raises_file_not_found_when_absent(tmp_path: Path) -> None:
    zip_path = _make_zip(tmp_path, "S1000001.zip", {"XBRL/PublicDoc/readme.txt": "no cal here"})
    with pytest.raises(FileNotFoundError):
        read_cal_xml(zip_path)


def test_read_cal_xml_raises_value_error_when_multiple(tmp_path: Path) -> None:
    zip_path = _make_zip(
        tmp_path,
        "S1000001.zip",
        {
            "XBRL/PublicDoc/a_cal.xml": "<a/>",
            "XBRL/PublicDoc/b_cal.xml": "<b/>",
        },
    )
    with pytest.raises(ValueError, match="expected exactly one"):
        read_cal_xml(zip_path)


def test_read_cal_xml_ignores_files_outside_public_doc(tmp_path: Path) -> None:
    zip_path = _make_zip(
        tmp_path,
        "S1000001.zip",
        {
            "XBRL/AuditDoc/decoy_cal.xml": "<decoy/>",
            "XBRL/PublicDoc/real_cal.xml": "<real/>",
        },
    )
    assert read_cal_xml(zip_path) == b"<real/>"


def test_read_instance_xbrl_returns_bytes(tmp_path: Path) -> None:
    zip_path = _make_zip(
        tmp_path,
        "S1000001.zip",
        {
            "XBRL/PublicDoc/jpcrp030000-asr-001_E00001-000_2024-03-31_01_2024-06-25_cal.xml": (
                "<link:linkbase/>"
            ),
            "XBRL/PublicDoc/jpcrp030000-asr-001_E00001-000_2024-03-31_01_2024-06-25.xbrl": (
                "<xbrl>instance</xbrl>"
            ),
        },
    )
    content = read_instance_xbrl(zip_path)
    assert content == b"<xbrl>instance</xbrl>"


def test_read_instance_xbrl_raises_file_not_found_when_absent(tmp_path: Path) -> None:
    zip_path = _make_zip(tmp_path, "S1000001.zip", {"XBRL/PublicDoc/readme.txt": "nothing"})
    with pytest.raises(FileNotFoundError):
        read_instance_xbrl(zip_path)
