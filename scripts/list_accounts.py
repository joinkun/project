#!/usr/bin/env python
"""canonical の基礎科目・小計・総合計の一覧を Markdown で出す(ステージ32 G、中間報告用)。

小計の構成は ``subtotal_members``(各基礎科目の係数)から求める。符号・側・親小計は
``conf/accounts/core.yaml``、日本語名は ``conf/accounts/labels_ja.yaml`` による。

    uv run python scripts/list_accounts.py > results/stage32/accounts.md
"""

from __future__ import annotations

from pathlib import Path

import yaml

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

_CORE = Path("conf/accounts/core.yaml")
_LABELS = Path("conf/accounts/labels_ja.yaml")
_SIGN = {"nonneg": "≥ 0", "nonpos": "≤ 0", "free": "自由"}
_SIDE = {"assets": "資産", "liabilities": "負債", "net_assets": "純資産"}


def main() -> None:
    d = yaml.safe_load(_CORE.read_text(encoding="utf-8"))
    ja: dict[str, str] = yaml.safe_load(_LABELS.read_text(encoding="utf-8"))["accounts"]
    spec = load_canonical_spec(_CORE)
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    basis = d["basis"]
    if [b["name"] for b in basis] != list(names):
        raise ValueError("core.yaml の basis の順序が状態ベクトルと一致しない")

    out = [
        "# canonical の科目一覧(`conf/accounts/core.yaml`)", "",
        f"基礎科目 {len(basis)} 個(状態ベクトルの順序)、小計 {len(d['subtotals'])} 個、"
        f"総合計 {len(d['totals'])} 個。`__OTHER__*` は親小計から他の基礎科目を差し引いた残余"
        "(差額定義)で、恒等式が構成的に成り立つ。", "",
        "## 基礎科目", "",
        "| # | 基礎科目 | 日本語 | 親小計 | 側 | 符号 | XBRL 要素数 |",
        "|---:|---|---|---|---|---|---:|",
    ]
    for i, b in enumerate(basis, 1):
        parent = b.get("parent_subtotal", "")
        out.append(
            f"| {i} | `{b['name']}` | {ja.get(b['name'], '')} | {ja.get(parent, parent)} "
            f"| {_SIDE.get(b.get('side', ''), b.get('side'))} | {_SIGN.get(b.get('sign', ''), '')} "
            f"| {len(b.get('members') or [])} |"
        )
    out += ["", "## 小計(基礎科目の和。係数 −1 はない)", "",
            "| # | 小計 | 日本語 | 側 | 構成する基礎科目(#) |", "|---:|---|---|---|---|"]
    for i, s in enumerate(d["subtotals"], 1):
        sid = s["element_id"]
        vec = members[sid]
        idx = [str(k + 1) for k, v in enumerate(vec) if v != 0]
        coef = {float(v) for v in vec if v != 0}
        note = "" if coef == {1.0} else f"(係数 {sorted(coef)})"
        out.append(f"| {i} | `{sid}` | {ja.get(sid, '')} | {_SIDE.get(s['side'], s['side'])} "
                   f"| {', '.join(idx)}{note} |")
    out += ["", "## 総合計", "", "| 区分 | 要素 | 日本語 |", "|---|---|---|"]
    for k, v in d["totals"].items():
        out.append(f"| {k} | `{v}` | {ja.get(v, '')} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
