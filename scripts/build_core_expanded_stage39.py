#!/usr/bin/env python
"""ステージ39 A: core.yaml から、有形固定資産と投資その他の資産を開いた canonical を作る。

規則(生成・推定の前に決めた):
- 開いた小計は基礎科目から外し、``subtotals`` に ``parent: jppfs_cor_NoncurrentAssets`` で加える
- 子のグループは、全書類(class 0・2、4,019)での出現率が 0.5 以上のものを基礎科目にする
  (``survey_expand_stage39.py``)。未満の子は残余 ``__OTHER__<小計>`` へ
- **純額で開く**: 建物・機械装置のグループには純額の要素と、総額・減価償却累計額の要素を
  両方入れる。集計は葉から祖先をたどって最初に当たったメンバーで止まるので、純額で報告する
  会社も総額 + 累計額で報告する会社も、グループの値は純額になる(累計額が負の科目にならない)
- 貸倒引当金(固定)の構成要素は投資その他の資産の子なので、親を投資その他の資産に移す
  (固定資産の残余で二重に引かれないため)

    uv run python scripts/build_core_expanded_stage39.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

_P = "jppfs_cor_"
PPE = f"{_P}PropertyPlantAndEquipment"
IOA = f"{_P}InvestmentsAndOtherAssets"
NCA = f"{_P}NoncurrentAssets"

# (名前, 日本語, 構成要素, 符号, 出現率(全書類)、class 2)
PPE_GROUPS = (
    ("land", "土地", ("Land",), "nonneg", 0.762, 0.426),
    ("buildings_net", "建物及び構築物(純額)", (
        "BuildingsAndStructuresNet", "BuildingsAndStructures",
        "AccumulatedDepreciationBuildingsAndStructures", "BuildingsNet", "Buildings",
        "AccumulatedDepreciationBuildings", "StructuresNet", "Structures",
        "AccumulatedDepreciationStructures"), "nonneg", 0.938, 0.794),
    ("machinery_net", "機械装置及び運搬具(純額)", (
        "MachineryEquipmentAndVehiclesNet", "MachineryEquipmentAndVehicles",
        "AccumulatedDepreciationMachineryEquipmentAndVehicles", "MachineryAndEquipmentNet",
        "MachineryAndEquipment", "AccumulatedDepreciationMachineryAndEquipment", "VehiclesNet",
        "Vehicles", "AccumulatedDepreciationVehicles"), "nonneg", 0.678, 0.559),
    ("construction_in_progress", "建設仮勘定", ("ConstructionInProgress",), "nonneg",
     0.564, 0.265),
)
IOA_GROUPS = (
    ("investment_securities", "投資有価証券", ("InvestmentSecurities",), "nonneg", 0.860, 0.691),
    ("deferred_tax_assets_nc", "繰延税金資産(固定)", ("DeferredTaxAssets",), "nonneg",
     0.851, 0.294),
)


def entry(name: str, members: tuple[str, ...], parent: str, sign: str, ratio: float | None,
          reason: str) -> dict[str, object]:
    return {"name": name, "members": [f"{_P}{m}" if not m.startswith(_P) else m for m in members],
            "parent_subtotal": parent, "side": "assets", "sign": sign,
            "sign_reason": reason, "ratio": ratio, "always_include": True,
            "include_reason": "ステージ39: 小計を開いた子(出現率 0.5 以上)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", type=Path, default=Path("conf/accounts/core.yaml"))
    ap.add_argument("--out", type=Path, default=Path("conf/accounts/core_expanded.yaml"))
    ap.add_argument("--labels", type=Path, default=Path("conf/accounts/labels_ja.yaml"))
    ap.add_argument("--labels-out", type=Path,
                    default=Path("conf/accounts/labels_ja_expanded.yaml"))
    args = ap.parse_args()
    doc = yaml.safe_load(args.core.read_text(encoding="utf-8"))
    new_basis = []
    for b in doc["basis"]:
        if b["name"] == PPE:
            for n, _ja, mem, sg, r, _r2 in PPE_GROUPS:
                new_basis.append(entry(n, mem, PPE, sg, r, "資産の構成要素(純額)"))
            new_basis.append(entry(f"__OTHER__{PPE}", (), PPE, "free", None,
                                   "残余科目(差額定義)"))
            continue
        if b["name"] == IOA:
            for n, _ja, mem, sg, r, _r2 in IOA_GROUPS:
                new_basis.append(entry(n, mem, IOA, sg, r, "資産の構成要素"))
            new_basis.append(entry(f"__OTHER__{IOA}", (), IOA, "free", None,
                                   "残余科目(差額定義)"))
            continue
        if b["name"] == "allowance_doubtful_noncurrent":
            b = {**b, "parent_subtotal": IOA}
        new_basis.append(b)
    doc["basis"] = new_basis
    subs = []
    for s in doc["subtotals"]:
        subs.append(s)
        if s["element_id"] == NCA:
            subs.append({"element_id": PPE, "side": "assets", "ratio": 0.988, "parent": NCA})
            subs.append({"element_id": IOA, "side": "assets", "ratio": 0.998, "parent": NCA})
    doc["subtotals"] = subs
    doc["name"] = "core_expanded"
    header = ("# ステージ39: scripts/build_core_expanded_stage39.py が core.yaml から生成"
              "(手で編集しない)。\n")
    args.out.write_text(header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False,
                                                width=1000), encoding="utf-8")
    lab = yaml.safe_load(args.labels.read_text(encoding="utf-8"))
    for n, ja, *_ in (*PPE_GROUPS, *IOA_GROUPS):
        lab["accounts"][n] = ja
    lab["accounts"][f"__OTHER__{PPE}"] = "有形固定資産その他"
    lab["accounts"][f"__OTHER__{IOA}"] = "投資その他の資産その他"
    args.labels_out.write_text(yaml.safe_dump(lab, allow_unicode=True, sort_keys=False),
                               encoding="utf-8")
    print(f"基礎科目 {len(new_basis)}、小計 {len(subs)} → {args.out}")


if __name__ == "__main__":
    main()
