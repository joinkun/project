"""会社独自の拡張要素を、局所名の完全一致で canonical のグループへ写像する(ステージ21-D)。

拡張要素の ID は ``jpcrp030000-asr_E01329-000_AccountsReceivableTradeAndContractAssets``
のように「様式の接頭辞 _ EDINET コード _ 局所名」からなる。局所名が標準要素
(``jppfs_cor_<局所名>``)の members と**完全一致**する場合に限り、同じグループへ写像する。
部分一致・類似度は使わない(恣意性が入るため)。

局所名が一致しても、cal ツリー上で最も近い小計が写像先グループの親小計と異なる場合は
別概念の可能性があるので写像しない(書類ごとに判定する)。タクソノミ改訂で標準要素が
新設される前に、各社が同じ局所名の拡張要素で表示していた例(契約資産を含む売掛金など)を
拾うのが目的。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from fsgen.parsing.calculation_link import CalcTree

_EXTENSION_ID = re.compile(r"^jp[a-z]+\d{6}-[a-z0-9]+_[A-Z]\d{5}-\d{3}_(?P<local>[A-Za-z0-9]+)$")
_STANDARD_PREFIX = "jppfs_cor_"


def extension_local_name(element_id: str) -> str | None:
    """拡張要素なら局所名を返す。標準要素や書式が違うものは None。"""
    m = _EXTENSION_ID.match(element_id)
    return m.group("local") if m else None


def nearest_subtotal(element_id: str, tree: CalcTree, subtotal_ids: frozenset[str]) -> str | None:
    """cal ツリーを親へ辿り、最初に当たる小計(自身は含まない)。無ければ None。"""
    u = element_id
    seen = {u}
    while True:
        parents = tree.parents_of(u)
        if not parents:
            return None
        u = parents[0].parent
        if u in subtotal_ids:
            return u
        if u in seen:  # 循環(不正な linkbase)への防御
            return None
        seen.add(u)


@dataclass(frozen=True)
class ExtensionRule:
    """局所名 -> グループ名、グループ名 -> 親小計。"""

    local_to_group: Mapping[str, str]
    group_parent: Mapping[str, str | None]

    @classmethod
    def from_members(
        cls, member_to_name: Mapping[str, str], group_parent: Mapping[str, str | None]
    ) -> ExtensionRule:
        local_to_group = {
            m.removeprefix(_STANDARD_PREFIX): name
            for m, name in member_to_name.items()
            if m.startswith(_STANDARD_PREFIX)
        }
        return cls(local_to_group, dict(group_parent))

    def candidate_group(self, element_id: str) -> str | None:
        """局所名だけで決まる写像先の候補(cal 上の位置は見ない)。"""
        local = extension_local_name(element_id)
        return None if local is None else self.local_to_group.get(local)

    def resolve(
        self, element_id: str, tree: CalcTree, subtotal_ids: frozenset[str]
    ) -> str | None:
        """写像先のグループ。局所名が一致し、かつ最も近い小計が親小計と一致するときだけ。"""
        group = self.candidate_group(element_id)
        if group is None:
            return None
        if nearest_subtotal(element_id, tree, subtotal_ids) != self.group_parent.get(group):
            return None
        return group
