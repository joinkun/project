"""会社独自の拡張要素を、局所名の完全一致で canonical のグループへ写像する(ステージ21-D)。

拡張要素の ID は ``jpcrp030000-asr_E01329-000_AccountsReceivableTradeAndContractAssets``
のように「様式の接頭辞 _ EDINET コード _ 局所名」からなる。局所名が標準要素
(``jppfs_cor_<局所名>``)の members と**完全一致**する場合に限り、同じグループへ写像する。
部分一致・類似度は使わない(恣意性が入るため)。

局所名が一致しても、cal ツリー上で最も近い小計が写像先グループの親小計と異なる場合は
別概念の可能性があるので写像しない(書類ごとに判定する)。タクソノミ改訂で標準要素が
新設される前に、各社が同じ局所名の拡張要素で表示していた例(契約資産を含む売掛金など)を
拾うのが目的。

ステージ22: 局所名の末尾から EDINET の接尾辞を 1 つ除去してから完全一致を判定する規則を
追加した(完全一致を先に試す)。除去する接尾辞は、金融庁『EDINET タクソノミの設定規約書』
(2023 年 12 月版、https://disclosure2dl.edinet-fsa.go.jp/guide/static/disclosure/download/ESE140124.pdf)
が要素名の接尾辞として定める**業種略号**(図表 1-4-6 語彙層の業種略号、図表 3-2-27
「name = {文字列(英数字)}{業種略号}」)に限る。CA・CL・NCL・IOA などの表示区分を表す略号は、
同規約書・『EDINET タクソノミの概要説明』・『フレームワーク設計書』のいずれにも一覧が無いため
含めない(推測で並べない)。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from fsgen.parsing.calculation_link import CalcTree

_EXTENSION_ID = re.compile(r"^jp[a-z]+\d{6}-[a-z0-9]+_[A-Z]\d{5}-\d{3}_(?P<local>[A-Za-z0-9]+)$")
_STANDARD_PREFIX = "jppfs_cor_"

# 『EDINET タクソノミの設定規約書』2023 年 12 月版 図表 1-4-6 語彙層の業種略号(21 個)。
EDINET_INDUSTRY_SUFFIXES: tuple[str, ...] = (
    "CTE",  # 一般商工業
    "CNS",  # 建設業
    "BNK",  # 銀行・信託業
    "CNA",  # 建設保証業
    "SEC",  # 第一種金融商品取引業
    "INS",  # 保険業
    "RWY",  # 鉄道事業
    "WAT",  # 海運事業
    "HWY",  # 高速道路事業
    "ELC",  # 電気通信事業
    "ELE",  # 電気事業
    "GAS",  # ガス事業
    "LIQ",  # 資産流動化業
    "IVT",  # 投資運用業
    "INV",  # 投資業
    "SPF",  # 特定金融業
    "MED",  # 社会医療法人
    "EDU",  # 学校法人
    "CMD",  # 商品先物取引業
    "LEA",  # リース事業
    "FND",  # 投資信託受益証券
)


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
    suffixes: tuple[str, ...] = ()  # 完全一致しないときに 1 つ除去してみる接尾辞

    @classmethod
    def from_members(
        cls,
        member_to_name: Mapping[str, str],
        group_parent: Mapping[str, str | None],
        suffixes: tuple[str, ...] = (),
    ) -> ExtensionRule:
        local_to_group = {
            m.removeprefix(_STANDARD_PREFIX): name
            for m, name in member_to_name.items()
            if m.startswith(_STANDARD_PREFIX)
        }
        return cls(local_to_group, dict(group_parent), suffixes)

    def match(self, element_id: str) -> tuple[str, str] | None:
        """(写像先の候補, 除去した接尾辞 "" = 完全一致)。cal 上の位置は見ない。"""
        local = extension_local_name(element_id)
        if local is None:
            return None
        if local in self.local_to_group:
            return self.local_to_group[local], ""
        for suffix in self.suffixes:
            base = local.removesuffix(suffix)
            if base != local and base in self.local_to_group:
                return self.local_to_group[base], suffix
        return None

    def candidate_group(self, element_id: str) -> str | None:
        """局所名だけで決まる写像先の候補(cal 上の位置は見ない)。"""
        m = self.match(element_id)
        return None if m is None else m[0]

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
