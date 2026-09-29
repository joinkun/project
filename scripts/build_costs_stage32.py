#!/usr/bin/env python
"""ステージ32 A・B: 頻度ベース・持続性ベース・頻度 + 非対称性のコスト重みを作る(生成の前に実行)。

- 版 2 ``costs_frequency.yaml``: 財務困窮企業が実際に行った施策の頻度 f から、該当する科目の
  該当する向きに c = 1 / f。頻度は **原典で確認できた Asquith, Gertner & Scharfstein (1994) の
  NBER WP 3942(1991)の値だけ**を使う(Ofek 1993 は本文を入手できず使わない)。
  分母は困窮企業 102 社全体(AGS の Table 3 は再建策をとった 76 社を分母にしているが、本研究の
  GC 企業は再建策をとったかで選ばれていないため)。対応づけられない科目・向きは 1.0。
- 版 3 ``costs_persistence.yaml``: 科目別の 1 年ラグの順位相関 ρ(``persistence_stage32.py``)から
  c = (1 − ρ_現金) / (1 − ρ)。1 − ρ を平均回帰の速さとみなし、遅いほど動かしにくいとする。
  増減は対称。現金預金 = 1.0。
- 版 4 ``costs_frequency_asym.yaml``: 版 2 に、自前の ABJ 型回帰の比 r = β1 / (β1 + β2) を
  w+ = c / √r、w− = c × √r として掛ける(w− / w+ = r、幾何平均は c のまま)。
  β1 と β1 + β2 がともに 5% 水準で正の科目だけ使い、それ以外は r = 1。

c = −log f も参考に計算して ``results/stage32/weights.md`` に載せるが、生成には使わない
(f = 1 で 0 になり、現金預金 = 1.0 に正規化する基準点がないため)。

    uv run python scripts/build_costs_stage32.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

_P = "jppfs_cor_"
_CASH = f"{_P}CashAndDeposits"
N_DISTRESSED = 102  # AGS の困窮企業(利払い能力の基準を満たした junk bond 発行企業)
NO_DATA = "該当する施策の頻度データなし。等重み(1.0)"
_Z = 1.959963984540054


@dataclass(frozen=True)
class Action:
    label: str
    count: int
    accounts: tuple[str, ...]
    direction: str  # "decrease" / "increase"
    note: str

    @property
    def freq(self) -> float:
        return self.count / N_DISTRESSED


# AGS (1991) NBER WP 3942 で確認した値のみ(ページは PDF のページ番号)
ACTIONS = (
    Action(
        "主要な与信枠の縮小(銀行が与信枠を減らすか、残高の返済を求める)", 22,
        ("short_term_loans", "long_term_loans"), "decrease",
        "AGS WP Table 3(p.32)「Reduction in main credit facility」22 社(76 社中 0.29)。"
        "本文(p.12)では 21 件。私的債務の再建 59 社(78%)の大半は条項の免除(37)・期限延長(28)で、"
        "残高を減らさない。元本の免除はほぼない(本文 p.3。例外は Zapata の 1 件、p.11)",
    ),
    Action(
        "公募債の交換の成功", 34, ("bonds_current", "bonds_noncurrent"), "decrease",
        "AGS WP Table 3(p.32)「Successful Exchange Offer」34 社(76 社中 0.45)。交換は現金・証券との"
        "交換で元本の削減か満期の延長をもたらす(本文 p.14)",
    ),
    Action(
        "総資産の 20% 超の資産売却", 21,
        (f"{_P}PropertyPlantAndEquipment", f"{_P}InvestmentsAndOtherAssets"), "decrease",
        "AGS WP 本文(p.17)「Twenty-one companies sell more than 20% of their assets」"
        "(102 社中)。売却は現金・非現金の対価の合計 / 前年度の総資産で測る。売却した資産の科目は"
        "区別されていない",
    ),
)
NOT_MAPPED = (
    ("設備投資の削減 83%(AGS WP p.4)", "フロー(設備投資)であり、有形固定資産の残高の増減は"
     "減価償却にも依存するので、残高のコストに写像しない"),
    ("銀行債務のリストラ 78%(AGS WP Table 3)", "条項の免除・期限延長が大半で残高を変えない。"
     "残高を減らす「与信枠の縮小」22 社だけを使う"),
    ("Chapter 11 の申立 42%(AGS WP p.23)", "BS の科目ではない"),
    ("Ofek (1993) の値(減配 47%、レイオフ 28%、資産リストラ 23%・現金流入 15%、債務リストラ 11%、"
     "破産 1%)", "本文を入手できず確認できないので使わない"),
)


def load_asym(path: Path) -> dict[str, tuple[float, str]]:
    t = pd.read_csv(path).set_index("account")
    out: dict[str, tuple[float, str]] = {}
    for acc, r in t.iterrows():
        up, down = r["asym_beta_up"], r["asym_beta_down"]
        su, sd = r["asym_se_up"], r["asym_se_down"]
        sig = (pd.notna(up) and pd.notna(down) and up / su > _Z and down / sd > _Z)
        if sig:
            ratio = float(up / down)
            out[str(acc)] = (ratio, f"自前の ABJ 型回帰(β1 = {up:.3f}、β1+β2 = {down:.3f}、"
                                    f"r = {ratio:.2f})")
        else:
            out[str(acc)] = (1.0, "非対称性を推定できない(β1 か β1+β2 が 5% 水準で正でない)。r = 1")
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persistence", type=Path, default=Path("results/stage32/persistence.csv"))
    ap.add_argument("--conf", type=Path, default=Path("conf/accounts"))
    ap.add_argument("--out", type=Path, default=Path("results/stage32/weights.md"))
    args = ap.parse_args()
    spec = load_canonical_spec(args.conf / "core.yaml")
    names = state_vector_names(spec, include_subtotals=False)
    base = yaml.safe_load((args.conf / "costs_equal.yaml").read_text(encoding="utf-8"))
    non_actionable = base["non_actionable"]
    pers = pd.read_csv(args.persistence).set_index("account")
    asym = load_asym(args.persistence)

    freq: dict[str, dict[str, tuple[float, str]]] = {n: {} for n in names}
    for a in ACTIONS:
        for acc in a.accounts:
            freq[acc][a.direction] = (a.freq, f"{a.label}: f = {a.count}/{N_DISTRESSED} = "
                                              f"{a.freq:.3f}。{a.note}")

    def cost_freq(n: str, direction: str, kind: str) -> tuple[float, str]:
        if direction not in freq[n]:
            return 1.0, NO_DATA
        f, src = freq[n][direction]
        return (1.0 / f if kind == "inv" else -math.log(f)), src

    rho_cash = float(pers.loc[_CASH, "rho1"])
    versions: dict[str, list[dict[str, object]]] = {"frequency": [], "persistence": [],
                                                    "frequency_asym": []}
    for n in names:
        wp, sp = cost_freq(n, "increase", "inv")
        wm, sm_ = cost_freq(n, "decrease", "inv")
        versions["frequency"].append({"name": n, "cost_increase": round(wp, 4),
                                      "cost_decrease": round(wm, 4),
                                      "source": f"w+: {sp} / w-: {sm_}"})
        rho = float(pers.loc[n, "rho1"])
        c = (1 - rho_cash) / (1 - rho)
        versions["persistence"].append({
            "name": n, "cost_increase": round(c, 4), "cost_decrease": round(c, 4),
            "source": f"持続性 ρ1 = {rho:.3f}(ρ現金 = {rho_cash:.3f})、c = (1−ρ現金)/(1−ρ)"})
        r, rsrc = asym[n]
        versions["frequency_asym"].append({
            "name": n, "cost_increase": round(wp / math.sqrt(r), 4),
            "cost_decrease": round(wm * math.sqrt(r), 4),
            "source": f"頻度(w+: {sp} / w-: {sm_})× 非対称性({rsrc})"})
    for v, rows in versions.items():
        doc = {"name": f"costs_{v}", "basis_costs": rows, "non_actionable": non_actionable}
        text = ("# ステージ32: scripts/build_costs_stage32.py が生成(手で編集しない)。\n"
                + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=1000))
        (args.conf / f"costs_{v}.yaml").write_text(text, encoding="utf-8")

    ja = yaml.safe_load(Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))[
        "accounts"]
    lines = ["# ステージ32 A: コスト重み(生成の前に作成)", "",
             "## A.2・A.3 頻度ベース(版 2)の対応づけ", "",
             "| 施策(AGS WP で確認) | 社数 / 102 | f | 科目 | 向き | c = 1/f | 参考 −log f |",
             "|---|---:|---:|---|---|---:|---:|"]
    for a in ACTIONS:
        lines.append(f"| {a.label} | {a.count} | {a.freq:.3f} | "
                     + "、".join(ja.get(x, x) for x in a.accounts)
                     + f" | 減少 | {1 / a.freq:.2f} | {-math.log(a.freq):.2f} |")
    n_mapped = len({acc for a in ACTIONS for acc in a.accounts})
    lines += ["", f"**対応づけた科目: {n_mapped} 個(減少の向きのみ)。対応づけられない科目: "
              f"{len(names) - n_mapped} 個**(32 個中。対応づけた科目の増加の向きも等重み)。", "",
              "使わなかった数値:", ""]
    lines += [f"- {k}: {v}" for k, v in NOT_MAPPED]
    lines += ["", "## 3 版の重み(w+ / w−)", "",
              "| # | 科目 | 版 2 頻度 | 版 3 持続性 | 版 4 頻度 + 非対称 | 版 4 の r |",
              "|---:|---|---|---|---|---:|"]
    for i, n in enumerate(names):
        f2 = versions["frequency"][i]
        f3 = versions["persistence"][i]
        f4 = versions["frequency_asym"][i]
        lines.append(
            f"| {i + 1} | {ja.get(n, n)} "
            f"| {f2['cost_increase']:.2f} / {f2['cost_decrease']:.2f} | {f3['cost_increase']:.2f} "
            f"| {f4['cost_increase']:.2f} / {f4['cost_decrease']:.2f} "
            f"| {asym[n][0]:.2f} |"
        )
    lines += ["", "非操作科目(利益剰余金・評価差額金・非支配株主持分)は、輸送では重み 0"
              "(`nearest_na0`)で、どの版でも同じ。"]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
