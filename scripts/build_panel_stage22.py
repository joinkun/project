#!/usr/bin/env python
"""ステージ22 B: 調整速度の推定に使うパネルを確定する。

観測の単位と差分の取り方はステージ21-D と同じ(同一書類の当期 - 前期、t = 当期末の年)。

企業単位の除外(この順に数える):
1. is_financial(残余資産比率 > 0.5、いずれかの書類)
2. 業種が金融(EDINET コードリストの提出者業種: 銀行業・証券・保険業・その他金融業)
3. 解析失敗・対象年の書類なし
4. 前期の BS が存在しない(連結初年度など)
5. 決算期変更(書類内の期末の差が 12 か月でない、または書類間で期末が不連続)
6. 恒等式が崩れた(r1〜r3 の残差が総資産比 1e-3 超、いずれかの書類・期)
フォールバックが発生しただけの社は除外しない(配下に項目が無ければ積み上げた値は正確)。

パネル:
- A(含む版): 除外後、対象のすべての年(既定は t = 2023, 2024)の差分がある社
- B(除く版): A から、重なる年(t-1 年の書類の当期と t 年の書類の前期)の水準が
  2 書類で一致しない社を除く

ステージ23 では ``build(..., years=(2022, 2023, 2024))`` として 3 年度で使う。

出力: ``<interim>/panel/{panel.csv,firms.csv}``、``results/stage22/panel.md``

    uv run python scripts/build_panel_stage22.py --interim data/interim/stage22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from build_delta_panel_stage21 import load_documents  # noqa: E402

_YEARS = (2023, 2024)
_ROUND_TOL = 1e-3  # plausibility.py の _ROUND_TOL と同じ(総資産比)
FINANCIAL_INDUSTRIES = ("銀行業", "証券、商品先物取引業", "保険業", "その他金融業")

REASONS: tuple[tuple[str, str], ...] = (
    ("financial", "is_financial(残余資産比率 > 0.5)"),
    ("financial_industry", "業種が金融(銀行・証券・保険・その他金融)"),
    ("parse_failed", "解析失敗・対象年の書類なし"),
    ("prior_absent", "前期の BS なし(連結初年度など)"),
    ("fiscal_change", "決算期変更(12 か月でない、または書類間で期末が不連続)"),
    ("identity_broken", "恒等式が崩れた(残差が総資産比 1e-3 超)"),
)


def build(
    docs: pd.DataFrame,
    names: list[str],
    industry33: pd.Series,
    years: tuple[int, ...] = _YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    ok = (docs["current_status"] == "ok") & (docs["prior_status"] == "ok")
    docs = docs.assign(
        financial=docs["is_financial"].astype(str) == "True",
        prior_absent=docs["prior_bs_absent"].astype(str) == "True",
        irregular=docs["months"].astype("Float64") != 12,
    )
    resid_cols = [f"{p}_{k}_rel" for p in ("current", "prior") for k in ("r1", "r2", "r3")]
    docs["identity_broken"] = docs[resid_cols].fillna(0).abs().gt(_ROUND_TOL).any(axis=1)

    ok_docs = docs[ok].sort_values("submitDateTime")
    n_dup = int(ok_docs.duplicated(["edinetCode", "t"], keep="last").sum())
    use = ok_docs.drop_duplicates(["edinetCode", "t"], keep="last")
    use = use[use["t"].isin(years)]

    firms = pd.DataFrame(index=sorted(set(docs["edinetCode"])))
    by_code = use.groupby("edinetCode")
    firms["industry33"] = industry33.reindex(firms.index).fillna("不明")
    firms["financial"] = docs.groupby("edinetCode")["financial"].any()
    firms["financial_industry"] = firms["industry33"].isin(FINANCIAL_INDUSTRIES)
    firms["parse_failed"] = ~firms.index.isin(use["edinetCode"])
    for col in ("prior_absent", "identity_broken"):
        firms[col] = by_code[col].any().reindex(firms.index, fill_value=False)
    irregular = by_code["irregular"].any().reindex(firms.index, fill_value=False)
    wide = use.pivot_table(
        index="edinetCode",
        columns="t",
        values=["current_instant", "prior_instant"],
        aggfunc="first",
    )
    chain_break = pd.Series(False, index=firms.index)
    for tp, t in zip(years, years[1:], strict=False):
        both = wide[("prior_instant", t)].notna() & wide[("current_instant", tp)].notna()
        broken = both & (wide[("prior_instant", t)] != wide[("current_instant", tp)])
        chain_break.loc[broken[broken].index] = True
    firms["fiscal_change"] = irregular | chain_break
    n_years = by_code["t"].nunique().reindex(firms.index, fill_value=0)

    lines = [
        f"同じ (社, 決算年) の重複書類: {n_dup} 件(提出日時が新しい方を使用)",
        "",
        f"企業(edinetCode): {len(firms)}",
        "",
        "| # | 除外理由 | 該当 | 順に除外 | 残り |",
        "|---|---|---:|---:|---:|",
    ]
    remaining = pd.Series(True, index=firms.index)
    for i, (col, label) in enumerate(REASONS, start=1):
        hit = firms[col] & remaining
        remaining &= ~firms[col]
        lines.append(
            f"| {i} | {label} | {int(firms[col].sum())} | {int(hit.sum())} "
            f"| {int(remaining.sum())} |"
        )
    firms["kept"] = remaining
    firms["panel_a"] = remaining & (n_years == len(years))

    lvl = use.set_index(["edinetCode", "t"])
    firms["mismatch_2023"] = False
    firms["mismatch_assets"] = False
    for code in firms.index[firms["panel_a"]]:
        for tp, t in zip(years, years[1:], strict=False):
            a = lvl.loc[(code, t), [f"prev_{n}" for n in names]].fillna(0).to_numpy()
            b = lvl.loc[(code, tp), [f"cur_{n}" for n in names]].fillna(0).to_numpy()
            if (a != b).any():
                firms.loc[code, "mismatch_2023"] = True
            if lvl.loc[(code, t), "prev_total_assets"] != lvl.loc[(code, tp), "cur_total_assets"]:
                firms.loc[code, "mismatch_assets"] = True
    firms["panel_b"] = firms["panel_a"] & ~firms["mismatch_2023"]
    pa = firms["panel_a"]
    lines += [
        "",
        f"対象のすべての年の差分がある社(パネル A): {int(pa.sum())}",
        f"  うち重なる年の水準が 2 書類で不一致: {int(firms['mismatch_2023'][pa].sum())}"
        f"(総資産まで変わる {int(firms['mismatch_assets'][pa].sum())})",
        f"パネル B(不一致を除く): {int(firms['panel_b'].sum())}",
    ]
    panel = use[use["edinetCode"].isin(firms.index[pa])].copy()
    return panel, firms, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=Path("data/interim/stage22"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--canonical", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--labels", type=Path, default=Path("data/interim/labels/labels.csv"))
    parser.add_argument("--industry", type=Path, default=Path("conf/accounts/industry.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/stage22/panel.md"))
    parser.add_argument("--title", default="ステージ22 B: 推定用パネル")
    parser.add_argument(
        "--years", type=int, nargs="+", default=list(_YEARS), help="差分の年(当期末の年)"
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["fy2024:", "2023_2024:_2023_2024"],
        help="提出期間の名前と接尾辞(名前:接尾辞)",
    )
    args = parser.parse_args()

    names = [b["name"] for b in yaml.safe_load(args.canonical.read_text())["basis"]]
    industry = yaml.safe_load(args.industry.read_text(encoding="utf-8"))
    ind33 = pd.Series(industry["edinet_industry"])
    sets = tuple(tuple(x.split(":", 1)) for x in args.sets)
    years = tuple(sorted(args.years))
    docs = load_documents(args.interim, args.raw_dir, names, sets)
    panel, firms, lines = build(docs, names, ind33, years)

    labels = pd.read_csv(args.labels, dtype={"edinetCode": str})
    firms["label"] = labels.groupby("edinetCode")["label"].max().reindex(firms.index)
    firms["coarse"] = firms["industry33"].map(industry["coarse_map"]).fillna("不明")

    out = [f"# {args.title}", "", *lines, ""]
    for key, title in (("panel_a", "A(不一致を含む)"), ("panel_b", "B(不一致を除く)")):
        sel = firms[firms[key]]
        cls = sel["label"].value_counts(dropna=False).sort_index()
        out += [
            f"## パネル {title}",
            "",
            f"- 社数 {len(sel)}、差分の期数 {len(years)}({'・'.join(map(str, years))})、"
            f"水準の期数 {len(years) + 1}({years[0] - 1}〜{years[-1]})、"
            f"差分の観測数 {len(years) * len(sel)}",
            "- class 別: "
            + "、".join(f"class {int(k) if pd.notna(k) else '不明'}: {v}" for k, v in cls.items()),
            "- 粗い業種: " + "、".join(f"{k} {v}" for k, v in sel["coarse"].value_counts().items()),
            "",
            "| 33 業種 | 社数 | class 2 |",
            "|---|---:|---:|",
        ]
        for ind, g in sel.groupby("industry33"):
            out.append(f"| {ind} | {len(g)} | {int((g['label'] == 2).sum())} |")
        out.append("")
    c2 = firms[firms["label"] == 2]
    out += ["## class 2 の行方", "", f"- ラベル上の class 2: {len(c2)} 社"]
    for col, label in REASONS:
        out.append(f"  - {label}: {int(c2[col].sum())}")
    out.append(f"- パネル A: {int(c2['panel_a'].sum())} 社、B: {int(c2['panel_b'].sum())} 社")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    (args.interim / "panel").mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.interim / "panel" / "panel.csv", index=False)
    firms.to_csv(args.interim / "panel" / "firms.csv", index_label="edinetCode")
    print("\n".join(out))


if __name__ == "__main__":
    main()
