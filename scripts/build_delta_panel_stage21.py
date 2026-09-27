#!/usr/bin/env python
"""ステージ21-D B: 同一書類内で差分を取るパネルを確定する。

観測の単位は書類。書類の当期末の年を t とし、差分 Δx_t = x_t - x_{t-1} と水準 x_{t-1} は
同じ書類の当期・前期から取る(前期は当期の表示方法・会計方針に組み替え済み)。
提出期間ではなく当期末の年で t を決めるので、同じ提出期間に 2 書類ある社
(決算期の遅延提出など)も、決算年が違えば別の t として使える。

企業単位の除外(この順に数える):
1. 金融業(いずれかの書類で is_financial)
2. 解析失敗
3. 前期の BS が存在しない(連結初年度など)
4. 決算期変更(書類内の当期末 - 前期末が 12 か月でない、または t-1 年の書類の当期末と
   t 年の書類の前期末が一致しない)
5. 小計が欠損しフォールバックが発生した(いずれかの書類・期)
同じ (社, t) に複数の書類がある場合は、提出日時が最も新しい書類を使う。

パネル:
- #1 バランス: 除外後、t = 2023, 2024 の両方の差分がある社
- #2 #1 から、2023 年の水準が 2 書類で一致しない社を除く

    uv run python scripts/build_delta_panel_stage21.py --interim data/interim/stage21/extmap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

_SETS: tuple[tuple[str, str], ...] = (("fy2024", ""), ("2023_2024", "_2023_2024"))
_YEARS = (2023, 2024)
_ROUND_TOL = 1e-3  # plausibility.py の _ROUND_TOL と同じ(総資産比)


def _months(earlier: str, later: str) -> int:
    return (int(later[:4]) - int(earlier[:4])) * 12 + int(later[5:7]) - int(earlier[5:7])


def load_documents(interim: Path, raw_dir: Path, names: list[str]) -> pd.DataFrame:
    """全書類の当期・前期の値と検証結果を 1 行 1 書類にまとめる。"""
    frames = []
    for set_name, suffix in _SETS:
        base = interim / set_name
        chk = pd.read_csv(base / "company_checks.csv", dtype={"docID": str, "edinetCode": str})
        cur = pd.read_csv(base / "values_current.csv", dtype={"docID": str})
        pri = pd.read_csv(base / "values_prior.csv", dtype={"docID": str})
        docs = pd.read_csv(
            raw_dir / "index" / f"document_list{suffix}.csv",
            dtype={"docID": str, "edinetCode": str},
        )[["docID", "filerName", "periodStart", "periodEnd", "submitDateTime"]]
        df = chk.merge(docs, on=["docID"], suffixes=("", "_doc"))
        df = df.merge(
            cur[["docID", "is_financial", "total_assets", *names]].rename(
                columns={n: f"cur_{n}" for n in [*names, "total_assets"]}
            ),
            on="docID",
        )
        df = df.merge(
            pri[["docID", "total_assets", *names]].rename(
                columns={n: f"prev_{n}" for n in [*names, "total_assets"]}
            ),
            on="docID",
        )
        df["set"] = set_name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["t"] = out["current_instant"].str[:4].astype("Int64")
    return out


def build(docs: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    lines: list[str] = []
    ok = (docs["current_status"] == "ok") & (docs["prior_status"] == "ok")
    docs = docs.copy()
    docs["financial"] = docs["is_financial"].astype(str) == "True"
    docs["prior_absent"] = docs["prior_bs_absent"].astype(str) == "True"
    docs["fallback"] = docs["current_fallback"].notna() | docs["prior_fallback"].notna()
    docs["irregular"] = docs["months"].astype("Float64") != 12

    # 同じ (社, t) の重複は提出日時が最も新しい書類を使う
    ok_docs = docs[ok].sort_values("submitDateTime")
    n_dup = int(ok_docs.duplicated(["edinetCode", "t"], keep="last").sum())
    lines.append(f"同じ (社, 決算年) の重複書類: {n_dup} 件(古い提出を除いた)")
    use = ok_docs.drop_duplicates(["edinetCode", "t"], keep="last")
    use = use[use["t"].isin(_YEARS)]

    firms = pd.DataFrame(index=sorted(set(docs["edinetCode"])))
    by_code = use.groupby("edinetCode")
    all_by_code = docs.groupby("edinetCode")
    firms["financial"] = all_by_code["financial"].any()
    firms["parse_failed"] = ~firms.index.isin(use["edinetCode"])
    firms["prior_absent"] = by_code["prior_absent"].any().reindex(firms.index, fill_value=False)
    irregular = by_code["irregular"].any().reindex(firms.index, fill_value=False)
    # 書類をまたぐ連続性: t=2024 の前期末 = t=2023 の当期末
    wide = use.pivot_table(
        index="edinetCode",
        columns="t",
        values=["current_instant", "prior_instant"],
        aggfunc="first",
    )
    chain_break = pd.Series(False, index=firms.index)
    if ("prior_instant", 2024) in wide and ("current_instant", 2023) in wide:
        both = wide[("prior_instant", 2024)].notna() & wide[("current_instant", 2023)].notna()
        broken = both & (wide[("prior_instant", 2024)] != wide[("current_instant", 2023)])
        chain_break.loc[broken[broken].index] = True
    firms["fiscal_change"] = irregular | chain_break
    firms["fallback"] = by_code["fallback"].any().reindex(firms.index, fill_value=False)
    n_years = by_code["t"].nunique().reindex(firms.index, fill_value=0)

    reasons = ["financial", "parse_failed", "prior_absent", "fiscal_change", "fallback"]
    labels_ja = {
        "financial": "金融業",
        "parse_failed": "解析失敗・対象年の書類なし",
        "prior_absent": "前期の BS なし(連結初年度など)",
        "fiscal_change": "決算期変更(12 か月でない、または書類間で期末が不連続)",
        "fallback": "小計の欠損によるフォールバック",
    }
    remaining = pd.Series(True, index=firms.index)
    lines += ["", f"企業(edinetCode): {len(firms)}", "", "| 除外理由 | 該当 | 順に除外 | 残り |"]
    lines.append("|---|---:|---:|---:|")
    for r in reasons:
        hit = firms[r] & remaining
        remaining &= ~firms[r]
        lines.append(
            f"| {labels_ja[r]} | {int(firms[r].sum())} | {int(hit.sum())} "
            f"| {int(remaining.sum())} |"
        )
    firms["kept"] = remaining
    firms["balanced"] = remaining & (n_years == len(_YEARS))

    # 代替案: フォールバックを「恒等式が崩れた場合だけ」除外する。崩れの判定は、丸め
    # (decimals=-6)の残差を拾わないよう、総資産比 _ROUND_TOL を超える残差とする。
    resid_cols = [f"{p}_{k}_rel" for p in ("current", "prior") for k in ("r1", "r2", "r3")]
    broken = use[resid_cols].fillna(0).abs().gt(_ROUND_TOL).any(axis=1) & use["fallback"]
    firms["fallback_broken"] = (
        use.assign(b=broken).groupby("edinetCode")["b"].any().reindex(firms.index, fill_value=False)
    )
    alt = firms[["financial", "parse_failed", "prior_absent", "fiscal_change"]].any(axis=1)
    firms["balanced_alt"] = ~alt & ~firms["fallback_broken"] & (n_years == len(_YEARS))

    # 2023 年の水準の不一致(t=2024 の書類の前期 vs t=2023 の書類の当期)
    lvl = use.set_index(["edinetCode", "t"])
    firms["mismatch_2023"] = False
    firms["mismatch_assets"] = False
    for code in firms.index[firms["balanced"] | firms["balanced_alt"]]:
        a = lvl.loc[(code, 2024), [f"prev_{n}" for n in names]].fillna(0).to_numpy()
        b = lvl.loc[(code, 2023), [f"cur_{n}" for n in names]].fillna(0).to_numpy()
        firms.loc[code, "mismatch_2023"] = bool((a != b).any())
        firms.loc[code, "mismatch_assets"] = bool(
            lvl.loc[(code, 2024), "prev_total_assets"] != lvl.loc[(code, 2023), "cur_total_assets"]
        )
    firms["panel1"] = firms["balanced"]
    firms["panel2"] = firms["balanced"] & ~firms["mismatch_2023"]
    firms["panel1_alt"] = firms["balanced_alt"]
    firms["panel2_alt"] = firms["balanced_alt"] & ~firms["mismatch_2023"]

    for key, label in (("panel1", "#1"), ("panel1_alt", "#1'")):
        sel = firms[key]
        lines += [
            "",
            f"バランスパネル {label}: {int(sel.sum())} 社"
            f"(うち 2023 年の水準が 2 書類で不一致 {int(firms['mismatch_2023'][sel].sum())}、"
            f"そのうち総資産まで変わる {int(firms['mismatch_assets'][sel].sum())})",
        ]
    lines += [
        "",
        "- #1・#2: 仕様どおり、フォールバックが発生した社をすべて除外",
        "- #1'・#2': フォールバックは、恒等式の残差が総資産比 1e-3 を超えた社だけ除外",
        "- #2・#2': それぞれ #1・#1' から 2023 年の水準の不一致を除く",
    ]
    keep = firms["kept"] | firms["balanced_alt"]
    panel = use[use["edinetCode"].isin(firms.index[keep])].copy()
    return panel, lines, firms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=Path("data/interim/stage21/extmap"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--canonical", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--labels", type=Path, default=Path("data/interim/labels/labels.csv"))
    parser.add_argument("--industry", type=Path, default=Path("conf/accounts/industry.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/stage21/panel_final.md"))
    args = parser.parse_args()

    names = [b["name"] for b in yaml.safe_load(args.canonical.read_text())["basis"]]
    docs = load_documents(args.interim, args.raw_dir, names)
    panel, lines, firms = build(docs, names)

    labels = pd.read_csv(args.labels, dtype={"edinetCode": str, "docID": str})
    # ラベルは fy2024 提出の書類単位。同じ社に複数あれば最大(2 > 1 > 0)を使う
    label = labels.groupby("edinetCode")["label"].max()
    industry = yaml.safe_load(args.industry.read_text(encoding="utf-8"))
    ind33 = pd.Series(industry["edinet_industry"])
    coarse = ind33.map(industry["coarse_map"])
    firms["label"] = label.reindex(firms.index)
    firms["industry33"] = ind33.reindex(firms.index).fillna("不明")
    firms["coarse"] = coarse.reindex(firms.index).fillna("不明")

    out_lines = ["# ステージ21-D B: 同一書類内で差分を取るパネル", "", *lines, ""]
    for key, title in (
        ("panel1", "#1 全社"),
        ("panel2", "#2 不一致を除く"),
        ("panel1_alt", "#1' 全社(フォールバックは恒等式が崩れた社だけ除外)"),
        ("panel2_alt", "#2' 不一致を除く(同上)"),
    ):
        sel = firms[firms[key]]
        n_obs = len(panel[panel["edinetCode"].isin(sel.index)])
        cls = sel["label"].value_counts(dropna=False).sort_index()
        out_lines += [
            f"## パネル {title}",
            "",
            f"- 社数 {len(sel)}、差分の期数 2(2023・2024)、水準の期数 3(2022〜2024)、"
            f"差分の観測数 {n_obs}",
            "- class 別: "
            + "、".join(f"class {int(k) if pd.notna(k) else '不明'}: {v}" for k, v in cls.items()),
            "- 粗い業種: " + "、".join(f"{k} {v}" for k, v in sel["coarse"].value_counts().items()),
            "",
            "| 33 業種 | 社数 | class 2 |",
            "|---|---:|---:|",
        ]
        for ind, g in sel.groupby("industry33"):
            out_lines.append(f"| {ind} | {len(g)} | {int((g['label'] == 2).sum())} |")
        out_lines.append("")

    # class 2 がどこで落ちたか
    c2 = firms[firms["label"] == 2]
    out_lines += ["## class 2 の行方", "", f"- ラベル上の class 2: {len(c2)} 社"]
    for col in ["financial", "parse_failed", "prior_absent", "fiscal_change", "fallback"]:
        out_lines.append(f"  - {col}: {int(c2[col].sum())}")
    out_lines += [
        f"  - うちフォールバックで恒等式が崩れた: {int(c2['fallback_broken'].sum())}",
        f"- パネル #1: {int(c2['panel1'].sum())} 社、#2: {int(c2['panel2'].sum())} 社、"
        f"#1': {int(c2['panel1_alt'].sum())} 社、#2': {int(c2['panel2_alt'].sum())} 社",
        "",
        "## 同じ提出期間に複数の書類がある社(決算期変更・遅延提出)",
        "",
        "| edinetCode | 社名 | docID | 期間 | 月数 | 決算年 t | class |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    multi = docs[docs.duplicated(["edinetCode", "set"], keep=False)].sort_values(
        ["edinetCode", "current_instant"]
    )
    for _, r in multi.iterrows():
        out_lines.append(
            f"| {r['edinetCode']} | {r['filerName']} | {r['docID']} "
            f"| {r['periodStart']}〜{r['periodEnd']} | {r['months']} | {r['t']} "
            f"| {firms['label'].get(r['edinetCode'])} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    panel_dir = args.interim / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_dir / "delta_panel.csv", index=False)
    firms.to_csv(panel_dir / "firms.csv", index_label="edinetCode")
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
