#!/usr/bin/env python
"""ステージ21-A E: 提出期間ごとの当期・前期の集約値を、edinetCode × 決算年で突合する。

入力は ``verify_prior_stage21.py`` の出力(``data/interim/stage21/<name>/``)。
``--sets`` に新しい提出期間から順に名前を並べる。k 番目(0 始まり)の提出期間の
当期は決算年 ``base_year - k``、前期は ``base_year - k - 1`` に割り当てる。

同じ決算年が 2 つの書類から得られる場合(新しい書類の前期と、古い書類の当期)は、
期末日・連結/単体の別・基礎科目の値が一致するかを検証する(E.2)。採用する値は
``--prefer`` で選ぶ(既定は ``current`` = その年度の書類の当期)。

非金融の判定は各書類の当期の ``is_financial``、ラベル(class)は
``--labels`` (最新の提出期間の labels.csv)から edinetCode で引く。

    uv run python scripts/build_panel_stage21.py --sets fy2024 --sets 2023_2024 \\
        --base-year 2024 --labels data/interim/labels/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

_BASIS_KEY = "basis"


@dataclass(frozen=True)
class Obs:
    edinet_code: str
    year: int
    set_name: str
    period: str  # "current" | "prior"
    doc_id: str
    instant: str
    role: str
    values: dict[str, int]  # 欠損は 0(下流の fillna(0) と同じ扱い)
    assets: int  # 資産側の基礎科目の和(= 総資産)


def _load_basis(canonical: Path) -> tuple[list[str], list[str]]:
    with canonical.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    names = [b["name"] for b in doc[_BASIS_KEY]]
    assets = [b["name"] for b in doc[_BASIS_KEY] if b["side"] == "assets"]
    return names, assets


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _months(earlier: str, later: str) -> int:
    ey, em = int(earlier[:4]), int(earlier[5:7])
    ly, lm = int(later[:4]), int(later[5:7])
    return (ly - ey) * 12 + (lm - em)


def load_set(
    interim: Path, set_name: str, year: int, names: list[str], asset_names: list[str]
) -> tuple[list[Obs], dict[str, int]]:
    """1 つの提出期間から、非金融・解析成功の社の当期・前期の観測を作る。"""
    base = interim / set_name
    checks = {r["docID"]: r for r in _read_csv(base / "company_checks.csv")}
    values = {
        period: {r["docID"]: r for r in _read_csv(base / f"values_{period}.csv")}
        for period in ("current", "prior")
    }
    stats: Counter[str] = Counter()
    by_code: dict[str, dict[str, str]] = {}
    for doc_id, chk in checks.items():
        cur = values["current"][doc_id]
        stats["documents"] += 1
        if chk["current_status"] != "ok" or chk["prior_status"] != "ok":
            stats["parse_failed"] += 1
            continue
        if cur["is_financial"] == "True":
            stats["financial"] += 1
            continue
        code = chk["edinetCode"]
        # 同一社の複数書類(決算期変更・訂正等)は、当期末が最も新しいものを残す。
        if code in by_code:
            stats["duplicate_edinet_code"] += 1
            if chk["current_instant"] <= by_code[code]["current_instant"]:
                continue
        by_code[code] = chk
    stats["non_financial_companies"] = len(by_code)

    obs: list[Obs] = []
    for code, chk in by_code.items():
        for period, y in (("current", year), ("prior", year - 1)):
            if period == "prior" and chk["prior_bs_absent"] == "True":
                stats["prior_bs_absent"] += 1
                continue
            row = values[period][chk["docID"]]
            vals = {n: int(row[n]) if row[n] else 0 for n in names}
            obs.append(
                Obs(
                    edinet_code=code,
                    year=y,
                    set_name=set_name,
                    period=period,
                    doc_id=chk["docID"],
                    instant=chk[f"{period}_instant"],
                    role=chk["used_role"],
                    values=vals,
                    assets=sum(vals[n] for n in asset_names),
                )
            )
    return obs, dict(stats)


def compare_overlap(
    pairs: list[tuple[Obs, Obs]], names: list[str]
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    """E.2: 同一社・同一決算年の 2 つの観測(新しい書類の前期, 古い書類の当期)を比べる。"""
    lines: list[str] = []
    kinds: Counter[str] = Counter()
    by_account: dict[str, list[float]] = defaultdict(list)
    details: list[dict[str, object]] = []
    for newer_prior, older_current in pairs:
        if newer_prior.instant != older_current.instant:
            kinds["期末日が異なる"] += 1
            continue
        if newer_prior.role != older_current.role:
            kinds["連結/単体の別が異なる"] += 1
            continue
        diff = {
            n: newer_prior.values[n] - older_current.values[n]
            for n in names
            if newer_prior.values[n] != older_current.values[n]
        }
        if not diff:
            kinds["完全一致"] += 1
            continue
        kinds["値が不一致"] += 1
        scale = abs(older_current.assets) or 1
        for n, d in diff.items():
            by_account[n].append(abs(d) / scale)
        details.append(
            {
                "edinetCode": newer_prior.edinet_code,
                "year": newer_prior.year,
                "doc_newer": newer_prior.doc_id,
                "doc_older": older_current.doc_id,
                "n_accounts": len(diff),
                "max_rel_diff": max(abs(d) for d in diff.values()) / scale,
                "assets_changed": newer_prior.assets != older_current.assets,
                "accounts": ";".join(sorted(diff)),
            }
        )
    total = len(pairs)
    lines.append(f"重複する決算年の観測: {total} 社")
    for kind in ("完全一致", "値が不一致", "期末日が異なる", "連結/単体の別が異なる"):
        lines.append(f"  {kind}: {kinds[kind]}")
    n_assets_changed = sum(bool(d["assets_changed"]) for d in details)
    lines.append(f"  値が不一致のうち、資産合計まで変わった社: {n_assets_changed}")
    small = sum(float(str(d["max_rel_diff"])) < 1e-3 for d in details)
    lines.append(f"  値が不一致のうち、最大のずれが総資産の 0.1% 未満: {small}")
    account_rows: list[dict[str, object]] = [
        {
            "account": n,
            "n_companies": len(v),
            "median_rel_diff": statistics.median(v),
            "max_rel_diff": max(v),
        }
        for n, v in sorted(by_account.items(), key=lambda kv: -len(kv[1]))
    ]
    return lines, account_rows, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", action="append", required=True)
    parser.add_argument("--base-year", type=int, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--prefer", choices=("current", "prior"), default="current")
    parser.add_argument("--canonical", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--interim", type=Path, default=Path("data/interim/stage21"))
    parser.add_argument("--results", type=Path, default=Path("results/stage21"))
    args = parser.parse_args()

    names, asset_names = _load_basis(args.canonical)

    all_obs: list[Obs] = []
    lines = ["# ステージ21-A E: パネルの構成", ""]
    for k, set_name in enumerate(args.sets):
        obs, stats = load_set(args.interim, set_name, args.base_year - k, names, asset_names)
        all_obs += obs
        lines.append(f"- {set_name}(当期 = {args.base_year - k} 年): {stats}")

    grouped: dict[tuple[str, int], list[Obs]] = defaultdict(list)
    for o in all_obs:
        grouped[(o.edinet_code, o.year)].append(o)

    # E.2 重複する決算年
    pairs: list[tuple[Obs, Obs]] = []
    for group in grouped.values():
        prior = [o for o in group if o.period == "prior"]
        current = [o for o in group if o.period == "current"]
        if prior and current:
            pairs.append((prior[0], current[0]))
    overlap_lines, account_rows, details = compare_overlap(pairs, names)
    lines += ["", "## E.2 重複する決算年の一致検証", "", *overlap_lines]

    # 採用する観測を 1 つに絞る
    adopted: dict[tuple[str, int], Obs] = {}
    for key, group in grouped.items():
        preferred = [o for o in group if o.period == args.prefer]
        adopted[key] = (preferred or group)[0]

    years = sorted({y for _, y in adopted}, reverse=True)
    by_code: dict[str, dict[int, Obs]] = defaultdict(dict)
    for (code, y), o in adopted.items():
        by_code[code][y] = o

    # E.3 決算期の連続性: 隣り合う決算年の期末日が 12 か月ずれているか
    irregular: set[str] = set()
    for code, per_year in by_code.items():
        for y in years:
            if y in per_year and y - 1 in per_year:
                if _months(per_year[y - 1].instant, per_year[y].instant) != 12:
                    irregular.add(code)
    # 1 書類内の当期・前期の間隔も見る(書類をまたがない決算期変更)
    for o in all_obs:
        if o.period == "prior":
            cur = grouped.get((o.edinet_code, o.year + 1), [])
            same_doc = [c for c in cur if c.doc_id == o.doc_id]
            if same_doc and _months(o.instant, same_doc[0].instant) != 12:
                irregular.add(o.edinet_code)

    labels: dict[str, str] = {}
    if args.labels is not None and args.labels.exists():
        labels = {r["edinetCode"]: r["label"] for r in _read_csv(args.labels)}

    lines += ["", "## E.1 パネルの構成(非金融、決算期変更を除く前 / 除いた後)", ""]
    n_years = len(years)
    for y in years:
        n_all = sum(1 for p in by_code.values() if y in p)
        n_reg = sum(1 for c, p in by_code.items() if y in p and c not in irregular)
        lines.append(f"- {y} 年: {n_all} / {n_reg} 社")
    balanced = [c for c, p in by_code.items() if all(y in p for y in years)]
    two_plus = [c for c, p in by_code.items() if len(p) >= 2]
    balanced_reg = [c for c in balanced if c not in irregular]
    two_plus_reg = [c for c in two_plus if c not in irregular]
    lines += [
        f"- 全 {n_years} 期に存在(バランスパネル): {len(balanced)} / {len(balanced_reg)} 社",
        f"- 2 期以上に存在: {len(two_plus)} / {len(two_plus_reg)} 社",
        f"- 決算期変更(隣り合う期末が 12 か月でない): {len(irregular)} 社",
    ]
    if labels:
        newest = args.base_year
        base = [c for c, p in by_code.items() if newest in p and c in labels]
        label_counts = Counter(labels[c] for c in base)
        lines.append(f"- {newest} 年の非金融でラベルがある社: {len(base)}({dict(label_counts)})")
        for cls in ("0", "2"):
            n_bal = sum(1 for c in balanced_reg if labels.get(c) == cls)
            n_two = sum(1 for c in two_plus_reg if labels.get(c) == cls)
            n_all = sum(1 for c in base if labels[c] == cls)
            lines.append(
                f"  - class {cls}: {n_all} 社 → 全期 {n_bal} 社、2 期以上 {n_two} 社"
                "(決算期変更を除く)"
            )

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "panel_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path, rows in (
        (args.results / "overlap_by_account.csv", account_rows),
        (args.interim / "overlap_mismatches.csv", details),
    ):
        with path.open("w", newline="", encoding="utf-8") as f:
            if rows:
                dict_writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                dict_writer.writeheader()
                dict_writer.writerows(rows)
    with (args.interim / "panel_long.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["edinetCode", "year", "instant", "used_role", "source", "docID", "irregular", *names]
        )
        for (code, y), o in sorted(adopted.items()):
            writer.writerow(
                [
                    code,
                    y,
                    o.instant,
                    o.role,
                    f"{o.set_name}:{o.period}",
                    o.doc_id,
                    code in irregular,
                    *(o.values[n] for n in names),
                ]
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
