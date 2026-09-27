#!/usr/bin/env python
"""ステージ21-A C: 前期(Prior1YearInstant)の値を当期と同じ規則で集約し、検証する。

1 つの提出期間(``--suffix``)について、取得済みの ZIP を当期・前期の両方で
``inspect_values.process_company`` に通し、以下を出力する。

- ``<interim>/values_{current,prior}.csv``: 集約値(``aggregated_core.csv`` と同じ列)
- ``<interim>/company_checks.csv``: 社ごとの検証結果(期末日、フォールバック、残差、欠損集合)
- ``<results>/fallback.csv``: 小計ごとの欠損(フォールバック)社数、当期 / 前期(C.2)
- ``<results>/residuals.csv``: r1〜r4 の非ゼロ社数と相対誤差の分布、当期 / 前期(C.3)
- ``<results>/sample10.md``: 無作為 10 社の詳細(C.1)
- ``<results>/summary.md``: 上記の要約

前期の BS が存在しない社(連結初年度など。前期の ``Assets`` が無い)は
``prior_bs_absent`` として別に数え、残差の集計からは除く。

    uv run python scripts/verify_prior_stage21.py --suffix "" --name fy2024
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from inspect_values import (  # noqa: E402
    CanonicalConfig,
    CompanyValues,
    Residuals,
    _load_calc_link_config,
    compute_residuals,
    load_canonical,
    months_between,
    process_company,
    write_aggregated_csv,
)

from fsgen.parsing.instance import Period  # noqa: E402

_SAMPLE_SEED = 0
_SAMPLE_SIZE = 10
_PERIODS: tuple[Period, ...] = ("current", "prior")


@dataclass(frozen=True)
class Pair:
    period_end: str
    current: CompanyValues
    prior: CompanyValues

    @property
    def prior_bs_absent(self) -> bool:
        return self.prior.parse_status == "ok" and self.prior.totals.get("assets") is None

    @property
    def months(self) -> int | None:
        if self.current.instant is None or self.prior.instant is None:
            return None
        return months_between(self.prior.instant, self.current.instant)


def _rel(residuals: Residuals | None, key: str) -> float | None:
    if residuals is None or not residuals.assets:
        return None
    value: int | None = getattr(residuals, key)
    return None if value is None else value / abs(residuals.assets)


def _quantiles(values: list[float]) -> str:
    if not values:
        return "-"
    qs = statistics.quantiles(values, n=100, method="inclusive") if len(values) > 1 else values * 99
    return f"中央値 {statistics.median(values):.2e} / p99 {qs[98]:.2e} / 最大 {max(values):.2e}"


def _missing_basis(r: CompanyValues) -> frozenset[str]:
    return frozenset(r.missing)


def collect(
    raw_dir: Path, suffix: str, canonical: CanonicalConfig, limit: int | None
) -> list[Pair]:
    role_map, prefixes = _load_calc_link_config()
    documents_dir = raw_dir / f"documents{suffix}"
    with (raw_dir / "index" / f"document_list{suffix}.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (documents_dir / f"{r['docID']}.zip").exists()]
    if limit is not None:
        rows = rows[:limit]

    pairs: list[Pair] = []
    for row in tqdm(rows, desc=f"verify prior{suffix}"):
        values = {
            period: process_company(
                row["docID"],
                row.get("edinetCode", ""),
                row.get("filerName", ""),
                row.get("periodEnd", ""),
                documents_dir,
                role_map,
                prefixes,
                canonical,
                period,
            )
            for period in _PERIODS
        }
        pairs.append(Pair(row.get("periodEnd", ""), values["current"], values["prior"]))
    return pairs


def write_company_checks(canonical: CanonicalConfig, pairs: list[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["docID", "edinetCode", "used_role", "periodEnd"]
        for period in _PERIODS:
            header += [
                f"{period}_status",
                f"{period}_instant",
                f"{period}_fallback",
                f"{period}_n_missing",
            ]
            header += [f"{period}_{k}_rel" for k in ("r1", "r2", "r3", "r4")]
        header += ["months", "prior_bs_absent", "same_missing_set", "missing_diff"]
        writer.writerow(header)
        for p in pairs:
            row: list[object] = [
                p.current.doc_id,
                p.current.edinet_code,
                p.current.used_role or "",
                p.period_end,
            ]
            for r in (p.current, p.prior):
                res = compute_residuals(canonical, r)
                row += [
                    r.parse_status,
                    r.instant or "",
                    ";".join(sorted(r.fallback_subtotals)),
                    len(r.missing),
                ]
                for k in ("r1", "r2", "r3", "r4"):
                    v = _rel(res, k)
                    row.append("" if v is None else f"{v:.3e}")
            diff = _missing_basis(p.current) ^ _missing_basis(p.prior)
            row += [
                "" if p.months is None else p.months,
                p.prior_bs_absent,
                not diff,
                ";".join(sorted(diff)),
            ]
            writer.writerow(row)


def summarize(canonical: CanonicalConfig, pairs: list[Pair], out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = [p for p in pairs if p.current.parse_status == "ok" and p.prior.parse_status == "ok"]
    absent = [p for p in ok if p.prior_bs_absent]
    usable = [p for p in ok if not p.prior_bs_absent]
    lines = [
        f"対象 {len(pairs)} 書類、当期・前期とも解析成功 {len(ok)}",
        f"前期の BS が存在しない社(前期の Assets 無し): {len(absent)}",
        "",
    ]

    # 期末日の間隔(E.3 の先取り)
    not_12 = [p for p in usable if p.months != 12]
    lines.append(f"当期末 - 前期末 が 12 か月でない社: {len(not_12)} / {len(usable)}")
    for p in not_12[:20]:
        lines.append(
            f"  {p.current.doc_id} {p.current.filer_name}: "
            f"{p.prior.instant} → {p.current.instant}({p.months} か月)"
        )

    # C.2 フォールバック(当期は全社、前期は BS がある社)
    with (out_dir / "fallback.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subtotal", "n_current", "n_prior", "n_companies"])
        lines += ["", "小計の欠損(フォールバック)社数: 当期 / 前期(前期の BS がある社のみ)"]
        for s in canonical.subtotals:
            n_cur = sum(s.element_id in p.current.fallback_subtotals for p in usable)
            n_pri = sum(s.element_id in p.prior.fallback_subtotals for p in usable)
            writer.writerow([s.element_id, n_cur, n_pri, len(usable)])
            lines.append(f"  {s.element_id}: {n_cur} / {n_pri}")
        any_cur = sum(bool(p.current.fallback_subtotals) for p in usable)
        any_pri = sum(bool(p.prior.fallback_subtotals) for p in usable)
        either = sum(
            bool(p.current.fallback_subtotals or p.prior.fallback_subtotals) for p in usable
        )
        writer.writerow(["ANY", any_cur, any_pri, len(usable)])
        writer.writerow(["ANY_EITHER_PERIOD", either, either, len(usable)])
        lines.append(f"  いずれかの小計が欠損: {any_cur} / {any_pri}(どちらかの期で {either})")
        n_same_fallback = sum(
            p.current.fallback_subtotals == p.prior.fallback_subtotals for p in usable
        )
        lines.append(f"  欠損する小計の集合が当期と前期で一致: {n_same_fallback} / {len(usable)}")

    # C.3 恒等式の残差
    with (out_dir / "residuals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["residual", "period", "n_evaluated", "n_nonzero", "rel_nonzero_summary"])
        lines += ["", "恒等式の残差(0 でない社数 / 評価社数、相対誤差 = 残差 / |Assets|)"]
        for key, label in (("r1", "資産"), ("r2", "負債"), ("r3", "純資産"), ("r4", "貸借一致")):
            for period in _PERIODS:
                res = [
                    compute_residuals(canonical, p.current if period == "current" else p.prior)
                    for p in usable
                ]
                evaluated = [x for x in res if x is not None and getattr(x, key) is not None]
                nonzero = [x for x in evaluated if getattr(x, key) != 0]
                rels = [v for x in nonzero if (v := _rel(x, key)) is not None]
                summary = _quantiles(rels)
                writer.writerow([key, period, len(evaluated), len(nonzero), summary])
                lines.append(
                    f"  {key}({label}) {period:7s}: {len(nonzero)} / {len(evaluated)}  {summary}"
                )

    # C.1 当期と前期の科目集合
    same = sum(_missing_basis(p.current) == _missing_basis(p.prior) for p in usable)
    lines += ["", f"欠損する基礎科目の集合が当期と前期で一致: {same} / {len(usable)}"]
    return "\n".join(lines)


def write_sample(canonical: CanonicalConfig, pairs: list[Pair], path: Path) -> None:
    rng = random.Random(_SAMPLE_SEED)
    # 推定の対象になりうる社(非金融で、前期の BS がある)から抽出する。
    candidates = sorted(
        (
            p
            for p in pairs
            if p.current.parse_status == "ok"
            and p.prior.parse_status == "ok"
            and not p.current.is_financial
            and not p.prior_bs_absent
        ),
        key=lambda p: p.current.doc_id,
    )
    sample = rng.sample(candidates, min(_SAMPLE_SIZE, len(candidates)))
    lines = [
        f"# C.1 前期の値の検証(無作為 {len(sample)} 社、seed={_SAMPLE_SEED})",
        "",
        f"抽出元: 非金融かつ前期の BS がある {len(candidates)} 社。",
        "",
        "残差は |Assets| に対する相対値。フォールバックは差額定義が使えなかった小計。",
        "",
        "| docID | 社名 | 連結/単体 | 前期末 | 当期末 | 前期 r1/r2/r3/r4 | 当期 r1/r2/r3/r4 "
        "| 前期フォールバック | 欠損科目の差 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in sample:
        cells: list[str] = []
        for r in (p.prior, p.current):
            res = compute_residuals(canonical, r)
            rels = [_rel(res, k) for k in ("r1", "r2", "r3", "r4")]
            cells.append(" / ".join("-" if v is None else f"{v:.0e}" for v in rels))
        diff = _missing_basis(p.current) ^ _missing_basis(p.prior)
        lines.append(
            f"| {p.current.doc_id} | {p.current.filer_name} | {p.current.used_role} "
            f"| {p.prior.instant} | {p.current.instant} | {cells[0]} | {cells[1]} "
            f"| {', '.join(sorted(p.prior.fallback_subtotals)) or '-'} "
            f"| {', '.join(sorted(diff)) or '-'} |"
        )
    lines += ["", "## 前期の集約値(円)", ""]
    names = [b.name for b in canonical.basis]
    lines.append("| 科目 | " + " | ".join(p.current.doc_id for p in sample) + " |")
    lines.append("|---|" + "---:|" * len(sample))
    for name in names:
        cells = [
            "欠損" if name in p.prior.missing else f"{p.prior.values.get(name, 0):,}"
            for p in sample
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--name", required=True, help="出力先の名前(例: fy2024)")
    parser.add_argument("--canonical", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim/stage21"))
    parser.add_argument("--results-root", type=Path, default=Path("results/stage21"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    canonical = load_canonical(args.canonical)
    pairs = collect(args.raw_dir, args.suffix, canonical, args.limit)

    interim = args.interim_root / args.name
    results = args.results_root / args.name
    write_aggregated_csv(canonical, [p.current for p in pairs], interim / "values_current.csv")
    write_aggregated_csv(canonical, [p.prior for p in pairs], interim / "values_prior.csv")
    write_company_checks(canonical, pairs, interim / "company_checks.csv")
    text = summarize(canonical, pairs, results)
    write_sample(canonical, pairs, results / "sample10.md")
    (results / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
