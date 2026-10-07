#!/usr/bin/env python
"""ステージ51 C・D・E・F: K(PCA の有無)と目標の設定(最近傍・密度モデル)の比較。

規則は ``results/stage51/criteria_cd.md``(生成の前に記録)。入力は
``metrics_stage39.py --gen-dir data/interim/stage51/gen --out data/interim/stage51/metrics/cd``
の出力と、生成の parquet・損失の記録。設定のディレクトリ名: ``k13_nn``(設定 1、ステージ50 の
``equal`` へのリンク)、``k31_nn``(2)、``k13_dens``(3)、``k31_dens``(4)。ないものは飛ばす。

出力(``results/stage51/``): ``comparison_cd.md``(E.1・F、項目別の違反)、``losses.md``(C.2)、
``assets_extreme.md``(総資産の極端な変化)、``summary_cd.json``。

    uv run python scripts/evaluate_stage51.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from compare_stage39 import boot, comp_level, corr, fmt_ci, verdict

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import CHECK_NAMES, EXTRA_CHECK_NAMES, subtotal_members
from fsgen.evaluation.paired import paired_test
from fsgen.viz.balance_sheet import build_bs_lines, load_labels

_M = Path("data/interim/stage51/metrics/cd")
_GEN = Path("data/interim/stage51/gen")
ALL_CHECKS = (*CHECK_NAMES, *EXTRA_CHECK_NAMES)
EXCL = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
CLS = {"c2": "class 2", "c1": "class 1"}
SETTINGS = {"k13_nn": "1 K = 13・最近傍(現行)", "k31_nn": "2 K = 31・最近傍",
            "k13_dens": "3 K = 13・密度モデル", "k31_dens": "4 K = 31・密度モデル"}
# (列, 良い向き, 表示名)
PRIMARY = (("net_asset_improvement", "大", "純資産の改善"),
           ("insolvency_resolved", "大", "解消率"))
SECONDARY = (("sparsity", "小", "sparsity"), ("validity_lightgbm", "大", "validity"),
             ("proximity_l2", "小", "proximity"))
BASE = "k13_nn"  # 比較の基準(ステージ52 は書き換えて使う)
BASE_DESC = "設定 1(K = 13・最近傍、現行)"  # 見出しでの基準の説明(BASE と合わせて書き換える)
CRITERIA = "`criteria_cd.md` §1"
BASE_COL = "現行"  # BS 表での基準の列名
BS_ORDER: tuple[str, ...] = ("k13_dens", "k31_nn")  # 採用がないときの BS 表の設定(先にあるもの)
OUT = {"comparison": "comparison_cd.md", "losses": "losses.md", "assets": "assets_extreme.md",
       "bs": "bs_cd.md", "summary": "summary_cd.json"}
STAGE = "ステージ51"
TITLE = "ステージ51 E.1・F: K と目標の設定の比較"
PAIRS_F2: tuple[tuple[str, str, str], ...] = (("k31_nn", "k13_nn", "最近傍での K の効果(2 対 1)"),
            ("k31_dens", "k13_dens", "密度モデルでの K の効果(4 対 3)"),
            ("k13_dens", "k13_nn", "K = 13 での目標の効果(3 対 1)"),
            ("k31_dens", "k31_nn", "K = 31 での目標の効果(4 対 2)"))
REAL_P5_P95 = (-0.360, 4.128)  # 再建群 72 社、複利換算(criteria_cd.md §8)


def available(cls: str) -> list[str]:
    return [s for s in SETTINGS if (_GEN / cls / s).is_dir()]


def load_gen(cls: str, s: str) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in sorted((_GEN / cls / s).glob("*__seed*.parquet"))])
    return g[(g["method"] == "PCFM") & ~g["docID"].isin(EXCL[cls])].reset_index(drop=True)


class Ctx:
    def __init__(self, cls: str, names: tuple[str, ...]) -> None:
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        self.rows = rows[~rows["docID"].isin(EXCL[cls])]
        self.comp = comp_level(self.rows)
        v = pd.read_csv(_M / f"viol_{cls}.csv", dtype={"docID": str})
        self.viol = v[~v["docID"].isin(EXCL[cls])]
        d = pd.read_parquet(_M / f"deltas_{cls}.parquet")
        d = d[(d["space"] == "native") & ~d["docID"].isin(EXCL[cls])]
        self.wide = {s: d[d["setting"] == s].pivot(index="docID", columns="account",
                                                   values="delta")[list(names)]
                     for s in d["setting"].unique()}
        refs = pd.read_parquet(_M / "refs.parquet")
        self.ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index(
            "account")["median"]

    def viol_items(self, s: str) -> dict[str, int]:
        v = self.viol[self.viol["setting"] == s]
        return {c: int(v[v["check"] == c]["rows"].sum()) for c in ALL_CHECKS}

    def compare(self, a: str, b: str, rng: np.random.Generator) -> dict[str, tuple[float, ...]]:
        ga = self.comp[self.comp["version"] == a].set_index("docID")
        gb = self.comp[self.comp["version"] == b].set_index("docID").reindex(ga.index)
        out: dict[str, tuple[float, ...]] = {}
        for col, better, lab in (*PRIMARY, *SECONDARY):
            res_ = col == "insolvency_resolved"
            sel = gb["insolvent_start"].astype(bool) if res_ else slice(None)
            r = paired_test(ga.loc[sel, col].to_numpy(dtype=float),
                            gb.loc[sel, col].to_numpy(dtype=float))
            out[lab] = (r.median_diff, r.ci_low, r.ci_high,
                        verdict(better, r.median_diff, r.ci_low, r.ci_high))
        wa, wb = self.wide[a], self.wide[b].loc[self.wide[a].index]
        d = corr(wa, self.ref) - corr(wb, self.ref)
        _, _, lo, hi = boot(wa, self.ref, wb, self.ref, rng)
        out["順位相関"] = (d, lo, hi, verdict("大", d, lo, hi))
        return out


def fc(t: tuple[float, ...]) -> str:
    return fmt_ci(float(t[0]), float(t[1]), float(t[2]), 3)


def wins(c: dict[str, tuple[float, ...]], nviol: int) -> tuple[bool, str]:
    keys = ("純資産の改善", "解消率", "順位相関")
    m = [int(c[k][3]) for k in keys]
    ok = nviol == 0 and sum(v == 1 for v in m) >= 2 and not any(v == -1 for v in m)
    return ok, " ".join({1: "+", -1: "−", 0: "0"}[v] for v in m)


def losses_section() -> list[str]:
    lines = [f"# {STAGE} C.2: 学習の状況(訓練損失と検証損失、class 2 の 5-fold)", "",
             "fold ごと・seed ごとの値の中央値 [範囲]。比 = 検証損失の最小 / 訓練損失の最小。"
             "過学習の兆候 = 50 epoch 目の検証損失が最小の 1.5 倍を超える(criteria_cd.md §3)。"
             "fold 0 は外れ値 2 社がテスト側にあり検証損失が大きい(ステージ44)ので、fold 1〜4 "
             "も分けて示す。",
             "", "| 設定 | 次元 | 比(全 fold) | 比(fold 1〜4) | 検証最小の epoch | 50 epoch 目 /"
                 " 最小 "
             "| 兆候のある fold × seed |", "|---|---:|---|---|---|---|---:|"]
    for s, lab in SETTINGS.items():
        files = sorted((_GEN / "c2" / s).glob("*__losses.json"))
        if not files:
            continue
        ratio, ratio14, ep, last, flag, dim = [], [], [], [], 0, 0
        for f in files:
            for k, v in json.loads(f.read_text(encoding="utf-8")).items():
                tr, va = np.array(v["train"]), np.array(v["val"])
                dim = int(v["dim"])
                ratio.append(va.min() / tr.min())
                if k != "0":
                    ratio14.append(va.min() / tr.min())
                ep.append(int(np.argmin(va)) + 1)
                last.append(va[-1] / va.min())
                flag += int(va[-1] > 1.5 * va.min())

        def fmt(a: list[float] | list[int], d: int = 2) -> str:
            return f"{np.median(a):.{d}f} [{np.min(a):.{d}f}, {np.max(a):.{d}f}]"
        lines.append(f"| {lab} | {dim} | {fmt(ratio)} | {fmt(ratio14)} | {fmt(ep, 0)} "
                     f"| {fmt(last)} | {flag} / {len(ratio)} |")
    return lines + [""]


def assets_section(names: tuple[str, ...]) -> tuple[list[str], dict[str, Any]]:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    m = subtotal_members(spec, names)["jppfs_cor_Assets"]
    real = pd.read_csv("results/stage51/asset_change.csv")
    real = real[real["group"].str.startswith("再建群")]["d5_compound"].to_numpy()
    lines = [f"# {STAGE}: 総資産の極端な変化(生成、t = 1 = 5 年)", "",
             "企業ごとの seed 中央値の ΔA / A0。実データは再建群 72 社(class 2 → 回復 15 社 + cl"
             "ass 1 → "
             "class 0 57 社)の複利 5 年換算。範囲外 = 実データの p5〜p95 "
             f"[{REAL_P5_P95[0]:+.3f}, {REAL_P5_P95[1]:+.3f}] の外。", "",
             "| 対象 | 社数 | p5 | p25 | p50 | p75 | p95 | 半減(< 0.5 倍) | 倍増(> 2 倍) | 範囲"
             "外(下 / 上) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]

    def row(lab: str, d: np.ndarray) -> str:
        p = np.percentile(d, [5, 25, 50, 75, 95])
        lo, hi = int((d < REAL_P5_P95[0]).sum()), int((d > REAL_P5_P95[1]).sum())
        return (f"| {lab} | {len(d)} | " + " | ".join(f"{v:+.3f}" for v in p)
                + f" | {int((d < -0.5).sum())} | {int((d > 1.0).sum())} | {lo} / {hi} |")

    lines.append(row("実データ(再建群 72 社)", real))
    out: dict[str, Any] = {}
    for cls in ("c2", "c1"):
        for s in available(cls):
            g = load_gen(cls, s)
            a0 = g[[f"x0_{n}" for n in names]].to_numpy() @ m
            a1 = g[[f"x1_{n}" for n in names]].to_numpy() @ m
            d = pd.Series((a1 - a0) / a0, index=g["docID"].to_numpy()).groupby(level=0).median()
            lines.append(row(f"{CLS[cls]} {SETTINGS[s]}", d.to_numpy()))
            out[f"{cls}_{s}"] = {"half": int((d < -0.5).sum()), "double": int((d > 1.0).sum()),
                                 "max": float(d.max()), "min": float(d.min())}
    return lines + [""], out


def bs_section(names: tuple[str, ...], best: str) -> list[str]:
    """E.4: 現行と ``best`` の BS 表(class 2、seed 0、乱数 seed 0 で 5 社)。"""
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    gens = {}
    for s in (BASE, best):
        (f,) = sorted((_GEN / "c2" / s).glob("*__seed0.parquet"))
        g = pd.read_parquet(f)
        gens[s] = g[g["method"] == "PCFM"].set_index("docID")
    pool = sorted(set(gens[BASE].index) - set(EXCL["c2"]))
    docs = [str(d) for d in np.random.default_rng(0).choice(pool, 5, replace=False)]
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    out = [f"# {STAGE} E.4: BS 表(class 2、seed 0、百万円、t = 1 は 5 年)", "",
           "企業は外れ値を除く class 2(55 社)から乱数(seed 0)で選んだ 5 社。"
           f"列は始点、{BASE_DESC}、"
           f"{SETTINGS[best]} の終点。", ""]
    for d in docs:
        r = gens[BASE].loc[d]
        cols = [("始点", r[x0c].to_numpy(float)), (BASE_COL, r[x1c].to_numpy(float)),
                (best, gens[best].loc[d][x1c].to_numpy(float))]
        bl = build_bs_lines(spec, names, [v * float(r["assets_start"]) for _, v in cols], labels)
        out += [f"## {r['filerName']}({d})", "",
                "| 科目 | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
        for line in bl:
            if line.values is None:
                out.append(f"| **{line.label}** |" + " |" * len(cols))
                continue
            lab = "&nbsp;&nbsp;" * line.indent + line.label
            bold = "**" if line.kind in ("subtotal", "total", "diff") else ""
            out.append(f"| {bold}{lab}{bold} | "
                       + " | ".join(f"{v / 1e6:,.0f}" for v in line.values) + " |")
        out.append("")
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage51"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    summary: dict[str, Any] = {}
    lines = [f"# {TITLE}", "",
             f"差は{BASE_DESC}に対する企業単位のペアの差の中央値 [95% 区間]"
             "(順位相関は企業の同時復元抽出 1,000 回)。判定は 純資産の改善・解消率・順位相関 の順"
             f"(+ 有意に優れる、− 有意に劣る、0 どちらでもない)。基準は {CRITERIA}。", ""]
    judg: dict[str, dict[str, bool]] = {}
    ctxs = {cls: Ctx(cls, names) for cls in ("c2", "c1")}
    for cls in ("c2", "c1"):
        cx = ctxs[cls]
        sts = available(cls)
        s_ = cx.comp.groupby("version")
        lines += [f"## {CLS[cls]}", "", "### 水準と判定", "",
                  "| 設定 | 違反 | 純資産の改善 | 差 | 解消 | 順位相関 | 差 | 判定 | 基準 |",
                  "|---|---:|---:|---|---:|---:|---|---|---|"]
        for s in sts:
            g = s_.get_group(s)
            ins = g[g["insolvent_start"].astype(bool)]
            nv = sum(cx.viol_items(s).values())
            cs = corr(cx.wide[s], cx.ref)
            if s == BASE:
                lines.append(f"| {SETTINGS[s]} | {nv} |"
                             f" {g['net_asset_improvement'].median():.3f} | — "
                             f"| {int((ins['insolvency_resolved'] >= 2 / 3).sum())}/{len(ins)} "
                             f"| {cs:+.3f} | — | — | — |")
                continue
            c = cx.compare(s, BASE, rng)
            ok, sym = wins(c, nv)
            judg.setdefault(s, {})[cls] = ok
            lines.append(
                f"| {SETTINGS[s]} | {nv} | {g['net_asset_improvement'].median():.3f} "
                f"| {fc(c['純資産の改善'])} "
                f"| {int((ins['insolvency_resolved'] >= 2 / 3).sum())}/{len(ins)} | {cs:+.3f} "
                f"| {fc(c['順位相関'])} | {sym} | {'満たす' if ok else '満たさない'} |")
        lines += ["", "### 副指標の差(対 設定 1)", "",
                  "| 設定 | sparsity | validity | proximity | 解消率 |", "|---|---|---|---|---|"]
        for s in sts:
            if s == BASE:
                continue
            c = cx.compare(s, BASE, rng)
            lines.append(f"| {SETTINGS[s]} | " + " | ".join(
                fc(c[k]) for k in ("sparsity", "validity", "proximity", "解消率")) + " |")
        lines += ["", "### 項目別の違反(行)", "",
                  "| 設定 | 行 | " + " | ".join(c.split("_")[0] for c in ALL_CHECKS) + " | 合計 |",
                  "|---|---:|" + "---:|" * (len(ALL_CHECKS) + 1)]
        for s in sts:
            it = cx.viol_items(s)
            n = int((cx.rows["version"] == s).sum())
            lines.append(f"| {SETTINGS[s]} | {n} | " + " | ".join(str(v) if v else "·"
                                                               for v in it.values())
                         + f" | {sum(it.values())} |")
        lines.append("")
    lines += ["## F.1 判定(両 class)", "", "| 設定 | class 2 | class 1 | 両 class |", "|---|---|"
                                                                                "---|---|"]
    passed = []
    for s, r in judg.items():
        both = r.get("c2", False) and r.get("c1", False)
        if both:
            passed.append(s)
        lines.append(f"| {SETTINGS[s]} | {'満たす' if r.get('c2') else '満たさない'} "
                     f"| {'満たす' if r.get('c1') else '満たさない'} |"
                     f" {'**成立**' if both else '不成立'} |")
    lines += ["", f"基準を満たした設定: {', '.join(SETTINGS[s] for s in passed) or 'なし'}", ""]
    summary["judgment"] = judg
    summary["passed"] = passed
    # F.2
    lines += ["## F.2 K の効果と目標の効果の分離(判定とは別に報告)", "",
              "| 比較 | class | 純資産の改善 | 解消率 | 順位相関 | sparsity | 記号(主指標) |",
              "|---|---|---|---|---|---|---|"]
    for a, b, lab in PAIRS_F2:
        for cls in ("c2", "c1"):
            if a not in available(cls) or b not in available(cls):
                continue
            c = ctxs[cls].compare(a, b, rng)
            _, sym = wins(c, 0)
            lines.append(f"| {lab} | {CLS[cls]} | {fc(c['純資産の改善'])} "
                         f"| {fc(c['解消率'])} | {fc(c['順位相関'])} "
                         f"| {fc(c['sparsity'])} | {sym} |")
    lines.append("")
    (args.out / OUT["comparison"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / OUT["losses"]).write_text("\n".join(losses_section()) + "\n", encoding="utf-8")
    al, asum = assets_section(names)
    (args.out / OUT["assets"]).write_text("\n".join(al) + "\n", encoding="utf-8")
    summary["assets"] = asum
    # E.4: 採用された設定、なければ設定 3、それもなければ設定 2(criteria_cd.md §8、§4 で 3 が未実
    # 施の場合)
    best = passed[0] if passed else next(s for s in BS_ORDER if s in available("c2"))
    (args.out / OUT["bs"]).write_text("\n".join(bs_section(names, best)) + "\n", encoding="utf-8")
    (args.out / OUT["summary"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
