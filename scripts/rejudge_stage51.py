#!/usr/bin/env python
"""ステージ51 A.4・B.2: 過去の判定を新しい主指標(v2)で再判定する(生成なし)。

規則は ``results/stage51/criteria_v2.md``(§4〜§6)。旧規則(各ステージの主指標)と新規則
(純資産の改善 ↑、解消率 ↑、順位相関 ↑、sparsity ↓)を同じコードで計算して並べる。
基準 1 は各ステージの時点で有効だった項目で数える(§6)。

入力: ステージ24・32・36 は ``metrics_stage39.py`` で計算し直した
``data/interim/stage51/metrics/``、
ステージ39・40・50 は各ステージの指標ファイル。

出力(``results/stage51/``): ``rejudge.md``、``rejudge.csv``(比較 × class × 指標の差)、
``violations_by_item.md``(B.2: 全設定の項目別の違反、現在の 15 項目の定義)。

    uv run python scripts/rejudge_stage51.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from compare_stage39 import boot, comp_level, corr, verdict

from fsgen.accounting.plausibility import CHECK_NAMES, EXTRA_CHECK_NAMES
from fsgen.evaluation.paired import paired_test

ALL_CHECKS = (*CHECK_NAMES, *EXTRA_CHECK_NAMES)
OUT2 = ("S100TSS9", "S100TWFQ")
OUT1 = ("S100VGDJ",)
# 指標キー → (列, 良い向き, 表示名)
METRICS = {"na": ("net_asset_improvement", "大", "純資産の改善"),
           "res": ("insolvency_resolved", "大", "解消率"),
           "asset": ("abs_asset_change", "小", "|総資産変化|"),
           "sparsity": ("sparsity", "小", "sparsity"),
           "validity": ("validity_lightgbm", "大", "validity"),
           "corr": ("", "大", "順位相関")}
SETS = {"v2": ("na", "res", "corr", "sparsity"),  # report_a.md の提案(criteria_v2.md §4)
        "confirmed": ("na", "res", "corr")}  # ユーザーが確定した主指標(sparsity は副)
NEW = SETS["v2"]


def checks_upto(stage: int) -> tuple[str, ...]:
    """基準 1 に数える項目(criteria_v2.md §6)。"""
    keep = [c for c in ALL_CHECKS if not c.startswith("8_")] if stage <= 44 else list(ALL_CHECKS)
    if stage <= 39:
        keep = [c for c in keep if not c.startswith(("14_", "15_"))]
    elif stage <= 40:
        keep = [c for c in keep if not c.startswith("15_")]
    return tuple(keep)


@dataclass(frozen=True)
class Stage:
    num: int
    dirs: dict[str, Path]  # D(空間)→ 指標のディレクトリ
    classes: tuple[str, ...]
    space: str  # 順位相関の空間(native / core32)
    settings: dict[str, tuple[str, str]]  # 名前 → (D, version)
    comparisons: tuple[tuple[str, str], ...]  # (比較する設定, 基準)
    old: tuple[str, ...]
    both: bool  # 両 class での成立を求める
    exclude: dict[str, tuple[str, ...]]
    note: str


M51 = Path("data/interim/stage51/metrics")
STAGES = (
    Stage(24, {"32": M51 / "s24"}, ("c2",), "native",
          {s: ("32", s) for s in ("equal", "estimated", "qualitative")},
          (("estimated", "equal"), ("qualitative", "equal")), ("asset", "validity"), False,
          {"c2": ()}, "R1 の設定、68 社(当時の評価)。当時の主指標は |総資産変化| と validity"),
    Stage(32, {"32": M51 / "s32"}, ("c2",), "native",
          {s: ("32", s) for s in ("equal", "frequency", "frequency_asym", "persistence")},
          (("frequency", "equal"), ("frequency_asym", "equal"), ("persistence", "equal")),
          ("na", "res", "asset"), False, {"c2": OUT2},
          "候補は frequency・frequency_asym(persistence は候補外)"),
    Stage(36, {"32": M51 / "s36"}, ("c2", "c1"), "native",
          {s: ("32", s) for s in ("equal", "estimated", "qualitative", "frequency",
                                  "persistence")},
          tuple((s, "equal") for s in ("estimated", "qualitative", "frequency", "persistence")),
          ("na", "res", "asset"), True, {"c2": OUT2, "c1": OUT1}, ""),
    Stage(39, {"32": Path("data/interim/stage39/metrics/core"),
               "38": Path("data/interim/stage39/metrics/expanded")}, ("c2", "c1"), "core32",
          {"1 core・最近傍": ("32", "nearest_na0"), "2 core・不均衡 OT": ("32", "unb0.1_na0"),
           "3 細分化・最近傍": ("38", "nearest_na0"), "4 細分化・不均衡 OT": ("38", "unb0.1_na0"),
           "5 細分化・持続性・最近傍": ("38", "pers_nearest_na0"),
           "6 細分化・持続性・不均衡 OT": ("38", "pers_unb0.1_na0")},
          (("2 core・不均衡 OT", "1 core・最近傍"), ("3 細分化・最近傍", "1 core・最近傍"),
           ("4 細分化・不均衡 OT", "3 細分化・最近傍"),
           ("5 細分化・持続性・最近傍", "3 細分化・最近傍"),
           ("6 細分化・持続性・不均衡 OT", "4 細分化・不均衡 OT")),
          ("na", "asset", "res", "corr"), True, {"c2": OUT2, "c1": OUT1}, ""),
    Stage(40, {"32": Path("data/interim/stage40/metrics/core"),
               "38": Path("data/interim/stage40/metrics/expanded")}, ("c2", "c1"), "core32",
          {"1 core": ("32", "nearest_na0"), "2 細分化・下限・等重み": ("38", "nearest_na0"),
           "3 細分化・下限・持続性": ("38", "pers_nearest_na0")},
          (("2 細分化・下限・等重み", "1 core"), ("3 細分化・下限・持続性", "1 core"),
           ("3 細分化・下限・持続性", "2 細分化・下限・等重み")),
          ("na", "asset", "corr"), True, {"c2": OUT2, "c1": OUT1},
          "停止規則(細分化が core に両 class で勝てなければ canonical の探索を終了)"),
    Stage(50, {"32": Path("data/interim/stage50/metrics")}, ("c2", "c1"), "native",
          {s: ("32", s) for s in ("equal", "t", "t_unb", "sub_all", "sub_trim", "lognorm",
                                  "t_pi12")},
          tuple((s, "equal") for s in ("t", "t_unb", "sub_all", "sub_trim", "lognorm", "t_pi12")),
          ("na", "asset", "corr"), True, {"c2": OUT2, "c1": OUT1},
          "候補は t、次に t_unb。他は感度分析"),
)


def load(st: Stage, cls: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series],
                                        pd.DataFrame]:
    rows, dels, viol = [], [], []
    refs: dict[str, pd.Series] = {}
    for d, path in st.dirs.items():
        if not (path / f"rows_{cls}.parquet").exists():
            continue
        r = pd.read_parquet(path / f"rows_{cls}.parquet")
        rows.append(r.assign(version=[f"{d}:{v}" for v in r["version"]]))
        x = pd.read_parquet(path / f"deltas_{cls}.parquet")
        dels.append(x[x["space"] == st.space].assign(
            setting=[f"{d}:{v}" for v in x[x["space"] == st.space]["setting"]]))
        v = pd.read_csv(path / f"viol_{cls}.csv", dtype={"docID": str})
        viol.append(v.assign(setting=[f"{d}:{s}" for s in v["setting"]]))
        f = pd.read_parquet(path / "refs.parquet")
        refs[d] = f[(f["cls"] == cls) & (f["space"] == st.space)].set_index("account")["median"]
    ex = st.exclude[cls]
    r_ = pd.concat(rows, ignore_index=True)
    x_ = pd.concat(dels, ignore_index=True)
    v_ = pd.concat(viol, ignore_index=True)
    return (r_[~r_["docID"].isin(ex)], x_[~x_["docID"].isin(ex)], refs,
            v_[~v_["docID"].isin(ex)])


def compare(comp: pd.DataFrame, dels: pd.DataFrame, refs: dict[str, pd.Series],
            a: tuple[str, str], b: tuple[str, str], rng: np.random.Generator
            ) -> dict[str, tuple[float, float, float, int]]:
    """指標ごとの (差, 下限, 上限, 判定)。"""
    ka, kb = f"{a[0]}:{a[1]}", f"{b[0]}:{b[1]}"
    ga = comp[comp["version"] == ka].set_index("docID")
    gb = comp[comp["version"] == kb].set_index("docID").reindex(ga.index)
    out: dict[str, tuple[float, float, float, int]] = {}
    for key, (col, better, _) in METRICS.items():
        if key == "corr":
            wa = dels[dels["setting"] == ka].pivot(index="docID", columns="account",
                                                   values="delta")
            wb = dels[dels["setting"] == kb].pivot(index="docID", columns="account",
                                                   values="delta").loc[wa.index]
            cols = list(refs[a[0]].index.intersection(wa.columns))
            wa, wb = wa[cols], wb[cols]
            d = corr(wa, refs[a[0]]) - corr(wb, refs[b[0]])
            _, _, lo, hi = boot(wa, refs[a[0]], wb, refs[b[0]], rng)
        else:
            sel = gb["insolvent_start"].astype(bool) if key == "res" else slice(None)
            pr = paired_test(ga.loc[sel, col].to_numpy(dtype=float),
                             gb.loc[sel, col].to_numpy(dtype=float))
            d, lo, hi = pr.median_diff, pr.ci_low, pr.ci_high
        out[key] = (float(d), float(lo), float(hi), verdict(better, d, lo, hi))
    return out


def win(marks: dict[str, int], keys: tuple[str, ...], viol: int) -> bool:
    m = [marks[k] for k in keys]
    return viol == 0 and sum(v == 1 for v in m) >= 2 and not any(v == -1 for v in m)


def sym(v: int) -> str:
    return {1: "+", -1: "−", 0: "0"}[v]


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage51"))
    ap.add_argument("--set", choices=tuple(SETS), default="v2",
                    help="新規則の主指標の集合(confirmed は rejudge_confirmed.md に書く)")
    args = ap.parse_args()
    global NEW
    NEW = SETS[args.set]
    suffix = "" if args.set == "v2" else f"_{args.set}"
    rng = np.random.default_rng(0)
    lab_new = "・".join(METRICS[k][2] for k in NEW)
    lines = [f"# ステージ51 A.4: 過去の判定の再検証(主指標 {args.set})", "",
             f"新規則: 主指標 = {lab_new} の {len(NEW)} つ。"
             "2 つ以上で有意に優れ、有意に劣るものがなく、基準 1(違反 0)を満たせば勝ち"
             "(`criteria_v2.md`)。旧規則は各ステージの当時の主指標で、同じ「2 つ以上で優れ、劣"
             "るものがない」。"
             "判定の記号は列の順(+ 有意に優れる、− 有意に劣る、0 どちらでもない)。"
             "違反は各ステージの時点の項目で数えた行数(§6)。", ""]
    det_rows, vrows = [], []
    for st in STAGES:
        lines += [f"## ステージ{st.num}", ""]
        if st.note:
            lines += [st.note, ""]
        old_lab = "・".join(METRICS[k][2] for k in st.old)
        new_lab = "・".join(METRICS[k][2] for k in NEW)
        lines += [f"| 比較 | class | 違反 | 旧({old_lab}) | 旧の勝ち | 新({new_lab}) | 新の勝ち |"
                  " 変化 |", "|---|---|---:|---|---|---|---|---|"]
        res: dict[tuple[str, str], dict[str, tuple[bool, bool]]] = {}
        allowed = checks_upto(st.num)
        for cls in st.classes:
            rows, dels, refs, viol = load(st, cls)
            comp = comp_level(rows)
            avail = set(comp["version"])
            for a, b in st.comparisons:
                sa, sb = st.settings[a], st.settings[b]
                if f"{sa[0]}:{sa[1]}" not in avail or f"{sb[0]}:{sb[1]}" not in avail:
                    continue
                c = compare(comp, dels, refs, sa, sb, rng)
                vv = viol[(viol["setting"] == f"{sa[0]}:{sa[1]}") & viol["check"].isin(allowed)]
                nv = int(vv["rows"].sum())
                marks = {k: v[3] for k, v in c.items()}
                wo, wn = win(marks, st.old, nv), win(marks, NEW, nv)
                res.setdefault((a, b), {})[cls] = (wo, wn)
                lines.append(f"| {a} 対 {b} | {cls} | {nv} |"
                             f" {' '.join(sym(marks[k]) for k in st.old)} "
                             f"| {'勝ち' if wo else '—'} | {' '.join(sym(marks[k]) for k in NEW)} "
                             f"| {'勝ち' if wn else '—'} | {'**変わる**' if wo != wn else ''} |")
                for k, (d, lo, hi, m) in c.items():
                    det_rows.append({"stage": st.num, "comparison": f"{a} 対 {b}", "cls": cls,
                                     "metric": METRICS[k][2], "diff": d, "lo": lo, "hi": hi,
                                     "mark": m, "violations": nv})
            # B.2: 項目別(現在の 15 項目の定義)
            for name, (d_, v_) in st.settings.items():
                key = f"{d_}:{v_}"
                if key not in avail:
                    continue
                vv = viol[viol["setting"] == key]
                n_rows = int((rows["version"] == key).sum())
                rec = {"stage": st.num, "cls": cls, "setting": name, "rows": n_rows}
                rec |= {c_: int(vv[vv["check"] == c_]["rows"].sum()) for c_ in ALL_CHECKS}
                vrows.append(rec)
        lines.append("")
        both_lines = []
        for (a, b), r in res.items():
            ok_old = all(v[0] for v in r.values()) if st.both else r.get("c2", (False,))[0]
            ok_new = all(v[1] for v in r.values()) if st.both else r.get("c2", (False, False))[1]
            if st.both and len(r) < 2:
                ok_old = ok_new = False
            both_lines.append(f"| {a} 対 {b} | {'成立' if ok_old else '不成立'} "
                              f"| {'成立' if ok_new else '不成立'} "
                              f"| {'**変わる**' if ok_old != ok_new else '変わらない'} |")
        cond = "両 class で成立" if st.both else "class 2 で成立(class 1 の出力なし)"
        lines += [f"**{cond}**", "", "| 比較 | 旧規則 | 新規則 | |", "|---|---|---|---|",
                  *both_lines, ""]
    (args.out / f"rejudge{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(det_rows).to_csv(args.out / "rejudge.csv", index=False)
    vdf = pd.DataFrame(vrows)
    vl = ["# ステージ51 B.2: 項目別の違反(行数)", "",
          "ステージ24・32・36・50 は現在の 15 項目の定義で数えた(24・32・36 は出力から計算し直"
          "した)。"
          "ステージ39・40 は当時の指標ファイルで、当時の項目で数えてある(8 は旧定義「資本金の半"
          "減」、"
          "39 には 14・15、40 には 15 がない)。8・14・15 は後から加えたり定義を変えたりした項目で、"
          "その時点の設計はこれらを満たすことを求めていない(`criteria_v2.md` §6)。"
          "外れ値は除く(ステージ24 は 68 社全体)。", "",
          "| ステージ | class | 設定 | 行 | " + " | ".join(c.split("_")[0] for c in ALL_CHECKS)
          + " | 合計 |", "|---|---|---|---:|" + "---:|" * (len(ALL_CHECKS) + 1)]
    for r in vdf.to_dict("records"):
        vals = [int(r[c]) for c in ALL_CHECKS]
        vl.append(f"| {r['stage']} | {r['cls']} | {r['setting']} | {r['rows']} | "
                  + " | ".join(str(v) if v else "·" for v in vals) + f" | {sum(vals)} |")
    vl += ["", "項目: " + "、".join(ALL_CHECKS)]
    (args.out / "violations_by_item.md").write_text("\n".join(vl) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
