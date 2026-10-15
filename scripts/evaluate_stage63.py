#!/usr/bin/env python
"""ステージ63: 事後選択の検証(A)、現金の端点解(B)、検定力(C.1)。

規則は ``results/stage63/criteria.md``(実行の前に記録)。ステージ61・62 の生成と指標に加えて、
``data/interim/stage63/all``(gapmed と現金の下限つきの生成)に ``metrics_stage39.py`` を
かけてから実行する。

    uv run python scripts/evaluate_stage63.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
import yaml
from compare_stage39 import corr
from evaluate_stage61 import CAL, gen_firms, loss, real_targets

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.evaluation.paired import paired_test

OUT = Path("results/stage63")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DESIGNS = ("base", "paid3", "paid10", "paid30", "gap", "invgap", "invrho", "gapmed")
_CASH = "jppfs_cor_CashAndDeposits"
_PANEL = Path("data/interim/stage23/panel/panel.csv")
REAL_GROUP = {"c2": "class 2 → 回復", "c1": "class 1 → class 0"}
# 生成の置き場所: (指標のルート, 生成のルート, 設定名の接頭辞)
ROOTS = {"s61": (Path("data/interim/stage61/metrics"), Path("data/interim/stage61/gen"), ""),
         "s62": (Path("data/interim/stage62/metrics"), Path("data/interim/stage62/all"), ""),
         "s63": (Path("data/interim/stage63/metrics"), Path("data/interim/stage63/all"), "")}


def excl() -> None:
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((ROOTS["s61"][1] / cls / "l2_l1").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))


def ctx_for(root: str, names: tuple[str, ...]) -> dict[str, e51.Ctx]:
    e51._M, e51._GEN, _ = ROOTS[root]
    return {c: e51.Ctx(c, names) for c in ("c2", "c1")}


def setting_of(des: str, dist: str, lt: float) -> tuple[str, str]:
    """(ルート, 設定名)。base はステージ61、6 設計はステージ62、gapmed はステージ63。"""
    if des == "base":
        return "s62", f"base__{dist}_l{lt:g}"
    if des == "gapmed":
        return "s63", f"gapmed__{dist}_l{lt:g}"
    if des.startswith("cash_"):
        return "s63", f"{des}__{dist}_l{lt:g}"
    return "s62", f"{des}__{dist}_l{lt:g}"


class Store:
    """企業ごとの値(較正指標・違反・純資産の改善)を、ルートごとの Ctx から集める。"""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
        self.members = subtotal_members(spec, names)
        self.ctx = {r: ctx_for(r, names) for r in ("s62", "s63")}
        self.cache: dict[tuple[str, str], pd.DataFrame] = {}

    def firms(self, cls: str, des: str, dist: str, lt: float) -> pd.DataFrame:
        root, s = setting_of(des, dist, lt)
        key = (cls, f"{root}:{s}")
        if key in self.cache:
            return self.cache[key]
        e51._M, e51._GEN, _ = ROOTS[root]
        f = gen_firms(cls, s, self.names, self.members)
        f = f[~f.index.isin(e51.EXCL[cls])].copy()
        c = self.ctx[root][cls]
        v = c.viol[c.viol["setting"] == s]
        f["viol"] = f.index.isin(set(v["docID"])).astype(float)
        comp = c.comp[c.comp["version"] == s].set_index("docID")
        f["nai"] = comp["net_asset_improvement"].reindex(f.index)
        self.cache[key] = f
        return f

    def wide(self, cls: str, des: str, dist: str, lt: float) -> pd.DataFrame:
        root, s = setting_of(des, dist, lt)
        return self.ctx[root][cls].wide[s]

    def ref(self, cls: str) -> pd.Series:
        return self.ctx["s62"][cls].ref


def summarize(f: pd.DataFrame, r: dict[str, float], w: dict[str, float]) -> dict[str, float]:
    g = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
         "d_assets": float(f["d_assets"].median())}
    return {**g, "loss": loss(g, r, w), "viol": int(f["viol"].sum()),
            "nai": float(f["nai"].median()), "n_changed": float(f["n_changed"].median())}


def pick(st: Store, cls: str, des: str, dist: str, idx: pd.Index | None,
         r: dict[str, float], w: dict[str, float]) -> tuple[float | None, dict[str, float]]:
    """違反 0 の λ̃ の中で L 最小(idx の企業に限る)。"""
    best: tuple[float | None, dict[str, float]] = (None, {})
    for lt in LAMS:
        f = st.firms(cls, des, dist, lt)
        if idx is not None:
            f = f.loc[f.index.intersection(idx)]
        s = summarize(f, r, w)
        if s["viol"] == 0 and (best[0] is None or s["loss"] < best[1]["loss"]):
            best = (lt, s)
    return best


def section_a(st: Store, reals: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    lines = ["# ステージ63 A: 第 1 項の重みの選択の検証", "",
             "## A.1・A.3 全設計(非対称 L1、違反 0 を制約にした λ̃*_0)", "",
             "| # | 設計 | class | λ̃*_0 | L | 払込資本↓(実データ) | 純資産の改善 | 順位相関 | "
             "変更科目数 |",
             "|---:|---|---|---:|---:|---|---:|---:|---:|"]
    for k, des in enumerate(DESIGNS):
        for cls in ("c2", "c1"):
            r, w, _ = reals[cls]
            lt, s = pick(st, cls, des, "l1a", None, r, w)
            if lt is None:
                lines.append(f"| {k} | {des} | {cls} | なし | — | — | — | — | — |")
                continue
            rc = corr(st.wide(cls, des, "l1a", lt), st.ref(cls))
            lines.append(f"| {k} | {des} | {cls} | {lt:g} | {s['loss']:.2f} "
                         f"| {s['paid_dec']:.0%}({r['paid_dec']:.0%}) | {s['nai']:.3f} | {rc:+.3f} "
                         f"| {s['n_changed']:.0f} |")
    lines += ["", "等重み L2 の参考(違反 0 の λ̃*_0):", "",
              "| 設計 | class | λ̃*_0 | L | 払込資本↓ | 純資産の改善 |", "|---|---|---:|---:|---"
                                                             ":|---:|"]
    for des in ("base", "paid10", "gap", "gapmed", "invrho"):
        for cls in ("c2", "c1"):
            r, w, _ = reals[cls]
            lt, s = pick(st, cls, des, "l2", None, r, w)
            lines.append(f"| {des} | {cls} | {lt if lt is None else f'{lt:g}'} "
                         f"| {s.get('loss', float('nan')):.2f} |"
                         f" {s.get('paid_dec', float('nan')):.0%} "
                         f"| {s.get('nai', float('nan')):.3f} |")
    # A.2: class 1 の 2 分割
    r, w, _ = reals["c1"]
    ids = st.firms("c1", "base", "l1a", 1.0).index
    res: dict[str, Any] = {"splits": []}

    def run_split(seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ids))
        a_idx, b_idx = ids[perm[: len(ids) // 2]], ids[perm[len(ids) // 2:]]
        out: dict[str, Any] = {}
        for tag, sel_idx, val_idx in (("A→B", a_idx, b_idx), ("B→A", b_idx, a_idx)):
            cands = []
            for des in DESIGNS:
                lt, s = pick(st, "c1", des, "l1a", sel_idx, r, w)
                if lt is not None:
                    cands.append((s["loss"], des, lt))
            if not cands:
                out[tag] = {"design": None}
                continue
            _, des_c, lt_c = min(cands)
            lt_b, _ = pick(st, "c1", "base", "l1a", sel_idx, r, w)
            fc = st.firms("c1", des_c, "l1a", lt_c).loc[val_idx]
            fb = st.firms("c1", "base", "l1a", lt_b if lt_b is not None else 3.0).loc[val_idx]
            sc, sb = summarize(fc, r, w), summarize(fb, r, w)
            pt = paired_test(fc["nai"].to_numpy(dtype=float), fb["nai"].to_numpy(dtype=float))
            ok = sc["viol"] == 0 and sc["loss"] < sb["loss"] and not (pt.ci_high < 0)
            out[tag] = {"design": des_c, "lam": lt_c, "base_lam": lt_b, "val_loss": sc["loss"],
                        "base_loss": sb["loss"], "val_viol": sc["viol"],
                        "nai_diff": (pt.median_diff, pt.ci_low, pt.ci_high),
                        "val_paid": sc["paid_dec"], "base_paid": sb["paid_dec"], "ok": ok}
        return out

    first = run_split(0)
    lines += ["", "## A.2 class 1 の 2 分割(seed 0、64 社ずつ)", "",
              "| 向き | 選ばれた設計(λ̃) | base の λ̃ | 検証側の L(選択 / base) | 検証側の違反 "
              "| 払込資本↓(選択 / base) | 純資産の改善の差 [95%] | 検証 |",
              "|---|---|---:|---|---:|---|---|---|"]
    for tag, o in first.items():
        if o["design"] is None:
            lines.append(f"| {tag} | 該当なし | — | — | — | — | — | — |")
            continue
        d0, lo, hi = o["nai_diff"]
        lines.append(f"| {tag} | {o['design']}({o['lam']:g}) | {o['base_lam']} "
                     f"| {o['val_loss']:.2f} / {o['base_loss']:.2f} | {o['val_viol']} "
                     f"| {o['val_paid']:.0%} / {o['base_paid']:.0%} "
                     f"| {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] | {'成立' if o['ok'] else '不成立'} |")
    freq: dict[str, int] = {}
    oks = 0
    total = 0
    for seed in range(100):
        o = run_split(1000 + seed)
        for v in o.values():
            if v["design"] is None:
                continue
            freq[v["design"]] = freq.get(v["design"], 0) + 1
            oks += int(v["ok"])
            total += 1
    lines += ["", f"**100 通りの分割(両向き、計 {total} 回)**: 選ばれた設計 "
              + "、".join(f"{k} {v}" for k, v in sorted(freq.items(), key=lambda t: -t[1]))
              + f"。検証の成立 {oks} / {total}({oks / max(total, 1):.0%})。", ""]
    res["first"] = first
    res["freq"] = freq
    res["ok_rate"] = oks / max(total, 1)
    return lines, res


def section_b2(names: tuple[str, ...]) -> list[str]:
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    lines = ["# ステージ63 B.2: 実データの現金の変化(2022 → 2024、2 年)", "",
             "比(正規化)= (現金 / 総資産)(2024) ÷ (現金 / 総資産)(2022)。比(名目)= 現金(2024) ÷"
             " 現金(2022)。"
             "始点の現金が正の企業。", "",
             "| 群 | 比 | n | 減少 | p1 | p5 | p10 | p25 | p50 | 比 < 0.5 | 比 < 0.1 | 0 になっ"
             "た |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for grp in REAL_GROUP.values():
        codes = pairs[pairs["group"] == grp]["edinetCode"].tolist()
        a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
        b = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
        c0, c1 = a[f"cur_{_CASH}"].fillna(0), b[f"cur_{_CASH}"].fillna(0)
        ok = c0 > 0
        for kind, rr in (("正規化", (c1 / b["cur_total_assets"]) / (c0 / a["cur_total_assets"])),
                         ("名目", c1 / c0)):
            v = rr[ok].to_numpy()
            q = np.percentile(v, [1, 5, 10, 25, 50])
            lines.append(f"| {grp}({len(codes)} 社) | {kind} | {len(v)} | {(v < 1).mean():.0%} | "
                         + " | ".join(f"{x:.3f}" for x in q)
                         + f" | {(v < 0.5).mean():.0%} | {(v < 0.1).mean():.0%} |"
                           f" {int((c1[ok] == 0).sum())} |")
    lines += ["", "β_cash = 0.11732(推定用パネルの 1 年、class 1+2 で現金が減った 191 組の名目"
                  "の比の 1 パーセンタイル。"
              "α と同じ方法、`criteria.md` §3)。", ""]
    return lines


def section_b3(st: Store, reals: dict[str, Any]) -> list[str]:
    lines = ["# ステージ63 B.3: 現金の下限 β_cash = 0.11732 の効果", "",
             "違反企業数は λ̃ ごと(15 項目のどれか)。λ̃*_0 = 違反 0 の中で L 最小。", ""]
    pairs = (("非対称 L1 base", "base", "cash_base", "l1a"), ("非対称 L1 paid10", "paid10",
                                                               "cash_paid10", "l1a"),
             ("非対称 L1 gap", "gap", "cash_gap", "l1a"), ("等重み L2 base", "base", "cash_base",
                                                           "l2"))
    for cls in ("c2", "c1"):
        r, w, _ = reals[cls]
        lines += [f"## {cls}", "", "| 設定 | 下限 | " + " | ".join(f"{x:g}" for x in LAMS)
                  + " | λ̃*_0 | L | 純資産の改善 | 払込資本↓ |",
                  "|---|---|" + "---:|" * (len(LAMS) + 4)]
        for lab, des0, des1, dist in pairs:
            for tag, des in (("なし", des0), ("あり", des1)):
                v = [int(st.firms(cls, des, dist, lt)["viol"].sum()) for lt in LAMS]
                lt0, s = pick(st, cls, des, dist, None, r, w)
                tail = (f"| {lt0:g} | {s['loss']:.2f} | {s['nai']:.3f} | {s['paid_dec']:.0%} |"
                        if lt0 is not None else "| なし | — | — | — |")
                lines.append(f"| {lab} | {tag} | " + " | ".join(str(x) for x in v) + f" {tail}")
        lines.append("")
    return lines


def section_b4(st: Store, names: tuple[str, ...]) -> list[str]:
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    star = json.loads(Path("results/stage61/summary.json").read_text(encoding="utf-8"))["star"]
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    lines = ["# ステージ63 B.4: 端点解(終点が始点の 1% 未満になった企業数)", "",
             "条件: 始点 |x0| > 1e-4(総資産比)かつ終点 |x1| < 0.01 |x0|。生成は各距離の λ̃*(ス"
             "テージ61)、"
             "「現金の下限」は非対称 L1 + β_cash を同じ λ̃* で。実データは 2 年(名目、総資産(202"
             "2)比)。", ""]
    for cls, grp in REAL_GROUP.items():
        codes = pairs[pairs["group"] == grp]["edinetCode"].tolist()
        a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
        b = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
        x0r = (a[[f"cur_{n}" for n in names]].fillna(0).to_numpy()
               / a["cur_total_assets"].to_numpy()[:, None])
        x1r = (b[[f"cur_{n}" for n in names]].fillna(0).to_numpy()
               / a["cur_total_assets"].to_numpy()[:, None])

        def count(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
            out: np.ndarray = ((np.abs(x0) > 1e-4) & (np.abs(x1) < 0.01 * np.abs(x0))).sum(axis=0)
            return out

        real = count(x0r, x1r)
        cols: dict[str, np.ndarray] = {}
        for d in ("l2", "l1s", "l1a"):
            e51._M, e51._GEN, _ = ROOTS["s61"]
            g = e51.load_gen(cls, f"{d}_l{star[cls][d]:g}")
            g = g[g["seed"] == 0]
            cols[d] = count(g[[f"x0_{n}" for n in names]].to_numpy(),
                            g[[f"x1_{n}" for n in names]].to_numpy())
        root, s = setting_of("cash_base", "l1a", star[cls]["l1a"])
        e51._M, e51._GEN, _ = ROOTS[root]
        g = e51.load_gen(cls, s)
        g = g[g["seed"] == 0]
        cols["l1a+cash"] = count(g[[f"x0_{n}" for n in names]].to_numpy(),
                                 g[[f"x1_{n}" for n in names]].to_numpy())
        n_gen = len(g)
        lines += [f"## {cls}(生成 {n_gen} 社、実データ {len(codes)} 社)", "",
                  "| 科目 | 等重み L2 | 対称 L1 | 非対称 L1 | 非対称 L1 + 現金の下限 | 実データ |",
                  "|---|---:|---:|---:|---:|---:|"]
        order = np.argsort(-(cols["l1a"] + cols["l1a+cash"] + cols["l1s"]))
        for j in order:
            if max(c[j] for c in cols.values()) == 0 and real[j] == 0:
                continue
            lines.append(f"| {ja.get(names[j], names[j])} | {cols['l2'][j]} | {cols['l1s'][j]} "
                         f"| {cols['l1a'][j]} | {cols['l1a+cash'][j]} | {real[j]} |")
        tot = {k: int(v.sum()) for k, v in cols.items()}
        lines += ["", "合計: " + "、".join(f"{k} {v}" for k, v in tot.items())
                  + f"、実データ {int(real.sum())}(企業数が違うので割合で比べる)。", ""]
    return lines


def section_c1(st: Store) -> list[str]:
    rng = np.random.default_rng(0)
    lines = ["# ステージ63 C.1: 検定力(class 1 の効果を 55 社で検出できる確率)", "",
             "class 1 の企業から 55 社を復元抽出(200 回)し、各回で企業の復元抽出(400 回)による"
             "順位相関の差"
             "(非対称 L1 − 等重み L2、元の定義)の 95% 区間の下限 > 0 の割合。", "",
             "| λ̃ | class 1 の差(128 社) | 検定力(n = 55) |", "|---:|---:|---:|"]
    ref = st.ref("c1")
    for lt in (0.003, 0.01, 0.03, 0.3, 1.0, 3.0, 10.0):
        wa = st.wide("c1", "base", "l1a", lt)
        wb = st.wide("c1", "base", "l2", lt).loc[wa.index]
        cols = list(wa.columns)
        ra = ref.loc[cols].to_numpy()
        a_arr, b_arr = wa.to_numpy(), wb.to_numpy()

        def diff(i: np.ndarray, a_arr: np.ndarray = a_arr, b_arr: np.ndarray = b_arr,
                 ra: np.ndarray = ra) -> float:
            from scipy.stats import spearmanr
            return float(spearmanr(np.median(a_arr[i], axis=0), ra).statistic
                         - spearmanr(np.median(b_arr[i], axis=0), ra).statistic)

        full = diff(np.arange(len(wa)))
        hits = 0
        for _ in range(200):
            s = rng.integers(0, len(wa), 55)
            inner = [diff(s[rng.integers(0, 55, 55)]) for _ in range(400)]
            hits += int(np.nanpercentile(inner, 2.5) > 0)
        lines.append(f"| {lt:g} | {full:+.3f} | {hits / 200:.2f} |")
    lines.append("")
    return lines


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    excl()
    st = Store(names)
    reals = {}
    for cls in ("c2", "c1"):
        r, se, n_real = real_targets(cls, names)
        reals[cls] = (r, {m: 1.0 / se[m] for m in CAL}, n_real)
    la, res = section_a(st, reals)
    (OUT / "selection.md").write_text("\n".join(la) + "\n", encoding="utf-8")
    (OUT / "cash_real.md").write_text("\n".join(section_b2(names)) + "\n", encoding="utf-8")
    (OUT / "cash_floor.md").write_text("\n".join(section_b3(st, reals)) + "\n", encoding="utf-8")
    (OUT / "endpoints.md").write_text("\n".join(section_b4(st, names)) + "\n", encoding="utf-8")
    (OUT / "power.md").write_text("\n".join(section_c1(st)) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                      encoding="utf-8")
    for f in ("selection.md", "cash_real.md", "cash_floor.md", "endpoints.md", "power.md"):
        print((OUT / f).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
