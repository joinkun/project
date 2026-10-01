#!/usr/bin/env python
"""ステージ44: 低次元化(PCA)、制約 (5)(8)(9)、自己株式の上限の必要性を確かめる。

入力: ``metrics_stage39.py --gen-dir data/interim/stage44/gen`` の出力
(``data/interim/stage44/metrics``)、
生成の parquet と損失の JSON。判定の基準は ``results/stage44/criteria.md``(生成の前に記録)。

出力(``results/stage44/``): ``pca.md``(A)、``constraints.md``(B)、``treasury_gen.md``(C.5・C.6)、
``summary.json``。

    uv run python scripts/evaluate_stage44.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_stage39 import boot, comp_level, corr, fmt_ci
from evaluate_stage34 import special_reason
from run_stage25 import summarize

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    orthonormal_basis,
    state_vector_names,
)
from fsgen.accounting.plausibility import (
    RETENTION_ACCOUNTS,
    distributable_proxy,
    subtotal_members,
)
from fsgen.accounting.subspace import build_reduced_basis

_M = Path("data/interim/stage44/metrics")
_GEN = Path("data/interim/stage44/gen")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
OUTLIERS = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
CLS = {"c2": "class 2(55 社)", "c1": "class 1(131 社)"}
_RE, _CS, _CSUR, _TS, _CASH = ("jppfs_cor_RetainedEarnings", "jppfs_cor_CapitalStock",
                               "jppfs_cor_CapitalSurplus", "jppfs_cor_TreasuryStock",
                               "jppfs_cor_CashAndDeposits")
GEN_YEARS, MATERIAL, HOLD = 5.0, 0.001, 1e-6
# 制約の境界(α × 始点、β × 始点)にちょうど張り付いた値を「下回った」と数えないための相対の余裕
_TOL = 1e-4
# 主指標(悪い向き): 大きいほど良い = "大"
PRIMARY = (("純資産の改善", "大"), ("債務超過の解消率(社ごとの 3 seed 平均)", "大"),
           ("|総資産変化|", "小"), ("sparsity", "小"))


def load_gen(cls: str, setting: str) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in
                   sorted((_GEN / cls / setting).glob("*__seed*.parquet"))])
    return g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIERS[cls])].reset_index(drop=True)


def worse(better: str, m: float, lo: float, hi: float) -> bool:
    if pd.isna(lo):
        return False
    return hi < 0 if better == "大" else lo > 0


class Ctx:
    """class ごとの指標・順位相関・違反を持つ。"""

    def __init__(self, cls: str, names: tuple[str, ...], rng: np.random.Generator) -> None:
        self.cls, self.names, self.rng = cls, names, rng
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        self.rows = rows[~rows["docID"].isin(OUTLIERS[cls])]
        self.comp = comp_level(self.rows)
        v = pd.read_csv(_M / f"viol_{cls}.csv", dtype={"docID": str})
        self.viol = v[~v["docID"].isin(OUTLIERS[cls])]
        d = pd.read_parquet(_M / f"deltas_{cls}.parquet")
        self.dels = d[(d["space"] == "native") & ~d["docID"].isin(OUTLIERS[cls])]
        refs = pd.read_parquet(_M / "refs.parquet")
        self.ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index(
            "account")["median"]

    def wide(self, s: str) -> pd.DataFrame:
        w = self.dels[self.dels["setting"] == s]
        return w.pivot(index="docID", columns="account", values="delta")[list(self.names)]

    def summary(self, s: str) -> pd.Series:
        summ, _ = summarize(self.comp, (), (s,), s)
        out: pd.Series = summ.set_index("version").loc[s]
        return out

    def n_viol(self, s: str) -> int:
        return int(self.viol[self.viol["setting"] == s]["rows"].sum())

    def compare(self, s: str, base: str = "final") -> dict[str, object]:
        """s − base: 主指標の差と悪化の判定、順位相関の差。"""
        _su, pr = summarize(self.comp, (), (base, s), base)
        pr = pr.set_index("metric")
        out: dict[str, object] = {}
        bad = []
        for m, better in PRIMARY:
            q = pr.loc[m]
            v, lo, hi = q["median_diff(version-equal)"], q["ci_low"], q["ci_high"]
            out[m] = fmt_ci(v, lo, hi)
            if worse(better, v, lo, hi):
                bad.append(m)
        q = pr.loc["validity LightGBM"]
        out["validity"] = fmt_ci(q["median_diff(version-equal)"], q["ci_low"], q["ci_high"])
        a, b = self.wide(s), self.wide(base)
        b = b.loc[a.index]
        ca, cb = corr(a, self.ref), corr(b, self.ref)
        _, _, dlo, dhi = boot(a, self.ref, b, self.ref, self.rng)
        out["corr"] = fmt_ci(ca - cb, dlo, dhi, 3)
        out["corr_worse"] = bool(dhi < 0)
        out["primary_worse"] = bad
        return out


def below(thr: float, x0: pd.Series) -> pd.Series:
    """thr × 始点を明確に下回る値の上限(境界への張り付きを除く: 相対 1e-4 と総資産比 1e-6 の大"
      "きい方)。"""
    out: pd.Series = thr * x0 - np.maximum(thr * x0 * _TOL, 1e-6)
    return out


def firm_share(flag: pd.Series, doc: pd.Series) -> tuple[int, int]:
    """3 seed の 2/3 以上で真の企業数と企業数。"""
    f = pd.Series(flag.to_numpy(), index=doc.to_numpy()).groupby(level=0).mean()
    return int((f >= 2 / 3).sum()), len(f)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage44"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    ctx = {cls: Ctx(cls, names, rng) for cls in ("c2", "c1")}
    summary: dict[str, object] = {}
    args.out.mkdir(parents=True, exist_ok=True)

    # ======================= A. PCA =======================
    c = build_constraint_matrix(spec, include_subtotals=False)
    u = orthonormal_basis(c)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")
              & agg["total_assets"].notna() & (agg["total_assets"] > 0)]
    xr = agg[list(names)].fillna(0.0).to_numpy() / agg["total_assets"].to_numpy()[:, None]
    a_lines = ["# ステージ44 A: 低次元化(PCA)の必要性", "",
               "## 基底の数値的な性質", "",
               "U は C(32 × 1、基礎科目の貸借の恒等式)の完全 SVD(`np.linalg.svd(C, full_matrices"
               "=True)`)の、"
               "rank(= 1)を超える 31 本の左特異ベクトル(`orthonormal_basis`)。QR は使っていない。"
               "PCA の基底 B = U V_k は、U の座標で実データ(非金融の全書類、総資産比)を非中心化 "
               "SVD した上位 k 方向。", "",
               "| 基底 | 次元 | max\\|CᵀB\\| | max\\|BᵀB − I\\| | 実データの再構成誤差(中央値) |",
               "|---|---:|---:|---:|---:|"]
    for lab, b in (("U(PCA なし)", u), *((f"U V_{k}", build_reduced_basis(c, xr, k))
                                         for k in (8, 13, 20))):
        rec = np.linalg.norm(xr - xr @ b @ b.T, axis=1) / np.maximum(np.linalg.norm(xr, axis=1),
                                                                    1e-12)
        a_lines.append(f"| {lab} | {b.shape[1]} | {np.abs(c.T @ b).max():.1e} "
                       f"| {np.abs(b.T @ b - np.eye(b.shape[1])).max():.1e} "
                       f"| {np.median(rec):.2e} |")
    a_lines += ["", "(この表の PCA は全書類で作った参考。生成では fold ごとに class 0 と訓練始"
                    "点で作る)", ""]
    settings_a = {"final": "U V_13(現行)", "null31": "U(PCA なし)", "k8": "U V_8", "k20": "U V_20"}
    # 恒等式の残差(FM の生の出力と、射影後)
    a_lines += ["## A.4 比較", ""]
    for cls in ("c2", "c1"):
        cx = ctx[cls]
        a_lines += [f"### {CLS[cls]}", "",
                    "| 基底 | 次元 | 変化分の恒等式の残差 FM / 射影後(最大) | 15 項目の違反 | "
                    "純資産の改善 | 解消 "
                    "| \\|総資産変化\\| | sparsity | validity | 順位相関 | 訓練損失の最小 [epoch] "
                    "| 検証損失の最小 [epoch の中央値、fold 1〜4] | 検証の最小 / 訓練の最小 "
                    "| seed 間の標準偏差(純資産の改善) |",
                    "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---:|"]
        for s, lab in settings_a.items():
            g = pd.concat([pd.read_parquet(f) for f in
                           sorted((_GEN / cls / s).glob("*__seed*.parquet"))])
            g = g[~g["docID"].isin(OUTLIERS[cls])]
            res = {}
            for meth in ("FM", "PCFM"):  # 変化分の恒等式の残差(始点自体の丸めの残差を除く)
                gg = g[g["method"] == meth]
                dx = (gg[[f"x1_{n}" for n in names]].to_numpy()
                      - gg[[f"x0_{n}" for n in names]].to_numpy())
                res[meth] = float(np.abs(dx @ c).max())
            losses = [json.loads(f.read_text()) for f in
                      sorted((_GEN / cls / s).glob("*__losses.json"))]
            # 外れ値 2 社(資本構成が学習データの範囲外)は fold 0 のテスト側にあり、fold 0 の検証損
            # 失は
            # その 2 社で決まる(約 100 倍)。過学習の判断は fold 1〜4 で行い、fold 0 は別に示す
            tr_min, tr_ep, va_min, va_ep, ratio, va0 = [], [], [], [], [], []
            for lf in losses:
                for fk, fold in lf.items():
                    t = np.asarray(fold["train"], dtype=float)
                    tr_min.append(t.min())
                    tr_ep.append(int(t.argmin()) + 1)
                    if fold["val"]:
                        v = np.asarray(fold["val"], dtype=float)
                        if fk == "0" and cls == "c2":
                            va0.append(v.min())
                            continue
                        va_min.append(v.min())
                        va_ep.append(int(v.argmin()) + 1)
                        ratio.append(v.min() / t.min())
            sm = cx.summary(s)
            w = cx.wide(s)
            cc = corr(w, cx.ref)
            r = cx.rows[cx.rows["version"] == s]
            sd = float(r.groupby("docID")["net_asset_improvement"].std().median())
            va_cell = (f"{np.mean(va_min):.3f} [{np.median(va_ep):.0f}、範囲 {min(va_ep)}"
                       f"〜{max(va_ep)}]"
                       f"(fold 0: {np.mean(va0):.1f})" if va_min else "—")
            ra_cell = f"{np.median(ratio):.2f}" if ratio else "—"
            a_lines.append(
                f"| {lab} | {next(iter(losses[0].values()))['dim']} "
                f"| {res['FM']:.1e} / {res['PCFM']:.1e} | {cx.n_viol(s)} "
                f"| {sm['net_asset_improvement']:.3f} "
                f"| {int(sm['resolved_firms(>=2/3 seeds)'])}/{int(sm['insolvent_start'])} "
                f"| {sm['abs_asset_change']:.3f} | {sm['sparsity']:.0f} "
                f"| {sm['validity_lightgbm']:.3f} | {cc:+.3f} "
                f"| {np.mean(tr_min):.4f} [{np.median(tr_ep):.0f}] | {va_cell} | {ra_cell} |"
                f" {sd:.4f} |")
        a_lines += ["", "各基底 − 現行(差の中央値 [95% CI]、順位相関は差 [95%]):", "",
                    "| 基底 | 純資産の改善 | 解消率 | \\|総資産変化\\| | sparsity | validity | "
                    "順位相関 "
                    "| 悪化した主指標 |",
                    "|---|---|---|---|---|---|---|---|"]
        for s in ("null31", "k8", "k20"):
            r = cx.compare(s)
            a_lines.append(f"| {settings_a[s]} | {r['純資産の改善']} "
                           f"| {r['債務超過の解消率(社ごとの 3 seed 平均)']} | {r['|総資産変化|']} "
                           f"| {r['sparsity']} | {r['validity']} | {r['corr']} "
                           f"| {'、'.join(r['primary_worse']) or 'なし'}"  # type: ignore[arg-type]
                           f"{'、順位相関' if r['corr_worse'] else ''} |")
            summary[f"A_{cls}_{s}"] = r
        a_lines.append("")
    ok = all(ctx[c_].n_viol("null31") == 0 and not summary[f"A_{c_}_null31"]["primary_worse"]  # type: ignore[index]
             and not summary[f"A_{c_}_null31"]["corr_worse"] for c_ in ("c2", "c1"))  # type: ignore[index]
    a_lines += ["## 判定(PCA なし)", "",
                "**PCA なし(U、31 次元)は、両 class で 15 項目の違反 0、主指標で有意に悪化する"
                "ものがなく、"
                "順位相関も有意に下がらない → 低次元化は不要(構成の単純化を提案)**" if ok else
                "**PCA なし(U、31 次元)は基準を満たさない → 低次元化が必要な根拠とする**", ""]
    summary["A_pca_unnecessary"] = ok
    (args.out / "pca.md").write_text("\n".join(a_lines) + "\n", encoding="utf-8")

    # ======================= B. 制約 =======================
    # 実データ: class 1・2(2022 年度)、特殊業種を除く、2022 → 2024 年度末
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wide_l = fy.pivot(index="edinetCode", columns="year", values="label")
    pan = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a23 = pan[pan["t"] == 2023].set_index("edinetCode")
    b24 = pan[pan["t"] == 2024].set_index("edinetCode")
    both = a23.index.intersection(b24.index)
    press = [cd for cd in both if wide_l.loc[cd, 2022] in (1, 2)] if len(both) else []
    reason = special_reason(fy, press)
    press = [cd for cd in press if reason.get(cd, "") == ""]

    def rv(df: pd.DataFrame, side: str, n: str) -> np.ndarray:
        out: np.ndarray = df.loc[press, f"{side}_{n}"].fillna(0.0).to_numpy()
        return out

    a1r = b24.loc[press, "cur_total_assets"].to_numpy()
    real: dict[str, float] = {}
    for n, key in ((_CS, "cs"), (_CSUR, "csur")):
        x0, x1 = rv(a23, "prev", n), rv(b24, "cur", n)
        held = x0 > 0
        real[key] = float(np.mean(x1[held] < 0.5 * x0[held]))
        real[f"{key}_n"] = float(held.sum())
    ret_bad = np.zeros(len(press), dtype=bool)
    for n in RETENTION_ACCOUNTS:
        x0, x1 = rv(a23, "prev", n), rv(b24, "cur", n)
        ret_bad |= (x0 > 0) & (x1 < 0.3 * x0)
    real["ret"] = float(ret_bad.mean())
    real["cash09"] = float((rv(b24, "cur", _CASH) / a1r > 0.9).sum())
    real["n"] = float(len(press))
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    rf = rf[(rf["reason"].fillna("") == "") & rf["in_panel"].astype(bool) & (rf["group"] == "Y")]
    re_real = rf["d_re"].to_numpy()  # 年率
    q_real = np.quantile(re_real, [0.5, 0.9, 0.99])

    def phenomena(cls: str, s: str) -> dict[str, float]:
        g = load_gen(cls, s)
        out: dict[str, float] = {}
        for n, key in ((_CS, "cs"), (_CSUR, "csur")):
            x0, x1 = g[f"x0_{n}"], g[f"x1_{n}"]
            held = x0 > HOLD  # 始点の浮動小数の端数(例 5e-19)を保有と数えない
            k, m = firm_share((x1 < below(0.5, x0)) & held, g["docID"])
            out[key] = k / m
            ratio = (x1[held] / x0[held])
            out[f"{key}_min"] = float(ratio.min())
        bad = pd.Series(False, index=g.index)
        for n in RETENTION_ACCOUNTS:  # core では 3 科目とも基礎科目
            x0, x1 = g[f"x0_{n}"], g[f"x1_{n}"]
            bad |= (x0 > HOLD) & (x1 < below(0.3, x0))
        k, m = firm_share(bad, g["docID"])
        out["ret"] = k / m
        x1a = g[[f"x1_{n}" for n in names]].to_numpy() @ members["jppfs_cor_Assets"]
        k, _m = firm_share(g[f"x1_{_CASH}"] / x1a > 0.9, g["docID"])
        out["cash09"] = float(k)
        dre = (g[f"x1_{_RE}"] - g[f"x0_{_RE}"]).groupby(g["docID"]).median() / GEN_YEARS
        q = np.quantile(dre, [0.5, 0.9, 0.99])
        out["re_p50"], out["re_p90"], out["re_p99"] = (float(v) for v in q)
        return out

    variants = {"final": "1 現行", "no_alpha": "2 (5) を外す", "no_beta": "3 (8) を外す",
                "no_cap": "4 (9) を外す", "no_all": "5 (5)(8)(9) を外す",
                "alpha0": "α = 0", "alpha03": "α = 0.3", "alpha10": "α = 1.0",
                "beta05": "β = 0.5", "beta10": "β = 1.0"}
    target = {"no_alpha": ("cs", "csur"), "no_beta": ("ret",), "no_cap": ("re_p90",),
              "no_all": ("cs", "csur", "ret", "re_p90")}
    b_lines = ["# ステージ44 B: 制約 (5)(8)(9) の必要性", "",
               "実装: (5) は資本金と資本剰余金の両方に α × 始点の下限、(8) は有形固定資産・棚卸"
               "資産・売上債権に β × 始点の"
               "下限、(9) は ΔRE ≤ γ × 総資産(始点)。生成は 5 年、実データは class 1・2(2022 年"
               "度、特殊業種を除く)"
               f"{int(real['n'])} 社の 2022 → 2024 年度末(2 年)。ΔRE の実データは再建群 15 社("
               f"年率)。", "",
               "## 実データ", "",
               f"- 資本金が 50% 未満に減った企業: {real['cs']:.1%}(資本金のある"
               f" {int(real['cs_n'])} 社中)。"
               f"資本剰余金: {real['csur']:.1%}({int(real['csur_n'])} 社中)",
               f"- 有形固定資産・棚卸資産・売上債権のいずれかが 30% 未満に減った企業:"
               f" {real['ret']:.1%}。"
               f"現金 / 総資産 > 0.9 の企業: {int(real['cash09'])} 社",
               f"- ΔRE / A(年率)の再建群: p50 {q_real[0]:+.4f}、p90 {q_real[1]:+.4f}、p99"
               f" {q_real[2]:+.4f}", ""]
    judg_b: dict[str, dict[str, dict[str, object]]] = {}
    for cls in ("c2", "c1"):
        cx = ctx[cls]
        ph = {s: phenomena(cls, s) for s in variants}
        b_lines += [f"## {CLS[cls]}", "",
                    "| 設定 | 15 項目の違反 | 純資産の改善 | 解消 | \\|総資産変化\\| | sparsity "
                    "| validity "
                    "| 資本金 < 50%(社の割合、最小比) | 資本剰余金 < 50% | 3 科目のいずれか < 30% "
                    "| 現金 > 0.9(社) | ΔRE/A 年率 p50 / p90 / p99 |",
                    "|---|---:|---:|---|---:|---:|---:|---|---|---:|---:|---|"]
        for s, lab in variants.items():
            sm, p_ = cx.summary(s), ph[s]
            b_lines.append(
                f"| {lab} | {cx.n_viol(s)} | {sm['net_asset_improvement']:.3f} "
                f"| {int(sm['resolved_firms(>=2/3 seeds)'])}/{int(sm['insolvent_start'])} "
                f"| {sm['abs_asset_change']:.3f} | {sm['sparsity']:.0f} "
                f"| {sm['validity_lightgbm']:.3f} | {p_['cs']:.1%}({p_['cs_min']:.2f}) "
                f"| {p_['csur']:.1%}({p_['csur_min']:.2f}) | {p_['ret']:.1%} | {p_['cash09']:.0f} "
                f"| {p_['re_p50']:+.4f} / {p_['re_p90']:+.4f} / {p_['re_p99']:+.4f} |")
        b_lines += [f"| 実データ | | | | | | | {real['cs']:.1%} | {real['csur']:.1%} "
                    f"| {real['ret']:.1%} | {int(real['cash09'])} "
                    f"| {q_real[0]:+.4f} / {q_real[1]:+.4f} / {q_real[2]:+.4f}(再建群) |", "",
                    "各設定 − 現行(差の中央値 [95% CI]):", "",
                    "| 設定 | 純資産の改善 | 解消率 | \\|総資産変化\\| | sparsity | validity | "
                    "順位相関 "
                    "| 悪化した主指標 | 現象の乖離(現行 → 外した場合) |",
                    "|---|---|---|---|---|---|---|---|---|"]
        for s, lab in variants.items():
            if s == "final":
                continue
            r = cx.compare(s)
            devs = []
            dev_worse = False
            for key in target.get(s, ()):
                ref_v = q_real[1] if key == "re_p90" else real[key]
                d0, d1 = abs(ph["final"][key] - ref_v), abs(ph[s][key] - ref_v)
                devs.append(f"{key} {d0:.3f} → {d1:.3f}")
                dev_worse |= d1 > d0 + 1e-12
            b_lines.append(f"| {lab} | {r['純資産の改善']} |"
                           f" {r['債務超過の解消率(社ごとの 3 seed 平均)']} "
                           f"| {r['|総資産変化|']} | {r['sparsity']} | {r['validity']} |"
                           f" {r['corr']} "
                           f"| {'、'.join(r['primary_worse']) or 'なし'}"  # type: ignore[arg-type]
                           f"{'、順位相関' if r['corr_worse'] else ''} |"
                           f" {'、'.join(devs) or '—'} |")
            if s in target:
                judg_b.setdefault(s, {})[cls] = {
                    "viol": cx.n_viol(s), "primary_worse": r["primary_worse"],
                    "corr_worse": r["corr_worse"], "dev_worse": dev_worse}
        b_lines.append("")
    b_lines += ["## B.4 判定(外してよいのは、両 class で (i)〜(iv) をすべて満たすとき)", "",
                "| 制約 | 違反 0 | 主指標の悪化なし | 現象の乖離が増えない | 順位相関の低下なし "
                "| 外してよいか |",
                "|---|---|---|---|---|---|"]
    for s in target:
        jj = judg_b[s]
        c1_ = all(jj[c_]["viol"] == 0 for c_ in ("c2", "c1"))
        c2_ = all(not jj[c_]["primary_worse"] for c_ in ("c2", "c1"))
        c3_ = all(not jj[c_]["dev_worse"] for c_ in ("c2", "c1"))
        c4_ = all(not jj[c_]["corr_worse"] for c_ in ("c2", "c1"))
        b_lines.append(f"| {variants[s]} | {'○' if c1_ else '×'} | {'○' if c2_ else '×'} "
                       f"| {'○' if c3_ else '×'} | {'○' if c4_ else '×'} "
                       f"| **{'外してよい' if c1_ and c2_ and c3_ and c4_ else '残す'}** |")
    # 参考(判定には使わない): ステージ31 の基準 2〜5(再建群から作った帯)を、現行と (9) を外した版
    # に当てる。
    # ステージ31 では、上限なしの R3 は基準 4(振替の大きさ)を満たさず、それが γ を導入した理由だった
    from evaluate_stage30 import _band_check
    from evaluate_stage31 import firm_table
    crit = json.loads(Path("results/stage31/criteria.json").read_text(encoding="utf-8"))
    yrs, thr_a = float(crit["years"]), float(crit["threshold_annual"])
    lo2, hi2 = crit["share_band_wilson"]
    p_hi = float(crit["gamma_5y_p90_all_boot95"][1])
    band4 = [crit["transfer_5y"]["band_lo"], crit["transfer_5y"]["band_hi"]]
    band5 = [crit["issue_5y"]["band_lo"], crit["issue_5y"]["band_hi"]]
    b_lines += ["## 参考: ステージ31 の基準 2〜5(class 2、57 社。判定には使わない)", "",
                f"基準 2 = RE が増える社の割合が Wilson {lo2:.3f}〜{hi2:.3f}、基準 3 = 利益部分"
                f"の最大 ≤ "
                f"{p_hi:.2f}、基準 4 = 振替(5 年)の帯 {band4[0]:.3f}〜{band4[1]:.3f}、"
                f"基準 5 = 増資の帯 {band5[0]:.3f}〜{band5[1]:.2f}。", "",
                "| 設定 | 2 RE が増える社 | 3 利益部分の最大 | 4 振替(社・中央値・p95) "
                "| 5 増資(社・中央値・p95) | 2 | 3 | 4 | 5 |", "|---|---|---:|---|---|---|---|--"
                                                       "-|---|"]

    def mk(x: bool | None) -> str:
        return "該当なし" if x is None else ("○" if x else "×")

    for s_ in ("final", "no_cap"):
        firm = firm_table(_GEN / "c2" / s_, names)
        up = firm["d_re"] / yrs > thr_a
        c3 = bool((firm["profit"] <= p_hi + 1e-9).all())
        c4, t_med, t_p95 = _band_check(firm.loc[firm["transfer"] / yrs > thr_a, "transfer"], band4)
        c5, i_med, i_p95 = _band_check(firm.loc[firm["d_paid"] / yrs > thr_a, "d_paid"], band5)
        n_t = int((firm["transfer"] / yrs > thr_a).sum())
        n_i = int((firm["d_paid"] / yrs > thr_a).sum())
        b_lines.append(
            f"| {variants[s_]} | {int(up.sum())}/{len(firm)} ({up.mean():.1%}) "
            f"| {firm['profit'].max():+.3f} | {n_t}・{t_med:.3f}・{t_p95:.3f} "
            f"| {n_i}・{i_med:.3f}・{i_p95:.3f} | {mk(lo2 <= up.mean() <= hi2)} | {mk(c3)} "
            f"| {mk(c4)} | {mk(c5)} |")
    b_lines.append("")
    summary["B"] = judg_b
    (args.out / "constraints.md").write_text("\n".join(b_lines) + "\n", encoding="utf-8")

    # ======================= C. 自己株式 =======================
    c_lines = ["# ステージ44 C.5・C.6: 自己株式の上限(分配可能額の代理)", "",
               "自己株式の増加 = \\|自己株式\\| が始点の総資産比 0.001 を超えて増えた(5 年)。保"
               "有 = 始点で "
               "\\|自己株式\\| ≥ 総資産比 1e-6。企業は 3 seed の 2/3 以上で増えた場合に数える。"
               "実データは class 1・2(2022 年度)で自己株式を保有する企業の 2 年"
               "(`treasury.md`)。", ""]
    settings_c = {"final": "1 不活性制約のみ(現行)", "tscap": "2 不活性制約 + 上限",
                  "tscap_only": "3 上限のみ"}
    judg_c = {}
    for cls in ("c2", "c1"):
        cx = ctx[cls]
        c_lines += [f"## {CLS[cls]}", "",
                    "| 設定 | 15 項目の違反 | 保有企業で増えた | 増加の規模 p50 / p90(保有企業"
                    "で増えた行) "
                    "| 非保有で出現 | 純資産の改善 | \\|総資産変化\\| | 主指標の差(悪化) |",
                    "|---|---:|---|---|---|---:|---:|---|"]
        rates = {}
        for s, lab in settings_c.items():
            g = load_gen(cls, s)
            t0, t1 = g[f"x0_{_TS}"].abs(), g[f"x1_{_TS}"].abs()
            hold = t0 >= HOLD
            up = (t1 - t0) > MATERIAL
            gh = g[hold]
            k, m = firm_share(up[hold], gh["docID"])
            sz = (t1 - t0)[hold & up]
            ke, me = firm_share(up[~hold], g.loc[~hold, "docID"])
            sm = cx.summary(s)
            worse_s = "—"
            if s != "final":
                r = cx.compare(s)
                worse_s = ("、".join(r["primary_worse"]) or "なし")  # type: ignore[arg-type]
                if s == "tscap":
                    judg_c[cls] = {"viol": cx.n_viol(s), "primary_worse": r["primary_worse"]}
            rates[s] = k / m if m else float("nan")
            c_lines.append(
                f"| {lab} | {cx.n_viol(s)} | {k}/{m}({k / m:.0%}) "
                f"| {(sz.quantile(0.5) if len(sz) else float('nan')):.4f} / "
                f"{(sz.quantile(0.9) if len(sz) else float('nan')):.4f} | {ke}/{me} "
                f"| {sm['net_asset_improvement']:.3f} | {sm['abs_asset_change']:.3f} | {worse_s} |")
        judg_c.setdefault(cls, {})["rates"] = rates
        # C.6: 債務超過の始点で D > 0
        g = load_gen(cls, "final").groupby("docID").first()
        x0 = g[[f"x0_{n}" for n in names]].to_numpy()
        na0 = x0 @ members["jppfs_cor_NetAssets"]
        dpos = sum(1 for row, na in zip(x0, na0, strict=True)
                   if na < 0 and distributable_proxy(row, names) > 0)
        c_lines += ["", f"C.6: 始点で債務超過の企業 {int((na0 < 0).sum())} 社のうち、代理 D > 0"
                        f" の企業 {dpos} 社"
                    "(D > 0 なら上限は方向制約(債務超過企業の自社株買い禁止)より緩く、包含しない"
                    ")。", ""]
        # 統合の確認: 上限があるとき、方向制約を外しても出力が変わらないか
        diffs = {}
        for a_, b_ in (("tscap_nodir", "tscap"), ("final_nodir", "final")):
            ga = load_gen(cls, a_).sort_values(["seed", "docID"]).reset_index(drop=True)
            gb = load_gen(cls, b_).sort_values(["seed", "docID"]).reset_index(drop=True)
            cols = [f"x1_{n}" for n in names]
            dd = np.abs(ga[cols].to_numpy() - gb[cols].to_numpy()).max(axis=1)
            diffs[a_] = (float(dd.max()), int((dd > 1e-6).sum()))
        c_lines += [f"統合の確認: 上限ありで方向制約を外した場合と外さない場合の最大差 "
                    f"{diffs['tscap_nodir'][0]:.1e}(差が 1e-6 を超える行"
                    f" {diffs['tscap_nodir'][1]})。"
                    f"上限なし(現行)で方向制約を外すと、最大差 {diffs['final_nodir'][0]:.3f}、"
                    f"変わる行 {diffs['final_nodir'][1]} / {len(load_gen(cls, 'final'))}。", ""]
        judg_c[cls]["integration"] = diffs
        judg_c[cls]["insolvent"] = int((na0 < 0).sum())
        judg_c[cls]["insolvent_dpos"] = dpos
    summary["C"] = judg_c
    (args.out / "treasury_gen.md").write_text("\n".join(c_lines) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                      default=str), encoding="utf-8")
    print("\n".join(a_lines + [""] + b_lines + [""] + c_lines))


if __name__ == "__main__":
    main()
