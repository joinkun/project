#!/usr/bin/env python
"""ステージ61 A: λ の較正・検証・距離の比較(``evaluate_stage51`` の評価を流用)。

規則は ``results/stage61/criteria.md``(実行の前に記録)。事前に ``qp_stage61.py`` と
``metrics_stage39.py --gen-dir data/interim/stage61/gen --out data/interim/stage61/metrics``
を実行する。

出力(``results/stage61/``): ``grid.csv``・``grid.md``・``calibration.md``・``distances.md``・
``figures/tradeoff_{c2,c1}.png``。

    uv run python scripts/evaluate_stage61.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import matplotlib
import numpy as np
import pandas as pd
from compare_stage39 import corr

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/stage61")
DATA = Path("data/interim/stage61")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DISTS = {"l2": "1 等重み L2", "l1s": "2 対称 L1", "l1a": "3 非対称 L1(提案)"}
DIST_EN = {"l2": "1 equal-weight L2 (primary)", "l1s": "2 symmetric L1",
           "l1a": "3 asymmetric L1 (proposed)"}
PRIMARY_DIST = "l2"
_CS, _CSUR, _RE = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus",
                   "jppfs_cor_RetainedEarnings")
GEN_Y, REAL_Y, THR, DELTA_Y = 5.0, 2.0, 0.001, 0.002
CAL = ("paid_dec", "re_up", "d_assets")
CAL_JA = {"paid_dec": "払込資本を減らした割合", "re_up": "利益剰余金が増えた割合",
          "d_assets": "総資産変化の中央値(年率)"}
REAL_GROUP = {"c2": "class 2 → 回復", "c1": "class 1 → class 0"}
PAID_GROUP = {"c2": "再建群(class 2 → 回復)", "c1": "再建群(class 1 → class 0)"}
# 参考パレット(dataviz の既定 categorical の 1〜3、全組で検証済みと記載のもの)とインク
COLORS = {"l2": "#2a78d6", "l1s": "#eb6834", "l1a": "#1baf7a"}
MARKERS = {"l2": "o", "l1s": "s", "l1a": "^"}
INK, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"


def real_firms(cls: str, names: tuple[str, ...]) -> pd.DataFrame:
    """実データの企業ごとの値(年率)。払込資本はステージ56 の annual_d_paid。"""
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    p = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})
    codes = pairs[pairs["group"] == REAL_GROUP[cls]]["edinetCode"].tolist()
    a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    b = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
    a0 = a["cur_total_assets"]
    d = pd.DataFrame({n: (b[f"cur_{n}"].fillna(0) - a[f"cur_{n}"].fillna(0)) / a0 for n in names})
    out = pd.DataFrame({"re_up": (d[_RE] / REAL_Y > THR).astype(float),
                        "d_assets": (b["cur_total_assets"] / a0 - 1.0) / REAL_Y,
                        "n_changed": (d.abs() / REAL_Y > DELTA_Y).sum(axis=1).astype(float)})
    paid = pd.read_csv("results/stage56/paidin.csv", dtype={"edinetCode": str})
    paid = paid[paid["group"] == PAID_GROUP[cls]].set_index("edinetCode")["annual_d_paid"]
    out["paid_dec"] = (paid.reindex(out.index) < -THR).astype(float).where(
        paid.reindex(out.index).notna())
    return out


def real_targets(cls: str, names: tuple[str, ...]) -> tuple[dict[str, float], dict[str, float],
                                                           float]:
    f = real_firms(cls, names)
    r = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
         "d_assets": float(f["d_assets"].median())}
    rng = np.random.default_rng(0)
    boots: dict[str, list[float]] = {m: [] for m in CAL}
    for _ in range(2000):
        s = f.iloc[rng.integers(0, len(f), len(f))]
        boots["paid_dec"].append(float(s["paid_dec"].mean()))
        boots["re_up"].append(float(s["re_up"].mean()))
        boots["d_assets"].append(float(s["d_assets"].median()))
    se = {m: float(np.std(v, ddof=1)) for m, v in boots.items()}
    return r, se, float(f["n_changed"].median())


def gen_firms(cls: str, s: str, names: tuple[str, ...], members: dict[str, np.ndarray]
              ) -> pd.DataFrame:
    g = e51.load_gen(cls, s)
    g = g[g["seed"] == 0]
    x0 = g[[f"x0_{n}" for n in names]].to_numpy()
    x1 = g[[f"x1_{n}" for n in names]].to_numpy()
    a0, a1 = x0 @ members["jppfs_cor_Assets"], x1 @ members["jppfs_cor_Assets"]
    pi = [names.index(_CS), names.index(_CSUR)]
    d = x1 - x0
    return pd.DataFrame({
        "paid_dec": ((d[:, pi].sum(axis=1)) / a0 / GEN_Y < -THR).astype(float),
        "paid_dec_norm": ((x1[:, pi].sum(axis=1) / a1 - x0[:, pi].sum(axis=1) / a0) / GEN_Y
                          < -THR).astype(float),
        "re_up": (d[:, names.index(_RE)] / a0 / GEN_Y > THR).astype(float),
        "d_assets": (a1 / a0 - 1.0) / GEN_Y,
        "n_changed": (np.abs(d) / a0[:, None] / GEN_Y > DELTA_Y).sum(axis=1).astype(float)},
        index=g["docID"].to_numpy())


def loss(g: dict[str, float], r: dict[str, float], w: dict[str, float]) -> float:
    return float(sum(w[m] * abs(g[m] - r[m]) for m in CAL))


def plot(cls: str, grid: pd.DataFrame, r: dict[str, float], n_real: float,
         lam_star: dict[str, float]) -> None:
    panels = (("paid_dec", "Share reducing paid-in capital", r["paid_dec"]),
              ("re_up", "Share with retained earnings up", r["re_up"]),
              ("d_assets", "Median total-asset change (annual)", r["d_assets"]),
              ("rank_corr", "Signed rank correlation (validation)", None),
              ("n_changed", "Accounts changed, median (validation)", n_real),
              ("viol_firms", "Firms violating any of 15 checks", 0.0))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    for ax, (col, title, ref) in zip(axes.ravel(), panels, strict=True):
        for d in DISTS:
            sub = grid[(grid["cls"] == cls) & (grid["dist"] == d)].sort_values("lam_tilde")
            ax.plot(sub["lam_tilde"], sub[col], color=COLORS[d], marker=MARKERS[d],
                    markersize=5, linewidth=2, label=DIST_EN[d])
            ls = sub[sub["lam_tilde"] == lam_star[d]]
            ax.plot(ls["lam_tilde"], ls[col], marker=MARKERS[d], markersize=11,
                    markerfacecolor="none", markeredgecolor=COLORS[d], linewidth=0)
        if ref is not None:
            ax.axhline(ref, color=INK, linestyle="--", linewidth=1.2)
            note = " required (0)" if col == "viol_firms" else " real data"
            ax.text(0.003, ref, note, color=INK, fontsize=8, va="bottom")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(True, color=GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("lambda tilde (log scale)", color=INK, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.955))
    name = "class 2 (n = 55)" if cls == "c2" else "class 1 (n = 128)"
    fig.suptitle(f"Trade-off over lambda, {name}. Large hollow marker = calibrated lambda*",
                 y=0.995, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"tradeoff_{cls}.png", dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    e51._M = DATA / "metrics"
    e51._GEN = DATA / "gen"
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / "l2_l1").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))
    rows: list[dict[str, Any]] = []
    reals: dict[str, Any] = {}
    firms: dict[tuple[str, str], pd.DataFrame] = {}
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    for cls in ("c2", "c1"):
        r, se, n_real = real_targets(cls, names)
        w = {m: 1.0 / se[m] for m in CAL}
        reals[cls] = {"r": r, "se": se, "w": w, "n_changed": n_real}
        for d in DISTS:
            for lt in LAMS:
                s = f"{d}_l{lt:g}"
                f = gen_firms(cls, s, names, members)
                f = f[~f.index.isin(e51.EXCL[cls])]
                firms[(cls, s)] = f
                gv = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                      "d_assets": float(f["d_assets"].median())}
                vi = ctx[cls].viol
                vfirms = vi[vi["setting"] == s]["docID"].nunique()
                comp = ctx[cls].comp
                cr = comp[comp["version"] == s]
                rows.append({"cls": cls, "dist": d, "lam_tilde": lt, "setting": s, **gv,
                             "paid_dec_norm": float(f["paid_dec_norm"].mean()),
                             "n_changed": float(f["n_changed"].median()),
                             "viol_rows": sum(ctx[cls].viol_items(s).values()),
                             "viol_firms": int(vfirms),
                             "net_asset_improvement": float(cr["net_asset_improvement"].median()),
                             "resolved": float(cr.loc[cr["insolvent_start"].astype(bool),
                                                      "insolvency_resolved"].mean()),
                             "rank_corr": float(corr(ctx[cls].wide[s], ctx[cls].ref)),
                             "loss": loss(gv, r, w),
                             "loss_equal": loss(gv, r, {m: 1.0 for m in CAL})})
    grid = pd.DataFrame(rows)
    qp_rows = []
    for f in sorted(OUT.glob("qp_*of8.csv")):
        q = pd.read_csv(f)
        q["cls"] = f.name.split("_")[1]
        qp_rows.append(q)
    qp = pd.concat(qp_rows)[["cls", "setting", "first_term_med", "move_cost_med"]]
    grid = grid.merge(qp, on=["cls", "setting"], how="left")
    grid.to_csv(OUT / "grid.csv", index=False)

    # ---------------- A.1・A.4 ----------------
    star: dict[str, dict[str, float]] = {}
    lines = ["# ステージ61 A.1・A.4: λ の較正と検証", "",
             "較正の乖離 L = Σ_m w_m |g_m − r_m|、w_m = 1 / SE_m(実データの企業単位のブートスト"
             "ラップ標準誤差)。"
             "λ̃* はグリッドの 10 点から、class・距離ごとに選んだ。"
             "年率は生成 ÷ 5、実データ ÷ 2。", ""]
    for cls in ("c2", "c1"):
        rv = reals[cls]
        lines += [f"## {cls}", "",
                  "実データ: " + "、".join(f"{CAL_JA[m]} {rv['r'][m]:.3f}(SE {rv['se'][m]:.3f})"
                                           for m in CAL)
                  + f"、変更された科目数の中央値 {rv['n_changed']:.0f}。", ""]
        star[cls] = {}
        for d, dlab in DISTS.items():
            sub = grid[(grid["cls"] == cls) & (grid["dist"] == d)].sort_values("lam_tilde")
            best = sub.loc[sub["loss"].idxmin()]
            best_eq = sub.loc[sub["loss_equal"].idxmin()]
            star[cls][d] = float(best["lam_tilde"])
            # λ̃* の不確かさ: 生成と実データの企業を同時に復元抽出
            rng = np.random.default_rng(0)
            rf = real_firms(cls, names)
            picks = []
            for _ in range(1000):
                ri = rf.iloc[rng.integers(0, len(rf), len(rf))]
                rr = {"paid_dec": float(ri["paid_dec"].mean()), "re_up": float(ri["re_up"].mean()),
                      "d_assets": float(ri["d_assets"].median())}
                idx = None
                losses = []
                for lt in LAMS:
                    f = firms[(cls, f"{d}_l{lt:g}")]
                    if idx is None:
                        idx = rng.integers(0, len(f), len(f))
                    fb = f.iloc[idx]
                    gb = {"paid_dec": float(fb["paid_dec"].mean()),
                          "re_up": float(fb["re_up"].mean()),
                          "d_assets": float(fb["d_assets"].median())}
                    losses.append(loss(gb, rr, rv["w"]))
                picks.append(LAMS[int(np.argmin(losses))])
            share = pd.Series(picks).value_counts(normalize=True)
            lines += [f"### {dlab}: **λ̃* = {best['lam_tilde']:g}**(等重みでは"
                      f" {best_eq['lam_tilde']:g})", "",
                      "λ̃* が選ばれた割合(1,000 回): "
                      + "、".join(f"{k:g} {v:.0%}" for k, v in share.sort_index().items()), "",
                      "| λ̃ | L | 払込資本↓(名目) | (正規化) | RE↑ | 総資産(年率) | 違反(社) "
                      "| 純資産の改善 | 解消率 | 順位相関(検証) | 変更科目数(検証) | 目標への距"
                      "離 | 移動コスト |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for rr_ in sub.itertuples():
                mark = " **←**" if rr_.lam_tilde == best["lam_tilde"] else ""
                lines.append(f"| {rr_.lam_tilde:g}{mark} | {rr_.loss:.2f} | {rr_.paid_dec:.0%} "
                             f"| {rr_.paid_dec_norm:.0%} | {rr_.re_up:.0%} | {rr_.d_assets:+.4f} "
                             f"| {rr_.viol_firms} | {rr_.net_asset_improvement:.3f} "
                             f"| {rr_.resolved:.2f} | {rr_.rank_corr:+.3f} | {rr_.n_changed:.0f} "
                             f"| {rr_.first_term_med:.3f} | {rr_.move_cost_med:.3f} |")
            # 検証(criteria §2)
            dist_rc = (1 - sub["rank_corr"]).to_numpy()
            dist_nc = (sub["n_changed"] - rv["n_changed"]).abs().to_numpy()
            pos = int(np.where(sub["lam_tilde"].to_numpy() == best["lam_tilde"])[0][0])
            one = int(np.where(sub["lam_tilde"].to_numpy() == 1.0)[0][0])

            def top_half(v: np.ndarray, i: int) -> bool:
                return bool((v < v[i]).sum() < len(v) / 2)

            va = {"順位相関": (top_half(dist_rc, pos), dist_rc[pos] <= dist_rc[one]),
                  "変更された科目数": (top_half(dist_nc, pos), dist_nc[pos] <= dist_nc[one])}
            star[cls][f"{d}_valid"] = float(all(a and (b or best["lam_tilde"] == 1.0)
                                                for a, b in va.values()))
            lines += ["", "**検証(較正に使わない指標)**: "
                      + "、".join(f"{k}: 上位半分 {'○' if a else '×'}・λ̃ = 1 以上に近い "
                                 f"{'—' if best['lam_tilde'] == 1.0 else ('○' if b else '×')}"
                                 for k, (a, b) in va.items()), ""]
    (OUT / "calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for cls in ("c2", "c1"):
        plot(cls, grid, reals[cls]["r"], reals[cls]["n_changed"],
             {d: star[cls][d] for d in DISTS})

    # ---------------- A.5 ----------------
    rng = np.random.default_rng(0)
    lines = ["# ステージ61 A.5: 距離の設計による差(各距離の λ̃* で)", "",
             "差は企業単位のペアの比較 [95% 区間](払込資本はブートストラップ 2,000 回の平均の差"
             "、主指標は"
             "ステージ51 の定義)。記号 = 純資産の改善・解消率・順位相関。", ""]
    verdict: dict[str, bool] = {}
    for cls in ("c2", "c1"):
        lines += [f"## {cls}", "",
                  "| 比較 | λ̃* | 払込資本↓ の差 | 記号 | 純資産の改善 | 順位相関 | L の差 |",
                  "|---|---|---|---|---|---|---:|"]
        absorbed = True
        for a, b in (("l1a", "l2"), ("l1s", "l2"), ("l1a", "l1s")):
            sa, sb = f"{a}_l{star[cls][a]:g}", f"{b}_l{star[cls][b]:g}"
            fa, fb = firms[(cls, sa)], firms[(cls, sb)].reindex(firms[(cls, sa)].index)
            dd = (fa["paid_dec"] - fb["paid_dec"]).to_numpy()
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            c = ctx[cls].compare(sa, sb, rng)
            sym = " ".join({1: "+", -1: "−", 0: "0"}[int(c[k][3])]
                           for k in ("純資産の改善", "解消率", "順位相関"))
            la = float(grid[(grid["cls"] == cls) & (grid["setting"] == sa)]["loss"].iloc[0])
            lb = float(grid[(grid["cls"] == cls) & (grid["setting"] == sb)]["loss"].iloc[0])
            sig = (lo > 0 or hi < 0) or sym != "0 0 0" or abs(la - lb) >= 1.0
            absorbed &= not sig
            lines.append(f"| {DISTS[a]} − {DISTS[b]} | {star[cls][a]:g} / {star[cls][b]:g} "
                         f"| {dd.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {sym} "
                         f"| {e51.fc(c['純資産の改善'])} | {e51.fc(c['順位相関'])} |"
                         f" {la - lb:+.2f} |")
        verdict[cls] = absorbed
        lines += ["", f"**距離の設計は λ に吸収されるか({cls})**:"
                      f" {'はい' if absorbed else 'いいえ'}。", ""]
        lam0 = star[cls][PRIMARY_DIST]
        lines += [f"参考: 主の距離の λ̃*({lam0:g})を共通に使った場合", "",
                  "| 距離 | 払込資本↓ | RE↑ | 総資産(年率) | 違反(社) | 純資産の改善 | 順位相関 "
                  "| L |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for d, dlab in DISTS.items():
            rr_ = grid[(grid["cls"] == cls) & (grid["setting"] == f"{d}_l{lam0:g}")].iloc[0]
            lines.append(f"| {dlab} | {rr_['paid_dec']:.0%} | {rr_['re_up']:.0%} "
                         f"| {rr_['d_assets']:+.4f} | {rr_['viol_firms']} "
                         f"| {rr_['net_asset_improvement']:.3f} | {rr_['rank_corr']:+.3f} "
                         f"| {rr_['loss']:.2f} |")
        lines.append("")
    (OUT / "distances.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"star": star, "reals": reals,
                                                  "absorbed": verdict}, ensure_ascii=False,
                                                 indent=1), encoding="utf-8")
    print((OUT / "calibration.md").read_text(encoding="utf-8")[:4000])
    print((OUT / "distances.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
