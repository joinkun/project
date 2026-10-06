#!/usr/bin/env python
"""ステージ57 A: 再建群の科目別の変化量と、それから作る重み(集計のみ、生成なし)。

規則は ``results/stage57/criteria.md``(集計の前に記録)。Δx_a / A = (x_a(2024) − x_a(2022)) /
総資産(2022)、名目額。代表値は平均(変化 0 も含む)。w_a = m_現金 / m_a(上限 100)、方向別は
max(±Δ, 0) の平均。非操作 3 科目は 0。

出力: ``results/stage57/changes.md``・``weights.md``・``weights.csv``、
``conf/accounts/costs_recovery.yaml``(方向なし)・``costs_recovery_dir.yaml``(方向別)。

    uv run python scripts/changes_stage57.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config

_PANEL = Path("data/interim/stage23/panel/panel.csv")
CASH = "jppfs_cor_CashAndDeposits"
CAP = 100.0
QS = (5, 25, 50, 75, 95)


def deltas(codes: list[str], names: tuple[str, ...]) -> pd.DataFrame:
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    b = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
    out = pd.DataFrame({n: (b[f"cur_{n}"].fillna(0) - a[f"cur_{n}"].fillna(0))
                        / a["cur_total_assets"] for n in names}, index=codes)
    return out


def weights(d: np.ndarray, icash: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(方向なし, 増加, 減少) の重み。d: (企業, 科目)。"""
    def w(m: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.minimum(np.where(m > 0, m[icash] / np.where(m > 0, m, 1.0), CAP),
                                     CAP)
        return out

    return (w(np.abs(d).mean(axis=0)), w(np.maximum(d, 0).mean(axis=0)),
            w(np.maximum(-d, 0).mean(axis=0)))


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage57"))
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    cc = load_cost_config(Path("conf/accounts/costs_equal.yaml"))
    na = set(cc.non_actionable)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    grp = pd.read_csv("results/stage51/asset_change.csv", dtype={"edinetCode": str})
    by = pairs.groupby("group")["edinetCode"].apply(list)
    sets = {"72 社": pairs["edinetCode"].tolist(),
            "15 社(class 2 → 回復)": by["class 2 → 回復"],
            "57 社(class 1 → class 0)": by["class 1 → class 0"],
            "非再建群 30 社": grp[grp["group"] == "非再建群(class 2 → class 2)"][
                "edinetCode"].tolist()}
    dd = {k: deltas(v, names) for k, v in sets.items()}
    icash = names.index(CASH)

    # ---------------- A.2・A.3・A.4 ----------------
    lines = ["# ステージ57 A: 再建群の科目別の変化量(2 年、Δx / 総資産(2022)、名目額)", "",
             "**2 年の変化**で、生成の 5 年とは期間が違う(限界)。|Δ| の分布は企業ごと。", ""]
    for k in ("72 社", "15 社(class 2 → 回復)", "57 社(class 1 → class 0)"):
        d = dd[k]
        lines += [f"## A.2 {k}", "",
                  "| 科目 | n | " + " | ".join(f"\\|Δ\\| p{q}" for q in QS)
                  + " | Δ の中央値 | 増加 / 減少 / 変化なし |",
                  "|---|---:|" + "---:|" * (len(QS) + 1) + "---|"]
        for n in names:
            v = d[n].to_numpy()
            av = np.abs(v)
            lines.append(f"| {ja.get(n, n)} | {len(v)} | "
                         + " | ".join(f"{np.percentile(av, q):.4f}" for q in QS)
                         + f" | {np.median(v):+.4f} | {(v > 0).sum()} / {(v < 0).sum()} /"
                           f" {(v == 0).sum()} |")
        lines.append("")
    rec, non = dd["72 社"], dd["非再建群 30 社"]
    lines += ["## A.3 再建群 対 非再建群(\\|Δ\\| の平均と中央値)", "",
              "| 科目 | 再建群 72 社 平均 | 非再建群 30 社 平均 | 比(再建 / 非再建) | 再建群 中"
              "央値 | 非再建群 中央値 |",
              "|---|---:|---:|---:|---:|---:|"]
    ratio = {}
    for n in names:
        mr, mn = np.abs(rec[n]).mean(), np.abs(non[n]).mean()
        ratio[n] = mr / mn if mn > 0 else np.inf
        lines.append(f"| {ja.get(n, n)} | {mr:.4f} | {mn:.4f} | {ratio[n]:.2f} "
                     f"| {np.abs(rec[n]).median():.4f} | {np.abs(non[n]).median():.4f} |")
    top = sorted(((v, n) for n, v in ratio.items() if np.isfinite(v)), reverse=True)[:5]
    lines += ["", "再建群で非再建群より大きく動く科目(平均の比の上位 5): "
              + "、".join(f"{ja.get(n, n)}({v:.2f})" for v, n in top), "",
              "## A.4 変化がちょうど 0 の企業(72 社)", "",
              "| 科目 | 変化なし | 割合 | \\|Δ\\| の中央値が 0 |", "|---|---:|---:|---|"]
    for n in names:
        z = int((rec[n] == 0).sum())
        lines.append(f"| {ja.get(n, n)} | {z} | {z / len(rec):.0%} "
                     f"| {'はい' if np.abs(rec[n]).median() == 0 else ''} |")
    (args.out / "changes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- A.5・A.6 重み ----------------
    d72 = rec.to_numpy()
    w0, wp, wm = weights(d72, icash)
    rng = np.random.default_rng(0)
    boots = np.array([weights(d72[rng.integers(0, len(d72), len(d72))], icash)[0]
                      for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    av = np.abs(d72)

    def ref(m: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.minimum(np.where(m > 0, m[icash] / np.where(m > 0, m, 1.0), CAP), CAP)
        return out

    med = ref(np.median(av, axis=0))
    p75 = ref(np.percentile(av, 75, axis=0))
    nzmed = ref(np.array([np.median(av[av[:, j] > 0, j]) if (av[:, j] > 0).any() else 0.0
                          for j in range(len(names))]))
    cond_p = ref(np.array([d72[d72[:, j] > 0, j].mean() if (d72[:, j] > 0).any() else 0.0
                           for j in range(len(names))]))
    w15, *_ = weights(dd["15 社(class 2 → 回復)"].to_numpy(), icash)
    w57, *_ = weights(dd["57 社(class 1 → class 0)"].to_numpy(), icash)
    nact = np.array([n in na for n in names])
    for w in (w0, wp, wm, lo, hi, med, p75, nzmed, cond_p, w15, w57):
        w[nact] = 0.0
    s55 = pd.read_csv("results/stage55/weights.csv").set_index("account")["w"]
    rho = pd.read_csv("results/stage32/persistence.csv").set_index("account")["rho1"]
    freq = load_cost_config(Path("conf/accounts/costs_frequency.yaml"))
    fq = pd.Series({s.name: (s.cost_increase + s.cost_decrease) / 2 for s in freq.accounts})
    wdf = pd.DataFrame({"account": names, "ja": [ja.get(n, n) for n in names], "w": w0,
                        "w_lo": lo, "w_hi": hi, "w_plus": wp, "w_minus": wm, "w_median": med,
                        "w_p75": p75, "w_nonzero_median": nzmed, "w_plus_conditional": cond_p,
                        "w_15": w15, "w_57": w57, "w_stage55": s55.reindex(names).to_numpy(),
                        "rho1": rho.reindex(names).to_numpy(),
                        "w_frequency": fq.reindex(names).to_numpy()})
    wdf.to_csv(args.out / "weights.csv", index=False)
    act = wdf[~nact]
    lines = ["# ステージ57 A.5〜A.7: 再建群の変化量からの重み", "",
             "w_a = m_現金 / m_a、m_a = 72 社の |Δx_a / A| の平均(変化 0 も含む)。上限 100。非"
             "操作 3 科目は 0。"
             "[ ] は企業の復元抽出 2,000 回の 95% 区間。w+ / w− = 増加・減少の向きの重み(max(±Δ,"
             " 0) の平均から)。"
             "参考は代表値を変えた版(生成に使わない)。", "",
             "| 科目 | **w**(平均) [95%] | w+ | w− | 参考: 中央値 | p75 | 非ゼロの中央値 | 15 社"
             " | 57 社 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in act.sort_values("w").itertuples():
        lines.append(f"| {r.ja} | **{r.w:.2f}** [{r.w_lo:.2f}, {r.w_hi:.2f}] | {r.w_plus:.2f} "
                     f"| {r.w_minus:.2f} | {r.w_median:.2f} | {r.w_p75:.2f} |"
                     f" {r.w_nonzero_median:.2f} "
                     f"| {r.w_15:.2f} | {r.w_57:.2f} |")
    r1557 = stats.spearmanr(act["w_15"], act["w_57"]).statistic
    lines += ["", f"15 社と 57 社の重みの順位相関: {r1557:+.3f}(29 科目)。", "",
              "## A.7 過去の推定との順位相関(29 科目、Spearman)", "",
              "| 比べる量 | ρ | p |", "|---|---:|---:|"]
    comps = (("w_stage55", "ステージ55 の学習された重み"), ("rho1", "持続性 ρ_a(ステージ32)"),
             ("w_frequency", "施策の頻度の重み(ステージ32 版 2)"),
             ("w_median", "参考: 中央値で作った重み"),
             ("w_nonzero_median", "参考: 非ゼロの中央値で作った重み"))
    for col, lab in comps:
        r = stats.spearmanr(act["w"], act[col], nan_policy="omit")
        lines.append(f"| {lab} | {r.statistic:+.3f} | {r.pvalue:.2g} |")
    lines.append("")
    (args.out / "weights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # costs yaml
    base = yaml.safe_load(Path("conf/accounts/costs_equal.yaml").read_text(encoding="utf-8"))
    for fname, wpos, wneg, label in (("costs_recovery", w0, w0, "方向なし"),
                                     ("costs_recovery_dir", wp, wm, "方向別")):
        doc = {**base, "name": fname, "basis_costs": []}
        for b in base["basis_costs"]:
            j = names.index(b["name"])
            doc["basis_costs"].append({**b, "cost_increase": float(wpos[j]) if not nact[j] else 1.0,
                                       "cost_decrease": float(wneg[j]) if not nact[j] else 1.0,
                                       "source": f"ステージ57: 再建群 72 社の |Δ/A| の平均から"
                                                 f"({label})"})
        Path(f"conf/accounts/{fname}.yaml").write_text(
            "# ステージ57: scripts/changes_stage57.py が生成(手で編集しない)。非操作科目は neare"
            "st_na0 で 0。\n"
            + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
