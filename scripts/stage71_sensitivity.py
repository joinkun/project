#!/usr/bin/env python
"""ステージ71 の感度(結果を見てから追加した。判定には使わない。lightgbm・sklearn のプロセス)。

1. 項目 5 の違反の大きさ: 𝓕⁺ の負債の上限で、終点は上限ちょうどに止まる。チェックは許容なしの
   「L(y) / L(x0) > 2」なので、10⁻¹³〜10⁻¹⁰ の数値誤差が違反になる。違反ごとに上限の超過量を測る。
2. 超過が QP の受け入れ許容(10⁻⁷ × max(1, |x|))以下の項目 5 を違反としない場合の、λ̃*_0 と主指標。
3. 終点のゼロ: QP の解は 0 のはずの科目に 10⁻¹² 程度の値を残す。|y| < 10⁻⁶(不活性の判定と同じ)を
   0 とみなした科目数と、0 に丸めた終点の識別 AUC。

生成はやり直さない。出力: ``results/stage71/sensitivity.md``。

    uv run python scripts/stage71_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from stage71_evaluate import LABELS, LAMS, S71, endpoints
from tables_stage69 import S67, Src, ctx, metrics, setup_excl, versus
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.evaluation.identification import identification_auc_repeated

OUT = Path("results/stage71")
ITEM5 = "5_負債が2倍超"
ZERO = 1e-6


def excess(src: Src, cls: str, names: tuple[str, ...], members: dict[str, np.ndarray]
           ) -> pd.Series:
    """企業ごとの負債の上限の超過量 L(y) − max(2 L(x0), 0.01 A(x0))(seed 0)。"""
    e51._GEN = src.gen
    g = e51.load_gen(cls, src.setting)
    g = g[g["seed"] == 0].set_index("docID")
    x0 = g[[f"x0_{n}" for n in names]].to_numpy(float)
    y = g[[f"x1_{n}" for n in names]].to_numpy(float)
    la, aa = members["jppfs_cor_Liabilities"], members["jppfs_cor_Assets"]
    cap = np.maximum(2.0 * (x0 @ la), 0.01 * (x0 @ aa))
    tol = 1e-7 * np.maximum(1.0, np.abs(np.concatenate([x0, y], axis=1)).max(axis=1))
    return pd.Series((y @ la - cap) - tol, index=g.index)


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    setup_excl(names)
    x, lab, _d, _ = load_tilde(names)
    xt = x[lab == 0]
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    lines = ["# ステージ71 の感度(結果を見てから追加、判定には使わない)", "",
             "## 1. 項目 5 の違反の大きさ", "",
             "| class | 項目 5 の違反(設定 × λ̃ × 企業) | 上限の超過量の最大 "
             "| QP の許容を超えるもの |",
             "|---|---:|---:|---:|"]
    for cls in ("c2", "c1"):
        c = ctx(Src(*S71, "a2__l1", "1"), cls, names)
        v = c.viol[c.viol["check"] == ITEM5].drop_duplicates(["setting", "docID"])
        exc = []
        for s_, g_ in v.groupby("setting"):
            e = excess(Src(*S71, str(s_), ""), cls, names, members)
            exc += [float(e.loc[d]) + 1e-7 for d in g_["docID"]]
        over = sum(1 for z in exc if z > 1e-7)
        lines.append(f"| {cls} | {len(exc)} | {max(exc) if exc else 0:.1e} | {over} |")
    lines += ["", "超過量は L(y) − max(2 L(x0), 0.01 A(x0))(総資産比)。チェックは許容なし、QP は "
              "10⁻⁷ × max(1, |x|) までの違反を受け入れる。", "",
              "## 2. QP の許容内の項目 5 を違反としない場合", "",
              "| 設定 | λ̃*_0(事前の規則 → 許容内を除く) | 違反 c2 / c1 | 純資産の改善 c2 / c1 "
              "| 差 c2 | 差 c1 | 記号 c2 / c1 |", "|---|---|---|---|---|---|---|"]
    base = Src(*S67, "s1__l1", "1")
    strict: dict[str, float] = json.loads(
        (OUT / "summary.json").read_text(encoding="utf-8"))["star"]
    tol_star: dict[str, float] = {}
    for st, lab_s in LABELS.items():
        best: tuple[float | None, float] = (None, np.inf)
        for lt in LAMS:
            src = Src(*S71, f"{st}__l{lt:g}", f"{lt:g}")
            c = ctx(src, "c1", names)
            v = c.viol[c.viol["setting"] == src.setting].drop_duplicates(["docID", "check"])
            e = excess(src, "c1", names, members)
            beyond = set(e.index[e > 0])
            real = v[(v["check"] != ITEM5) | v["docID"].isin(beyond)]
            f = gen_firms("c1", src.setting, names, members)
            f = f[~f.index.isin(e51.EXCL["c1"])]
            lv = loss({"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                       "d_assets": float(f["d_assets"].median())}, r1, w1)
            if real["docID"].nunique() == 0 and lv < best[1]:
                best = (lt, lv)
        if best[0] is None:
            lines.append(f"| {lab_s} | {strict[st]:g} → なし | — | — | — | — | — |")
            continue
        tol_star[st] = best[0]
        src = Src(*S71, f"{st}__l{best[0]:g}", f"{best[0]:g}")
        mm = {cls: metrics(src, cls, names, members) for cls in ("c2", "c1")}
        vv = {cls: versus(src, base, cls, names, np.random.default_rng(0)) for cls in ("c2", "c1")}

        def sym(s: list[int]) -> str:
            return " ".join({1: "+", -1: "−", 0: "0"}[int(z)] for z in s)

        def ci(t: tuple[float, float, float]) -> str:
            return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
        lines.append(f"| {lab_s} | {strict[st]:g} → {best[0]:g} | {mm['c2']['viol']} / "
                     f"{mm['c1']['viol']} | {mm['c2']['nai']:.3f} / {mm['c1']['nai']:.3f} "
                     f"| {ci(vv['c2']['nai'])} | {ci(vv['c1']['nai'])} | {sym(vv['c2']['sign'])} / "
                     f"{sym(vv['c1']['sign'])} |")
        print(lines[-1], flush=True)
    lines += ["", "違反の数は 15 項目のチェックそのまま(許容内の項目 5 を含む)。", "",
              f"## 3. 終点のゼロ(|y| < {ZERO:g} を 0 とみなす)と、0 に丸めた終点の識別 AUC", "",
              f"健全企業のちょうど 0 の科目数は {float((xt == 0).sum(axis=1).mean()):.2f}。λ̃ は"
              "事前の規則の λ̃*_0。", "",
              "| class | 設定 | 0 とみなす科目数 | 識別 AUC(丸め後、LightGBM / LogReg) |",
              "|---|---|---:|---|"]
    for cls in ("c2", "c1"):
        rows = [("1 従来(ステージ69)", base)] + [
            (lab_s, Src(*S71, f"{st}__l{strict[st]:g}", f"{strict[st]:g}"))
            for st, lab_s in LABELS.items()]
        for lab_s, src in rows:
            y = endpoints(src, cls, names)
            yz = np.where(np.abs(y) < ZERO, 0.0, y)
            r = identification_auc_repeated(yz, xt, 3)
            lines.append(f"| {cls} | {lab_s} | {(yz == 0).sum(axis=1).mean():.2f} "
                         f"| {r['lightgbm'].mean():.3f} / {r['logreg'].mean():.3f} |")
            print(lines[-1], flush=True)
    lines.append("")
    (OUT / "sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
