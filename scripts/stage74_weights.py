#!/usr/bin/env python
"""ステージ74: リークのないプールのコスト(差分・対数変化率)と下限 x̲(numpy・pandas だけのプロセス)。

規則は ``results/stage74/criteria.md`` §2・§3。プールは 57 社(class 1 → class 0)、15 社(class 2 →
回復)、10 社(15 社から class 1 の順位相関の参照 = ステージ38 の T2 の組と重なる 5 社を除く)。

- 差分のコスト: ステージ70 と同じ ``stage70_weights.costs``(c = 1/s、残余は 10³ で上限)
- 対数変化率のコスト: 名目額で両期とも非ゼロ・同符号の科目の g = log(|X_2024| / |X_2022|)、
  c_grow = 1/median(g > 0)、c_shrink = 1/median(−g, g < 0)、n < 5 は同じプールの中央値で代替
- 下限 x̲_a: パネルの 2024 年度の全企業で、総資産比 |x_a| が非ゼロのものの 10 パーセンタイル(≥ 10⁻⁴)

出力: ``data/interim/stage74/weights.json``、``results/stage74/costs.md``。

    uv run python scripts/stage74_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from prep_stage59 import N_MIN, deltas
from stage70_weights import FROZEN, costs
from targets_stage68 import sources
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

OUT = Path("results/stage74")
DATA = Path("data/interim/stage74")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
FLOOR_Q, FLOOR_MIN = 10.0, 1e-4
SHOW = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus", "jppfs_cor_CashAndDeposits",
        "jppfs_cor_PropertyPlantAndEquipment", "short_term_loans", "long_term_loans",
        "accounts_payable_other", "non_controlling_interests", "jppfs_cor_RetainedEarnings")


def pools() -> dict[str, list[str]]:
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    p57 = pairs[pairs["group"] == "class 1 → class 0"]["edinetCode"].tolist()
    p15 = pairs[pairs["group"] == "class 2 → 回復"]["edinetCode"].tolist()
    r38 = pd.read_parquet("data/interim/stage38/real_deltas.parquet")
    t2 = set(r38[(r38["label_prev"] == 1) & (r38["label_cur"] == 0)]["edinetCode"].astype(str))
    p10 = [c for c in p15 if c not in t2]
    if len(p10) != 10:
        raise ValueError(f"10 社のプールが {len(p10)} 社")
    return {"57": p57, "15": p15, "10": p10}


def log_costs(codes: list[str], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """対数変化率の尺度とコスト(行 0 = 絶対値の拡大、行 1 = 縮小)。"""
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    b = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
    xa = a[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float)
    xb = b[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float)
    ok = (xa != 0) & (xb != 0) & (np.sign(xa) == np.sign(xb))
    g = np.where(ok, np.log(np.abs(np.where(ok, xb, 1.0)) / np.abs(np.where(ok, xa, 1.0))), 0.0)
    k = len(names)
    n = np.zeros((2, k), dtype=int)
    s = np.full((2, k), np.nan)
    for j in range(k):
        for r, v in enumerate((g[ok[:, j] & (g[:, j] > 0), j], -g[ok[:, j] & (g[:, j] < 0), j])):
            n[r, j] = len(v)
            if len(v):
                s[r, j] = float(np.median(v))
    frozen = np.array([nm in FROZEN for nm in names])
    with np.errstate(divide="ignore"):
        c = 1.0 / s
    good = (n >= N_MIN) & ~frozen[None, :] & np.isfinite(c)
    fill = float(np.median(c[good]))
    c = np.where(good, c, fill)
    c[:, frozen] = 0.0
    return {"n": n, "s": s, "c": c, "fill": np.array(fill),
            "n_sub": np.array(int(((n < N_MIN) & ~frozen[None, :]).sum()))}


def main() -> None:  # noqa: PLR0915
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    pl = pools()
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    p24 = p[p["t"] == 2024]
    x24 = p24[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float) / p24[
        "cur_total_assets"].to_numpy(float)[:, None]
    floor = np.array([max(float(np.percentile(np.abs(v[v != 0]), FLOOR_Q)) if (v != 0).any()
                          else FLOOR_MIN, FLOOR_MIN) for v in x24.T])
    x, lab, docs, _ = load_tilde(names)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    docs_a = np.array(docs)
    src_codes = {code_of.get(str(d), "") for cls in ("c2", "c1")
                 for d in docs_a[sources(cls, lab, docs_a)]}
    out: dict[str, object] = {"names": list(names), "floor": floor.tolist(), "pools": {},
                              "loo": {}}
    res: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for g, codes in pl.items():
        ab = costs(deltas(codes, names), names)
        rl = log_costs(codes, names)
        res[g] = {"abs": ab, "rel": rl}
        out["pools"][g] = {"codes": codes, "abs_plus": ab["c"][0].tolist(),  # type: ignore[index]
                           "abs_minus": ab["c"][1].tolist(), "rel_grow": rl["c"][0].tolist(),
                           "rel_shrink": rl["c"][1].tolist()}
        for code in sorted(src_codes & set(codes)):
            rest = [c for c in codes if c != code]
            ab2, rl2 = costs(deltas(rest, names), names), log_costs(rest, names)
            out["loo"][f"{g}:{code}"] = {  # type: ignore[index]
                "abs_plus": ab2["c"][0].tolist(), "abs_minus": ab2["c"][1].tolist(),
                "rel_grow": rl2["c"][0].tolist(), "rel_shrink": rl2["c"][1].tolist()}
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / "weights.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    lines = ["# ステージ74 (a): リークのないプールのコスト", "",
             "差分 = 総資産比の構成の差(ステージ70 の c = 1/s)。対数 = 名目額の対数変化率"
             "(両期とも非ゼロ・同符号)、c_grow / c_shrink は絶対値の拡大 / 縮小。資本金のように"
             "正の科目では、拡大 = 増加(c⁺)、縮小 = 減少(c⁻)。", "",
             "| プール | 社数 | n < 5 で代替した方向(差分 / 対数、62 方向中) | 自分を除いた推定 |",
             "|---|---:|---|---|"]
    loo_keys: dict[str, object] = out["loo"]  # type: ignore[assignment]
    for g, codes in pl.items():
        lines.append(f"| {g} 社 | {len(codes)} | {int(res[g]['abs']['n_sub'])} / "
                     f"{int(res[g]['rel']['n_sub'])} | "
                     f"{', '.join(k.split(':')[1] for k in loo_keys if k.startswith(g + ':'))} |")
    for g in ("57", "10"):
        ab, rl = res[g]["abs"], res[g]["rel"]
        lines += ["", f"## {g} 社のプール", "",
                  "| 科目 | 差分 n+ / n− | 差分 c⁺ / c⁻ | 差分 c⁻/c⁺ | 対数 n 拡大 / 縮小 "
                  "| 対数 c_grow / c_shrink | 対数 c_shrink/c_grow | 下限 x̲ |",
                  "|---|---|---|---:|---|---|---:|---:|"]
        for nm in SHOW:
            j = names.index(nm)
            cp, cm = ab["c"][0, j], ab["c"][1, j]
            cg, cs = rl["c"][0, j], rl["c"][1, j]
            lines.append(f"| {ja.get(nm, nm)} | {ab['n'][0, j]} / {ab['n'][1, j]} "
                         f"| {cp:.1f} / {cm:.1f} | {cm / cp if cp else float('nan'):.2f} "
                         f"| {rl['n'][0, j]} / {rl['n'][1, j]} | {cg:.2f} / {cs:.2f} "
                         f"| {cs / cg if cg else float('nan'):.2f} | {floor[j]:.1e} |")
    lines += ["", "下限 x̲ は 2024 年度の全企業(パネル)で、総資産比の非ゼロ値の 10 パーセンタイル"
              f"(≥ {FLOOR_MIN:g})。", ""]
    (OUT / "costs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
