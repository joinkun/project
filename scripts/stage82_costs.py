#!/usr/bin/env python
"""ステージ82: 基準単価 θ の 5 方式と、生存バイアスのプール(numpy・pandas だけのプロセス)。

規則は ``results/stage82/criteria.md`` §1・§7。

- fold f0〜f4・full(57 社、ステージ75 と同じ分け方)で 5 方式の θ を推定する。
- full では、回復群 72 社・滞留群・苦境全体(回復 + 滞留)の R1 の θ も推定する。
- 実験 1: 5 方式の θ の 64 方向の相関(log、スピアマン)と、KDE0 の代替の数。
- 実験 5: 回復群と滞留群の θ の比(上位の科目)。

出力: ``data/interim/stage82/theta.json``、``results/stage82/exp1_theta_corr.csv``・
``exp1_theta.csv``・``exp5_survival_gap.csv``。

    uv run python scripts/stage82_costs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from prep_stage59 import deltas
from stage82_common import DATA, OUT

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.unit_costs import MODES, THETA_MAX, estimate_theta

_PANEL = Path("data/interim/stage23/panel/panel.csv")
SIGMA_FLOOR = 1e-3  # qp_stage60.SIGMA_FLOOR と同じ
FROZEN = ("valuation_difference_securities",)


def start(codes: list[str], names: tuple[str, ...]) -> np.ndarray:
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    out: np.ndarray = a[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float) / a[
        "cur_total_assets"].to_numpy(float)[:, None]
    return out


def main() -> None:  # noqa: PLR0915
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    sigma = np.maximum(sd["sigma_1"].to_numpy(float), SIGMA_FLOOR)
    frozen = np.array([n in FROZEN for n in names])
    w75 = json.loads(Path("data/interim/stage75/weights.json").read_text(encoding="utf-8"))
    folds = w75["folds"]
    p57 = sorted(c for f in folds.values() for c in f)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    rec72 = sorted(pairs["edinetCode"])
    r38 = pd.read_parquet("data/interim/stage38/real_deltas.parquet").astype({"edinetCode": str})
    l22 = r38[r38["t"] == 2023].set_index("edinetCode")["label_prev"]
    l24 = r38[r38["t"] == 2024].set_index("edinetCode")["label_cur"]
    both = l22.index.intersection(l24.index)
    pan = pd.read_csv(_PANEL, dtype={"edinetCode": str}, usecols=["edinetCode", "t",
                                                                   "cur_total_assets"])
    ok = set(pan[(pan["t"] == 2022) & (pan["cur_total_assets"] > 0)]["edinetCode"]) & set(
        pan[(pan["t"] == 2024) & (pan["cur_total_assets"] > 0)]["edinetCode"])
    stag = sorted(c for c in both if l22[c] >= 1 and l24[c] >= 1 and c in ok)
    pools = {f"{f}": [c for c in p57 if c not in set(codes)] for f, codes in folds.items()}
    pools["full"] = p57
    out: dict[str, object] = {"names": list(names), "sigma": sigma.tolist(), "configs": {},
                              "pools": {"stag": stag, "rec72": rec72}}

    def thetas(codes: list[str], modes: tuple[str, ...]) -> dict[str, dict[str, object]]:
        d, x0 = deltas(codes, names), start(codes, names)
        res = {}
        for mode in modes:
            r = estimate_theta(d, x0, sigma, mode)
            th = r.theta.copy()
            th[:, frozen] = 0.0
            res[mode] = {"theta": th.tolist(), "n_movers": r.n_movers.tolist(),
                         "n_fallback": r.n_fallback, "n_pool": len(codes)}
        return res
    for cfg, codes in pools.items():
        out["configs"][cfg] = thetas(codes, MODES)  # type: ignore[index]
    sp = {"rec72": thetas(rec72, ("R1",))["R1"], "stag": thetas(stag, ("R1",))["R1"],
          "alldist": thetas(sorted(set(rec72) | set(stag)), ("R1",))["R1"]}
    out["survival"] = sp
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / "theta.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    # 実験 1: 64 方向の θ(full)と相関
    full = out["configs"]["full"]  # type: ignore[index]
    rows = []
    for j, n in enumerate(names):
        for r, sg in enumerate(("+", "−")):
            row: dict[str, object] = {"account": n, "label": ja.get(n, n), "dir": sg}
            for mode in MODES:
                row[mode] = full[mode]["theta"][r][j]
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "exp1_theta.csv", index=False)
    act = ~tab["account"].isin(FROZEN)
    corr = tab.loc[act, list(MODES)].rank().corr()
    corr.round(3).to_csv(OUT / "exp1_theta_corr.csv")
    capped = {m: int((tab.loc[act, m] >= THETA_MAX).sum()) for m in MODES}
    fb = {cfg: full_cfg["KDE0"]["n_fallback"]
          for cfg, full_cfg in out["configs"].items()}  # type: ignore[attr-defined]
    print("相関(スピアマン、評価差額金を除く 62 方向)\n", corr.round(3))
    print("上限 10³ に当たる方向:", capped, "KDE0 の R1 への代替:", fb)
    (OUT / "exp1_theta_meta.json").write_text(json.dumps(
        {"capped": capped, "kde0_fallback": fb, "n_directions": int(act.sum())},
        ensure_ascii=False, indent=1), encoding="utf-8")
    # 実験 5: 回復群と滞留群の θ の比
    gap = []
    tr, ts = np.array(sp["rec72"]["theta"]), np.array(sp["stag"]["theta"])
    for j, n in enumerate(names):
        if n in FROZEN:
            continue
        for r, sg in enumerate(("+", "−")):
            gap.append({"account": n, "label": ja.get(n, n), "dir": sg,
                        "theta_rec72": tr[r, j], "theta_stag": ts[r, j],
                        "ratio_stag_over_rec": ts[r, j] / tr[r, j] if tr[r, j] > 0 else np.nan,
                        "n_rec": sp["rec72"]["n_movers"][r][j],  # type: ignore[index]
                        "n_stag": sp["stag"]["n_movers"][r][j]})  # type: ignore[index]
    g = pd.DataFrame(gap)
    g["abs_log_ratio"] = np.abs(np.log(g["ratio_stag_over_rec"]))
    g.sort_values("abs_log_ratio", ascending=False).to_csv(OUT / "exp5_survival_gap.csv",
                                                           index=False)
    print(f"滞留群 {len(stag)} 社、回復群 {len(rec72)} 社")


if __name__ == "__main__":
    main()
