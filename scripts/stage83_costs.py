#!/usr/bin/env python
"""ステージ83: KDE0 の θ(生存バイアスのプール)と、KDE0 の代替の一覧(numpy・pandas だけのプロセス)。

規則は ``results/stage83/criteria_stage83.md`` §1・§2。fold と 57 社の θ はステージ82 の
``theta.json`` をそのまま使い、生存バイアスのプール(回復群 72 社・滞留群・苦境全体)だけを KDE0 で
推定し直す。``stage82_ot.py``・``stage82_qp.py`` に ``--data data/interim/stage83`` を渡すと、
苦境全体のプール(``KDE0p_all``)はこの KDE0 の θ を使う。

出力: ``data/interim/stage83/theta.json``、``results/stage83/kde0_fallback.csv``・
``exp5_survival_gap.csv``。

    uv run python scripts/stage83_costs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from prep_stage59 import deltas
from stage82_costs import FROZEN, start

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.unit_costs import KDE_MIN_N, estimate_theta, kde_density_at_zero

D82, DATA, OUT = Path("data/interim/stage82"), Path("data/interim/stage83"), Path("results/stage83")


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    tj = json.loads((D82 / "theta.json").read_text(encoding="utf-8"))
    sigma = np.array(tj["sigma"], dtype=float)
    frozen = np.array([n in FROZEN for n in names])
    w75 = json.loads(Path("data/interim/stage75/weights.json").read_text(encoding="utf-8"))
    folds = w75["folds"]
    p57 = sorted(c for f in folds.values() for c in f)
    pools = {f: [c for c in p57 if c not in set(codes)] for f, codes in folds.items()}
    pools["full"] = p57
    rec72, stag = tj["pools"]["rec72"], tj["pools"]["stag"]
    pools |= {"rec72": rec72, "stag": stag, "alldist": sorted(set(rec72) | set(stag))}
    # KDE0 の代替の一覧
    rows = []
    surv: dict[str, dict[str, object]] = {}
    for cfg, codes in pools.items():
        d, x0 = deltas(codes, names), start(codes, names)
        z = np.abs(d) / (np.abs(x0) + sigma[None, :])
        for r, sg in enumerate(("+", "−")):
            for j, n in enumerate(names):
                if frozen[j]:
                    continue
                mv = (1.0 if r == 0 else -1.0) * d[:, j] > 0
                k = int(mv.sum())
                if k == 0:
                    reason = "動いた企業なし(θ_max)"
                elif k < KDE_MIN_N:
                    reason = "n < 5(R1 に戻す)"
                elif kde_density_at_zero(z[mv, j]) <= 0:
                    reason = "帯域 0(R1 に戻す)"
                else:
                    continue
                rows.append({"pool": cfg, "account": n, "label": ja.get(n, n), "dir": sg,
                             "n_movers": k, "reason": reason})
        if cfg in ("rec72", "stag", "alldist"):
            res = estimate_theta(d, x0, sigma, "KDE0")
            th = res.theta.copy()
            th[:, frozen] = 0.0
            surv[cfg] = {"theta": th.tolist(), "n_movers": res.n_movers.tolist(),
                         "n_fallback": res.n_fallback, "n_pool": len(codes)}
    fb = pd.DataFrame(rows)
    for cfg in ("f0", "f1", "f2", "f3", "f4", "full"):
        n_fb = int((fb["pool"] == cfg).sum() - ((fb["pool"] == cfg)
                                                 & (fb["reason"].str.startswith("動いた"))).sum())
        if n_fb != tj["configs"][cfg]["KDE0"]["n_fallback"]:
            raise ValueError(f"{cfg}: 代替の数がステージ82 と一致しない")
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    fb.to_csv(OUT / "kde0_fallback.csv", index=False)
    out = {**tj, "survival": surv, "note": "survival は KDE0(ステージ83)"}
    (DATA / "theta.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    gap = []
    tr, ts = np.array(surv["rec72"]["theta"]), np.array(surv["stag"]["theta"])
    for j, n in enumerate(names):
        if frozen[j]:
            continue
        for r, sg in enumerate(("+", "−")):
            gap.append({"account": n, "label": ja.get(n, n), "dir": sg, "theta_rec72": tr[r, j],
                        "theta_stag": ts[r, j],
                        "ratio_stag_over_rec": ts[r, j] / tr[r, j] if tr[r, j] > 0 else np.nan,
                        "n_rec": surv["rec72"]["n_movers"][r][j],  # type: ignore[index]
                        "n_stag": surv["stag"]["n_movers"][r][j]})  # type: ignore[index]
    g = pd.DataFrame(gap)
    g["abs_log_ratio"] = np.abs(np.log(g["ratio_stag_over_rec"]))
    g.sort_values("abs_log_ratio", ascending=False).to_csv(OUT / "exp5_survival_gap.csv",
                                                           index=False)
    print(fb.groupby(["pool", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
