#!/usr/bin/env python
"""ステージ83 パートC: 再建企業を始点にした out-of-fold の輸送計画(POT・numpy のプロセス)。

規則は ``results/stage83/criteria_stage83.md`` §3。

- fold f のホールドアウト D_f(57 社)には、D_f を除いて推定した θ(cfg f)を使う。
- class 2 → 回復の 15 社(c2rec)には、57 社の全体の θ(full)を使う。
- コスト C_ij = ψ_i(x_j − x_i)(利益剰余金・評価差額金は 0)。主モデル(KDE0p、stage1_penalty)と比較の
  KDE0(base)。対象は健全企業(class 0)。**同一企業の組(自社の再建後の書類)は除く。**
- 輸送はスパース OT(γ' = 0.1、τ' = τ1 log κ / (κ(κ − 1))、κ = n1/n0、n0 = 群の社数)。
- 比較の基準の 1-NN は、同じコストで最小の 1 社(主モデルのコスト)。

出力: ``data/interim/stage83/bt_plan_{群}_{コスト}.npz``(``codes``・``idx``・``pt``・``nn``)。

    uv run python scripts/stage83_backtest_ot.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stage82_common import COSTS, GAMMA, TAU1, ZERO_OT  # noqa: E402
from stage83_common import DATA, recovery_groups  # noqa: E402
from stage83_common import compositions as comps  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.models.sparse_transport import solve_sparse_semi_unbalanced  # noqa: E402
from fsgen.models.unit_costs import inactive_mask, individual_theta, pairwise_cost  # noqa: E402


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    tj = json.loads((DATA / "theta.json").read_text(encoding="utf-8"))
    sigma = np.array(tj["sigma"], dtype=float)
    zero = np.array([n in ZERO_OT for n in names])
    x, lab, docs, _ = load_tilde(names)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    xt = x[lab == 0]
    tgt_codes = np.array([code_of.get(str(d), "") for d in np.array(docs)[lab == 0]])
    for grp, codes in recovery_groups().items():
        x0, _x1, _ta, ok = comps(codes, names)
        cfg = "full" if grp == "c2rec" else grp
        mask = np.array(ok)[:, None] == tgt_codes[None, :]
        n0, n1 = len(x0), len(xt)
        kappa = n1 / n0
        tau_q = TAU1 * np.log(kappa) / (kappa * (kappa - 1.0))
        for cost in ("KDE0p", "KDE0"):
            sp = COSTS[cost]
            th = np.array(tj["configs"][cfg][sp.mode]["theta"], dtype=float)
            rows = np.stack([individual_theta(th, r, sigma, indiv_scale=sp.indiv,
                                              inactive=inactive_mask(r, names) if sp.penalty
                                              else None) for r in x0])
            rows[:, :, zero] = 0.0
            c = pairwise_cost(x0, xt, rows)
            ct = c / np.median(c[~mask])
            ct[mask] = 10.0 * ct.max()
            res = solve_sparse_semi_unbalanced(ct, GAMMA, tau_q, allowed=~mask)
            p = res.p
            kmax = int((p > 0).sum(axis=1).max())
            idx = np.argsort(-p, axis=1)[:, :kmax]
            pt = np.take_along_axis(p, idx, axis=1)
            if (pt[np.take_along_axis(mask, idx, axis=1)] > 0).any():
                raise ValueError("同一企業の組に質量がある")
            nn = np.where(mask, np.inf, ct).argmin(axis=1)
            np.savez(DATA / f"bt_plan_{grp}_{cost}.npz", codes=np.array(ok), idx=idx, pt=pt,
                     nn=nn)
            print(f"[{grp} {cost}] {n0} 社(除外 {len(codes) - n0})、|S| 中央値 "
                  f"{np.median((pt > 0).sum(axis=1)):.1f}、反復 {res.n_iter}", flush=True)


if __name__ == "__main__":
    main()
