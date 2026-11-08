#!/usr/bin/env python
"""ステージ79: A1・A2 の Sinkhorn と、R1・A1 のスパース OT の輸送計画(POT のプロセス)。

規則は ``results/stage79/criteria.md`` §3。

- Sinkhorn: ステージ75 と同じ(片側不均衡、ε = 0.001、τ1 = 0.05、同一企業のマスク)。
  候補は累積 0.9・最大 10 社。A2 の地上コストは、組ごとの区分線形の費用
  Ψ_a(t) = c₁ min(t, b) + c₂ (t − b)₊ の和。
- スパース OT: ``fsgen.models.sparse_transport``。τ' = τ1 log κ / (κ(κ − 1))、κ = n1/n0
  (criteria.md §7: 1 社の全質量を 1 つの対応先が受ける負荷で、KL と勾配が等しい)。γ' は格子から
  |S(i)| の中央値が 5 に最も近いもの。候補は非ゼロのすべて(カットオフなし)。
- ``--diag``: 57 社の全体の R1 で、Sinkhorn の確率の減衰を診断する(指示 2-1)。

出力: ``data/interim/stage79/plan_{cls}_{cfg}.npz``(キー ``A1__idx``・``A1__pt``・``A2__…``・
``R1sp__idx``・``R1sp__pt``・``A1sp__…``)、``results/stage79/sparse_gamma_{cls}_{cfg}.csv``、
``results/stage79/sinkhorn_decay.md``。

    uv run python scripts/stage79_ot.py --cls c1 --cfgs f0,f1
    uv run python scripts/stage79_ot.py --diag
"""

from __future__ import annotations

import os
import sys

# POT の torch バックエンドと sklearn の読み込みを止める(precompute_transport.py と同じ)
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from prep_stage59 import l1_cost  # noqa: E402
from stage71_ot import candidates  # noqa: E402
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.models.sparse_transport import solve_sparse_semi_unbalanced  # noqa: E402
from fsgen.models.transport import solve_semi_unbalanced_normalized  # noqa: E402

DATA = Path("data/interim/stage79")
OUT = Path("results/stage79")
EPS, TAU1 = 0.001, 0.05
# criteria.md §7 で下側に広げた格子
GAMMAS = (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
TARGET_SUPPORT, STOP_SUPPORT = 5.0, 8.0
SIGMA_FLOOR = 1e-3  # qp_stage60.SIGMA_FLOOR と同じ
_ZERO_OT = ("jppfs_cor_RetainedEarnings", "valuation_difference_securities")


def a2_cost(xs: np.ndarray, xt: np.ndarray, e: dict[str, list[float]], sigma: np.ndarray,
            zero: np.ndarray) -> np.ndarray:
    """組 (i, j) ごとの Σ_a Ψ_a(|x_j,a − x_i,a|)(区切り b = m × (|x_i,a| + σ_a))。"""
    c1 = np.array([e["c1_plus"], e["c1_minus"]], dtype=float)
    c2 = np.array([e["c2_plus"], e["c2_minus"]], dtype=float)
    m = np.array([e["m_plus"], e["m_minus"]], dtype=float)
    c1[:, zero] = c2[:, zero] = 0.0
    out = np.empty((len(xs), len(xt)))
    for s in range(0, len(xs), 8):
        x0 = xs[s:s + 8]
        d = xt[None, :, :] - x0[:, None, :]
        up = d > 0
        t = np.abs(d)
        b = np.where(up, m[0], m[1]) * (np.abs(x0) + sigma[None, :])[:, None, :]
        k1, k2 = np.where(up, c1[0], c1[1]), np.where(up, c2[0], c2[1])
        out[s:s + 8] = (k1 * np.minimum(t, b) + k2 * np.maximum(t - b, 0.0)).sum(axis=2)
    return out


def normalized(c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ct: np.ndarray = c / np.median(c[~mask])
    ct[mask] = 10.0 * ct.max()
    return ct


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"))
    ap.add_argument("--cfgs", default="", help="カンマ区切り(f0〜f4、full)")
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    zero = np.array([n in _ZERO_OT for n in names])
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    sigma = np.maximum(sd["sigma_1"].to_numpy(float), SIGMA_FLOOR)
    wj = json.loads((DATA / "weights.json").read_text(encoding="utf-8"))
    x, lab, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    xt = x[lab == 0]
    tgt_codes = np.array([code_of.get(str(d), "") for d in docs_a[lab == 0]])
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    def setup(cls: str) -> tuple[np.ndarray, list[str], np.ndarray]:
        m = sources(cls, lab, docs_a)
        src = [str(d) for d in docs_a[m]]
        sc = np.array([code_of.get(d, "") for d in src])
        mask = (sc[:, None] == tgt_codes[None, :]) & (sc[:, None] != "")
        return x[m], src, mask

    def lin(cfg: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
        e = wj["configs"][cfg][kind]
        cp, cm = np.array(e["plus"], dtype=float), np.array(e["minus"], dtype=float)
        cp[zero] = cm[zero] = 0.0
        return cp, cm

    if args.diag:
        rows = []
        for cls in ("c2", "c1"):
            xs, _, mask = setup(cls)
            ct = normalized(l1_cost(xs, xt, *lin("full", "R1")), mask)
            n0, n1 = ct.shape
            r = solve_semi_unbalanced_normalized(ct, np.full(n0, 1 / n0), np.full(n1, 1 / n1),
                                                 EPS, TAU1, max_iter=2_000_000)
            p = r.plan / r.plan.sum(axis=1, keepdims=True)
            order = np.argsort(ct, axis=1)
            by_rank = np.take_along_axis(p, order, axis=1)
            ps = -np.sort(-p, axis=1)
            cum = np.cumsum(ps, axis=1)
            row: dict[str, Any] = {"cls": cls, "n0": n0, "n1": n1,
                                      "exact_zero": float((p == 0).mean()),
                                      "below_1e-300": float(((p > 0) & (p < 1e-300)).mean())}
            for k in (1, 2, 5, 10, 50, 100, 1000):
                v = by_rank[:, k - 1]
                lv = np.log10(np.where(v > 0, v, np.nan))
                row[f"rank{k}"] = (float(np.nanmedian(lv)), float(np.nanmin(lv)),
                                   float(np.nanmax(lv)), float((v == 0).mean()))
            for q in (0.9, 0.99, 0.999):
                kq = (cum < q).sum(axis=1) + 1
                row[f"mass{q}"] = (float(np.median(kq)), int(kq.max()))
            rows.append(row)
        lines = ["# ステージ79: Sinkhorn の確率の減衰(R1、57 社の全体、ε = 0.001、τ1 = 0.05)", "",
                 "p_ij = 行で正規化した輸送確率。順位はコスト C̃_ij の小さい順(1 位 = 最も近い)。"
                 "値は log10 p の中央値 [最小, 最大](0 の行は除く)と、厳密に 0 の割合。", "",
                 "| class | 1 位 | 2 位 | 5 位 | 10 位 | 50 位 | 100 位 | 1000 位 |",
                 "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            cells = []
            for k in (1, 2, 5, 10, 50, 100, 1000):
                lmed, lo, hi, z0 = row[f"rank{k}"]
                cells.append(f"{lmed:.1f} [{lo:.1f}, {hi:.1f}](0: {z0:.0%})")
            lines.append(f"| {row['cls']} | " + " | ".join(cells) + " |")
        lines += ["", "| class | 累積 90% に必要な候補数(中央値 / 最大) | 99% | 99.9% "
                  "| 全体で厳密に 0 の割合 |",
                  "|---|---|---|---|---:|"]
        for row in rows:
            cells = [f"{row[f'mass{q}'][0]:.0f} / {row[f'mass{q}'][1]}"
                     for q in (0.9, 0.99, 0.999)]
            lines.append(f"| {row['cls']} | " + " | ".join(cells)
                         + f" | {row['exact_zero']:.1%} |")
        lines.append("")
        (OUT / "sinkhorn_decay.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return
    if args.cls is None or not args.cfgs:
        raise SystemExit("--cls と --cfgs、または --diag を指定する")
    xs, src, mask = setup(args.cls)
    n0, n1 = len(xs), len(xt)
    a, b = np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1)
    kappa = n1 / n0
    tau_q = TAU1 * np.log(kappa) / (kappa * (kappa - 1.0))  # criteria.md §7
    for cfg in args.cfgs.split(","):
        store: dict[str, np.ndarray] = {"source_docs": np.array(src)}
        costs = {"A1": normalized(l1_cost(xs, xt, *lin(cfg, "A1")), mask),
                 "A2": normalized(a2_cost(xs, xt, wj["configs"][cfg]["A2"], sigma, zero), mask),
                 "R1": normalized(l1_cost(xs, xt, *lin(cfg, "R1")), mask)}
        for kind in ("A1", "A2"):
            t0 = time.perf_counter()
            r = solve_semi_unbalanced_normalized(costs[kind], a, b, EPS, TAU1, max_iter=2_000_000)
            p = r.plan / r.plan.sum(axis=1, keepdims=True)
            if (p[mask] > 1e-12).any():
                raise ValueError("除外した組に質量がある")
            idx, pt, kk = candidates(p)
            store[f"{kind}__idx"], store[f"{kind}__pt"], store[f"{kind}__k"] = idx, pt, kk
            print(f"[{args.cls} {cfg} {kind} Sinkhorn] 実効サポート "
                  f"{np.median(1.0 / (pt ** 2).sum(axis=1)):.2f}({time.perf_counter() - t0:.0f}s)",
                  flush=True)
        grid_rows = []
        for kind in ("R1", "A1"):
            best: tuple[float, float, np.ndarray] | None = None
            for g in GAMMAS:
                t0 = time.perf_counter()
                res = solve_sparse_semi_unbalanced(costs[kind], g, tau_q, allowed=~mask)
                nz = (res.p > 0).sum(axis=1)
                ess = 1.0 / (res.p ** 2).sum(axis=1)
                med = float(np.median(nz))
                grid_rows.append({"kind": kind, "gamma": g, "tau": tau_q, "nz_med": med,
                                  "nz_max": int(nz.max()), "nz_min": int(nz.min()),
                                  "ess_med": float(np.median(ess)), "iters": res.n_iter,
                                  "residual": res.residual,
                                  "seconds": round(time.perf_counter() - t0, 1)})
                print(f"[{args.cls} {cfg} {kind} sparse γ'={g:g}] |S| 中央値 {med:.1f}、最大 "
                      f"{int(nz.max())}、反復 {res.n_iter}({time.perf_counter() - t0:.0f}s)",
                      flush=True)
                if best is None or abs(med - TARGET_SUPPORT) < abs(best[1] - TARGET_SUPPORT):
                    best = (g, med, res.p)
                if med >= STOP_SUPPORT:
                    break
            assert best is not None
            g, med, p = best
            kmax = int((p > 0).sum(axis=1).max())
            order = np.argsort(-p, axis=1)[:, :kmax]
            pt = np.take_along_axis(p, order, axis=1)
            store[f"{kind}sp__idx"], store[f"{kind}sp__pt"] = order, pt
            store[f"{kind}sp__gamma"] = np.array(g)
            if not np.allclose(pt.sum(axis=1), 1.0):
                raise ValueError("スパース OT の行の和が 1 でない")
        pd.DataFrame(grid_rows).to_csv(OUT / f"sparse_gamma_{args.cls}_{cfg}.csv", index=False)
        np.savez(DATA / f"plan_{args.cls}_{cfg}.npz", **store)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
