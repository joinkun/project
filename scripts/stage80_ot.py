#!/usr/bin/env python
"""ステージ80: γ' = 0.1 のスパース OT の計画と、A1 のコストでの命題 1 の検算(POT のプロセス)。

規則は ``results/stage80/criteria.md`` §1・§4。τ' はステージ79 §7(τ' = τ1 log κ / (κ(κ − 1)))。
コストは、abs がステージ75、R1・A1 がステージ79 の ``weights.json``(利益剰余金・評価差額金は 0)。
同一企業の組は候補から除き、C̃ = C / median(C[~mask])。

出力: ``data/interim/stage80/plan_{cls}_{cfg}.npz``(キー ``{abs,R1,A1}sp__idx``・``__pt``)、
``results/stage80/sparse_plans.csv``、``results/stage80/verify_a1.md``(``--verify``)。

    uv run python scripts/stage80_ot.py --cls c1 --cfgs f0,f1
    uv run python scripts/stage80_ot.py --verify
"""

from __future__ import annotations

import os
import sys

# POT の torch バックエンドと sklearn の読み込みを止める(precompute_transport.py と同じ)
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import ot  # noqa: E402
import pandas as pd  # noqa: E402
from prep_stage59 import l1_cost  # noqa: E402
from targets_stage68 import sources  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.errors import SolverNotConvergedError  # noqa: E402
from fsgen.models.sparse_transport import solve_sparse_semi_unbalanced  # noqa: E402

DATA = Path("data/interim/stage80")
OUT = Path("results/stage80")
TAU1, GAMMA = 0.05, 0.1
_ZERO_OT = ("jppfs_cor_RetainedEarnings", "valuation_difference_securities")
FOLDS = ("f0", "f1", "f2", "f3", "f4")


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"))
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    zero = np.array([n in _ZERO_OT for n in names])
    w75 = json.loads(Path("data/interim/stage75/weights.json").read_text(encoding="utf-8"))
    w79 = json.loads(Path("data/interim/stage79/weights.json").read_text(encoding="utf-8"))
    x, lab, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    xt = x[lab == 0]
    tgt_codes = np.array([code_of.get(str(d), "") for d in docs_a[lab == 0]])
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    def lin(cfg: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
        e = (w75 if kind == "abs" else w79)["configs"][cfg][kind]
        cp, cm = np.array(e["plus"], dtype=float), np.array(e["minus"], dtype=float)
        cp[zero] = cm[zero] = 0.0
        return cp, cm

    def setup(cls: str) -> tuple[np.ndarray, list[str], np.ndarray]:
        m = sources(cls, lab, docs_a)
        src = [str(d) for d in docs_a[m]]
        sc = np.array([code_of.get(d, "") for d in src])
        mask = (sc[:, None] == tgt_codes[None, :]) & (sc[:, None] != "")
        return x[m], src, mask

    def normalized(c: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ct: np.ndarray = c / np.median(c[~mask])
        ct[mask] = 10.0 * ct.max()
        return ct

    if args.verify:
        lines = ["# ステージ80: 命題 1 の検算(A1 のコスト、57 社の全体、同一企業のマスクあり)", "",
                 "バランス輸送の厳密解(`ot.emd`)。非ゼロは π_ij > 1e-15、"
                 "実効サポートは行で正規化した p に対する 1/Σp²。", "",
                 "| class | n0 | n1 | n1/n0 | ⌈n1/n0⌉ | 行の非ゼロ数 最小 / 平均 / 最大 "
                 "| 非ゼロの総数(上界 n0 + n1 − 1) | 実効サポート 最小 / 中央値 / 平均 |",
                 "|---|---:|---:|---:|---:|---|---|---|"]
        for cls in ("c2", "c1"):
            xs, _, mask = setup(cls)
            ct = normalized(l1_cost(xs, xt, *lin("full", "A1")), mask)
            n0, n1 = ct.shape
            p, log = ot.emd(np.full(n0, 1 / n0), np.full(n1, 1 / n1), ct, numItermax=10_000_000,
                            log=True)
            if log.get("warning"):
                raise SolverNotConvergedError(f"ot.emd: {log['warning']}")
            nz = (p > 1e-15).sum(axis=1)
            q = p / p.sum(axis=1, keepdims=True)
            eff = 1.0 / (q ** 2).sum(axis=1)
            lines.append(f"| {cls} | {n0} | {n1} | {n1 / n0:.3f} | {math.ceil(n1 / n0)} "
                         f"| {nz.min()} / {nz.mean():.3f} / {nz.max()} "
                         f"| {nz.sum()}({n0 + n1 - 1}) | {eff.min():.3f} / "
                         f"{np.median(eff):.3f} / {eff.mean():.3f} |")
        lines.append("")
        (OUT / "verify_a1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return
    if args.cls is None or not args.cfgs:
        raise SystemExit("--cls と --cfgs、または --verify を指定する")
    xs, src, mask = setup(args.cls)
    n0, n1 = len(xs), len(xt)
    kappa = n1 / n0
    tau_q = TAU1 * np.log(kappa) / (kappa * (kappa - 1.0))
    rows = []
    for cfg in args.cfgs.split(","):
        store: dict[str, np.ndarray] = {"source_docs": np.array(src)}
        for kind in (("A1",) if cfg == "full" else ("abs", "R1", "A1")):
            t0 = time.perf_counter()
            res = solve_sparse_semi_unbalanced(normalized(l1_cost(xs, xt, *lin(cfg, kind)), mask),
                                               GAMMA, tau_q, allowed=~mask)
            p = res.p
            nz = (p > 0).sum(axis=1)
            kmax = int(nz.max())
            order = np.argsort(-p, axis=1)[:, :kmax]
            pt = np.take_along_axis(p, order, axis=1)
            if not np.allclose(pt.sum(axis=1), 1.0):
                raise ValueError("行の和が 1 でない")
            store[f"{kind}sp__idx"], store[f"{kind}sp__pt"] = order, pt
            ess = 1.0 / (pt ** 2).sum(axis=1)
            rows.append({"cls": args.cls, "cfg": cfg, "kind": kind, "gamma": GAMMA, "tau": tau_q,
                         "nz_med": float(np.median(nz)), "nz_max": kmax,
                         "ess_med": float(np.median(ess)), "iters": res.n_iter,
                         "residual": res.residual})
            print(f"[{args.cls} {cfg} {kind}] |S| 中央値 {np.median(nz):.1f}、最大 {kmax}、"
                  f"ESS {np.median(ess):.2f}({time.perf_counter() - t0:.0f}s)", flush=True)
        np.savez(DATA / f"plan_{args.cls}_{cfg}.npz", **store)  # type: ignore[arg-type]
    pd.DataFrame(rows).to_csv(OUT / f"sparse_plans_{args.cls}_{args.cfgs.replace(',', '_')}.csv",
                              index=False)


if __name__ == "__main__":
    main()
