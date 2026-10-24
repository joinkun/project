#!/usr/bin/env python
"""ステージ71: 𝓕⁺(負債の上限つき)の上で、輸送計画を直接組み込む生成(cvxpy のプロセス)。

規則は ``results/stage71/criteria.md``。新設計(ステージ70: 72 社・c = 1/s・非支配株主持分を動かす、
利益剰余金のコスト 0)、第 1 項は全 32 科目・m = 10、制約は s47_final + 現金の下限 + 負債の上限。

- ``a2``: τ1 = 0.01 の argmax(ステージ70 の目標)
- ``f1``: 事後アンサンブル。S(i) の各候補 y_j を目標に ``RecourseQP`` を解き、p̃ で重み付き平均
- ``f2``: 期待 L1 接近。min Σ_j p̃ Σ_a √w |y − y_j| + λ Ψ(y − x) + 10⁻⁴ Σ w (y − x)²
- ``f3`` / ``f3h0`` / ``f3t05``: F3 のゼロ整合重心(ハミング ≤ 2・τ1 = 0.05 / 0・0.05 / ≤ 2・0.5)

    uv run python scripts/stage71_qp.py --cls c1 --setting f1 --shard 0/2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from qp_stage60 import SIGMA_FLOOR, s47_final
from qp_stage66 import BETA_CASH, LAMS, lam_scale, sources
from stage70_qp import move_cost
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import (
    CanonicalSpec,
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import (
    INACTIVE_ACCOUNTS,
    LinearConstraint,
    build_linear_constraints,
    decompose_re_change,
    inactive_accounts,
    subtotal_members,
)
from fsgen.accounting.projection_qp import load_direction_constraints
from fsgen.accounting.recourse_qp import RecourseQP, RecourseWeights
from fsgen.errors import InfeasibleProjectionError, SolverNotConvergedError

DATA = Path("data/interim/stage71")
OUT = Path("results/stage71")
_CANONICAL = Path("conf/accounts/core.yaml")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
FROZEN = ("valuation_difference_securities",)
EPS_REG = 1e-4
_TOL = 1e-7
SETTINGS = ("a2", "f1", "f2", "f3", "f3h0", "f3t05")


def liability_cap(names: tuple[str, ...], members: dict[str, np.ndarray], x0: np.ndarray
                  ) -> tuple[LinearConstraint, ...]:
    """負債の上限 L(y) ≤ max(2 L(x0), 0.01 A(x0))(L(x0) ≤ 0 なら課さない)。項目 5 と同じ集合。"""
    la = members["jppfs_cor_Liabilities"]
    l0 = float(la @ x0)
    if l0 <= 0:
        return ()
    cap = max(2.0 * l0, 0.01 * float(members["jppfs_cor_Assets"] @ x0))
    return (LinearConstraint("Liabilities<=max(2*start,0.01*Assets)", -la, -cap),)


class ExpectedL1:
    """F2: min Σ_j p̃_j Σ_a √w_a |y_a − y_ja| + λ Σ_a (c⁺u + c⁻v) + ε Σ_a w_a (y_a − x_a)²。"""

    def __init__(self, spec: CanonicalSpec, x0: np.ndarray, w: np.ndarray, cp_: np.ndarray,
                 cm_: np.ndarray, lam: float, extra: tuple[LinearConstraint, ...],
                 frozen: tuple[str, ...], k: int) -> None:
        names = state_vector_names(spec, include_subtotals=False)
        d = len(names)
        self.x0, self.names, self.spec, self.extra = x0, names, spec, extra
        self.cmat = build_constraint_matrix(spec, include_subtotals=False)
        self.y = cp.Variable(d)
        u, v = cp.Variable(d, nonneg=True), cp.Variable(d, nonneg=True)
        self.t = cp.Variable((k, d), nonneg=True)
        self.targets = cp.Parameter((k, d))
        self.pt = cp.Parameter(k, nonneg=True)
        sw = np.sqrt(w)
        cons = [self.cmat.T @ self.y == 0, self.y - x0 == u - v]
        for j in range(k):
            cons += [self.t[j] >= cp.multiply(sw, self.y - self.targets[j]),
                     self.t[j] >= -cp.multiply(sw, self.y - self.targets[j])]
        idx = {n: i for i, n in enumerate(names)}
        self.signs = [(idx[b.name], b.sign) for b in spec.basis if b.sign in ("nonneg", "nonpos")]
        for i, sg in self.signs:
            cons.append(self.y[i] >= 0 if sg == "nonneg" else self.y[i] <= 0)
        self.frozen = [idx[n] for n in frozen]
        for i in self.frozen:
            cons.append(self.y[i] == x0[i])
        for lc in extra:
            cons.append(lc.a @ self.y >= lc.rhs)
        obj = (cp.sum(self.pt @ self.t) + lam * (cp_ @ u + cm_ @ v)
               + EPS_REG * cp.sum(cp.multiply(w, cp.square(self.y - x0))))
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.n_fallback = 0

    def _violation(self, y: np.ndarray) -> float:
        v = float(np.abs(self.cmat.T @ y).max())
        for i, sg in self.signs:
            v = max(v, -y[i] if sg == "nonneg" else y[i])
        for i in self.frozen:
            v = max(v, abs(y[i] - self.x0[i]))
        for lc in self.extra:
            v = max(v, float(lc.rhs - lc.a @ y))
        return v

    def solve(self, targets: np.ndarray, pt: np.ndarray) -> np.ndarray:
        self.targets.value, self.pt.value = targets, pt
        tol = _TOL * max(1.0, float(np.abs(self.x0).max()))
        last = ""
        for i_s, solver in enumerate((cp.CLARABEL, cp.ECOS)):
            try:
                self.prob.solve(solver=solver)
            except cp.error.SolverError as e:
                last = str(e)
                continue
            if self.prob.status == "infeasible":
                raise InfeasibleProjectionError("F2 が実行不可能")
            if self.prob.status == "optimal" and self.y.value is not None:
                y = np.asarray(self.y.value, dtype=float)
                if self._violation(y) <= tol:
                    self.n_fallback += i_s
                    return y
                last = f"違反 {self._violation(y):.2e}"
            else:
                last = str(self.prob.status)
        raise SolverNotConvergedError(f"F2 が解けない: {last}")


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--setting", choices=SETTINGS, required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    st = args.setting
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cfg = dataclasses.replace(s47_final(), cash_floor=BETA_CASH)
    dcs = load_direction_constraints(_CANONICAL)
    na_coef = members["jppfs_cor_NetAssets"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    w_full = 1.0 / np.maximum(sd["sigma_1"].to_numpy(), SIGMA_FLOOR) ** 2
    m = np.array([10.0 if n in _PAID else 1.0 for n in names])
    w = w_full / m ** 2
    wj = json.loads(Path("data/interim/stage70/weights.json").read_text(encoding="utf-8"))
    cp0, cm0 = move_cost(wj["pools"]["72"], names, False)
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    code_of = agg["edinetCode"].astype(str)
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]

    def load(cls: str, src: list[str]) -> dict[str, np.ndarray]:
        """設定ごとの目標(a2・f3*)または候補(f1・f2)。"""
        if st == "a2":
            z = np.load("data/interim/stage70/targets_" + cls + "_main.npz", allow_pickle=True)
            key = "t0.01__argmax"
        else:
            z = np.load(DATA / f"plan_{cls}.npz", allow_pickle=True)
            key = {"f3": "t0.05__face2", "f3h0": "t0.05__face0", "f3t05": "t0.5__face2"}.get(
                st, "")
        if [str(v) for v in z["source_docs"]] != src:
            raise ValueError("始点の並びが一致しない")
        if st in ("f1", "f2"):
            return {"idx": np.asarray(z["t0.05__idx"]), "pt": np.asarray(z["t0.05__pt"])}
        return {"tau": np.asarray(z[key], dtype=float)}

    def psi(d: np.ndarray) -> np.ndarray:
        out: np.ndarray = (np.where(d > 0, cp0, cm0) * np.abs(d)).sum(axis=-1)
        return out

    m1 = sources("c1", x, lab, docs_a)
    x1 = x[m1]
    t1 = load("c1", [str(d) for d in docs_a[m1]])
    if st in ("f1", "f2"):
        cand = xt[t1["idx"]]  # (n, 10, 32)
        d = cand - x1[:, None, :]
        first = (0.5 * (w * d ** 2).sum(axis=2) if st == "f1"
                 else (np.sqrt(w) * np.abs(d)).sum(axis=2))
        num, den = (t1["pt"] * first).sum(axis=1), (t1["pt"] * psi(d)).sum(axis=1)
        s = float(np.median(num[den > 0] / den[den > 0]))
    else:
        s = lam_scale(x1, t1["tau"], RecourseWeights(w, na_coef, 0.0, cp0, cm0, 0.0))
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(dd) for dd in docs_a[mask]]
    tg = load(args.cls, src)
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    lams = list(LAMS)[i_sh::n_sh]
    print(f"[{args.cls} {st}] s = {s:.4g}、λ̃ = {lams}", flush=True)
    summary = []
    for lt in lams:
        lam = lt * s
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        n_infeas, n_fb, n_cand = 0, 0, 0
        for i, doc in enumerate(src):
            x0 = xs[i]
            lk = f"72:{code_of.get(doc, '')}"
            cpi, cmi = move_cost(wj["loo"][lk], names, False) if lk in wj["loo"] else (cp0, cm0)
            inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
            frozen_i = FROZEN + tuple(a for a in inactive if a not in FROZEN)
            extra = build_linear_constraints(spec, names, x0, cfg) + liability_cap(
                names, members, x0)
            status = "ok"
            try:
                if st == "f2":
                    k = tg["pt"][i] > 0
                    model = ExpectedL1(spec, x0, w, cpi, cmi, lam, extra, frozen_i, int(k.sum()))
                    y = model.solve(xt[tg["idx"][i][k]], tg["pt"][i][k])
                    n_fb += model.n_fallback
                    n_cand += int(k.sum())
                else:
                    rw = RecourseWeights(w, na_coef, 0.0, cpi, cmi, 0.0)
                    qp = RecourseQP(spec, x0, rw, lam, extra_constraints=extra,
                                    direction_constraints=dcs, frozen_accounts=frozen_i)
                    if st == "f1":
                        k = tg["pt"][i] > 0
                        ys = [qp.solve(xt[j]).y for j in tg["idx"][i][k]]
                        y = (tg["pt"][i][k][:, None] * np.array(ys)).sum(axis=0)
                        n_cand += int(k.sum())
                    else:
                        y = qp.solve(tg["tau"][i]).y
                    n_fb += qp.n_fallback
            except InfeasibleProjectionError:
                y, status = x0.copy(), "infeasible"
                n_infeas += 1
            tr, pr = decompose_re_change(x0, y, names)
            row: dict[str, object] = {
                "tag": "nearest_na0", "config": "s47_final", "fold": 0, "method": "PCFM",
                "docID": doc, "filerName": str(filer.get(doc, "")),
                "assets_start": float(agg.loc[doc, "total_assets"]), "status": status,
                "active": "", "re_branch": -1, "re_transfer": float(tr), "re_profit": float(pr),
                "n_inactive": len(inactive)}
            for j_, n in enumerate(names):
                row[f"x0_{n}"] = float(x0[j_])
                row[f"x1_{n}"] = float(y[j_])
            rows.append(row)
        out_dir = DATA / "gen" / args.cls / f"{st}__l{lt:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        for seed in (0, 1, 2):
            df.assign(seed=seed).to_parquet(out_dir / f"nearest_na0__seed{seed}.parquet",
                                            index=False)
        summary.append({"cls": args.cls, "setting": st, "lam_tilde": lt, "lam": lam, "s": s,
                        "n_infeasible": n_infeas, "n_fallback": n_fb, "n_candidates": n_cand,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  λ̃ = {lt:g}: 不可能 {n_infeas}、代替ソルバ {n_fb}"
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    (OUT / "qp").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "qp" / f"{args.cls}_{st}_{i_sh}of{n_sh}.csv", index=False)


if __name__ == "__main__":
    main()
