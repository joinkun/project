#!/usr/bin/env python
"""ステージ79: A1・A2 の Sinkhorn と、R1・A1 のスパース OT による F1 の生成(cvxpy のプロセス)。

規則は ``results/stage79/criteria.md`` §2・§4。その他はステージ75 の F1 と同じ(𝓕⁺、m = 10、
新設計、利益剰余金の移動コスト 0)。

- ``A1_f1``: A1 のコスト、Sinkhorn の候補(``A1__idx``・``A1__pt``)。
- ``A2_f1``: A2 の区分線形のコスト、Sinkhorn の候補。候補 j の QP の c± は、目標までの移動 t で
  評価した平均の傾き Ψ(t)/t(移動の向きの側)。逆の向きの側は小さい区間の傾き c₁。
- ``R1sp_f1``・``A1sp_f1``: スパース OT の非ゼロの候補すべて(重み = p_ij、カットオフなし)。

λ = λ̃ × s。s は class 1 の F1 の定義(候補の重みつきの D / Ψ の中央値)。

出力: ``data/interim/stage79/gen_{cfg}/{cls}/{設定}__l{λ̃}``、``results/stage79/qp/``。

    uv run python scripts/stage79_qp.py --cls c1 --cfg f0 --setting A2_f1 --shard 0/2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qp_stage60 import SIGMA_FLOOR, s47_final
from qp_stage66 import BETA_CASH, LAMS, sources
from stage72_qp import liability_cap
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    INACTIVE_ACCOUNTS,
    build_linear_constraints,
    decompose_re_change,
    inactive_accounts,
    subtotal_members,
)
from fsgen.accounting.projection_qp import load_direction_constraints
from fsgen.accounting.recourse_qp import RecourseQP, RecourseWeights
from fsgen.errors import InfeasibleProjectionError

DATA = Path("data/interim/stage79")
OUT = Path("results/stage79")
_CANONICAL = Path("conf/accounts/core.yaml")
_PAID = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus")
_RE = "jppfs_cor_RetainedEarnings"
FROZEN = ("valuation_difference_securities",)
# 設定 → (コスト, 計画のキー)
SETTINGS = {"A1_f1": ("A1", "A1"), "A2_f1": ("A2", "A2"), "R1sp_f1": ("R1", "R1sp"),
            "A1sp_f1": ("A1", "A1sp")}


class A2Slopes:
    """A2 の区分線形の費用と、候補ごとの平均の傾き。"""

    def __init__(self, e: dict[str, list[float]], sigma: np.ndarray, re_idx: int) -> None:
        self.c1 = np.array([e["c1_plus"], e["c1_minus"]], dtype=float)
        self.c2 = np.array([e["c2_plus"], e["c2_minus"]], dtype=float)
        self.m = np.array([e["m_plus"], e["m_minus"]], dtype=float)
        self.c1[:, re_idx] = self.c2[:, re_idx] = 0.0
        self.sigma = sigma

    def psi(self, x0: np.ndarray, d: np.ndarray) -> np.ndarray:
        """x0 (…, D) から d (…, D) 動かす科目ごとの費用。"""
        up = d > 0
        t = np.abs(d)
        b = np.where(up, self.m[0], self.m[1]) * (np.abs(x0) + self.sigma)
        k1 = np.where(up, self.c1[0], self.c1[1])
        k2 = np.where(up, self.c2[0], self.c2[1])
        out: np.ndarray = k1 * np.minimum(t, b) + k2 * np.maximum(t - b, 0.0)
        return out

    def slopes(self, x0: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """目標までの移動 d の候補で使う (c⁺, c⁻)。移動の向きの側だけ平均の傾きにする。"""
        t = np.abs(d)
        avg = np.where(t > 0, self.psi(x0, d) / np.where(t > 0, t, 1.0), 0.0)
        cp = np.where(d > 0, avg, self.c1[0])
        cm = np.where(d < 0, avg, self.c1[1])
        return cp, cm


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls", choices=("c2", "c1"), required=True)
    ap.add_argument("--cfg", choices=("f0", "f1", "f2", "f3", "f4"), required=True)
    ap.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    kind, pkey = SETTINGS[args.setting]
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cfg = dataclasses.replace(s47_final(), cash_floor=BETA_CASH)
    dcs = load_direction_constraints(_CANONICAL)
    na_coef = members["jppfs_cor_NetAssets"]
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    sigma = np.maximum(sd["sigma_1"].to_numpy(float), SIGMA_FLOOR)
    w = 1.0 / sigma ** 2 / np.array([10.0 if n in _PAID else 1.0 for n in names]) ** 2
    wj = json.loads((DATA / "weights.json").read_text(encoding="utf-8"))
    re_idx = names.index(_RE)
    a2 = A2Slopes(wj["configs"][args.cfg]["A2"], sigma, re_idx) if kind == "A2" else None
    cp = cm = np.zeros(len(names))
    if a2 is None:
        e = wj["configs"][args.cfg][kind]
        cp, cm = np.array(e["plus"], dtype=float), np.array(e["minus"], dtype=float)
        cp[re_idx] = cm[re_idx] = 0.0
    x, lab, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]

    def plan(cls: str, src: list[str]) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(DATA / f"plan_{cls}_{args.cfg}.npz", allow_pickle=True)
        if [str(v) for v in z["source_docs"]] != src:
            raise ValueError("始点の並びが一致しない")
        return np.asarray(z[f"{pkey}__idx"]), np.asarray(z[f"{pkey}__pt"])

    def move_cost(x0: np.ndarray, d: np.ndarray) -> np.ndarray:
        if a2 is not None:
            out: np.ndarray = a2.psi(x0, d).sum(axis=-1)
            return out
        return (np.where(d > 0, cp, cm) * np.abs(d)).sum(axis=-1)

    m1 = sources("c1", x, lab, docs_a)
    x1 = x[m1]
    idx1, pt1 = plan("c1", [str(d) for d in docs_a[m1]])
    d1 = xt[idx1] - x1[:, None, :]
    num = (pt1 * 0.5 * (w * d1 ** 2).sum(axis=2)).sum(axis=1)
    den = (pt1 * move_cost(x1[:, None, :], d1)).sum(axis=1)
    s = float(np.median(num[den > 0] / den[den > 0]))
    mask = sources(args.cls, x, lab, docs_a)
    xs = x[mask]
    src = [str(dd) for dd in docs_a[mask]]
    idx, pt = plan(args.cls, src)
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    lams = list(LAMS)[i_sh::n_sh]
    print(f"[{args.cls} {args.cfg} {args.setting}] s = {s:.4g}、λ̃ = {lams}", flush=True)
    summary = []
    for lt in lams:
        lam = lt * s
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []
        n_infeas, n_fb = 0, 0
        for i, doc in enumerate(src):
            x0 = xs[i]
            inactive = inactive_accounts(x0, names, INACTIVE_ACCOUNTS)
            frozen_i = FROZEN + tuple(a for a in inactive if a not in FROZEN)
            extra = build_linear_constraints(spec, names, x0, cfg) + liability_cap(members, x0)
            status = "ok"
            k = pt[i] > 0
            try:
                ys = []
                if a2 is None:
                    qp = RecourseQP(spec, x0, RecourseWeights(w, na_coef, 0.0, cp, cm, 0.0), lam,
                                    extra_constraints=extra, direction_constraints=dcs,
                                    frozen_accounts=frozen_i)
                    ys = [qp.solve(xt[j]).y for j in idx[i][k]]
                    n_fb += qp.n_fallback
                else:
                    for j in idx[i][k]:
                        cpj, cmj = a2.slopes(x0, xt[j] - x0)
                        qpj = RecourseQP(spec, x0, RecourseWeights(w, na_coef, 0.0, cpj, cmj, 0.0),
                                         lam, extra_constraints=extra, direction_constraints=dcs,
                                         frozen_accounts=frozen_i)
                        ys.append(qpj.solve(xt[j]).y)
                        n_fb += qpj.n_fallback
                y = (pt[i][k][:, None] * np.array(ys)).sum(axis=0)
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
        out_dir = DATA / f"gen_{args.cfg}" / args.cls / f"{args.setting}__l{lt:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        for seed in (0, 1, 2):
            df.assign(seed=seed).to_parquet(out_dir / f"nearest_na0__seed{seed}.parquet",
                                            index=False)
        summary.append({"cls": args.cls, "cfg": args.cfg, "setting": args.setting,
                        "lam_tilde": lt, "lam": lam, "s": s, "n_infeasible": n_infeas,
                        "n_fallback": n_fb, "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  λ̃ = {lt:g}: 不可能 {n_infeas}、代替ソルバ {n_fb}"
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    (OUT / "qp").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(
        OUT / "qp" / f"{args.cls}_{args.cfg}_{args.setting}_{i_sh}of{n_sh}.csv", index=False)


if __name__ == "__main__":
    main()
