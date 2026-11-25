#!/usr/bin/env python
"""ステージ83 パートC: 実在の再建企業への out-of-fold のバックテストの指標。

lightgbm・sklearn のプロセス。

規則は ``results/stage83/criteria_stage83.md`` §3。``stage83_backtest_qp.py`` の生成 ŷ を、総資産で
割り直した構成 ŷ / A(ŷ) にして、実績 x₁(2024 年度の構成)と比べる。

出力: ``results/stage83/backtest_firms.csv``(企業ごと)・``backtest_summary.csv``・
``backtest_ratios.csv``・``backtest_cases.md``、
``data/interim/stage83/backtest_arrays.npz``(図 2 用)。

    uv run python scripts/stage83_backtest_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stage76_feasibility_map import N_FOLD, SEED_CLF, make_clf
from stage82_common import COSTS
from stage82_final import f3, pct, ratios
from stage83_common import DATA, FOLDS, OUT, recovery_groups
from stage83_common import compositions as comps
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.evaluation.classifiers import strict_sklearn_convergence
from fsgen.models.unit_costs import inactive_mask, individual_theta, psi

_RE = "jppfs_cor_RetainedEarnings"
EPS = (1e-2, 5e-3, 2e-2)
CASE_ACCOUNTS = ("jppfs_cor_CashAndDeposits", "receivables_trade",
                 "jppfs_cor_PropertyPlantAndEquipment", "short_term_loans", "long_term_loans",
                 "jppfs_cor_CapitalStock", "jppfs_cor_RetainedEarnings")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    idx = {n: i for i, n in enumerate(names)}
    tj = json.loads((DATA / "theta.json").read_text(encoding="utf-8"))
    sigma = np.array(tj["sigma"], dtype=float)
    x, lab, docs, _ = load_tilde(names)
    xt = x[lab == 0]
    med_healthy = np.median(xt, axis=0)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    codes_all = np.array([code_of.get(d, d) for d in docs])
    uc = np.random.default_rng(SEED_CLF).permutation(np.unique(codes_all))
    fold_of = {c: i for i, part in enumerate(np.array_split(uc, N_FOLD)) for c in part}
    fold = np.array([fold_of[c] for c in codes_all])
    yv = (lab >= 1).astype(int)
    prior = float(yv.sum() / (1 - yv).sum())
    models = []
    for f in range(N_FOLD):
        clf = make_clf("lightgbm")
        with strict_sklearn_convergence():
            clf.fit(x[fold != f], yv[fold != f])
        models.append(clf)

    def pdist(v: np.ndarray, code: str) -> float:
        f = fold_of.get(code)
        ps = ([models[f].predict_proba(v[None, :])[0, 1]] if f is not None
              else [m.predict_proba(v[None, :])[0, 1] for m in models])
        p = float(np.mean(ps))
        o = p / max(1.0 - p, 1e-12) * prior
        return o / (1.0 + o)

    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {"x0": [], "x1": [], "y": [], "code": [], "cls": []}
    groups = recovery_groups()
    for grp in (*FOLDS, "c2rec"):
        x0, x1, ta, ok = comps(groups[grp], names)
        cfg = "full" if grp == "c2rec" else grp
        th = np.array(tj["configs"][cfg]["KDE0"]["theta"], dtype=float)
        sp = COSTS["KDE0p"]
        plan = np.load(DATA / f"bt_plan_{grp}_KDE0p.npz", allow_pickle=True)
        nn = np.asarray(plan["nn"])
        # モデル → (構成 ŷ / A(ŷ), 実行不可能の数, A(ŷ)(始点の総資産比))
        gens: dict[str, tuple[np.ndarray, int, np.ndarray]] = {}
        for cost in ("KDE0p", "KDE0"):
            g = pd.read_parquet(DATA / f"bt_gen_{grp}_{cost}.parquet")
            for kind, gk in g.groupby("lam_kind"):
                gk = gk.set_index("code").loc[ok]
                y = gk[[f"y_{n}" for n in names]].to_numpy(float)
                ay = y @ members["jppfs_cor_Assets"]
                gens[f"{cost}|{kind}"] = (y / ay[:, None], int((gk["status"] != "ok").sum()), ay)
        ones = np.ones(len(x0))
        gens["1-NN"] = (xt[nn], 0, ones)
        gens["健全企業の中央値"] = (np.repeat(med_healthy[None, :], len(x0), axis=0), 0, ones)
        for i, code in enumerate(ok):
            t = individual_theta(th, x0[i], sigma, indiv_scale=sp.indiv,
                                 inactive=inactive_mask(x0[i], names))
            t[:, idx[_RE]] = 0.0
            dr = x1[i] - x0[i]
            base_psi = float(psi(t, dr))
            for model, (yy, n_inf, ay_) in gens.items():
                dg = yy[i] - x0[i]
                r: dict[str, Any] = {"group": grp, "cls": "c2" if grp == "c2rec" else "c1",
                                     "code": code, "model": model, "n_infeasible": n_inf}
                for e in EPS:
                    act = np.abs(dr) >= e
                    r[f"dsa_{e:g}"] = (float((np.sign(dg[act]) == np.sign(dr[act])).mean())
                                       if act.any() else np.nan)
                den = np.linalg.norm(dg) * np.linalg.norm(dr)
                r["cos"] = float(dg @ dr / den) if den > 0 else np.nan
                r["rho"] = spearman(dg, dr)
                r["err_l1"] = 1.0 - float(np.abs(yy[i] - x1[i]).sum() / np.abs(dr).sum())
                r["err_psi"] = (1.0 - float(psi(t, x1[i] - yy[i])) / base_psi
                                if base_psi > 0 else np.nan)
                r["p_x0"], r["p_x1"], r["p_y"] = (pdist(v, code) for v in (x0[i], x1[i], yy[i]))
                r["ta0"], r["ta1"] = float(ta[i, 0]), float(ta[i, 1])
                for n in CASE_ACCOUNTS:
                    r[f"x0_{n}"], r[f"x1_{n}"] = x0[i, idx[n]], x1[i, idx[n]]
                    r[f"y_{n}"] = yy[i, idx[n]]
                for lab_, v in (("x0", x0[i]), ("x1", x1[i]), ("y", yy[i])):
                    r[f"{lab_}_NetAssets"] = float(v @ members["jppfs_cor_NetAssets"])
                    rv, _exc = ratios(v[None, :], members, idx)
                    for k_, val in rv.items():
                        r[f"{lab_}|{k_}"] = float(val[0])
                r["y_assets_ratio"] = float(ay_[i])
                if model == "KDE0p|main":
                    arrays["x0"].append(x0[i])
                    arrays["x1"].append(x1[i])
                    arrays["y"].append(yy[i])
                    arrays["code"].append(np.array(code))
                    arrays["cls"].append(np.array(r["cls"]))
                rows.append(r)
        print(f"  {grp}: {len(ok)} 社", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "backtest_firms.csv", index=False)
    np.savez(DATA / "backtest_arrays.npz",
             **{k: np.array(v) for k, v in arrays.items()})  # type: ignore[arg-type]
    # 要約
    summ = []
    met = [f"dsa_{e:g}" for e in EPS] + ["cos", "rho", "err_l1", "err_psi"]
    for (cls, model), g in df.groupby(["cls", "model"]):
        r = {"cls": cls, "model": model, "n": len(g), "n_infeasible": int(g["n_infeasible"].max())}
        for m in met:
            v = g[m].dropna()
            r |= {f"{m}_med": float(v.median()), f"{m}_q1": float(v.quantile(0.25)),
                  f"{m}_q3": float(v.quantile(0.75)), f"{m}_pos": float((v > 0).mean()),
                  f"{m}_n": int(len(v))}
        r |= {f"p_{k}_med": float(g[f"p_{k}"].median()) for k in ("x0", "x1", "y")}
        summ.append(r)
    sd = pd.DataFrame(summ)
    sd.to_csv(OUT / "backtest_summary.csv", index=False)
    rr = []
    ratio_names = [c.split("|", 1)[1] for c in df.columns if c.startswith("x0|")]
    for (cls, model), g in df.groupby(["cls", "model"]):
        for k in ratio_names:
            a0, a1, ay = g[f"x0|{k}"], g[f"x1|{k}"], g[f"y|{k}"]
            ok_ = a0.notna() & a1.notna() & ay.notna()
            d1, dy = (a1 - a0)[ok_], (ay - a0)[ok_]
            rr.append({"cls": cls, "model": model, "ratio": k, "n": int(ok_.sum()),
                       "x0_med": float(a0[ok_].median()), "x1_med": float(a1[ok_].median()),
                       "y_med": float(ay[ok_].median()),
                       "corr_change": (float(np.corrcoef(d1, dy)[0, 1]) if ok_.sum() > 2
                                       else np.nan),
                       "mae": float((ay - a1)[ok_].abs().mean())})
    pd.DataFrame(rr).to_csv(OUT / "backtest_ratios.csv", index=False)
    # 事例(class ごとに代表例と乖離の例、主モデル)
    lines = ["# ステージ83 パートC: 再建企業の事例(主モデル、out-of-fold)", "",
             "構成は各年度の総資産比(%)。生成 ŷ は ŷ / A(ŷ)。"
             "総資産は百万円(生成は始点の総資産 × A(ŷ))。",
             "代表例 = DSA(ε = 10⁻²)と ERR^{L1} の順位の平均が中央値に最も近い企業、"
             "乖離の例 = ERR^{L1} が最も小さい企業(criteria §3)。", ""]
    ja = {"jppfs_cor_CashAndDeposits": "現金及び預金", "receivables_trade": "売上債権",
          "jppfs_cor_PropertyPlantAndEquipment": "有形固定資産", "short_term_loans": "短期借入金",
          "long_term_loans": "長期借入金", "jppfs_cor_CapitalStock": "資本金",
          "jppfs_cor_RetainedEarnings": "利益剰余金", "NetAssets": "純資産合計"}
    for cls in ("c2", "c1"):
        g = df[(df["cls"] == cls) & (df["model"] == "KDE0p|main")].copy()
        rk = (g["dsa_0.01"].rank(pct=True) + g["err_l1"].rank(pct=True)) / 2
        rep = g.loc[(rk - 0.5).abs().idxmin()]
        bad = g.loc[g["err_l1"].idxmin()]
        for tag, r in (("代表例", rep), ("乖離の例", bad)):
            lines += [f"## {cls} {tag}: {r['code']}(DSA {f3(r['dsa_0.01'])}、Cos {f3(r['cos'])}、"
                      f"ERR^L1 {f3(r['err_l1'])}、ERR^ψ {f3(r['err_psi'])})", "",
                      "| 科目 | 再建前 x₀ | 実績 x₁ | 生成 ŷ |", "|---|---:|---:|---:|"]
            for n in (*CASE_ACCOUNTS, "NetAssets"):
                lines.append(f"| {ja[n]} | {pct(r[f'x0_{n}'])} | {pct(r[f'x1_{n}'])} "
                             f"| {pct(r[f'y_{n}'])} |")
            lines += [f"| 総資産(百万円) | {r['ta0'] / 1e6:,.0f} | {r['ta1'] / 1e6:,.0f} "
                      f"| {r['ta0'] * r['y_assets_ratio'] / 1e6:,.0f} |",
                      f"| 苦境の確率 | {pct(r['p_x0'])} | {pct(r['p_x1'])} | {pct(r['p_y'])} |", ""]
    (OUT / "backtest_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(sd[["cls", "model", "n", "dsa_0.01_med", "cos_med", "err_l1_med", "err_l1_pos",
              "err_psi_med"]].round(3).to_string())


if __name__ == "__main__":
    main()
