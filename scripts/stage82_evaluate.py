#!/usr/bin/env python
"""ステージ82: 実験 1〜3 の D1 の比較と、追加の指標・主モデルの選択(lightgbm・sklearn のプロセス)。

規則は ``results/stage82/criteria.md`` §5・§6。集計は ``stage79_evaluate.run``
(ステージ75 と同じ較正・参照・条件)。追加の指標は、各設定の λ̃*_f での終点について
fold ごとに計算し、fold で平均する。

- P_distress: LightGBM(苦境 vs 健全、企業単位の 5 分割、乱数 42)の確率の中央値(学習データの有病率に
  戻したもの。ステージ76 と同じ)。
- R_void: 健全企業が Ψ60(z − y) ≤ r90 に 1 社もない終点の割合(Ψ60 はステージ60 の class ごとの c±、
  r90 は健全企業 2,000 社の最近傍の 90% 点)。
- 始点ゼロ科目の発生率(|y| ≥ 10⁻⁶・≥ 10⁻³)。

出力: ``results/stage82/exp{1,2,3}/``(``comparison_report.md``・``summary.json``・``extras.csv``)、
``results/stage82/selection.json``。

    uv run python scripts/stage82_evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from stage76_feasibility_map import N_FOLD, SEED_CLF, make_clf, psi
from stage79_evaluate import Setting as Setting79
from stage79_evaluate import run
from stage82_common import BASE, DATA, EXP1, EXP2, EXP3, FOLDS, OUT, Setting
from tables_stage69 import setup_excl
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.evaluation.classifiers import strict_sklearn_convergence
from fsgen.models.unit_costs import emergence_rate

RULES = "`results/stage82/criteria.md`"
EXPS: dict[str, tuple[dict[str, Setting], str]] = {
    "exp1": (EXP1, "# ステージ82 実験 1: 基準単価 θ の 5 方式(D1、5 fold の交差適合)"),
    "exp2": (EXP2, "# ステージ82 実験 2: 個体別の単価 × 不活性科目の扱い(D1)"),
    "exp3": (EXP3, "# ステージ82 実験 3: 輸送の方式のアブレーション(D1)"),
}


def to79(st: Setting) -> Setting79:
    return Setting79(DATA / st.group, st.tag, "", plan_dir=DATA, argmax=st.ot == "pure_lp",
               plan_suffix=f"{st.cost}_{st.ot}")


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    for key, (sets, title) in EXPS.items():
        base_name = next(k for k, v in sets.items() if v == BASE)
        run({k: to79(v) for k, v in sets.items()}, OUT / key, title, RULES, base_key=base_name)
        print(f"{key} の比較を書いた", flush=True)
    # 追加の指標
    setup_excl(names)
    x, lab, docs, _ = load_tilde(names)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    codes = np.array([code_of.get(d, d) for d in docs])
    uc = np.random.default_rng(SEED_CLF).permutation(np.unique(codes))
    fold_of = {c: i for i, part in enumerate(np.array_split(uc, N_FOLD)) for c in part}
    fold = np.array([fold_of[c] for c in codes])
    yv = (lab >= 1).astype(int)
    prior = float(yv.sum() / (1 - yv).sum())
    models = []
    for f in range(N_FOLD):
        clf = make_clf("lightgbm")
        with strict_sklearn_convergence():
            clf.fit(x[fold != f], yv[fold != f])
        models.append(clf)
    healthy = lab == 0
    xh, ch = x[healthy], codes[healthy]
    q_i = np.random.default_rng(1).choice(len(xh), size=2000, replace=False)
    c60: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    r90: dict[str, float] = {}
    for cls in ("c2", "c1"):
        meta = json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(encoding="utf-8"))
        cp, cm = np.array(meta["c_plus"], float), np.array(meta["c_minus"], float)
        c60[cls] = (cp, cm)
        nn = np.empty(len(q_i))
        for s in range(0, len(q_i), 100):
            qq = q_i[s:s + 100]
            dd = psi(xh[None, :, :] - xh[qq][:, None, :], cp, cm)
            dd[ch[None, :] == ch[qq][:, None]] = np.inf
            nn[s:s + 100] = dd.min(axis=1)
        r90[cls] = float(np.quantile(nn, 0.9))

    def extras(st: Setting, cfg: str, lt: float, cls: str) -> dict[str, float]:
        e51._GEN = DATA / st.group / f"gen_{cfg}"
        g = e51.load_gen(cls, f"{st.tag}__l{lt:g}")
        g = g[g["seed"] == 0]
        x0 = g[[f"x0_{n}" for n in names]].to_numpy(float)
        y = g[[f"x1_{n}" for n in names]].to_numpy(float)
        gc = np.array([code_of.get(str(d), str(d)) for d in g["docID"]])
        gf = np.array([fold_of[c] for c in gc])
        p = np.empty(len(y))
        for f in range(N_FOLD):
            sel = gf == f
            if sel.any():
                p[sel] = models[f].predict_proba(y[sel])[:, 1]
        o = p / np.clip(1.0 - p, 1e-12, None) * prior
        dd = psi(xh[None, :, :] - y[:, None, :], *c60[cls])
        dd[ch[None, :] == gc[:, None]] = np.inf
        return {"p_distress": float(np.median(o / (1.0 + o))),
                "r_void": float((dd.min(axis=1) > r90[cls]).mean()),
                "emerge_any": emergence_rate(x0, y, names, 1e-6),
                "emerge_mat": emergence_rate(x0, y, names, 1e-3)}
    rows = []
    for key, (sets, _t) in EXPS.items():
        sm = json.loads((OUT / key / "summary.json").read_text(encoding="utf-8"))
        for name, st in sets.items():
            stars = sm["res"][name]["star"]
            for cfg in FOLDS:
                if stars[cfg] is None:
                    continue
                for cls in ("c2", "c1"):
                    rows.append({"exp": key, "setting": name, "fold": cfg, "cls": cls,
                                 **extras(st, cfg, float(stars[cfg]), cls)})
            print(f"  {key} {name}: 追加の指標", flush=True)
    ex = pd.DataFrame(rows)
    for key in EXPS:
        e = ex[ex["exp"] == key].groupby(["setting", "cls"])[
            ["p_distress", "r_void", "emerge_any", "emerge_mat"]].mean().reset_index()
        e.to_csv(OUT / key / "extras.csv", index=False)
    # 主モデルの選択(criteria §6)
    sm2 = json.loads((OUT / "exp2" / "summary.json").read_text(encoding="utf-8"))
    order = list(EXP2)
    e2 = ex[ex["exp"] == "exp2"].groupby("setting")["emerge_mat"].mean()

    def ok(name: str) -> dict[str, Any]:
        a = [sm2["summary"][name][c] for c in ("c2", "c1")]
        if not all(x_["ok"] for x_ in a):
            return {"c1": False, "c2": False, "c4": False}
        return {"c1": all(x_["viol_max"] == 0 for x_ in a),
                "c2": all(-1 not in x_["sign69"] for x_ in a),
                "c4": all(x_["ess"] >= 3 and x_["auc"] <= (0.762 if c == "c2" else 0.736)
                          and x_["e_zero"] >= 6.70 for x_, c in zip(a, ("c2", "c1"), strict=True))}
    cand = [n for n in order if all(ok(n).values())]
    best = order[0]
    if cand:
        lo = min(float(e2[n]) for n in cand)
        best = next(n for n in order if n in cand and float(e2[n]) <= lo + 0.005)
    sm1 = json.loads((OUT / "exp1" / "summary.json").read_text(encoding="utf-8"))
    k = sm1["summary"]["KDE0"]
    kde_better = (all(-1 not in k[c].get("signR1", [-1]) for c in ("c2", "c1"))
                  and any(1 in k[c].get("signR1", []) for c in ("c2", "c1")))
    st = EXP2[best]
    stars = [v for v in sm2["res"][best]["star"].values() if v is not None]
    sel = {"best": best, "best_setting": {"cost": st.cost, "ot": st.ot, "qp": st.qp},
           "lambda_tilde": float(np.median(stars)) if stars else None,
           "conditions": {n: ok(n) for n in order},
           "emerge_mat_mean": {n: float(e2[n]) for n in order},
           "candidates": cand, "kde0_over_r1": kde_better}
    (OUT / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps(sel, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
