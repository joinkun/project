#!/usr/bin/env python
"""ステージ83: パートA・B の D1 の比較、追加の指標、ゲートの判定(lightgbm・sklearn のプロセス)。

規則は ``results/stage83/criteria_stage83.md`` §1・§2・§5。集計は ``stage79_evaluate.run``、
追加の指標はステージ82 と同じ定義(P_distress・R_void・始点ゼロ科目の発生率)。

出力: ``results/stage83/partA/``・``partB/``(``comparison_report.md``・``summary.json``・
``extras.csv``)、``results/stage83/gate.json``。

    uv run python scripts/stage83_evaluate.py
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
from stage83_common import D82, DATA, FOLDS, NAMES, OUT
from tables_stage69 import setup_excl
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.evaluation.classifiers import strict_sklearn_convergence
from fsgen.models.unit_costs import emergence_rate

RULES = "`results/stage83/criteria_stage83.md`"
MAIN = NAMES["KDE0p"]
PART_A = {
    MAIN: Setting79(DATA / "a", "KDE0p_prop_base_t123", "", plan_dir=DATA,
                    plan_suffix="KDE0p_prop"),
    NAMES["KDE0"]: Setting79(D82 / "e1", "KDE0_prop_base_t123", "", plan_dir=D82,
                             plan_suffix="KDE0_prop"),
    NAMES["R1p"]: Setting79(D82 / "e2", "R1p_prop_base_t123", "", plan_dir=D82,
                            plan_suffix="R1p_prop"),
}
OT_NAMES = {"pure_lp": "pure_lp", "top5": "topk_hard_5", "top8": "topk_hard_8",
            "sinkhorn": "sinkhorn", "row": "quad_row_only", "col": "quad_col_only"}
PART_B = {"proposed_quad_unbalanced": PART_A[MAIN],
          **{v: Setting79(DATA / "b", f"KDE0p_{k}_base_t123", "", plan_dir=DATA,
                          argmax=k == "pure_lp", plan_suffix=f"KDE0p_{k}")
             for k, v in OT_NAMES.items()}}


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    parts = {"partA": (PART_A, MAIN, "# ステージ83 パートA: KDE0 + stage1_penalty の D1"),
             "partB": (PART_B, "proposed_quad_unbalanced",
                       "# ステージ83 パートB: 輸送の方式のアブレーション"
                       "(KDE0 + stage1_penalty、D1)")}
    for key, (sets, base, title) in parts.items():
        run(sets, OUT / key, title, RULES, base_key=base)
        print(f"{key} の比較を書いた", flush=True)
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
        c60[cls] = (np.array(meta["c_plus"], float), np.array(meta["c_minus"], float))
        nn = np.empty(len(q_i))
        for s in range(0, len(q_i), 100):
            qq = q_i[s:s + 100]
            dd = psi(xh[None, :, :] - xh[qq][:, None, :], *c60[cls])
            dd[ch[None, :] == ch[qq][:, None]] = np.inf
            nn[s:s + 100] = dd.min(axis=1)
        r90[cls] = float(np.quantile(nn, 0.9))

    def extras(st: Setting79, cfg: str, lt: float, cls: str) -> dict[str, float]:
        e51._GEN = st.data / f"gen_{cfg}"
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
    for key, (sets, _b, _t) in parts.items():
        sm = json.loads((OUT / key / "summary.json").read_text(encoding="utf-8"))
        rows = []
        for name, st in sets.items():
            for cfg in FOLDS:
                lt = sm["res"][name]["star"][cfg]
                if lt is None:
                    continue
                for cls in ("c2", "c1"):
                    rows.append({"setting": name, "fold": cfg, "cls": cls,
                                 **extras(st, cfg, float(lt), cls)})
            print(f"  {key} {name}: 追加の指標", flush=True)
        pd.DataFrame(rows).groupby(["setting", "cls"])[
            ["p_distress", "r_void", "emerge_any", "emerge_mat"]].mean().reset_index().to_csv(
            OUT / key / "extras.csv", index=False)
    # ゲート(criteria §5)
    sm = json.loads((OUT / "partA" / "summary.json").read_text(encoding="utf-8"))
    ex = pd.read_csv(OUT / "partA" / "extras.csv").set_index(["setting", "cls"])
    a = [sm["summary"][MAIN][c] for c in ("c2", "c1")]
    r1p = [sm["summary"][NAMES["R1p"]][c] for c in ("c2", "c1")]
    qp = pd.concat([pd.read_csv(f) for f in sorted((OUT / "qp").glob("*.csv"))])
    main_qp = qp[(qp["setting"].isin(["KDE0p_prop_base_t123", "KDE0p_all_prop_base_t123"]))]
    stars = [v for v in sm["res"][MAIN]["star"].values() if v is not None]
    g1 = all(x_["viol_max"] == 0 for x_ in a) and int(main_qp["n_infeasible"].sum()) == 0
    g2 = all(float(ex.loc[(MAIN, c), "emerge_mat"]) < 0.01 for c in ("c2", "c1"))
    c4 = all(x_["ess"] >= 3 and x_["auc"] <= (0.762 if c == "c2" else 0.736)
             and x_["e_zero"] >= 6.70 for x_, c in zip(a, ("c2", "c1"), strict=True))
    g3_69 = (all(-1 not in x_["sign69"] for x_ in a) and all(1 in x_["sign69"] for x_ in a)
             and c4 and all(x_["viol_max"] == 0 for x_ in a))
    # R1p の記号は「R1p − 主モデル」なので、+ が主モデルの「劣る」にあたる
    g3_r1 = all(1 not in x_.get("signR1", [1]) for x_ in r1p)
    gate: dict[str, Any] = {
        "gate1_no_violation_no_infeasible": g1, "gate2_emergence_below_1pct": g2,
        "gate3_vs_stage69_conditions_1_4": g3_69, "gate3_not_worse_than_R1p": g3_r1,
        "passed": bool(g1 and g2 and g3_69 and g3_r1),
        "lambda_tilde": float(np.median(stars)) if len(stars) == len(FOLDS) else None,
        "stars": sm["res"][MAIN]["star"],
        "n_infeasible_main": int(main_qp["n_infeasible"].sum()),
        "emerge_mat": {c: float(ex.loc[(MAIN, c), "emerge_mat"]) for c in ("c2", "c1")},
        "sign69": {c: x_["sign69"] for c, x_ in zip(("c2", "c1"), a, strict=True)},
        "sign_R1p_minus_main": {c: x_.get("signR1") for c, x_ in zip(("c2", "c1"), r1p,
                                                                      strict=True)},
        "ess": [x_["ess"] for x_ in a], "auc": [x_["auc"] for x_ in a],
        "e_zero": [x_["e_zero"] for x_ in a]}
    (OUT / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
