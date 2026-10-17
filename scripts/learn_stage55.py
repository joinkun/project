#!/usr/bin/env python
"""ステージ55 B: 実際の遷移から対角の距離の重みを学習する(numpy・scipy のみ)。

規則は ``results/stage55/criteria.md``(学習の前に記録)。d_w(x, y) = Σ_a w_a (x_a − y_a)²、w ≥ 0、
非操作 3 科目を除く 29 科目。損失(主)は triplet の滑らかな版 softplus((d(正例) − d(負例)) / τ)
+ α Σ (w − 1)²、代替はロジスティック回帰 σ(β0 − d)。α は入れ子の 5 分割で選ぶ。分割は LOO(主)と
群(class 1 で学習 → class 2 で検証)。

出力(``results/stage55/``): ``accuracy.md``・``accuracy.csv``(B.4・D.2)、``weights.md``・
``weights.csv``(B.5)、``conf/accounts/costs_learned.yaml``(C の生成用、72 社すべてで学習)。

    uv run python scripts/learn_stage55.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, log1p
from sklearn.model_selection import KFold
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.cost import load_cost_config

ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
KINDS = ("近い企業", "無作為")


@dataclass
class Data:
    fpos: np.ndarray  # (72, A) 正例の (x − y)²
    fneg: dict[str, np.ndarray]  # 種類 → (72, M, A)
    group: np.ndarray  # 72


def softplus(z: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.where(z > 30, z, log1p(np.exp(np.minimum(z, 30))))
    return out


def fit_triplet(fpos: np.ndarray, fneg: np.ndarray, alpha: float) -> np.ndarray:
    diff = fpos[:, None, :] - fneg  # (n, M, A)
    tau = float(np.median(np.abs(diff.sum(axis=2)))) or 1.0
    a = diff.shape[2]

    def obj(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = diff @ w / tau
        loss = float(softplus(z).mean() + alpha * np.sum((w - 1) ** 2))
        g = (expit(z)[..., None] * diff).mean(axis=(0, 1)) / tau + 2 * alpha * (w - 1)
        return loss, g

    res = minimize(obj, np.ones(a), jac=True, method="L-BFGS-B", bounds=[(0, None)] * a,
                   options={"maxiter": 5000})
    if not res.success:
        raise RuntimeError(f"L-BFGS-B が収束しない: {res.message}")
    out: np.ndarray = res.x
    return out


def fit_logistic(fpos: np.ndarray, fneg: np.ndarray, alpha: float) -> np.ndarray:
    feats = np.concatenate([fpos, fneg.reshape(-1, fneg.shape[2])])
    y = np.r_[np.ones(len(fpos)), np.zeros(fneg.shape[0] * fneg.shape[1])]
    sw = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (1 - y).sum()))  # 釣り合いの重み
    scale = float(np.median(feats.sum(axis=1))) or 1.0
    a = feats.shape[1]

    def obj(p: np.ndarray) -> tuple[float, np.ndarray]:
        b0, w = p[0], p[1:]
        z = b0 - feats @ w / scale
        pr = expit(z)
        loss = float(np.mean(sw * (softplus(-z) * y + softplus(z) * (1 - y)))
                     + alpha * np.sum((w - 1) ** 2))
        r = sw * (pr - y) / len(y)
        g = np.r_[r.sum(), -(r @ feats) / scale + 2 * alpha * (w - 1)]
        return loss, g

    res = minimize(obj, np.r_[0.0, np.ones(a)], jac=True, method="L-BFGS-B",
                   bounds=[(None, None)] + [(0, None)] * a, options={"maxiter": 5000})
    if not res.success:
        raise RuntimeError(f"L-BFGS-B が収束しない: {res.message}")
    out: np.ndarray = res.x[1:]
    return out


FITTERS = {"triplet": fit_triplet, "logistic": fit_logistic}


def accuracy(w: np.ndarray, fpos: np.ndarray, fneg: np.ndarray) -> np.ndarray:
    """企業ごとの順位精度(負例のうち正例より遠いものの割合。同点は 0.5)。"""
    dp = fpos @ w
    dn = fneg @ w
    out: np.ndarray = ((dn > dp[:, None]) + 0.5 * (dn == dp[:, None])).mean(axis=1)
    return out


def choose_alpha(loss: str, fpos: np.ndarray, fneg: np.ndarray) -> float:
    best, best_a = -1.0, ALPHAS[-1]
    for a in sorted(ALPHAS, reverse=True):  # 同点は大きい α(等重みに近い方)
        accs = []
        for tr, te in KFold(5, shuffle=True, random_state=0).split(fpos):
            w = FITTERS[loss](fpos[tr], fneg[tr], a)
            accs.append(accuracy(w, fpos[te], fneg[te]))
        m = float(np.concatenate(accs).mean())
        if m > best + 1e-12:
            best, best_a = m, a
    return best_a


def boot_ci(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(v), size=(2000, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def load(names: tuple[str, ...], cols: list[int]) -> Data:
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    neg = pd.read_csv("results/stage55/negatives.csv", dtype={"edinetCode": str, "neg_docID": str})
    x, label, docs, _k = load_tilde(names)
    pool = pd.DataFrame(x[label == 0], index=np.array(docs)[label == 0])
    xt = pairs[[f"xt_{n}" for n in names]].to_numpy()[:, cols]
    xt2 = pairs[[f"xt2_{n}" for n in names]].to_numpy()[:, cols]
    fneg = {}
    for k in KINDS:
        rows = []
        for i, c in enumerate(pairs["edinetCode"]):
            ids = neg[(neg["edinetCode"] == c) & (neg["kind"] == k)]["neg_docID"]
            rows.append((pool.loc[ids].to_numpy()[:, cols] - xt[i]) ** 2)
        fneg[k] = np.array(rows)
    return Data((xt2 - xt) ** 2, fneg, pairs["group"].to_numpy())


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage55"))
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    cc = load_cost_config(Path("conf/accounts/costs_equal.yaml"))
    cols = [i for i, n in enumerate(names) if n not in cc.non_actionable]
    acts = [names[i] for i in cols]
    data = load(names, cols)
    rng = np.random.default_rng(0)
    n = len(data.fpos)
    rows: list[dict[str, Any]] = []
    weights_loo: list[np.ndarray] = []
    for loss in ("triplet", "logistic"):
        for train_kind in KINDS:
            fn = data.fneg[train_kind]
            # LOO
            pred = {k: np.zeros(n) for k in KINDS}
            train_acc = []
            for i in range(n):
                tr = np.array([j for j in range(n) if j != i])
                a = choose_alpha(loss, data.fpos[tr], fn[tr])
                w = FITTERS[loss](data.fpos[tr], fn[tr], a)
                if loss == "triplet" and train_kind == "近い企業":
                    weights_loo.append(w)
                train_acc.append(float(accuracy(w, data.fpos[tr], fn[tr]).mean()))
                for k in KINDS:
                    pred[k][i] = accuracy(w, data.fpos[i:i + 1], data.fneg[k][i:i + 1])[0]
            # 群で分ける(class 1 → class 2)
            tr = np.where(data.group == "class 1 → class 0")[0]
            te = np.where(data.group == "class 2 → 回復")[0]
            a = choose_alpha(loss, data.fpos[tr], fn[tr])
            wg = FITTERS[loss](data.fpos[tr], fn[tr], a)
            splits = (("LOO", np.arange(n), None), ("群(class 1 → class 2)", te, wg))
            for split, te_idx, getp in splits:
                for k in KINDS:
                    if getp is None:
                        acc = pred[k]
                        tr_acc = float(np.mean(train_acc))
                    else:
                        acc = accuracy(getp, data.fpos[te_idx], data.fneg[k][te_idx])
                        tr_acc = float(accuracy(getp, data.fpos[tr], fn[tr]).mean())
                    eq = accuracy(np.ones(len(cols)), data.fpos[te_idx], data.fneg[k][te_idx])
                    lo, hi = boot_ci(acc, rng)
                    dlo, dhi = boot_ci(acc - eq, rng)
                    rows.append({"loss": loss, "train_neg": train_kind, "split": split,
                                 "eval_neg": k, "n": len(acc), "train_acc": tr_acc,
                                 "acc": float(acc.mean()), "lo": lo, "hi": hi,
                                 "equal_acc": float(eq.mean()), "diff": float((acc - eq).mean()),
                                 "dlo": dlo, "dhi": dhi})
    acc_df = pd.DataFrame(rows)
    acc_df.to_csv(args.out / "accuracy.csv", index=False)
    main_row = acc_df[(acc_df["loss"] == "triplet") & (acc_df["train_neg"] == "近い企業")
                      & (acc_df["split"] == "LOO") & (acc_df["eval_neg"] == "近い企業")].iloc[0]
    crit0 = bool(main_row["lo"] > 0.5)
    lines = ["# ステージ55 B.4・D.2: 順位精度", "",
             "順位精度 = 検証の企業ごとに、10 個の負例のうち正例(実際の終点)より遠いものの割合"
             "の平均"
             "(ランダムなら 0.5)。[ ] は企業の復元抽出 2,000 回の 95% 区間。等重み = 学習しない"
             "現行の距離。"
             "学習データの精度は、LOO では 72 回の学習データでの平均。", "",
             "| 損失 | 学習の負例 | 分割 | 検証の負例 | 社数 | 学習データ | **検証** | 等重み |"
             " 学習 − 等重み |",
             "|---|---|---|---|---:|---:|---|---:|---|"]
    for r in acc_df.itertuples():
        lines.append(f"| {r.loss} | {r.train_neg} | {r.split} | {r.eval_neg} | {r.n} "
                     f"| {r.train_acc:.3f} | **{r.acc:.3f}** [{r.lo:.3f}, {r.hi:.3f}] "
                     f"| {r.equal_acc:.3f} | {r.diff:+.3f} [{r.dlo:+.3f}, {r.dhi:+.3f}] |")
    lines += ["", f"**基準 0**(triplet・近い企業で学習・LOO・近い企業で検証): 順位精度 "
              f"{main_row['acc']:.3f} [{main_row['lo']:.3f}, {main_row['hi']:.3f}] → "
              f"{'50% を有意に上回る(満たす)' if crit0 else '満たさない'}。"
              f"等重み(学習なし)は {main_row['equal_acc']:.3f}、学習 − 等重み "
              f"{main_row['diff']:+.3f} [{main_row['dlo']:+.3f}, {main_row['dhi']:+.3f}]。", ""]
    (args.out / "accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- B.5 重み(72 社すべてで学習)----------------
    a_full = choose_alpha("triplet", data.fpos, data.fneg["近い企業"])
    w_full = fit_triplet(data.fpos, data.fneg["近い企業"], a_full)
    wl = np.array(weights_loo)
    p32 = pd.read_csv("results/stage32/persistence.csv").set_index("account")["rho1"]
    est = load_cost_config(Path("conf/accounts/costs_estimated.yaml"))
    freq = load_cost_config(Path("conf/accounts/costs_frequency.yaml"))
    pers = load_cost_config(Path("conf/accounts/costs_persistence.yaml"))

    def mean_cost(cfg: Any) -> pd.Series:
        return pd.Series({s.name: (s.cost_increase + s.cost_decrease) / 2 for s in cfg.accounts})

    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    wdf = pd.DataFrame({"account": acts, "ja": [ja.get(a, a) for a in acts], "w": w_full,
                        "w_loo_lo": np.percentile(wl, 5, axis=0),
                        "w_loo_hi": np.percentile(wl, 95, axis=0),
                        "rho1": p32.reindex(acts).to_numpy(),
                        "cost_estimated": mean_cost(est).reindex(acts).to_numpy(),
                        "cost_frequency": mean_cost(freq).reindex(acts).to_numpy(),
                        "cost_persistence": mean_cost(pers).reindex(acts).to_numpy()})
    wdf.to_csv(args.out / "weights.csv", index=False)
    lines = ["# ステージ55 B.5: 学習された重み(triplet・近い企業の負例・72 社すべて)", "",
             f"α = {a_full}(5 分割の交差検証で選択)。等重みは全科目 1 なので、比 = w。"
             "LOO の範囲 = 72 回の LOO の学習での 5〜95% 点。", "",
             "| 科目 | w(= 等重みとの比) | LOO の範囲 | 持続性 ρ(ステージ32) |",
             "|---|---:|---|---:|"]
    for r in wdf.sort_values("w", ascending=False).itertuples():
        lines.append(f"| {r.ja} | {r.w:.3f} | [{r.w_loo_lo:.3f}, {r.w_loo_hi:.3f}] |"
                     f" {r.rho1:.3f} |")
    lines += ["", "**過去の推定との順位相関(Spearman、29 科目)**", "",
              "| 過去の推定 | ρ | p |", "|---|---:|---:|"]
    for col, lab in (("rho1", "持続性 ρ_a(ステージ32)"),
                     ("cost_persistence", "持続性の重み(ステージ32 版 3)"),
                     ("cost_estimated", "推定ベースの重み(部分調整、ステージ24)"),
                     ("cost_frequency", "施策の頻度の重み(ステージ32 版 2)")):
        r = stats.spearmanr(wdf["w"], wdf[col], nan_policy="omit")
        lines.append(f"| {lab} | {r.statistic:+.3f} | {r.pvalue:.2g} |")
    lines += ["", f"重みが 1 から有意に離れた科目(LOO の 5〜95% 点が 1 を含まない): "
              f"{int(((wdf['w_loo_lo'] > 1) | (wdf['w_loo_hi'] < 1)).sum())} / {len(wdf)}", ""]
    (args.out / "weights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 生成用の重み(非操作 3 科目は nearest_na0 で 0 に上書きされる)
    base = yaml.safe_load(Path("conf/accounts/costs_equal.yaml").read_text(encoding="utf-8"))
    wmap = dict(zip(acts, w_full, strict=True))
    for b in base["basis_costs"]:
        if b["name"] in wmap:
            b["cost_increase"] = b["cost_decrease"] = float(wmap[b["name"]])
            b["source"] = "ステージ55: 再建群 72 社の遷移から学習(triplet、近い企業の負例)"
    base["name"] = "costs_learned"
    Path("conf/accounts/costs_learned.yaml").write_text(
        "# ステージ55: scripts/learn_stage55.py が生成(手で編集しない)。\n"
        + yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(open(args.out / "accuracy.md", encoding="utf-8").read())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
