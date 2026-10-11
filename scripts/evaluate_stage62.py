#!/usr/bin/env python
"""ステージ62 B: 第 1 項の重み m_a の効果(λ̃ = 3 で固定した比較と、設計ごとの再較正)。

規則は ``results/stage62/criteria.md`` §2。事前に ``qp_stage61.py --m-file`` で生成し、
``data/interim/stage62/all``(設計と base のリンク)に ``metrics_stage39.py`` をかける。

    uv run python scripts/evaluate_stage62.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from compare_stage39 import corr
from evaluate_stage61 import CAL, gen_firms, loss, real_targets

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

OUT = Path("results/stage62")
DATA = Path("data/interim/stage62")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DESIGNS = ("base", "paid3", "paid10", "paid30", "gap", "invgap", "invrho")
DISTS = {"l2": "等重み L2", "l1a": "非対称 L1"}
LAM_FIX = 3.0


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    e51._M = DATA / "metrics"
    e51._GEN = DATA / "all"
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / "base__l2_l1").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    rows: list[dict[str, Any]] = []
    firms: dict[tuple[str, str], pd.DataFrame] = {}
    reals = {}
    for cls in ("c2", "c1"):
        r, se, n_real = real_targets(cls, names)
        w = {m: 1.0 / se[m] for m in CAL}
        reals[cls] = (r, w, n_real)
        for des in DESIGNS:
            for d in DISTS:
                for lt in LAMS:
                    s = f"{des}__{d}_l{lt:g}"
                    f = gen_firms(cls, s, names, members)
                    f = f[~f.index.isin(e51.EXCL[cls])]
                    firms[(cls, s)] = f
                    gv = {"paid_dec": float(f["paid_dec"].mean()),
                          "re_up": float(f["re_up"].mean()),
                          "d_assets": float(f["d_assets"].median())}
                    cr = ctx[cls].comp[ctx[cls].comp["version"] == s]
                    vi = ctx[cls].viol
                    rows.append({"cls": cls, "design": des, "dist": d, "lam_tilde": lt,
                                 "setting": s, **gv,
                                 "paid_dec_norm": float(f["paid_dec_norm"].mean()),
                                 "n_changed": float(f["n_changed"].median()),
                                 "viol_firms": int(vi[vi["setting"] == s]["docID"].nunique()),
                                 "net_asset_improvement":
                                     float(cr["net_asset_improvement"].median()),
                                 "resolved": float(cr.loc[cr["insolvent_start"].astype(bool),
                                                          "insolvency_resolved"].mean()),
                                 "rank_corr": float(corr(ctx[cls].wide[s], ctx[cls].ref)),
                                 "loss": loss(gv, r, w)})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid_m.csv", index=False)
    rng = np.random.default_rng(0)
    lines = ["# ステージ62 B.3: 第 1 項の重み m_a の効果(λ̃ = 3 で固定)", "",
             "差は base(m = 1)に対する企業単位のペアのブートストラップ(2,000 回)。記号 = 純資産"
             "の改善・"
             "解消率・順位相関(ステージ51 の有意の定義)。実データ: 払込資本↓ 36%(class 2)・23%(c"
             "lass 1)。",
             ""]
    for cls in ("c2", "c1"):
        r, _w, n_real = reals[cls]
        for d, dlab in DISTS.items():
            lines += [f"## {cls}、{dlab}", "",
                      "| m の設計 | 払込資本↓(名目) | 差 [95%] | (正規化) | RE↑ | 純資産の改善 |"
                      " 解消率 "
                      "| 順位相関 | 変更科目数 | 違反(社) | L | 記号 |",
                      "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            base_s = f"base__{d}_l{LAM_FIX:g}"
            fb = firms[(cls, base_s)]
            for des in DESIGNS:
                s = f"{des}__{d}_l{LAM_FIX:g}"
                row = grid[(grid["cls"] == cls) & (grid["setting"] == s)].iloc[0]
                fa = firms[(cls, s)]
                dd = (fa["paid_dec"] - fb["paid_dec"].reindex(fa.index)).to_numpy()
                bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                if des == "base":
                    diff, sym = "—", "—"
                else:
                    c = ctx[cls].compare(s, base_s, rng)
                    sym = " ".join({1: "+", -1: "−", 0: "0"}[int(c[k][3])]
                                   for k in ("純資産の改善", "解消率", "順位相関"))
                    diff = f"{dd.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]"
                lines.append(f"| {des} | {row['paid_dec']:.0%} | {diff} |"
                             f" {row['paid_dec_norm']:.0%} "
                             f"| {row['re_up']:.0%} | {row['net_asset_improvement']:.3f} "
                             f"| {row['resolved']:.2f} | {row['rank_corr']:+.3f} "
                             f"| {row['n_changed']:.0f} | {row['viol_firms']} | {row['loss']:.2f} "
                             f"| {sym} |")
            lines += [f"(実データ: 払込資本↓ {r['paid_dec']:.0%}、RE↑ {r['re_up']:.0%}、変更科目数 "
                      f"{n_real:.0f})", ""]
    # B.4: 設計ごとの再較正
    lines += ["# B.4: m を変えた後の再較正(λ̃* と、違反 0 を制約にした λ̃*_0)", "",
              "| class | 距離 | m の設計 | λ̃* | L | 違反(社) | λ̃*_0 | L | 純資産の改善 | 払込"
              "資本↓ "
              "| 順位相関 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary: dict[str, Any] = {}
    for cls in ("c2", "c1"):
        for d, dlab in DISTS.items():
            for des in DESIGNS:
                sub = grid[(grid["cls"] == cls) & (grid["dist"] == d) & (grid["design"] == des)]
                b = sub.loc[sub["loss"].idxmin()]
                ok = sub[sub["viol_firms"] == 0]
                b0 = ok.loc[ok["loss"].idxmin()] if len(ok) else None
                summary[f"{cls}_{d}_{des}"] = {"star": float(b["lam_tilde"]),
                                               "star0": None if b0 is None
                                               else float(b0["lam_tilde"])}
                tail = "| なし | — | — | — | — |"
                if b0 is not None:
                    tail = (f"| **{b0['lam_tilde']:g}** | {b0['loss']:.2f} "
                            f"| {b0['net_asset_improvement']:.3f} | {b0['paid_dec']:.0%} "
                            f"| {b0['rank_corr']:+.3f} |")
                lines.append(f"| {cls} | {dlab} | {des} | {b['lam_tilde']:g} | {b['loss']:.2f} "
                             f"| {b['viol_firms']} {tail}")
    lines.append("")
    (OUT / "m_effect.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary_m.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
